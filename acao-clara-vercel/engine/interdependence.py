"""Núcleo 03 — Interdependência entre mercados e ativos.

Constrói a matriz de dependências da empresa (correlações contemporânea,
Spearman, móvel e defasada; cointegração Engle–Granger), estima a
sensibilidade por regressão multivariada

    R = α + β1·X1 + β2·X2 + ... + βn·Xn

e converte as expectativas estatísticas de cada variável externa em retorno
esperado para a ação. Correlação não é causalidade: por isso a saída informa
estabilidade da relação no tempo, significância e desempenho fora da amostra.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, stats
from .context import Context
from .estimate import CoreEstimate

RIDGE_ALPHA = 15.0
SHRINKAGE = 0.5           # encolhimento aplicado às expectativas das variáveis externas
LAG_PREFERENCE = 0.05     # ganho mínimo de correlação para preferir uma defasagem
MAX_DRIVERS = 8


def load_driver_spec() -> dict:
    try:
        return json.loads((config.DATA_DIR / "drivers.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"registry": {}, "sector_defaults": {}, "companies": {}}


def drivers_for(symbol: str, sector: str | None = None) -> list[str]:
    """Variáveis externas programadas para a empresa (ou para o setor dela)."""
    spec = load_driver_spec()
    ticker = symbol.upper().removesuffix(".SA")
    company = spec.get("companies", {}).get(ticker)
    if company:
        return list(company.get("drivers", []))[:MAX_DRIVERS]
    if sector:
        return list(spec.get("sector_defaults", {}).get(sector, []))[:MAX_DRIVERS]
    return ["ibov", "usdbrl", "sp500", "vix"][:MAX_DRIVERS]


def driver_label(key: str) -> str:
    spec = load_driver_spec()
    return spec.get("registry", {}).get(key, {}).get("label", key)


def resolve_series(data, keys: list[str], macro: dict[str, pd.Series]) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Materializa as séries dos fatores pedidos (mercado/commodity ou macro)."""
    spec = load_driver_spec().get("registry", {})
    out: dict[str, pd.Series] = {}
    kinds: dict[str, str] = {}
    symbols = {key: spec[key]["symbol"] for key in keys if key in spec and spec[key].get("kind") == "factor"}
    frames = data.histories(symbols.values(), years=6) if symbols else {}
    reverse = {symbol: key for key, symbol in symbols.items()}
    for symbol, frame in frames.items():
        key = reverse.get(symbol)
        if key:
            out[key] = pd.Series(frame["close"].to_numpy(dtype=float), index=pd.DatetimeIndex([pd.Timestamp(d) for d in frame["date"]]))
            kinds[key] = "factor"
    for key in keys:
        if key in out:
            continue
        entry = spec.get(key, {})
        macro_key = entry.get("key", key) if entry.get("kind") == "macro" else key
        series = macro.get(macro_key)
        if series is not None and len(series.dropna()):
            out[key] = series.astype(float)
            kinds[key] = "macro"
    return out, kinds


def _median_gap_days(index) -> float:
    """Intervalo médio, em dias, entre observações da série."""
    try:
        stamps = pd.DatetimeIndex(index)
    except (TypeError, ValueError):
        return 1.0
    if len(stamps) < 3:
        return 1.0
    gaps = np.diff(stamps.values).astype("timedelta64[D]").astype(float)
    return float(np.median(gaps))


def expected_move(series: pd.Series, horizon_days: int) -> dict:
    """Expectativa estatística de variação da variável externa (AR com reversão à média)."""
    values = series.dropna()
    if len(values) < 60:
        return {"ok": False, "reason": "histórico curto"}
    array = values.to_numpy(dtype=float)
    median_gap = _median_gap_days(values.index)
    stepped = median_gap > 15 or len(np.unique(array[-120:])) < 20
    if stepped:
        monthly = values.resample("MS").last().dropna()
        if len(monthly) < 24:
            return {"ok": False, "reason": "série mensal curta"}
        changes = np.diff(monthly.to_numpy(dtype=float))
        steps = max(1, int(round(horizon_days / 30.0)))
        lag = changes[:-1]
        target = changes[1:]
        design = np.column_stack([np.ones_like(lag), lag])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        current = changes[-1]
        total = 0.0
        for _ in range(steps):
            current = beta[0] + beta[1] * current
            total += current
        base = float(array[-1])
        return {
            "ok": True,
            "method": f"AR(1) em variações mensais (série com intervalo médio de {median_gap:.0f} dias)",
            "expected_pct": float(np.clip(SHRINKAGE * total / base, -0.9, 2.0)) if base else 0.0,
            "expected_change": float(total),
            "steps": steps,
        }
    logs = np.log(array)
    log_returns = np.diff(logs)
    result = stats.ar_forecast(log_returns, horizon=max(1, int(round(horizon_days * 252 / 365))))
    if not result.get("ok"):
        return {"ok": False, "reason": result.get("reason", "AR não estimado")}
    return {
        "ok": True,
        "method": f"AR({result['order']}) em log-retornos diários",
        "expected_pct": float(np.clip(SHRINKAGE * result["cumulative_return"], -0.9, 2.0)),
        "raw_ar_return": float(result["cumulative_return"]),
        "shrunk_by": SHRINKAGE,
        "order": result["order"],
    }


def dependency_matrix(context: Context, horizon_td: int) -> dict[str, dict]:
    """Correlações, defasagens, estabilidade e cointegração de cada variável externa.

    As correlações comparam janelas iguais: o retorno da ação em `horizon_td`
    pregões contra a variação da variável externa na mesma janela, com
    defasagens de 0 a 60 dias. Também é registrada a correlação diária.
    """
    close = context.close
    stock_window = stats.window_returns(close.to_numpy(), horizon_td)
    stock_daily = stats.log_returns(close.to_numpy())
    matrix: dict[str, dict] = {}
    for key, series in context.drivers.items():
        aligned = context.aligned_driver(key)
        if aligned is None:
            continue
        driver_window = stats.window_returns(aligned, horizon_td)
        driver_daily = stats.log_returns(aligned)
        pearson, pvalue, n = stats.correlation(stock_window, driver_window)
        spearman, _, _ = stats.correlation(stock_window, driver_window, method="spearman")
        daily_pearson, _, _ = stats.correlation(stock_daily, driver_daily)
        rolling = stats.rolling_correlation(stock_window, driver_window, window=config.CORR_WINDOW)
        rolling_values = rolling[np.isfinite(rolling)]
        lags = {}
        for lag in config.LAGS_TESTED:
            coef, pval, _ = stats.correlation(stock_window, stats.window_returns(aligned, horizon_td, lag))
            lags[lag] = {"corr": None if not np.isfinite(coef) else round(float(coef), 4), "pvalue": None if not np.isfinite(pval) else round(float(pval), 5)}
        best_lag = max(lags.items(), key=lambda item: abs(item[1]["corr"] or 0.0))[0]
        if abs(lags[best_lag]["corr"] or 0.0) - abs(lags[0]["corr"] or 0.0) < LAG_PREFERENCE:
            best_lag = 0
        cointegration = stats.engle_granger(np.log(close.to_numpy()), np.log(aligned))
        matrix[key] = {
            "label": driver_label(key),
            "window": int(horizon_td),
            "pearson": None if not np.isfinite(pearson) else round(float(pearson), 4),
            "pearson_pvalue": None if not np.isfinite(pvalue) else round(float(pvalue), 5),
            "spearman": None if not np.isfinite(spearman) else round(float(spearman), 4),
            "pearson_daily": None if not np.isfinite(daily_pearson) else round(float(daily_pearson), 4),
            "n": int(n),
            "rolling_corr_last": None if rolling_values.size == 0 else round(float(rolling_values[-1]), 4),
            "rolling_corr_std": None if rolling_values.size == 0 else round(float(np.std(rolling_values)), 4),
            "rolling_corr_min": None if rolling_values.size == 0 else round(float(np.min(rolling_values)), 4),
            "rolling_corr_max": None if rolling_values.size == 0 else round(float(np.max(rolling_values)), 4),
            "stability": _round_or_none(stats.correlation_stability(stock_window, driver_window)),
            "lags": {str(lag): value for lag, value in lags.items()},
            "best_lag": int(best_lag),
            "cointegration": {
                "ok": cointegration.get("ok", False),
                "pvalue": _round_or_none(cointegration.get("pvalue")),
                "cointegrated": bool(cointegration.get("cointegrated", False)),
            },
            "expected_move": expected_move(series, context.horizon_days),
        }
    return matrix


def _round_or_none(value, digits: int = 5):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else round(value, digits)


def run(context: Context) -> CoreEstimate:
    estimate = CoreEstimate(core="interdependence", label="Interdependência de mercados")
    if not context.drivers:
        return CoreEstimate.skipped("interdependence", estimate.label, "Nenhuma variável externa disponível para esta empresa.")
    matrix = dependency_matrix(context, context.horizon_td)
    estimate.data["matrix"] = matrix
    if not matrix:
        return CoreEstimate.skipped("interdependence", estimate.label, "As variáveis externas não se alinharam ao calendário do ativo.")
    columns: dict[str, np.ndarray] = {}
    for key, item in matrix.items():
        aligned = context.aligned_driver(key)
        if aligned is None:
            continue
        # Regressores na mesma janela do alvo: variação da variável em h pregões.
        columns[key] = stats.window_returns(aligned, context.horizon_td, int(item.get("best_lag", 0)))
    design_frame = pd.DataFrame(columns)
    target = stats.window_returns(context.close.to_numpy(), context.horizon_td)
    frame = design_frame.copy()
    frame["_y"] = target
    frame = frame.dropna()
    if len(frame) < config.REGRESSION_MIN_SAMPLES:
        estimate.status = "skip"
        estimate.notes.append(f"Histórico conjunto insuficiente ({len(frame)} observações) para a regressão multivariada.")
        return estimate
    design = frame.drop(columns=["_y"]).to_numpy()
    outcome = frame["_y"].to_numpy()
    train_end, test_start = stats.purged_split(len(frame), context.horizon_td)
    train = stats.ridge_regression(design[:train_end], outcome[:train_end], alpha=RIDGE_ALPHA)
    r2_oos = (
        stats.out_of_sample_r2(train["model"], design[test_start:], outcome[test_start:])
        if train.get("ok")
        else float("nan")
    )
    full = stats.ridge_regression(design, outcome, alpha=RIDGE_ALPHA)
    if not full.get("ok"):
        return CoreEstimate.skipped("interdependence", estimate.label, "Regressão multivariada não pôde ser estimada.")
    # `coefficients` já vem por unidade da variável (coeficiente bruto da regressão).
    betas = np.asarray(full["coefficients"], dtype=float)
    names = list(frame.drop(columns=["_y"]).columns)
    contributions: dict[str, float] = {}
    for index, key in enumerate(names):
        move = matrix.get(key, {}).get("expected_move", {})
        if not move.get("ok"):
            continue
        beta = float(betas[index])
        contributions[key] = beta * float(move.get("expected_pct", 0.0))
        matrix[key]["beta"] = round(beta, 5)
        matrix[key]["contribution_pp"] = round(contributions[key] * 100, 3)
    raw_expected = float(sum(contributions.values()))
    expected, shrink_detail = stats.shrink_forecast(
        raw_expected, float(r2_oos) if np.isfinite(r2_oos) else 0.0, float(np.nanstd(outcome))
    )
    estimate.data["shrinkage"] = shrink_detail
    for key, value in contributions.items():
        move = matrix[key]["expected_move"]
        estimate.add_signal(
            f"{matrix[key]['label']} (defasagem {matrix[key].get('best_lag', 0)}d)",
            value * 100,
            "alta" if value > 0 else "baixa",
            f"beta {matrix[key].get('beta', 0):+.3f} · variação esperada da variável {move.get('expected_pct', 0):+.2%} ({move.get('method', '')})",
        )
    stabilities = [item.get("stability") for item in matrix.values() if item.get("stability") is not None]
    stability_ratio = float(np.mean(stabilities)) if stabilities else 0.0
    cointegrated_ratio = float(np.mean([item["cointegration"].get("cointegrated", False) for item in matrix.values()])) if matrix else 0.0
    estimate.expected_return = expected
    estimate.sigma = float(full["residual_std"])
    r2_oos_value = float(r2_oos) if np.isfinite(r2_oos) else 0.0
    estimate.confidence = round(
        float(
            np.clip(
                0.12
                + 2.5 * max(0.0, r2_oos_value)
                - 0.15 * min(1.0, max(0.0, -r2_oos_value))
                + 0.25 * stability_ratio
                + 0.10 * cointegrated_ratio
                + 0.05 * min(1.0, len(names) / 6.0),
                0.0,
                0.85,
            )
        ),
        3,
    )
    estimate.data.update(
        {
            "model": "regressão Ridge multivariada com defasagem escolhida por correlação",
            "alpha": RIDGE_ALPHA,
            "r2_in_sample": round(float(full["r2"]), 5),
            "r2_out_of_sample": None if not np.isfinite(r2_oos) else round(float(r2_oos), 5),
            "n": int(full["n"]),
            "variables": names,
            "shrunk_by": SHRINKAGE,
            "stability_ratio": round(stability_ratio, 3),
        }
    )
    estimate.notes.append(
        "As expectativas das variáveis externas vêm de modelos AR com reversão à média e são encolhidas pela metade; "
        "correlação histórica não garante causalidade nem estabilidade no futuro."
    )
    estimate.sort_signals()
    estimate.status = "ok"
    return estimate
