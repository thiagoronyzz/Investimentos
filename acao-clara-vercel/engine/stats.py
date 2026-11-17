"""Ferramentas estatísticas compartilhadas pelos núcleos do motor.

Tudo implementado com numpy/scipy/sklearn — sem IA generativa. Inclui:
correlações (Pearson, Spearman e móvel), teste ADF (cointegração Engle–Granger),
AR(p) com seleção por AIC, GARCH(1,1) por máxima verossimilhança, Monte Carlo e
correção de múltiplos testes de Benjamini–Hochberg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Séries básicas
# ---------------------------------------------------------------------------
def returns(close: pd.Series | np.ndarray, periods: int = 1) -> np.ndarray:
    values = np.asarray(pd.Series(close).astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.full_like(values, np.nan, dtype=float)
        out[periods:] = values[periods:] / values[:-periods] - 1.0
    return out


def log_returns(close: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(pd.Series(close).astype(float))
    out = np.full_like(values, np.nan, dtype=float)
    out[1:] = np.log(values[1:] / values[:-1])
    return out


def forward_returns(close: pd.Series | np.ndarray, horizon: int) -> np.ndarray:
    """Retorno realizado `horizon` barras à frente (NaN onde o futuro é desconhecido)."""
    values = np.asarray(pd.Series(close).astype(float))
    out = np.full_like(values, np.nan, dtype=float)
    out[:-horizon] = values[horizon:] / values[:-horizon] - 1.0
    return out


def window_returns(values: np.ndarray, window: int, lag: int = 0) -> np.ndarray:
    """Retorno acumulado em `window` barras, opcionalmente defasado em `lag` barras.

    Comparar janelas iguais é o que dá sentido econômico ao teste: o retorno da
    ação em h pregões é explicado pela variação da variável externa na mesma
    janela (e não pela variação de um único dia).
    """
    array = np.asarray(values, dtype=float)
    out = np.full_like(array, np.nan, dtype=float)
    if len(array) <= window:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = array[window:] / array[:-window] - 1.0
    if lag:
        shifted = np.full_like(out, np.nan, dtype=float)
        shifted[lag:] = out[:-lag]
        return shifted
    return out


def zscore(series: pd.Series | np.ndarray, window: int = 756) -> np.ndarray:
    values = np.asarray(pd.Series(series).astype(float))
    frame = pd.Series(values)
    mean = frame.rolling(window, min_periods=max(30, window // 6)).mean()
    std = frame.rolling(window, min_periods=max(30, window // 6)).std()
    out = ((frame - mean) / std.replace(0.0, np.nan)).to_numpy()
    return np.where(np.isfinite(out), out, np.nan)


def slope_per_period(series: pd.Series | np.ndarray, window: int = 21) -> float:
    """Velocidade da mudança: inclinação média por período nos últimos `window`."""
    values = np.asarray(pd.Series(series).astype(float).dropna().to_numpy()[-window:])
    if len(values) < 5:
        return float("nan")
    x = np.arange(len(values), dtype=float)
    if np.nanstd(values) == 0:
        return 0.0
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def annualized_vol(returns: np.ndarray, window: int | None = None) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if window:
        values = values[-window:]
    if len(values) < 10:
        return float("nan")
    return float(np.std(values, ddof=1) * math.sqrt(252))


# ---------------------------------------------------------------------------
# Correlação
# ---------------------------------------------------------------------------
def correlation(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> tuple[float, float, int]:
    """Retorna (coeficiente, p-valor, n). Valores ausentes são descartados aos pares."""
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    n = len(frame)
    if n < 20 or frame.x.std() == 0 or frame.y.std() == 0:
        return float("nan"), float("nan"), n
    if method == "spearman":
        coef, pvalue = sps.spearmanr(frame.x, frame.y)
    else:
        coef, pvalue = sps.pearsonr(frame.x, frame.y)
    return float(coef), float(pvalue), int(n)


def rolling_correlation(x: np.ndarray, y: np.ndarray, window: int = 252) -> np.ndarray:
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    return frame.x.rolling(window, min_periods=window // 2).corr(frame.y).to_numpy()


def correlation_stability(x: np.ndarray, y: np.ndarray, periods: int = 3) -> float:
    """Fração de subperíodos com correlação no mesmo sinal e >= 50% da total."""
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    total, _, _ = correlation(frame.x.to_numpy(), frame.y.to_numpy())
    if not np.isfinite(total) or len(frame) < 60:
        return float("nan")
    chunks = np.array_split(np.arange(len(frame)), periods)
    hits = 0
    for chunk in chunks:
        if len(chunk) < 20:
            continue
        coef, _, _ = correlation(frame.x.to_numpy()[chunk], frame.y.to_numpy()[chunk])
        if np.isfinite(coef) and np.sign(coef) == np.sign(total) and abs(coef) >= 0.5 * abs(total):
            hits += 1
    return hits / max(1, len(chunks))


# ---------------------------------------------------------------------------
# Estacionariedade / cointegração
# ---------------------------------------------------------------------------
# Âncoras (estatística t do ADF x p-valor) para o caso com constante e n grande.
_ADF_ANCHORS = (
    (-3.90, 0.001), (-3.44, 0.010), (-3.13, 0.025), (-2.87, 0.050), (-2.57, 0.100),
    (-2.23, 0.200), (-1.70, 0.400), (-1.28, 0.600), (-0.79, 0.800), (-0.07, 0.950), (0.60, 0.999),
)


def _adf_pvalue(tstat: float) -> float:
    if tstat <= _ADF_ANCHORS[0][0]:
        return _ADF_ANCHORS[0][1] * 0.5
    if tstat >= _ADF_ANCHORS[-1][0]:
        return 0.999
    for (t0, p0), (t1, p1) in zip(_ADF_ANCHORS, _ADF_ANCHORS[1:]):
        if t0 <= tstat <= t1:
            weight = (tstat - t0) / (t1 - t0)
            return float(math.exp(math.log(p0) + weight * (math.log(p1) - math.log(p0))))
    return 0.5


def adf_test(series: np.ndarray, max_lags: int = 8) -> dict:
    """Teste ADF (Dickey–Fuller aumentado) com constante, lag escolhido por AIC.

    O p-valor é uma aproximação por interpolação das tabelas de MacKinnon;
    serve para ordenar evidências, não como valor exato.
    """
    values = np.asarray(pd.Series(series).astype(float).dropna().to_numpy())
    if len(values) < 60:
        return {"ok": False, "reason": "série curta demais"}
    best = None
    for lags in range(0, max_lags + 1):
        delta = np.diff(values)
        y = delta[lags:]
        if len(y) < 40:
            continue
        columns = [np.ones_like(y), values[lags:-1]]
        for lag in range(1, lags + 1):
            columns.append(delta[lags - lag:len(delta) - lag])
        design = np.column_stack(columns)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ beta
        dof = len(y) - design.shape[1]
        if dof <= 2:
            continue
        sigma2 = float(residuals @ residuals) / dof
        try:
            covariance = sigma2 * np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError:
            continue
        stderr = math.sqrt(max(covariance[1, 1], 1e-18))
        tstat = float(beta[1] / stderr)
        aic = len(y) * math.log(max(sigma2, 1e-18)) + 2 * design.shape[1]
        if best is None or aic < best["aic"]:
            best = {"aic": aic, "tstat": tstat, "lags": lags, "n": len(y)}
    if best is None:
        return {"ok": False, "reason": "regressão ADF não convergiu"}
    pvalue = _adf_pvalue(best["tstat"])
    return {"ok": True, "tstat": best["tstat"], "pvalue": pvalue, "lags": best["lags"], "n": best["n"], "stationary": pvalue < 0.05}


def engle_granger(stock: np.ndarray, driver: np.ndarray, max_lags: int = 8) -> dict:
    """Cointegração Engle–Granger: ADF sobre o resíduo da regressão em níveis."""
    frame = pd.DataFrame({"y": np.asarray(stock, dtype=float), "x": np.asarray(driver, dtype=float)}).dropna()
    if len(frame) < 120:
        return {"ok": False, "reason": "histórico insuficiente para cointegração"}
    design = np.column_stack([np.ones(len(frame)), frame.x.to_numpy()])
    beta, *_ = np.linalg.lstsq(design, frame.y.to_numpy(), rcond=None)
    residuals = frame.y.to_numpy() - design @ beta
    result = adf_test(residuals, max_lags=max_lags)
    result.update({"ok": result.get("ok", False), "beta": float(beta[1]), "cointegrated": bool(result.get("stationary", False))})
    return result


# ---------------------------------------------------------------------------
# Modelos de série temporal
# ---------------------------------------------------------------------------
def ar_forecast(series: np.ndarray, horizon: int, max_order: int = 5) -> dict:
    """AR(p) em retornos com ordem escolhida por AIC; retorna retorno acumulado previsto."""
    values = np.asarray(pd.Series(series).astype(float).dropna().to_numpy())
    if len(values) < 120:
        return {"ok": False, "reason": "série curta demais"}
    best = None
    for order in range(1, max_order + 1):
        if len(values) <= order + 40:
            continue
        design = np.column_stack([np.ones(len(values) - order)] + [values[order - lag - 1:-lag - 1] for lag in range(order)])
        target = values[order:]
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ beta
        sigma2 = float(residuals @ residuals) / max(1, len(target) - design.shape[1])
        aic = len(target) * math.log(max(sigma2, 1e-18)) + 2 * design.shape[1]
        if best is None or aic < best["aic"]:
            best = {"aic": aic, "order": order, "beta": beta, "history": values, "sigma": math.sqrt(sigma2)}
    if best is None:
        return {"ok": False, "reason": "AR não estimado"}
    order, beta = best["order"], best["beta"]
    window = list(best["history"][-order:])
    total, path = 0.0, []
    for _ in range(horizon):
        x = np.array([1.0] + window[::-1][:order], dtype=float)
        step = float(x @ beta)
        window = (window + [step])[-order:]
        total += step
        path.append(total)
    return {
        "ok": True,
        "order": order,
        "cumulative_return": float(np.clip(total, -0.95, 3.0)),
        "sigma": float(best["sigma"]),
        "path": [float(v) for v in path],
        "aic": float(best["aic"]),
    }


@dataclass
class GarchFit:
    ok: bool
    omega: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    long_run_vol: float = float("nan")
    current_vol: float = float("nan")
    forecast_vol: float = float("nan")
    reason: str = ""

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "persistence": round(self.alpha + self.beta, 4),
            "long_run_vol_anual": _round(self.long_run_vol),
            "current_vol_anual": _round(self.current_vol),
            "forecast_vol_anual": _round(self.forecast_vol),
            "reason": self.reason,
        }


def garch11(returns: np.ndarray, horizon: int) -> GarchFit:
    """GARCH(1,1) por máxima verossimilhança; projeta a volatilidade no horizonte.

    A variância incondicional é fixada na variância amostral (ω = σ²·(1−α−β)),
    simplificação usual que torna a estimação estável; α e β são escolhidos por
    uma grade inicial seguida de refinamento por Nelder–Mead.
    """
    values = np.asarray(pd.Series(returns).astype(float).dropna().to_numpy())
    if len(values) < 250:
        return GarchFit(ok=False, reason="histórico insuficiente para GARCH")
    variance = float(np.var(values, ddof=1))
    if variance <= 0:
        return GarchFit(ok=False, reason="variância nula")

    def variance_path(alpha: float, beta: float):
        if alpha <= 0.0 or beta <= 0.0 or alpha + beta >= 0.999:
            return None
        omega = variance * (1.0 - alpha - beta)
        sigma2 = np.empty_like(values)
        sigma2[0] = variance
        for index in range(1, len(values)):
            sigma2[index] = omega + alpha * values[index - 1] ** 2 + beta * sigma2[index - 1]
            if not np.isfinite(sigma2[index]) or sigma2[index] <= 0:
                return None
        return sigma2, omega

    def negloglik(alpha: float, beta: float) -> float:
        built = variance_path(alpha, beta)
        if built is None:
            return 1e10
        sigma2, _ = built
        return float(0.5 * np.sum(math.log(2 * math.pi) + np.log(sigma2) + values**2 / sigma2))

    best = None
    for alpha in (0.01, 0.03, 0.06, 0.09, 0.14, 0.20):
        for beta in (0.70, 0.80, 0.88, 0.93, 0.97):
            if alpha + beta >= 0.995:
                continue
            value = negloglik(alpha, beta)
            if best is None or value < best[0]:
                best = (value, alpha, beta)
    if best is None:
        return GarchFit(ok=False, reason="grade inicial vazia")
    try:
        refined = optimize.minimize(
            lambda params: negloglik(params[0], params[1]),
            [best[1], best[2]],
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-3},
        )
        alpha, beta = float(refined.x[0]), float(refined.x[1])
        if refined.fun < best[0] and 0 < alpha < 0.5 and 0 < beta < 0.995 and alpha + beta < 0.995:
            pass
        else:
            alpha, beta = best[1], best[2]
    except Exception:  # pragma: no cover - falha numérica
        alpha, beta = best[1], best[2]
    built = variance_path(alpha, beta)
    if built is None:
        return GarchFit(ok=False, reason="parâmetros fora do domínio")
    sigma2, omega = built
    long_run = math.sqrt(variance)
    total, current = 0.0, omega + alpha * values[-1] ** 2 + beta * sigma2[-1]
    for _ in range(max(1, horizon)):
        total += current
        current = omega + (alpha + beta) * current
    return GarchFit(
        ok=True,
        omega=omega,
        alpha=alpha,
        beta=beta,
        long_run_vol=long_run * math.sqrt(252),
        current_vol=math.sqrt(sigma2[-1]) * math.sqrt(252),
        forecast_vol=math.sqrt(total / max(1, horizon)) * math.sqrt(252),
    )


def monte_carlo(
    daily_drift: float,
    daily_vol: float,
    steps: int,
    paths: int = 4000,
    seed: int = 0,
    tail_df: float = 6.0,
    vol_persistence: float = 0.0,
) -> dict:
    """Simula trajetórias de preço (GBM com choques Student-t e vol persistente)."""
    rng = np.random.default_rng(seed)
    steps = max(1, int(steps))
    paths = max(200, int(paths))
    daily_vol = max(daily_vol, 1e-5)
    base = rng.standard_t(tail_df, size=(paths, steps))
    base /= math.sqrt(tail_df / (tail_df - 2))
    if vol_persistence:
        # Choque de volatilidade por trajetória que reverte à média ao longo do tempo.
        level = rng.normal(0.0, vol_persistence, size=paths)
        for index in range(steps):
            base[:, index] *= np.clip(1.0 + level, 0.35, 2.5)
            level = level * 0.97
    steps_returns = daily_drift + daily_vol * base
    log_paths = np.cumsum(np.log1p(np.clip(steps_returns, -0.6, 3.0)), axis=1)
    wealth = np.exp(log_paths)
    final = wealth[:, -1]
    running_max = np.maximum.accumulate(wealth, axis=1)
    drawdowns = wealth / running_max - 1.0
    return {
        "final": final,
        "mean": float(np.mean(final)),
        "median": float(np.median(final)),
        "std": float(np.std(final)),
        "p_up": float(np.mean(final > 1.0)),
        "quantiles": {f"p{int(q*100)}": float(np.quantile(final, q)) for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)},
        "max_drawdown_mean": float(np.mean(drawdowns.min(axis=1))),
        "max_drawdown_p95": float(np.quantile(drawdowns.min(axis=1), 0.05)),
        "var95": float(np.quantile(final, 0.05) - 1.0),
        "cvar95": float(np.mean(final[final <= np.quantile(final, 0.05)]) - 1.0),
        "paths": paths,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Regressões utilitárias
# ---------------------------------------------------------------------------
def ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> dict:
    frame = pd.DataFrame(x)
    frame["_y"] = y
    frame = frame.dropna()
    if len(frame) < 60:
        return {"ok": False, "reason": "amostras insuficientes"}
    design = frame.drop(columns=["_y"]).to_numpy()
    target = frame["_y"].to_numpy()
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(design, target)
    residuals = target - model.predict(design)
    ss_res = float(residuals @ residuals)
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return {
        "ok": True,
        "model": model,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(frame)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "intercept": float(model[-1].intercept_),
        "coefficients": (model[-1].coef_ / model[0].scale_).tolist(),
    }


def shrink_forecast(raw: float, oos_r2: float, horizon_sigma: float, floor: float = 0.25) -> tuple[float, dict]:
    """Encolhe a previsão de uma regressão pelo seu desempenho fora da amostra.

    Duas defesas contra regressão espúria:

    1. **Encolhimento por habilidade** — o sinal é multiplicado por
       ``floor + (1 - floor) · max(0, R² fora da amostra)``. Um modelo sem
       poder explicativo real conserva apenas o piso, em vez de empurrar a
       projeção com toda a força do ajuste dentro da amostra.
    2. **Teto econômico** — nenhuma condição isolada justifica, sozinha, um
       movimento maior que um desvio-padrão do retorno do próprio horizonte.

    Retorna o valor ajustado e o detalhe da transformação (para auditoria).
    """
    skill = float(np.clip(oos_r2, 0.0, 1.0)) if np.isfinite(oos_r2) else 0.0
    factor = floor + (1.0 - floor) * skill
    shrunk = float(raw) * factor
    sigma = float(horizon_sigma) if np.isfinite(horizon_sigma) and horizon_sigma > 0 else float("nan")
    capped = shrunk
    cap_hit = False
    if np.isfinite(sigma):
        capped = float(np.clip(shrunk, -sigma, sigma))
        cap_hit = abs(capped - shrunk) > 1e-12
    if not np.isfinite(capped):
        capped = 0.0
    return capped, {
        "raw": float(raw),
        "oos_r2": skill,
        "shrink_factor": round(factor, 4),
        "shrunk": shrunk,
        "horizon_sigma": sigma if np.isfinite(sigma) else None,
        "capped_at_sigma": cap_hit,
        "final": capped,
    }


def purged_split(n: int, horizon: int, train_ratio: float = 0.7) -> tuple[int, int]:
    """Divisão cronológica treino/teste **com purga** entre os blocos.

    Quando o rótulo é o retorno dos próximos `horizon` pregões, os últimos
    `horizon` exemplos do treino têm rótulos que se sobrepõem no tempo ao
    período de teste. Sem purga, o "fora da amostra" vaza informação do treino
    e o R² fica otimista — e é justamente esse R² que alimenta os pesos
    dinâmicos da Camada B e a confiança dos núcleos.

    Retorna `(fim_do_treino, inicio_do_teste)`: treine com `[:fim]` e teste com
    `[inicio:]`. O intervalo entre eles fica vazio de propósito.
    """
    n = int(n)
    horizon = max(0, int(horizon))
    if n < 40:
        return n, n
    min_test = max(20, int(n * 0.10))
    min_train = max(20, int(n * 0.30))
    train_end = int(n * train_ratio)
    # 1) A purga completa cabe sem sacrificar o teste: caso ideal.
    if n - (train_end + horizon) >= min_test:
        return train_end, train_end + horizon
    # 2) Cabe purga completa, mas é preciso recuar o treino.
    candidate = n - min_test - horizon
    if candidate >= min_train:
        return candidate, candidate + horizon
    # 3) Não há espaço para a purga completa: reduz o quanto der, mantendo
    #    treino e teste mínimos (horizontes longos com histórico curto).
    test_start = n - min_test
    train_end = max(min_train, test_start - horizon)
    return min(train_end, test_start - 1), test_start


def out_of_sample_r2(model, x: np.ndarray, y: np.ndarray) -> float:
    frame = pd.DataFrame(x)
    frame["_y"] = y
    frame = frame.dropna()
    if len(frame) < 20:
        return float("nan")
    design = frame.drop(columns=["_y"]).to_numpy()
    target = frame["_y"].to_numpy()
    if np.var(target) == 0:
        return float("nan")
    predictions = model.predict(design)
    return float(1.0 - ((target - predictions) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def lasso_selection(x: np.ndarray, y: np.ndarray, names: list[str], alpha: float = 0.002) -> dict:
    frame = pd.DataFrame(x)
    frame["_y"] = y
    frame = frame.dropna()
    if len(frame) < 80 or frame.shape[1] < 2:
        return {"ok": False, "reason": "amostras insuficientes"}
    design = frame.drop(columns=["_y"]).to_numpy()
    target = frame["_y"].to_numpy()
    model = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000))
    model.fit(design, target)
    coefficients = (model[-1].coef_ / model[0].scale_).tolist()
    selected = [names[index] for index, value in enumerate(coefficients) if abs(value) > 1e-8]
    return {"ok": True, "selected": selected, "coefficients": coefficients, "n": int(len(frame))}


def logistic_probability(x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> dict:
    frame = pd.DataFrame(x)
    frame["_y"] = y
    frame = frame.dropna()
    target = frame["_y"].to_numpy()
    if len(frame) < 120 or len(np.unique(target)) < 2:
        return {"ok": False, "reason": "amostras insuficientes ou rótulo único"}
    design = frame.drop(columns=["_y"]).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=penalty))
    model.fit(design, target)
    return {"ok": True, "model": model, "n": int(len(frame)), "base_rate": float(target.mean())}


# ---------------------------------------------------------------------------
# Múltiplos testes
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Retorna máscara de rejeição pelo procedimento BH (controle de FDR)."""
    pvalues = np.asarray(pvalues, dtype=float)
    mask = np.isfinite(pvalues)
    rejected = np.zeros(len(pvalues), dtype=bool)
    valid = pvalues[mask]
    if valid.size == 0:
        return rejected
    order = np.argsort(valid)
    ranked = valid[order]
    m = len(ranked)
    thresholds = (np.arange(1, m + 1) / m) * q
    passed = ranked <= thresholds
    if not passed.any():
        return rejected
    last = int(np.max(np.flatnonzero(passed)))
    selected = np.zeros(m, dtype=bool)
    selected[: last + 1] = True
    restored = np.zeros(m, dtype=bool)
    restored[order] = selected
    rejected[mask] = restored
    return rejected


def _round(value: float, digits: int = 6):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)
