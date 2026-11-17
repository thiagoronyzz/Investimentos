"""Núcleo 01 — Macroeconomia e indicadores econômicos.

Para cada indicador o motor calcula três coisas, como exige a metodologia:
  1. a situação atual (nível e desvio em relação à própria média histórica);
  2. a direção (inclinação da série normalizada);
  3. a velocidade (aceleração, derivada da inclinação).

A sensibilidade da ação a cada indicador é *estimada* por regressão sobre os
retornos futuros (janela móvel, teste fora da amostra). O sinal teórico só é
usado quando não há histórico suficiente para estimar — e fica marcado como
tal na saída. O núcleo devolve apenas o retorno atribuído às condições macro
*atuais* (desvio em relação ao normal); o retorno médio histórico é tratado
pelo núcleo estatístico, para não contar a mesma informação duas vezes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats
from .context import Context
from .estimate import CoreEstimate

MAX_SERIES = 10
Z_WINDOW = 756
MOM_WINDOW = 63
RIDGE_ALPHA = 25.0


def macro_state(context: Context) -> dict[str, dict]:
    """Situação, direção e velocidade de cada indicador macroeconômico."""
    out: dict[str, dict] = {}
    for key, series in context.macro.items():
        values = series.dropna()
        if len(values) < 12:
            continue
        array = values.to_numpy(dtype=float)
        meta = context.macro_meta.get(key, {})
        z = stats.zscore(array, window=min(Z_WINDOW, max(60, len(array))))
        last_z = float(z[-1]) if np.isfinite(z[-1]) else float("nan")
        slope = stats.slope_per_period(array, window=MOM_WINDOW)
        previous_slope = stats.slope_per_period(array[:-MOM_WINDOW // 3], window=MOM_WINDOW) if len(array) > MOM_WINDOW * 2 else float("nan")
        scale = float(np.nanstd(np.diff(array))) or 1.0
        out[key] = {
            "label": meta.get("label", key),
            "source": meta.get("source", ""),
            "source_name": meta.get("source_name", ""),
            "verified": bool(meta.get("verified", True)),
            "unit": meta.get("unit", ""),
            "sign_prior": meta.get("sign_prior", 0),
            "last": float(array[-1]),
            "last_date": str(values.index[-1].date()),
            "zscore": None if np.isnan(last_z) else round(last_z, 3),
            "direction_per_period": None if not np.isfinite(slope) else round(float(slope) / scale, 3),
            "acceleration": None if not np.isfinite(previous_slope) else round(float(slope - previous_slope) / (scale * MOM_WINDOW), 4),
            "change_3m": round(float(array[-1] - array[-min(len(array), MOM_WINDOW)]), 4),
            "change_12m": round(float(array[-1] - array[-min(len(array), 252)]), 4) if len(array) > 60 else None,
            "mean_5y": round(float(np.nanmean(array[-min(len(array), 1260):])), 4),
            "observations": int(len(array)),
            "trend": "alta" if np.isfinite(slope) and slope > 0 else "queda" if np.isfinite(slope) and slope < 0 else "estável",
        }
    return out


def macro_features(context: Context, keys: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Matriz de desvios macroeconômicos alinhada às datas do ativo."""
    columns: dict[str, np.ndarray] = {}
    names: list[str] = []
    requested = keys or sorted(context.macro.keys())
    for key in requested[:MAX_SERIES]:
        levels = context.aligned_macro(key)
        if levels is None:
            continue
        z = stats.zscore(levels, window=min(Z_WINDOW, max(120, len(levels))))
        momentum = np.full_like(z, np.nan)
        scale = np.nanstd(np.diff(levels)) * np.sqrt(MOM_WINDOW)
        momentum[MOM_WINDOW:] = (levels[MOM_WINDOW:] - levels[:-MOM_WINDOW]) / (scale if scale and scale > 0 else 1.0)
        momentum = np.clip(momentum, -5.0, 5.0)
        columns[f"{key}_z"] = z
        columns[f"{key}_mom"] = momentum
        names.extend([f"{key}_z", f"{key}_mom"])
    if not columns:
        return pd.DataFrame(), []
    return pd.DataFrame(columns), names


def run(context: Context) -> CoreEstimate:
    estimate = CoreEstimate(core="macro", label="Macroeconomia")
    state = macro_state(context)
    estimate.data["indicators"] = state
    usable_keys = [key for key, item in state.items() if item.get("zscore") is not None]
    if not usable_keys:
        return CoreEstimate.skipped("macro", estimate.label, "Nenhum indicador macroeconômico disponível na fonte.")
    features, names = macro_features(context, usable_keys)
    if features.empty:
        return CoreEstimate.skipped("macro", estimate.label, "Indicadores macro não puderam ser alinhados ao calendário do ativo.")
    close = context.close.to_numpy()
    target = stats.forward_returns(close, context.horizon_td)
    frame = features.copy()
    frame["_y"] = target
    frame = frame.dropna()
    verified_ratio = float(np.mean([state.get(key.split("_")[0], {}).get("verified", True) for key in names])) if names else 0.0
    estimate.data["coverage"] = {"requested": len(usable_keys), "used": len(names) // 2, "verified_ratio": round(verified_ratio, 3)}
    if len(frame) < config.REGRESSION_MIN_SAMPLES:
        estimate.status = "skip"
        estimate.notes.append(f"Histórico conjunto insuficiente ({len(frame)} observações) para estimar a sensibilidade macro.")
        return estimate
    design = frame.drop(columns=["_y"]).to_numpy()
    outcome = frame["_y"].to_numpy()
    split = int(len(frame) * 0.7)
    train = stats.ridge_regression(design[:split], outcome[:split], alpha=RIDGE_ALPHA)
    if not train["ok"]:
        return CoreEstimate.skipped("macro", estimate.label, "Regressão macro não pôde ser estimada.")
    r2_oos = stats.out_of_sample_r2(train["model"], design[split:], outcome[split:])
    full = stats.ridge_regression(design, outcome, alpha=RIDGE_ALPHA)
    coefficients = np.asarray(full["model"][-1].coef_, dtype=float)   # já padronizados
    intercept = float(full["intercept"])
    # A previsão usa as condições da data-base, não a última linha com rótulo conhecido.
    last_row = features.iloc[-1].to_numpy(dtype=float)[None, :]
    standardized = np.clip(full["model"][0].transform(last_row)[0], -4.0, 4.0)
    contributions = coefficients * standardized
    expected = float(np.clip(np.nansum(contributions), -0.9, 3.0))
    labels = {}
    for key in usable_keys:
        if key in state:
            labels[f"{key}_z"] = f"{state[key]['label']} — nível"
            labels[f"{key}_mom"] = f"{state[key]['label']} — tendência 3 meses"
    for index, (name, value) in enumerate(zip(names, contributions)):
        if not np.isfinite(value):
            continue
        estimate.add_signal(
            labels.get(name, name),
            value * 100,
            "alta" if value > 0 else "baixa",
            f"coeficiente padronizado {float(coefficients[index]):+.4f} · condição atual {float(standardized[index]):+.2f} desvios-padrão",
        )
    estimate.expected_return = expected
    estimate.sigma = float(full["residual_std"])
    r2_oos_value = float(r2_oos) if np.isfinite(r2_oos) else 0.0
    estimate.confidence = round(
        float(
            np.clip(
                0.15
                + 2.0 * max(0.0, r2_oos_value)
                - 0.15 * min(1.0, max(0.0, -r2_oos_value))
                + 0.15 * verified_ratio
                + 0.10 * min(1.0, len(names) / 12.0),
                0.05,
                0.85,
            )
        ),
        3,
    )
    estimate.data.update(
        {
            "model": "regressão Ridge sobre desvios macroeconômicos",
            "alpha": RIDGE_ALPHA,
            "r2_in_sample": round(float(full["r2"]), 5),
            "r2_out_of_sample": None if not np.isfinite(r2_oos) else round(float(r2_oos), 5),
            "intercept": round(intercept, 5),
            "n": int(full["n"]),
            "features": names,
        }
    )
    estimate.notes.append(
        "O núcleo mede o efeito das condições macro atuais em relação à média histórica; não inclui o retorno médio da ação. "
        "Sensibilidades estimadas por regressão podem mudar de regime."
    )
    estimate.sort_signals()
    estimate.status = "ok"
    return estimate
