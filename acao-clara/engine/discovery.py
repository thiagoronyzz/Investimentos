"""Descoberta automática de variáveis relevantes (sem IA).

O motor varre um universo amplo de variáveis — fatores de mercado, commodities,
índices externos e séries macroeconômicas — testando, para cada par
(variável, defasagem):

  1. correlação contemporânea e com defasagem de 1, 5, 10, 20 e 60 dias;
  2. significância estatística (teste t da correlação);
  3. estabilidade da relação em subperíodos;
  4. poder explicativo fora da amostra.

Como testar centenas de relações encontra correlações por acaso, aplica-se a
correção de Benjamini–Hochberg (controle da taxa de falsa descoberta) antes de
aceitar qualquer variável. As relações aprovadas entram no Núcleo 03 como
candidatas adicionais, sem substituir as relações economicamente programadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats
from .context import Context

OOS_R2_FLOOR = -0.05   # relações claramente piores que a média fora da amostra são descartadas


def candidate_variables(context: Context) -> dict[str, np.ndarray]:
    """Variáveis candidatas na mesma janela do horizonte analisado.

    Comparar janelas iguais evita o ruído de correlacionar um retorno de 90 dias
    com a variação de um único dia da variável externa.
    """
    window = context.horizon_td
    out: dict[str, np.ndarray] = {}
    for key in list(context.drivers) + list(context.candidates):
        aligned = context.aligned_driver(key)
        if aligned is not None:
            out[f"driver:{key}"] = stats.window_returns(aligned, window)
    for key in context.macro:
        aligned = context.aligned_macro(key)
        if aligned is not None:
            out[f"macro:{key}"] = stats.window_returns(aligned, window)
    return out


def scan(context: Context, max_candidates: int = config.DISCOVERY_MAX_CANDIDATES, top: int = 12) -> dict:
    candidates = dict(list(candidate_variables(context).items())[:max_candidates])
    if len(candidates) < 2:
        return {"ok": False, "reason": "poucas variáveis candidatas", "tests": 0, "discoveries": []}
    target = stats.window_returns(context.close.to_numpy(), context.horizon_td)
    rows: list[dict] = []
    for name, values in candidates.items():
        for lag in config.LAGS_TESTED:
            shifted = _shift(values, lag)
            coef, pvalue, n = stats.correlation(target, shifted)
            if not np.isfinite(coef) or not np.isfinite(pvalue):
                continue
            rows.append(
                {
                    "variable": name,
                    "lag": lag,
                    "corr": float(coef),
                    "pvalue": float(pvalue),
                    "n": int(n),
                    "stability": stats.correlation_stability(target, shifted),
                    "oos_r2": _oos_r2(target, shifted, context.horizon_td),
                }
            )
    if not rows:
        return {"ok": False, "reason": "nenhum teste válido", "tests": 0, "discoveries": []}
    pvalues = np.array([row["pvalue"] for row in rows])
    rejected = stats.benjamini_hochberg(pvalues, q=config.FDR_Q)
    discoveries = []
    for row, keep in zip(rows, rejected):
        if not keep:
            continue
        if not np.isfinite(row["stability"]) or row["stability"] < 0.66:
            continue
        if np.isfinite(row["oos_r2"]) and row["oos_r2"] < OOS_R2_FLOOR:
            continue
        discoveries.append(row)
    discoveries.sort(key=lambda item: (abs(item["corr"]), item["oos_r2"] or 0.0), reverse=True)
    # Mantém apenas a melhor defasagem de cada variável.
    best_per_variable: dict[str, dict] = {}
    for item in discoveries:
        best_per_variable.setdefault(item["variable"], item)
    return {
        "ok": True,
        "tests": len(rows),
        "candidates": len(candidates),
        "fdr_q": config.FDR_Q,
        "accepted_before_filters": int(rejected.sum()),
        "discoveries": list(best_per_variable.values())[:top],
        "rejected_examples": sorted(
            [row for row in rows if abs(row["corr"]) > 0.08], key=lambda item: abs(item["corr"]), reverse=True
        )[:8],
        "note": "Variáveis aprovadas no teste de falsa descoberta e com estabilidade entre subperíodos. Correlação não implica causalidade.",
    }


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return values
    out = np.full_like(values, np.nan, dtype=float)
    out[lag:] = values[:-lag]
    return out


def _oos_r2(target: np.ndarray, values: np.ndarray, horizon: int = 0) -> float:
    frame = pd.DataFrame({"x": values, "y": target}).dropna()
    if len(frame) < 100:
        return float("nan")
    train_end, test_start = stats.purged_split(len(frame), horizon)
    train = stats.ridge_regression(frame.x.to_numpy()[:train_end, None], frame.y.to_numpy()[:train_end], alpha=5.0)
    if not train.get("ok"):
        return float("nan")
    return stats.out_of_sample_r2(
        train["model"], frame.x.to_numpy()[test_start:, None], frame.y.to_numpy()[test_start:]
    )


def suggested_additions(context: Context, discoveries: list[dict], limit: int = 3) -> list[str]:
    """Variáveis descobertas que ainda não estão programadas para a empresa."""
    known = set(context.drivers)
    extra = []
    for item in discoveries:
        variable = item["variable"]
        kind, _, key = variable.partition(":")
        if kind != "driver" or key in known:
            continue
        extra.append(key)
        if len(extra) >= limit:
            break
    return extra
