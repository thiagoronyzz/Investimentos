"""Núcleo 05 — Fluxo financeiro e comportamento institucional.

Constrói indicadores próprios de pressão compradora e vendedora a partir de
preço e volume (volume relativo, OBV, linha de acumulação/distribuição, MFI,
razão volume de alta/baixa) e estima, por regressão, quanto esses indicadores
costumam preceder o retorno no horizonte escolhido. O objetivo é capturar
divergências entre preço e movimentação financeira — por exemplo, preço estável
com volume comprador crescente.

Limitação declarada: fluxo de investidores estrangeiros, posição vendida
(short interest), aluguel de ações, fluxos de ETFs e mudanças de participação
acionária não estão disponíveis nas fontes gratuitas usadas; quando a fonte
não fornece, o núcleo trabalha apenas com preço e volume e avisa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats
from .context import Context
from .estimate import CoreEstimate

RIDGE_ALPHA = 20.0
WEIGHTS = {"rel_volume": 0.15, "obv": 0.30, "accumulation": 0.25, "mfi": 0.15, "up_down": 0.15}


def flow_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de fluxo calculados apenas com dados até cada data."""
    close = history["close"].astype(float).reset_index(drop=True)
    volume = history.get("volume")
    volume = pd.Series(volume).astype(float).reset_index(drop=True) if volume is not None else pd.Series(np.nan, index=close.index)
    out = pd.DataFrame(index=close.index)
    change = close.diff()
    out["rel_volume"] = volume / volume.rolling(60).mean().replace(0.0, np.nan) - 1.0
    signed = np.sign(change.fillna(0.0)) * volume.fillna(0.0)
    obv = signed.cumsum()
    obv_change = obv.diff(20)
    out["obv"] = obv_change / obv_change.rolling(120).std().replace(0.0, np.nan)
    typical = (history.get("high", close).astype(float) + history.get("low", close).astype(float) + close) / 3.0
    direction = np.sign(typical.diff().fillna(0.0))
    accumulation = (direction * volume.fillna(0.0)).cumsum()
    accumulation_change = accumulation.diff(20)
    out["accumulation"] = accumulation_change / accumulation_change.rolling(120).std().replace(0.0, np.nan)
    typical_return = typical.pct_change()
    money_flow = typical_return * volume.fillna(0.0)
    positive = money_flow.clip(lower=0.0).rolling(14).sum()
    negative = (-money_flow.clip(upper=0.0)).rolling(14).sum()
    ratio = positive / negative.replace(0.0, np.nan)
    out["mfi"] = (100.0 - 100.0 / (1.0 + ratio)) / 50.0 - 1.0
    up_volume = volume.where(change > 0, 0.0).rolling(20).sum()
    down_volume = volume.where(change < 0, 0.0).rolling(20).sum()
    out["up_down"] = (up_volume - down_volume) / (up_volume + down_volume).replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def composite_score(indicators: pd.DataFrame) -> pd.Series:
    """Índice único de pressão compradora/vendedora em [-1, 1]."""
    parts = []
    total_weight = 0.0
    for key, weight in WEIGHTS.items():
        if key not in indicators:
            continue
        series = indicators[key]
        if series.notna().sum() < 60:
            continue
        scaled = np.tanh(series / max(series.std() or 1.0, 1e-9))
        parts.append(weight * scaled.fillna(0.0))
        total_weight += weight
    if not parts or total_weight == 0:
        return pd.Series(np.nan, index=indicators.index)
    return (sum(parts) / total_weight).clip(-1.0, 1.0)


def run(context: Context) -> CoreEstimate:
    estimate = CoreEstimate(core="flows", label="Fluxo financeiro")
    history = context.history
    if "volume" not in history or history["volume"].notna().sum() < 120:
        return CoreEstimate.skipped("flows", estimate.label, "A fonte não forneceu volume negociado suficiente para este ativo.")
    indicators = flow_indicators(history)
    score = composite_score(indicators)
    if score.dropna().empty:
        return CoreEstimate.skipped("flows", estimate.label, "Indicadores de fluxo indisponíveis.")
    score_z = stats.zscore(score.to_numpy(), window=504)
    target = stats.forward_returns(context.close.to_numpy(), context.horizon_td)
    frame = pd.DataFrame({"score": score_z, "rel_volume": indicators["rel_volume"].to_numpy(), "y": target}).dropna()
    current_score = float(score.iloc[-1]) if np.isfinite(score.iloc[-1]) else 0.0
    estimate.data["current"] = {
        "composite": round(current_score, 4),
        "zscore": round(float(score_z[-1]), 3) if np.isfinite(score_z[-1]) else None,
        "pressure": "compradora" if current_score > 0.1 else "vendedora" if current_score < -0.1 else "equilibrada",
        "indicators": {key: round(float(value), 4) if np.isfinite(value) else None for key, value in indicators.iloc[-1].items()},
        "divergence": _divergence(context.close, history.get("volume")),
    }
    if len(frame) < config.REGRESSION_MIN_SAMPLES:
        estimate.status = "skip"
        estimate.notes.append(f"Histórico insuficiente ({len(frame)} observações) para estimar a sensibilidade ao fluxo.")
        return estimate
    design = frame[["score", "rel_volume"]].to_numpy()
    outcome = frame["y"].to_numpy()
    split = int(len(frame) * 0.7)
    train = stats.ridge_regression(design[:split], outcome[:split], alpha=RIDGE_ALPHA)
    r2_oos = stats.out_of_sample_r2(train["model"], design[split:], outcome[split:]) if train.get("ok") else float("nan")
    full = stats.ridge_regression(design, outcome, alpha=RIDGE_ALPHA)
    if not full.get("ok"):
        return CoreEstimate.skipped("flows", estimate.label, "Regressão de fluxo não pôde ser estimada.")
    scale = full["model"][0].scale_
    coefficients = np.asarray(full["coefficients"], dtype=float) / scale
    last = frame[["score", "rel_volume"]].to_numpy()[-1]
    center = np.nanmean(design, axis=0)
    contributions = coefficients * (last - center)
    expected = float(np.clip(float(np.nansum(contributions)), -0.9, 3.0))
    estimate.add_signal("Pressão compradora/vendedora (composto)", contributions[0] * 100, "alta" if contributions[0] > 0 else "baixa", f"índice atual {current_score:+.2f} · coeficiente {coefficients[0]:+.4f}")
    estimate.add_signal("Volume relativo", contributions[1] * 100, "alta" if contributions[1] > 0 else "baixa", f"coeficiente {coefficients[1]:+.4f}")
    estimate.expected_return = expected
    estimate.sigma = float(full["residual_std"])
    estimate.confidence = round(float(np.clip(0.15 + 2.0 * max(0.0, r2_oos if np.isfinite(r2_oos) else 0.0) + 0.10 * min(1.0, len(frame) / 800.0), 0.0, 0.7)), 3)
    estimate.data.update(
        {
            "model": "regressão Ridge sobre o índice de fluxo e volume relativo",
            "r2_out_of_sample": None if not np.isfinite(r2_oos) else round(float(r2_oos), 5),
            "n": int(full["n"]),
            "missing_sources": [
                "fluxo de investidores estrangeiros",
                "posição vendida (short interest)",
                "aluguel de ações",
                "fluxos de ETFs",
                "mudanças na participação acionária",
            ],
        }
    )
    estimate.notes.append(
        "Volume alto não permite concluir acumulação ou distribuição por si só; o núcleo mede apenas a relação histórica "
        "entre esses indicadores e o retorno futuro, que pode não se manter."
    )
    estimate.sort_signals()
    estimate.status = "ok"
    return estimate


def _divergence(close: pd.Series, volume) -> dict:
    """Divergência entre movimento de preço e movimento de volume (60 pregões)."""
    if volume is None:
        return {"ok": False}
    price_change = close.astype(float).pct_change(5)
    volume_change = pd.Series(volume).astype(float).pct_change(5)
    frame = pd.DataFrame({"p": price_change, "v": volume_change}).dropna().tail(60)
    if len(frame) < 20:
        return {"ok": False}
    coef, _, _ = stats.correlation(frame.p.to_numpy(), frame.v.to_numpy())
    price_direction = float(np.sign(close.astype(float).iloc[-1] - close.astype(float).iloc[-21])) if len(close) > 21 else 0.0
    volume_trend = float(np.mean(frame.v.tail(20).to_numpy()))
    return {
        "ok": True,
        "corr_price_volume": None if not np.isfinite(coef) else round(float(coef), 4),
        "price_direction_20d": int(price_direction),
        "volume_trend_20d": round(volume_trend, 4),
        "reading": "preço estável com volume crescente" if abs(price_direction) < 1 and volume_trend > 0.1 else "sem divergência clara",
    }
