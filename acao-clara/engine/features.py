"""Variáveis técnicas calculadas apenas com dados disponíveis até a data-base.

Inclui as seis variáveis do modelo clássico (retornos de 5/20/60 pregões,
volatilidade de 20 e distância das médias de 20 e 60) e variáveis adicionais
usadas pelo motor quantitativo (força de tendência, posição no intervalo anual,
z-score de volume e distância do topo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = ["ret5", "ret20", "ret60", "vol20", "ma20", "ma60"]
EXTRA_FEATURES = ["ret120", "vol60", "vol_ratio", "range_pos", "max_drawdown_120", "rsi14"]


def technical_features(close: pd.Series, volume: pd.Series | None = None, extra: bool = True) -> pd.DataFrame:
    price = pd.Series(close).astype(float).reset_index(drop=True)
    out = pd.DataFrame(index=price.index)
    for periods in (5, 20, 60):
        out[f"ret{periods}"] = price.pct_change(periods)
    out["vol20"] = price.pct_change().rolling(20).std()
    for periods in (20, 60):
        out[f"ma{periods}"] = price / price.rolling(periods).mean() - 1.0
    if extra:
        out["ret120"] = price.pct_change(120)
        out["vol60"] = price.pct_change().rolling(60).std()
        volume_series = pd.Series(volume).astype(float).reset_index(drop=True) if volume is not None else pd.Series(np.nan, index=price.index)
        mean_volume = volume_series.rolling(60).mean()
        out["vol_ratio"] = volume_series / mean_volume.replace(0.0, np.nan) - 1.0
        rolling_min = price.rolling(252, min_periods=60).min()
        rolling_max = price.rolling(252, min_periods=60).max()
        span = (rolling_max - rolling_min).replace(0.0, np.nan)
        out["range_pos"] = (price - rolling_min) / span
        out["max_drawdown_120"] = price / price.rolling(120, min_periods=40).max() - 1.0
        delta = price.diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        relative = gain / loss.replace(0.0, np.nan)
        out["rsi14"] = 100.0 - 100.0 / (1.0 + relative)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_matrix(close: pd.Series, volume: pd.Series | None = None, extra: bool = True) -> pd.DataFrame:
    return technical_features(close, volume, extra=extra)
