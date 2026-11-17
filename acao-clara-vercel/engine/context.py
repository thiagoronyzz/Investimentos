"""Contexto de análise: tudo que os núcleos podem enxergar em uma data.

A regra central do motor é a ausência de vazamento de informação: um núcleo
só recebe dados cuja data é <= `as_of`. O mesmo objeto é usado na projeção
corrente (as_of = último fechamento) e no backtesting (as_of = data passada),
o que garante que a validação reproduza exatamente o que o motor faria na época.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from . import config


@dataclass
class Context:
    symbol: str
    name: str
    as_of: Any                      # datetime.date do último dado permitido
    history: pd.DataFrame           # colunas: date, open, high, low, close_raw, close, volume
    drivers: dict[str, pd.Series] = field(default_factory=dict)   # fator -> série de fechamento
    candidates: dict[str, pd.Series] = field(default_factory=dict)  # séries só para a descoberta de variáveis
    macro: dict[str, pd.Series] = field(default_factory=dict)     # chave -> série de nível
    macro_meta: dict[str, dict] = field(default_factory=dict)
    fundamentals: dict | None = None
    peers: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    horizon_days: int = config.DEFAULT_HORIZON
    horizon_td: int = 0
    sector: str | None = None
    currency: str = "BRL"
    demo: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.horizon_td:
            self.horizon_td = config.horizon_to_trading(self.horizon_days)

    # ------------------------------------------------------------------
    @property
    def close(self) -> pd.Series:
        return self.history["close"].astype(float).reset_index(drop=True)

    @property
    def price(self) -> float:
        return float(self.history["close"].iloc[-1])

    @property
    def dates(self) -> pd.Series:
        return self.history["date"].reset_index(drop=True)

    def truncate(self, as_of) -> "Context":
        """Devolve uma cópia do contexto limitada à data `as_of`."""
        history = self.history[self.history["date"] <= as_of]
        drivers = {key: series[series.index <= pd.Timestamp(as_of)] for key, series in self.drivers.items()}
        candidates = {key: series[series.index <= pd.Timestamp(as_of)] for key, series in self.candidates.items()}
        macro = {key: series[series.index <= pd.Timestamp(as_of)] for key, series in self.macro.items()}
        events = [event for event in self.events if str(event.get("date", "")) <= str(as_of)]
        return Context(
            symbol=self.symbol,
            name=self.name,
            as_of=as_of,
            history=history.reset_index(drop=True),
            drivers=drivers,
            candidates=candidates,
            macro=macro,
            macro_meta=dict(self.macro_meta),
            fundamentals=self.fundamentals,
            peers=dict(self.peers),
            events=events,
            horizon_days=self.horizon_days,
            horizon_td=self.horizon_td,
            sector=self.sector,
            currency=self.currency,
            demo=self.demo,
            notes=list(self.notes),
        )

    def aligned_driver(self, key: str) -> np.ndarray | None:
        """Retorna a série do fator realinhada às datas do ativo (sem lookahead)."""
        series = self.drivers.get(key)
        if series is None:
            series = self.candidates.get(key)
        if series is None or series.empty:
            return None
        index = pd.Index([pd.Timestamp(d) for d in self.dates])
        aligned = series.reindex(index.union(series.index)).sort_index().ffill().reindex(index)
        values = aligned.to_numpy(dtype=float)
        if np.sum(np.isfinite(values)) < 60:
            return None
        return values

    def aligned_macro(self, key: str) -> np.ndarray | None:
        series = self.macro.get(key)
        if series is None or series.empty:
            return None
        index = pd.Index([pd.Timestamp(d) for d in self.dates])
        aligned = series.reindex(series.index.union(index)).sort_index().ffill(limit=120).reindex(index)
        values = aligned.to_numpy(dtype=float)
        if np.sum(np.isfinite(values)) < 30:
            return None
        return values

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "as_of": str(self.as_of),
            "bars": int(len(self.history)),
            "horizon_days": self.horizon_days,
            "horizon_trading_days": self.horizon_td,
            "drivers": sorted(self.drivers),
            "macro_series": sorted(self.macro),
            "events": len(self.events),
            "has_fundamentals": self.fundamentals is not None,
            "demo": self.demo,
        }


def series_from_rows(rows: list, label: str = "valor") -> pd.Series:
    """Converte [[data, valor], ...] em Series indexada por Timestamp."""
    if not rows:
        return pd.Series(dtype=float)
    dates, values = [], []
    for row in rows:
        try:
            dates.append(pd.Timestamp(str(row[0])))
            values.append(float(str(row[1]).replace(",", ".")))
        except (ValueError, TypeError, IndexError):
            continue
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=label).sort_index()
    return series[~series.index.duplicated(keep="last")]


def to_jsonable(value: Any) -> Any:
    """Converte valores numpy em tipos nativos para serialização JSON."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, date)):
        return str(value)
    return value
