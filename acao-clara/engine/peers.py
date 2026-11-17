"""Comparáveis de setor: medianas de múltiplos e indicadores de empresas pares.

A comparação setorial é exigida pelo Núcleo 02 ("comparar a empresa com sua
própria média, com o setor e com modelos de negócio semelhantes"). Como não há
histórico longo de múltiplos nas fontes gratuitas usadas, o motor compara com a
mediana *atual* dos pares do mesmo setor e informa quantas empresas entraram na
amostra — sem pares suficientes, a etapa é pulada em vez de chutada.
"""
from __future__ import annotations

import json
import random

import numpy as np

from . import config

MULTIPLE_KEYS = ("pe", "pb", "ev_ebitda")
INDICATOR_KEYS = ("roe", "net_margin", "revenue_growth", "earnings_growth", "net_debt_ebitda", "dividend_yield")
PEER_CACHE_TTL = 24 * 3600


def load_sector_map() -> dict[str, str]:
    try:
        data = json.loads((config.DATA_DIR / "sectors.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(ticker).upper(): str(sector) for ticker, sector in data.items()}


def sector_of(symbol: str, fallback: str | None = None) -> str | None:
    ticker = symbol.upper().removesuffix(".SA")
    return load_sector_map().get(ticker, fallback)


def _median(values: list[float]) -> float | None:
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else None


def _demo_peers(data, symbol: str, sector: str, limit: int) -> dict | None:
    from .synth import synthetic_fundamentals, synthetic_universe

    peers = [item["symbol"] for item in synthetic_universe() if item["symbol"] != symbol.upper()]
    same = [peer for peer in peers if (synthetic_fundamentals(peer) or {}).get("sector") == sector]
    sample = same[:limit] if len(same) >= 2 else peers[:limit]
    if not sample:
        return None
    return _build(symbol, sector, sample, {peer: synthetic_fundamentals(peer) for peer in sample})


def _build(symbol: str, sector: str, sample: list[str], fundamentals: dict[str, dict]) -> dict | None:
    rows = []
    for peer in sample:
        data_peer = fundamentals.get(peer) or {}
        if not data_peer:
            continue
        row = {"symbol": peer}
        for key in MULTIPLE_KEYS + INDICATOR_KEYS:
            row[key] = data_peer.get(key)
        rows.append(row)
    if len(rows) < 2:
        return None
    out: dict = {"sector": sector, "count": len(rows), "peers": rows}
    for key in MULTIPLE_KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        if key == "pe":
            values = [value for value in values if 0 < value < 120]
        elif key == "pb":
            values = [value for value in values if 0 < value < 40]
        elif key == "ev_ebitda":
            values = [value for value in values if -20 < value < 80]
        out[key] = _median(values)
    for key in INDICATOR_KEYS:
        out[f"{key}_median"] = _median([row[key] for row in rows if row.get(key) is not None])
    return out


def sample_peers(data, symbol: str, limit: int = 6) -> dict | None:
    """Mediana de múltiplos e indicadores entre empresas do mesmo setor."""
    symbol = symbol.upper()
    cached = data.cache_get("peers", symbol) if hasattr(data, "cache_get") else None
    if cached:
        return cached
    sector = sector_of(symbol)
    if data.demo:
        sector = sector or (data.fundamentals(symbol) or {}).get("sector") or "Geral"
        result = _demo_peers(data, symbol, sector, limit)
        if result:
            result["source"] = "modo demonstração"
        return result
    if not sector:
        return None
    sector_map = load_sector_map()
    candidates = [f"{ticker}.SA" for ticker, value in sector_map.items() if value == sector and f"{ticker}.SA" != symbol]
    if not candidates:
        return None
    random.Random(symbol).shuffle(candidates)
    picked, fundamentals = [], {}
    for candidate in candidates:
        payload = data.fundamentals(candidate)
        if not payload:
            continue
        picked.append(candidate)
        fundamentals[candidate] = payload
        if len(picked) >= limit:
            break
    result = _build(symbol, sector, picked, fundamentals)
    if result:
        result["source"] = "mediana atual dos pares na fonte brapi.dev"
        data.cache_put("peers", symbol, result)
    return result
