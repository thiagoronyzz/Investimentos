"""Camada de dados do motor.

Todas as fontes externas passam por aqui, com cache em disco, tempo de espera
curto e falha silenciosa: quando uma fonte não responde, o núcleo dependente
recebe "sem dados" e o motor reduz o peso dele em vez de quebrar.

Fontes usadas:
  * Yahoo Finance (endpoints públicos não oficiais) — preços de ações e de
    fatores de mercado (índices, câmbio, commodities, juros);
  * brapi.dev — lista de ativos da B3, cotações e fundamentos;
  * Banco Central do Brasil (SGS) — Selic, IPCA, câmbio, atividade;
  * FRED (CSV público) — juros e inflação dos EUA.

Nada é inventado: se a série não vier, ela não entra no cálculo.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from . import config
from .config import CACHE_TTL_HISTORY, CACHE_TTL_MACRO, CACHE_TTL_NEWS, CACHE_TTL_QUOTE, HTTP_TIMEOUT

TZ = "America/Sao_Paulo"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:120]


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn").lower()


class DataSourceError(Exception):
    """Falha de fonte externa — nunca deve derrubar a requisição."""


class MarketData:
    """Acesso unificado às fontes, com cache em disco e modo demonstração."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        demo: bool | None = None,
        session: requests.Session | None = None,
        ttl: dict | None = None,
    ) -> None:
        self.demo = bool(demo if demo is not None else os.environ.get("ACAO_CLARA_DEMO") == "1")
        self.cache_dir = Path(cache_dir) if cache_dir else config.RUNTIME_DIR / "cache"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self.ttl = {
            "history": CACHE_TTL_HISTORY,
            "macro": CACHE_TTL_MACRO,
            "quote": CACHE_TTL_QUOTE,
            "news": CACHE_TTL_NEWS,
            **(ttl or {}),
        }
        self.health: dict[str, dict[str, Any]] = {}
        self._macro_cache: dict | None = None
        if not self.demo:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_path(self, kind: str, key: str) -> Path:
        return self.cache_dir / f"{kind}__{_slug(key)}.json"

    def _cache_get(self, kind: str, key: str) -> Any | None:
        if self.demo:
            return None
        path = self._cache_path(kind, key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - float(raw.get("stored_at", 0)) > self.ttl.get(kind, 3600):
            return None
        return raw.get("payload")

    def _cache_put(self, kind: str, key: str, payload: Any) -> None:
        if self.demo:
            return
        try:
            self._cache_path(kind, key).write_text(
                json.dumps({"stored_at": time.time(), "payload": payload}, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

    def cache_get(self, kind: str, key: str) -> Any | None:
        return self._cache_get(kind, key)

    def cache_put(self, kind: str, key: str, payload: Any) -> None:
        self._cache_put(kind, key, payload)

    def _note(self, source: str, status: str, detail: str = "", items: int | None = None) -> None:
        entry = self.health.setdefault(source, {"status": status, "detail": detail, "checked_at": datetime.now().isoformat(timespec="seconds")})
        entry["status"] = status
        entry["detail"] = detail
        if items is not None:
            entry["items"] = items

    # ------------------------------------------------------------------
    # HTTP genérico
    # ------------------------------------------------------------------
    def _get(self, url: str, params: dict | None = None, source: str = "http") -> Any:
        if self.demo:
            raise DataSourceError("modo demonstração: fontes externas desligadas")
        try:
            response = self.session.get(
                url,
                params={**(params or {}), "_": str(int(time.time()))},
                headers={"Accept": "application/json, text/csv, text/plain, */*", "Cache-Control": "no-cache"},
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            if "json" in response.headers.get("Content-Type", "") or url.endswith("formato=json"):
                self._note(source, "ok")
                return response.json()
            self._note(source, "ok")
            return response.text
        except (requests.RequestException, ValueError) as exc:
            self._note(source, "erro", f"{type(exc).__name__}: {exc}"[:180])
            raise DataSourceError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Preços
    # ------------------------------------------------------------------
    def history(self, symbol: str, years: int = 10, use_cache: bool = True) -> pd.DataFrame | None:
        """Histórico diário (fechamento ajustado) de qualquer símbolo suportado."""
        if self.demo:
            from .synth import synthetic_history

            return synthetic_history(symbol, years=years)
        key = f"{symbol}:{years}y"
        if use_cache:
            cached = self._cache_get("history", key)
            if cached is not None:
                return _frame_from_rows(cached)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        try:
            data = self._get(url, {"range": f"{years}y", "interval": "1d", "events": "div,splits"}, source="yahoo")
        except DataSourceError:
            return None
        frame = chart_to_frame(data)
        if frame is None or frame.empty:
            self._note("yahoo", "erro", f"sem histórico para {symbol}")
            return None
        if use_cache:
            self._cache_put("history", key, frame_to_rows(frame))
        return frame

    def histories(self, symbols: Iterable[str], years: int = 10) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self.history(symbol, years=years)
            if frame is not None and not frame.empty:
                out[symbol] = frame
        return out

    def quote(self, symbol: str) -> dict | None:
        if self.demo:
            frame = self.history(symbol)
            if frame is None or frame.empty:
                return None
            last = frame.iloc[-1]
            return {"price": float(last.close), "time": datetime.combine(last.date, datetime.min.time()).isoformat()}
        try:
            data = self._get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", {"range": "5d", "interval": "1d"}, source="yahoo")
        except DataSourceError:
            return None
        meta = (data.get("chart", {}) or {}).get("result", [{}])[0].get("meta", {})
        price, stamp = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
        if price is None or not stamp:
            return None
        return {"price": float(price), "time": datetime.fromtimestamp(stamp, _tz()).isoformat()}

    def search(self, query: str) -> list[dict]:
        if self.demo:
            from .synth import synthetic_universe

            q = _strip_accents(query.strip())
            return [
                {"symbol": item["symbol"], "name": item["name"], "exchange": "B3 (demonstração)"}
                for item in synthetic_universe()
                if q and (q in _strip_accents(item["symbol"]) or q in _strip_accents(item["name"]))
            ][:10]
        try:
            data = self._get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                {"q": query.strip(), "quotesCount": 20, "newsCount": 0, "enableFuzzyQuery": "false"},
                source="yahoo",
            )
        except DataSourceError:
            return []
        results = []
        for item in data.get("quotes", []):
            symbol = str(item.get("symbol", ""))
            if item.get("quoteType") == "EQUITY" and symbol.upper().endswith(".SA"):
                results.append(
                    {
                        "symbol": symbol,
                        "name": item.get("longname") or item.get("shortname") or symbol,
                        "exchange": item.get("exchDisp", "B3"),
                    }
                )
        return results[:10]

    def universe(self) -> dict:
        if self.demo:
            from .synth import synthetic_universe

            items = synthetic_universe()
            self._note("brapi", "demonstração", "universo sintético", len(items))
            return {"items": items, "source": "modo demonstração (dados sintéticos)", "total_in_source": len(items), "demo": True}
        try:
            data = self._get("https://brapi.dev/api/quote/list", {"type": "stock", "limit": 1000}, source="brapi")
        except DataSourceError:
            raise DataSourceError("Não foi possível obter a lista de ações da B3. Tente novamente.")
        if data.get("totalPages", 1) > 1:
            raise DataSourceError("A lista da fonte veio incompleta; tente mais tarde.")
        items, seen = [], set()
        for row in data.get("stocks", []):
            ticker = str(row.get("stock", "")).upper()
            if not re.fullmatch(r"[A-Z0-9]{4,10}(?:3|4|5|6|11)", ticker) or ticker in seen:
                continue
            seen.add(ticker)
            items.append({"symbol": ticker + ".SA", "name": row.get("name") or ticker})
        if not items:
            raise DataSourceError("A fonte não retornou ações disponíveis para o ranking.")
        return {"items": items, "source": "brapi.dev: lista de ativos classificados como ações", "total_in_source": data.get("totalCount", len(items))}

    # ------------------------------------------------------------------
    # Fundamentos e notícias
    # ------------------------------------------------------------------
    def fundamentals(self, symbol: str) -> dict | None:
        if self.demo:
            from .synth import synthetic_fundamentals

            return synthetic_fundamentals(symbol)
        ticker = symbol.upper().removesuffix(".SA")
        try:
            data = self._get(f"https://brapi.dev/api/quote/{ticker}", {"fundamentals": "true", "token": os.environ.get("BRAPI_TOKEN", "")}, source="brapi")
        except DataSourceError:
            return None
        results = data.get("results") or []
        if not results:
            return None
        raw = results[0]
        fund = raw.get("fundamentals") or {}
        return normalize_fundamentals(raw, fund)

    def news(self, symbol: str, limit: int = 25) -> list[dict]:
        if self.demo:
            from .synth import synthetic_news

            return synthetic_news(symbol)
        try:
            data = self._get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                {"q": symbol.upper(), "quotesCount": 0, "newsCount": limit},
                source="yahoo",
            )
        except DataSourceError:
            return []
        items = []
        for item in (data.get("news") or [])[:limit]:
            stamp = item.get("providerPublishTime")
            when = datetime.fromtimestamp(stamp, _tz()).date().isoformat() if stamp else None
            items.append({"title": item.get("title") or "", "publisher": item.get("publisher") or "", "date": when, "url": item.get("link") or ""})
        return items

    # ------------------------------------------------------------------
    # Macroeconomia
    # ------------------------------------------------------------------
    def macro(self, force: bool = False) -> dict[str, dict]:
        """Séries macroeconômicas indexadas por data (valores, não retornos)."""
        if self._macro_cache is not None and not force:
            return self._macro_cache
        if self.demo:
            from .synth import synthetic_macro

            self._note("macro", "demonstração", "séries macroeconômicas sintéticas")
            self._macro_cache = synthetic_macro()
            return self._macro_cache

        spec = _load_json(config.DATA_DIR / "macro.json") or {"bcb": [], "fred": [], "yahoo": []}
        series: dict[str, dict] = {}
        cached = self._cache_get("macro", "macro") if not force else None
        if cached:
            series = {key: payload for key, payload in cached.items() if payload.get("rows")}
            self._note("macro", "ok (cache)", f"{len(series)} séries em cache")
        else:
            for entry in spec.get("bcb", []):
                rows, name, serie_id = self._bcb_candidates(entry.get("candidates") or [])
                if rows:
                    series[entry["key"]] = {
                        "label": entry.get("label", entry["key"]),
                        "source": f"BCB SGS {serie_id}",
                        "source_name": name,
                        "verified": _keywords_match(name, [word for candidate in entry.get("candidates", []) for word in candidate.get("expect", [])]),
                        "unit": entry.get("unit", ""),
                        "sign_prior": entry.get("sign", 0),
                        "rows": rows,
                    }
            for entry in spec.get("fred", []):
                rows = self._fred_series(entry["id"])
                if entry.get("transform") == "yoy" and rows:
                    rows = _to_year_over_year(rows)
                if rows:
                    series[entry["key"]] = {
                        "label": entry.get("label", entry["id"]),
                        "source": f"FRED {entry['id']}",
                        "source_name": entry["id"],
                        "verified": True,
                        "unit": entry.get("unit", ""),
                        "sign_prior": entry.get("sign", 0),
                        "rows": rows,
                    }
            yahoo_symbols = [entry["symbol"] for entry in spec.get("yahoo", [])]
            frames = self.histories(yahoo_symbols, years=5)
            for entry in spec.get("yahoo", []):
                frame = frames.get(entry["symbol"])
                if frame is None:
                    continue
                series[entry["key"]] = {
                    "label": entry.get("label", entry["symbol"]),
                    "source": f"Yahoo {entry['symbol']}",
                    "source_name": config.FACTOR_SYMBOLS.get(entry["symbol"], entry["symbol"]),
                    "verified": True,
                    "unit": entry.get("unit", ""),
                    "sign_prior": entry.get("sign", 0),
                    "rows": frame_to_rows(frame[["date", "close"]]),
                }
            self._cache_put("macro", "macro", series)
        self._macro_cache = series
        return series

    def _bcb_candidates(self, candidates: list[dict], last: int = 1500) -> tuple[list[list[str, str]], str, int | None]:
        """Tenta cada código candidato e usa o primeiro cujo nome confere com o esperado."""
        fallback: tuple[list[list[str, str]], str, int] | None = None
        for candidate in candidates:
            serie_id = int(candidate.get("id", 0))
            if not serie_id:
                continue
            rows, name = self._bcb_series(serie_id, last=last)
            if not rows:
                continue
            if _keywords_match(name, candidate.get("expect", [])):
                return rows, name, serie_id
            if fallback is None:
                fallback = (rows, name, serie_id)
        if fallback:
            return fallback
        return [], "", None

    def _bcb_series(self, serie_id: int, last: int = 1500) -> tuple[list[list[str, str]], str]:
        try:
            data = self._get(
                f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{int(serie_id)}/dados/ultimos/{last}",
                {"formato": "json"},
                source="bcb",
            )
        except DataSourceError:
            return [], ""
        rows = []
        if isinstance(data, list):
            for item in data:
                date = str(item.get("data", ""))
                value = str(item.get("valor", ""))
                if not date or not value:
                    continue
                try:
                    iso = datetime.strptime(date, "%d/%m/%Y").date().isoformat()
                except ValueError:
                    continue
                rows.append([iso, value])
        name = ""
        try:
            meta = self._get(f"https://api.bcb.gov.br/dados/serie/metadados/{int(serie_id)}", {"formato": "json"}, source="bcb")
            name = json.dumps(meta, ensure_ascii=False)[:400]
        except DataSourceError:
            pass
        return rows, name

    def _fred_series(self, serie_id: str) -> list[list[str, str]]:
        try:
            text = self._get(f"https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": serie_id}, source="fred")
        except DataSourceError:
            return []
        if not isinstance(text, str):
            return []
        rows = []
        for line in text.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            date, value = parts[0].strip(), parts[1].strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or value in ("", "."):
                continue
            rows.append([date, value])
        return rows

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------
    def health_report(self) -> dict:
        return {
            "demo": self.demo,
            "sources": self.health,
            "cache_dir": str(self.cache_dir),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }


# ---------------------------------------------------------------------------
# Auxiliares de conversão
# ---------------------------------------------------------------------------
def _tz():
    from zoneinfo import ZoneInfo

    return ZoneInfo(TZ)


def chart_to_frame(data: dict) -> pd.DataFrame | None:
    chart = (data or {}).get("chart", {})
    if chart.get("error") or not chart.get("result"):
        return None
    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return None
    indicators = result.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adj = ((indicators.get("adjclose") or [{}])[0] or {}).get("adjclose")
    close_raw = quote.get("close") or [None] * len(timestamps)
    close = adj if adj else close_raw
    if not close:
        return None
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(TZ).date,
            "open": quote.get("open", [None] * len(timestamps)),
            "high": quote.get("high", [None] * len(timestamps)),
            "low": quote.get("low", [None] * len(timestamps)),
            "close_raw": close_raw,
            "close": close,
            "volume": quote.get("volume", [None] * len(timestamps)),
        }
    )
    today = datetime.now(_tz()).date()
    # A barra do dia só entra depois do fechamento + 30 min de folga.
    market_end = (result.get("meta", {}) or {}).get("currentTradingPeriod", {}).get("regular", {}).get("end")
    now = datetime.now(_tz())
    closed = bool(
        market_end
        and now.timestamp() >= market_end + 1800
        and datetime.fromtimestamp(market_end, _tz()).date() == today
    )
    frame = frame[frame.date <= today] if closed else frame[frame.date < today]
    for column in ("open", "high", "low", "close_raw", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame.close > 0].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return frame if len(frame) else None


def frame_to_rows(frame: pd.DataFrame) -> list[list]:
    rows = []
    for row in frame.itertuples(index=False):
        values = [str(getattr(row, column)) for column in frame.columns]
        rows.append(values)
    return {"columns": list(frame.columns), "rows": rows}


def _frame_from_rows(payload: dict) -> pd.DataFrame | None:
    try:
        frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    except (KeyError, TypeError, ValueError):
        return None
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"]).date
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["close"]).reset_index(drop=True)


def _to_year_over_year(rows: list[list[str, str]]) -> list[list[str, str]]:
    """Converte um índice em variação acumulada em 12 meses (usada no CPI dos EUA)."""
    frame = pd.DataFrame(rows, columns=["date", "value"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna()
    frame["index"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("index").sort_index()
    yoy = (frame["value"] / frame["value"].shift(12) - 1.0) * 100.0
    yoy = yoy.dropna()
    return [[timestamp.date().isoformat(), f"{float(value):.4f}"] for timestamp, value in yoy.items()]


def _keywords_match(text: str, keywords: Iterable[str]) -> bool:
    if not keywords:
        return True
    haystack = _strip_accents(text)
    return any(_strip_accents(word) in haystack for word in keywords)


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def normalize_fundamentals(raw: dict, fund: dict) -> dict:
    """Padroniza os fundamentos vindos da fonte em um dicionário único.

    Nomes de campos variam entre versões da API; cada campo tenta várias
    chaves possíveis e, se nenhuma existir, fica ausente (nunca é chutado).
    """
    def pick(*keys):
        for key in keys:
            value = fund.get(key, raw.get(key))
            value = config.safe_float(value)
            if value is not None:
                return value
        return None

    out = {
        "pe": pick("pe", "priceEarnings", "trailingPE", "forwardPE"),
        "pb": pick("pbr", "priceBook", "priceToBook"),
        "ps": pick("psr", "priceToSales"),
        "ev_ebitda": pick("evToEbitda", "enterpriseToEbitda"),
        "roe": pick("returnOnEquity", "roe"),
        "roic": pick("returnOnInvestedCapital", "roic"),
        "net_margin": pick("profitMargin", "netMargin"),
        "gross_margin": pick("grossMargins", "grossMargin"),
        "ebitda_margin": pick("ebitdaMargins", "ebitdaMargin"),
        "revenue_growth": pick("revenueGrowth", "revenueQuarterlyGrowth"),
        "earnings_growth": pick("earningsGrowth", "earningsQuarterlyGrowth"),
        "debt_to_equity": pick("debtToEquity", "totalDebtToEquity"),
        "net_debt_ebitda": pick("netDebtToEbitda"),
        "current_ratio": pick("currentRatio"),
        "dividend_yield": pick("dividendsYield", "dividendYield"),
        "payout": pick("payoutRatio", "payout"),
        "free_cash_flow": pick("freeCashflow", "freeCashFlow"),
        "ebitda": pick("ebitda"),
        "revenue": pick("revenue", "totalRevenue"),
        "net_income": pick("netIncome"),
        "market_cap": pick("marketCap"),
        "total_cash": pick("totalCash"),
        "total_debt": pick("totalDebt"),
        "earnings_yield": pick("earningsYield"),
        "beta": pick("beta"),
        "sector": fund.get("sector") or raw.get("sector") or None,
        "name": raw.get("longName") or raw.get("name") or None,
        "source": "brapi.dev fundamentals",
    }
    # Percentuais vêm como fração em algumas fontes e como número inteiro em outras.
    for key in ("roe", "net_margin", "gross_margin", "ebitda_margin", "dividend_yield", "payout", "revenue_growth", "earnings_growth"):
        value = out.get(key)
        if value is not None and abs(value) > 3:
            out[key] = value / 100.0
    return {key: value for key, value in out.items() if value is not None or key in ("sector", "name", "source")}
