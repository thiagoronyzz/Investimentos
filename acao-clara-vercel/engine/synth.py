"""Gerador de universo sintético coerente.

Serve para duas coisas:
  1. testes automatizados que precisam rodar sem internet;
  2. o "modo demonstração" (`ACAO_CLARA_DEMO=1`), que permite exercitar todo o
     motor — inclusive a descoberta de variáveis — quando as fontes externas
     estão indisponíveis.

Os números são fictícios por construção, mas são *coerentes entre si*: os
retornos das ações são gerados a partir dos fatores macro/mercado com
sensibilidades conhecidas, algumas com defasagem, e há também uma variável
espúria (que só se correlaciona por acaso) para exercitar a correção de
múltiplos testes. Nada aqui é usado no modo normal do site.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import zlib

import numpy as np
import pandas as pd

TZ = ZoneInfo("America/Sao_Paulo")
_DAYS = 2650
_STATE: dict | None = None

# símbolo: (setor, preço inicial, volume base, sensibilidades, defasagens)
_SPECS = {
    "PETR4.SA": dict(name="Petrobras PN (demonstração)", sector="Petróleo e gás", price=34.0, volume=9.0e7,
                     loadings={"oil": 0.55, "usdbrl": 0.30, "ibov": 0.42, "vix": -0.25}, lags={"oil": 1}),
    "VALE3.SA": dict(name="Vale ON (demonstração)", sector="Mineração", price=61.0, volume=7.0e7,
                     loadings={"copper": 0.40, "china": 0.34, "usdbrl": 0.26, "ibov": 0.38}, lags={"copper": 2}),
    "WEGE3.SA": dict(name="WEG ON (demonstração)", sector="Bens industriais", price=42.0, volume=1.2e7,
                     loadings={"ibov": 0.30, "us10y": -0.06, "usdbrl": 0.18, "copper": -0.12}, lags={}),
    "ITUB4.SA": dict(name="Itaú Unibanco PN (demonstração)", sector="Bancos", price=33.0, volume=6.0e7,
                     loadings={"ibov": 0.55, "selic": 0.012, "usdbrl": -0.10}, lags={}),
    "MGLU3.SA": dict(name="Magazine Luiza ON (demonstração)", sector="Varejo", price=10.5, volume=1.1e8,
                     loadings={"ibov": 0.48, "selic": -0.018, "ipca": -0.02}, lags={}),
    "SUZB3.SA": dict(name="Suzano ON (demonstração)", sector="Papel e celulose", price=54.0, volume=2.5e7,
                     loadings={"usdbrl": 0.45, "china": 0.22, "ibov": 0.25}, lags={}),
    "ABEV3.SA": dict(name="Ambev ON (demonstração)", sector="Bebidas", price=12.8, volume=5.0e7,
                     loadings={"ibov": 0.40, "ipca": -0.015, "usdbrl": -0.12}, lags={}),
    "B3SA3.SA": dict(name="B3 ON (demonstração)", sector="Serviços financeiros", price=11.5, volume=9.0e7,
                     loadings={"ibov": 0.70, "selic": -0.010, "vix": -0.35}, lags={}),
    "ELET6.SA": dict(name="Eletrobras PNB (demonstração)", sector="Energia elétrica", price=44.0, volume=1.0e7,
                     loadings={"ibov": 0.33, "selic": -0.014}, lags={}),
    "CCRO3.SA": dict(name="CCR ON (demonstração)", sector="Logística e transporte", price=10.2, volume=6.0e7,
                     loadings={"ibov": 0.36, "pib": 0.30, "ipca": 0.02}, lags={}),
    "PRIO3.SA": dict(name="PRIO ON (demonstração)", sector="Petróleo e gás", price=38.0, volume=3.0e7,
                     loadings={"oil": 0.72, "usdbrl": 0.24, "ibov": 0.30}, lags={"oil": 1}),
    "AZUL4.SA": dict(name="Azul PN (demonstração)", sector="Aviação", price=1.6, volume=1.3e8,
                     loadings={"usdbrl": -0.55, "oil": -0.35, "ibov": 0.55}, lags={}),
    "TOTS3.SA": dict(name="TOTVS ON (demonstração)", sector="Tecnologia", price=31.0, volume=8.0e6,
                     loadings={"ibov": 0.30, "us10y": -0.09, "nasdaq": 0.20}, lags={}),
    "JBSS3.SA": dict(name="JBS ON (demonstração)", sector="Alimentos", price=39.0, volume=2.0e7,
                     loadings={"usdbrl": 0.40, "ibov": 0.22, "china": 0.15}, lags={}),
    "SPUR3.SA": dict(name="Variável espúria (demonstração)", sector="Outros", price=22.0, volume=4.0e6,
                     loadings={"ibov": 0.20, "spurious": 0.90}, lags={}),
}


def _build() -> dict:
    rng = np.random.default_rng(20250117)
    dates = pd.bdate_range(end=datetime.now(TZ).date(), periods=_DAYS).date
    n = _DAYS

    def ar1(target_std: float, phi: float = 0.94) -> np.ndarray:
        """AR(1) de média zero normalizado para o desvio-padrão diário pedido."""
        shocks = rng.normal(0.0, 1.0, n)
        out = np.empty(n)
        out[0] = shocks[0]
        for i in range(1, n):
            out[i] = phi * out[i - 1] + shocks[i]
        out -= out.mean()
        return out / (out.std() or 1.0) * target_std

    # Persistência baixa: retornos diários reais são quase não correlacionados;
    # phi alto inflaria artificialmente a variância de retornos longos.
    factors = {
        "ibov": ar1(0.0105, 0.25),
        "oil": ar1(0.0185, 0.22),
        "copper": ar1(0.0130, 0.25),
        "china": ar1(0.0115, 0.24),
        "usdbrl": ar1(0.0095, 0.28),
        "spurious": rng.normal(0.0, 0.012, n),
    }
    # Correlaciona levemente commodities e China para o motor ter o que separar.
    factors["china"] = 0.55 * factors["china"] + 0.45 * factors["copper"]
    factors["copper"] = 0.8 * factors["copper"] + 0.2 * factors["ibov"]

    levels = {
        "ibov": 100000.0,
        "oil": 78.0,
        "copper": 4.1,
        "china": 3100.0,
        "usdbrl": 5.2,
        "us10y": 3.8,
        "selic": 13.0,
        "ipca": 5.2,
        "pib": 2.1,
        "vix": 16.0,
        "nasdaq": 16500.0,
        "spurious": 100.0,
    }
    factors["nasdaq"] = 0.6 * factors["ibov"] + ar1(0.009, 0.25)
    factors["vix"] = ar1(0.05, 0.93)

    # Séries de nível mensais/trimestrais (juros, inflação, PIB).
    monthly_dates = pd.date_range(end=pd.Timestamp(dates[-1]), periods=_DAYS // 20 + 4, freq="MS").date
    selic = np.clip(np.cumsum(rng.normal(0.0, 0.22, len(monthly_dates))) + 9.5, 1.75, 15.0)
    ipca = np.clip(6.0 + np.cumsum(rng.normal(0.0, 0.28, len(monthly_dates))) * 0.6, 1.5, 12.5)
    quarterly_dates = pd.date_range(end=pd.Timestamp(dates[-1]), periods=_DAYS // 60 + 4, freq="QS").date
    pib = np.clip(2.0 + np.cumsum(rng.normal(0.0, 0.25, len(quarterly_dates))) * 0.5, -3.0, 6.0)

    us10y = np.clip(3.6 + np.cumsum(rng.normal(0.0, 0.045, n)) * 0.5, 0.4, 6.0)
    vix = np.clip(15.0 + np.cumsum(factors["vix"]) * 0.35, 9.0, 65.0)
    nasdaq = levels["nasdaq"] * np.exp(np.cumsum(factors["nasdaq"]))

    def step_series(monthly: np.ndarray, month_dates) -> np.ndarray:
        """Expande uma série periódica para a grade diária (último valor conhecido)."""
        keys = np.array(sorted(set(month_dates)), dtype="datetime64[D]")
        target = np.array(list(dates), dtype="datetime64[D]")
        pos = np.clip(np.searchsorted(keys, target, side="right") - 1, 0, len(monthly) - 1)
        return monthly[pos]

    selic_daily = step_series(selic, monthly_dates)
    ipca_daily = step_series(ipca, monthly_dates)
    pib_daily = step_series(pib, quarterly_dates)

    def price_from_returns(returns: np.ndarray, start: float) -> np.ndarray:
        return start * np.exp(np.cumsum(returns))

    stocks: dict[str, dict] = {}
    for symbol, spec in _SPECS.items():
        ret = np.full(n, 0.00040)   # deriva anual de ~10%
        for factor, beta in spec["loadings"].items():
            series = factors.get(factor)
            if series is None:
                continue
            if factor == "selic":
                series = np.diff(selic_daily, prepend=selic_daily[0]) / 12.0
            elif factor == "ipca":
                series = np.diff(ipca_daily, prepend=ipca_daily[0]) / 6.0
            elif factor == "pib":
                series = np.diff(pib_daily, prepend=pib_daily[0]) / 2.0
            elif factor == "us10y":
                series = np.diff(us10y, prepend=us10y[0])
            elif factor == "vix":
                series = np.diff(vix, prepend=vix[0]) / vix[:-1].mean()
            lag = spec["lags"].get(factor, 0)
            aligned = np.roll(series, lag)
            aligned[:lag] = 0.0
            ret = ret + beta * aligned
        idio = rng.normal(0.0, 0.0130, n)
        # Regimes de volatilidade: tornam o GARCH e a detecção de regime úteis.
        regime = np.ones(n)
        for _ in range(6):
            start = int(rng.integers(0, n - 60))
            regime[start:start + 60] *= rng.uniform(1.3, 2.0)
        ret = ret + idio * regime
        close = price_from_returns(ret, spec["price"])
        volume = spec["volume"] * (1.0 + 6.0 * np.abs(ret)) * np.exp(rng.normal(0.0, 0.18, n))
        high = close * (1 + np.abs(rng.normal(0.0, 0.008, n)))
        low = close * (1 - np.abs(rng.normal(0.0, 0.008, n)))
        open_ = np.roll(close, 1) * (1 + rng.normal(0.0, 0.004, n))
        open_[0] = close[0]
        stocks[symbol] = {
            "spec": spec,
            "frame": pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close_raw": close, "close": close, "volume": volume}),
        }

    factor_frames = {
        "^BVSP": price_from_returns(factors["ibov"] + 0.00030, levels["ibov"]),
        "BRL=X": price_from_returns(factors["usdbrl"], levels["usdbrl"]),
        "CL=F": price_from_returns(factors["oil"] + 0.00010, levels["oil"]),
        "BZ=F": price_from_returns(factors["oil"] * 1.02, levels["oil"] * 1.05),
        "HG=F": price_from_returns(factors["copper"] + 0.00012, levels["copper"]),
        "GC=F": price_from_returns(0.25 * factors["oil"] + rng.normal(0, 0.006, n), 2350.0),
        "SI=F": price_from_returns(0.4 * factors["copper"] + rng.normal(0, 0.011, n), 27.0),
        "NG=F": price_from_returns(ar1(0.03, 0.9), 2.9),
        "ZC=F": price_from_returns(ar1(0.012, 0.93), 440.0),
        "ZS=F": price_from_returns(ar1(0.012, 0.93), 1100.0),
        "DX-Y.NYB": price_from_returns(factors["usdbrl"] * 0.7, 104.0),
        "^TNX": us10y * 10.0,
        "^GSPC": price_from_returns(0.8 * factors["ibov"] + rng.normal(0, 0.004, n), 5300.0),
        "^IXIC": nasdaq,
        "000001.SS": price_from_returns(factors["china"] + 0.00010, levels["china"]),
        "^VIX": vix,
        "^STOXX50E": price_from_returns(0.6 * factors["ibov"] + rng.normal(0, 0.005, n), 4900.0),
    }
    factors_frames = {
        symbol: pd.DataFrame({"date": dates, "open": values, "high": values, "low": values, "close_raw": values, "close": values, "volume": np.full(n, 1.0e6)})
        for symbol, values in factor_frames.items()
    }

    def rows(values, digits=4):
        return [[str(d), f"{float(v):.{digits}f}"] for d, v in zip(dates, values)]

    macro = {
        "selic": dict(label="Selic (demonstração)", source="sintético", source_name="Selic", verified=True, unit="% a.a.", sign_prior=-1, rows=[[str(d), f"{v:.2f}"] for d, v in zip(monthly_dates, selic)]),
        "ipca_12m": dict(label="IPCA 12 meses (demonstração)", source="sintético", source_name="IPCA", verified=True, unit="% a.a.", sign_prior=-1, rows=[[str(d), f"{v:.2f}"] for d, v in zip(monthly_dates, ipca)]),
        "usdbrl": dict(label="Dólar (demonstração)", source="sintético", source_name="USD/BRL", verified=True, unit="R$", sign_prior=0, rows=rows(price_from_returns(factors["usdbrl"], levels["usdbrl"]))),
        "us10y": dict(label="Juros 10 anos EUA (demonstração)", source="sintético", source_name="US 10Y", verified=True, unit="%", sign_prior=-1, rows=rows(us10y)),
        "pib": dict(label="PIB (demonstração)", source="sintético", source_name="PIB", verified=True, unit="% a.a.", sign_prior=1, rows=[[str(d), f"{v:.2f}"] for d, v in zip(quarterly_dates, pib)]),
        "ibov": dict(label="Ibovespa (demonstração)", source="sintético", source_name="Ibovespa", verified=True, unit="pontos", sign_prior=1, rows=rows(price_from_returns(factors["ibov"], levels["ibov"]), 2)),
        "vix": dict(label="VIX (demonstração)", source="sintético", source_name="VIX", verified=True, unit="pontos", sign_prior=-1, rows=rows(vix, 2)),
        "oil": dict(label="Petróleo WTI (demonstração)", source="sintético", source_name="WTI", verified=True, unit="US$", sign_prior=0, rows=rows(price_from_returns(factors["oil"], levels["oil"]))),
        "spurious": dict(label="Série espúria (demonstração)", source="sintético", source_name="Espúria", verified=False, unit="", sign_prior=0, rows=rows(price_from_returns(factors["spurious"], levels["spurious"]))),
    }
    return {"dates": dates, "stocks": stocks, "factor_frames": factors_frames, "macro": macro, "factors": factors}


def _state() -> dict:
    global _STATE
    if _STATE is None:
        _STATE = _build()
    return _STATE


def synthetic_universe() -> list[dict]:
    state = _state()
    return [{"symbol": symbol, "name": data["spec"]["name"]} for symbol, data in state["stocks"].items()]


def synthetic_history(symbol: str, years: int = 10) -> pd.DataFrame | None:
    state = _state()
    symbol = symbol.upper()
    if symbol in state["stocks"]:
        frame = state["stocks"][symbol]["frame"].copy()
    elif symbol in state["factor_frames"]:
        frame = state["factor_frames"][symbol].copy()
    else:
        return None
    cutoff = len(frame) - int(years * 252)
    return frame.iloc[max(0, cutoff):].reset_index(drop=True)


def synthetic_macro() -> dict:
    return {key: dict(payload) for key, payload in _state()["macro"].items()}


def synthetic_fundamentals(symbol: str) -> dict | None:
    """Fundamentos fictícios, porém internamente consistentes entre si e com o preço."""
    state = _state()
    symbol = symbol.upper()
    entry = state["stocks"].get(symbol)
    if entry is None:
        return None
    spec = entry["spec"]
    price = float(entry["frame"]["close"].iloc[-1])
    rng = np.random.default_rng(zlib.crc32(symbol.encode("utf-8")))
    shares = float(np.clip(rng.normal(1.5e9, 5.0e8), 3.0e8, 4.0e9))
    market_cap = price * shares
    # P/L sorteado primeiro: garante múltiplos plausíveis e coerentes entre si.
    pe = float(np.clip(rng.normal(11.0, 3.5), 5.0, 22.0))
    net_margin = float(np.clip(rng.normal(0.12, 0.04), 0.03, 0.25))
    net_income = market_cap / pe
    revenue = net_income / net_margin
    ebitda_margin = net_margin + float(np.clip(rng.normal(0.09, 0.03), 0.03, 0.20))
    ebitda = revenue * ebitda_margin
    price_to_book = float(np.clip(rng.normal(2.6, 1.0), 0.7, 7.0))
    book_value = market_cap / price_to_book
    net_debt_ebitda = float(np.clip(rng.normal(1.5, 0.8), -0.5, 4.5))
    total_debt = ebitda * max(net_debt_ebitda, 0.0) + ebitda * 0.05
    total_cash = total_debt * float(np.clip(rng.normal(0.35, 0.15), 0.05, 0.8))
    dividend_yield = float(np.clip(rng.normal(0.05, 0.025), 0.0, 0.14))
    return {
        "pe": round(pe, 2),
        "pb": round(price_to_book, 2),
        "ps": round(market_cap / revenue, 2),
        "ev_ebitda": round((market_cap + total_debt - total_cash) / ebitda, 2),
        "roe": round(net_income / book_value, 4),
        "roic": round(ebitda * (1 - 0.34) / (total_debt + book_value), 4),
        "net_margin": round(net_margin, 4),
        "gross_margin": round(float(np.clip(ebitda_margin + rng.normal(0.15, 0.06), 0.10, 0.75)), 4),
        "ebitda_margin": round(ebitda_margin, 4),
        "revenue_growth": round(float(np.clip(rng.normal(0.12, 0.06), -0.05, 0.35)), 4),
        "earnings_growth": round(float(np.clip(rng.normal(0.13, 0.10), -0.30, 0.60)), 4),
        "debt_to_equity": round(total_debt / book_value, 3),
        "net_debt_ebitda": round(net_debt_ebitda, 2),
        "current_ratio": round(float(np.clip(rng.normal(1.6, 0.5), 0.6, 4.0)), 2),
        "dividend_yield": round(dividend_yield, 4),
        "payout": round(float(np.clip(dividend_yield * market_cap / net_income, 0.0, 1.2)), 3) if net_income > 0 else None,
        "free_cash_flow": float(net_income * np.clip(rng.normal(0.85, 0.15), 0.3, 1.2)),
        "ebitda": float(ebitda),
        "revenue": float(revenue),
        "net_income": float(net_income),
        "market_cap": float(market_cap),
        "total_cash": float(total_cash),
        "total_debt": float(total_debt),
        "beta": round(float(np.clip(rng.normal(1.05, 0.3), 0.3, 2.2)), 2),
        "sector": spec["sector"],
        "name": spec["name"],
        "source": "sintético (modo demonstração)",
    }


def synthetic_news(symbol: str) -> list[dict]:
    today = datetime.now(TZ).date()
    base = [
        ("Banco central reduz juros básicos em 0,50 ponto", 3),
        ("Inflação oficial acelera e pressiona consumo", 9),
        ("Sanções a produtor elevam preço do petróleo no mercado internacional", 5),
        ("Governo anuncia tarifa de importação sobre o aço", 12),
        ("China divulga dados de atividade industrial acima do esperado", 6),
        ("Companhia aprova recompra de ações e dividendos extraordinários", 2),
        ("Dólar fecha em alta com aversão global a risco", 1),
        ("Agência eleva projeção de crescimento do PIB brasileiro", 15),
        ("Empresa conclui aquisição estratégica no exterior", 20),
        ("Greve de funcionários interrompe operação em unidade", 4),
    ]
    out = []
    for index, (title, ago) in enumerate(base):
        out.append(
            {
                "title": title,
                "publisher": "Fonte sintética",
                "date": (today - timedelta(days=ago)).isoformat(),
                "url": "",
                "demo": True,
            }
        )
    return out
