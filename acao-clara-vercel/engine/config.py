"""Parâmetros centrais do motor quantitativo.

Tudo que é "escolha de projeto" (horizontes, pesos-base por horizonte, limites
de contribuição, janelas estatísticas) fica aqui, para que o comportamento do
motor seja auditável em um único lugar.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / "runtime"

# ---------------------------------------------------------------------------
# Horizontes
# ---------------------------------------------------------------------------
HORIZONS_DAYS = (7, 30, 90, 180, 365)
TRADING_DAYS_PER_YEAR = 252
DEFAULT_HORIZON = 90

HORIZON_LABELS = {
    7: "Curtíssimo prazo",
    30: "Curto prazo",
    90: "Médio prazo",
    180: "Médio/longo prazo",
    365: "Longo prazo",
}


def horizon_to_trading(days: int) -> int:
    """Converte horizonte em dias corridos para pregões (aproximação 252/365)."""
    return max(2, int(round(days * TRADING_DAYS_PER_YEAR / 365)))


# ---------------------------------------------------------------------------
# Pesos-base de cada núcleo por horizonte (Camada B, antes do ajuste dinâmico)
# ---------------------------------------------------------------------------
CORE_WEIGHTS = {
    7: {
        "fundamentals": 0.05,
        "macro": 0.10,
        "interdependence": 0.20,
        "events": 0.15,
        "flows": 0.20,
        "statistics": 0.25,
        "simulation": 0.05,
    },
    30: {
        "fundamentals": 0.10,
        "macro": 0.15,
        "interdependence": 0.20,
        "events": 0.15,
        "flows": 0.15,
        "statistics": 0.20,
        "simulation": 0.05,
    },
    90: {
        "fundamentals": 0.20,
        "macro": 0.15,
        "interdependence": 0.20,
        "events": 0.10,
        "flows": 0.10,
        "statistics": 0.15,
        "simulation": 0.10,
    },
    180: {
        "fundamentals": 0.30,
        "macro": 0.15,
        "interdependence": 0.15,
        "events": 0.05,
        "flows": 0.05,
        "statistics": 0.15,
        "simulation": 0.15,
    },
    365: {
        "fundamentals": 0.35,
        "macro": 0.15,
        "interdependence": 0.12,
        "events": 0.03,
        "flows": 0.03,
        "statistics": 0.12,
        "simulation": 0.20,
    },
}

# Limites que impedem um único núcleo de dominar a projeção.
MAX_CORE_WEIGHT = 0.40
MIN_CORE_WEIGHT = 0.02
# O ajuste dinâmico por desempenho histórico pode multiplicar o peso-base por
# no máximo este fator (para cima e para baixo).
WEIGHT_ADJUST_CAP = 2.0

# ---------------------------------------------------------------------------
# Janelas estatísticas
# ---------------------------------------------------------------------------
MIN_HISTORY_BARS = 320          # mínimo para o motor quantitativo operar
REGRESSION_MIN_SAMPLES = 120    # mínimo para regressões de sensibilidade
CORR_WINDOW = 252               # janela de correlação móvel
LAGS_TESTED = (0, 1, 5, 10, 20, 60)
DISCOVERY_MAX_CANDIDATES = 60
FDR_Q = 0.10                    # Benjamini–Hochberg sobre os testes de descoberta
SUBPERIODS_STABILITY = 3

# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------
MC_PATHS = 4000
MC_SEED = 20240607
SCENARIO_MASSES = (0.10, 0.25, 0.35, 0.20, 0.10)
SCENARIO_NAMES = (
    "Muito otimista",
    "Otimista",
    "Neutro",
    "Pessimista",
    "Muito pessimista",
)
CONFIDENCE_RANGE_QUANTILES = (0.10, 0.90)

# ---------------------------------------------------------------------------
# Limites de sanidade (Camada C)
# ---------------------------------------------------------------------------
MAX_ABS_RETURN = 3.0            # trava absoluta de retorno projetado (+300%)
MIN_PRICE_RATIO = 0.05
SHOCK_SIGMAS = 1.0              # choque padrão nos testes de sensibilidade

# Meia-vida (em pregões) da convergência do preço ao valor justo estimado.
VALUATION_CONVERGENCE_TAU = 500

# Meia-vida (dias corridos) do efeito de um evento.
EVENT_DECAY_DAYS = 45
EVENT_MAX_IMPACT = 0.60         # saturação do impacto agregado de eventos (60 p.p.)

# ---------------------------------------------------------------------------
# Cache e fontes
# ---------------------------------------------------------------------------
CACHE_TTL_HISTORY = 6 * 3600
CACHE_TTL_MACRO = 6 * 3600
CACHE_TTL_QUOTE = 15 * 60
CACHE_TTL_NEWS = 3 * 3600
HTTP_TIMEOUT = 18
USER_AGENT = "Mozilla/5.0 (compatible; AcaoClara/2.0; motor quantitativo)"

# Séries do Yahoo usadas como fatores de mercado/commodities (Núcleo 03).
FACTOR_SYMBOLS = {
    "^BVSP": "Ibovespa",
    "BRL=X": "Dólar (USD/BRL)",
    "CL=F": "Petróleo WTI",
    "BZ=F": "Petróleo Brent",
    "HG=F": "Cobre (proxy de atividade industrial)",
    "GC=F": "Ouro",
    "SI=F": "Prata",
    "NG=F": "Gás natural",
    "ZC=F": "Milho",
    "ZS=F": "Soja",
    "DX-Y.NYB": "Índice dólar (DXY)",
    "^TNX": "Juros 10 anos EUA (x10)",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "000001.SS": "Xangai Composite (China)",
    "^VIX": "VIX (aversão a risco)",
    "^STOXX50E": "Euro Stoxx 50",
}

# Sinal teórico (apenas quando não há dado suficiente para estimar a
# sensibilidade por regressão). +1 = alta do fator tende a ajudar a ação.
FACTOR_PRIOR_SIGN = {
    "^BVSP": +1,
    "BRL=X": 0,        # depende da empresa (exportadora x importadora)
    "CL=F": 0,
    "BZ=F": 0,
    "HG=F": 0,
    "GC=F": 0,
    "SI=F": 0,
    "NG=F": 0,
    "ZC=F": 0,
    "ZS=F": 0,
    "DX-Y.NYB": -1,
    "^TNX": -1,
    "^GSPC": +1,
    "^IXIC": +1,
    "000001.SS": 0,
    "^VIX": -1,
    "^STOXX50E": +1,
}

# Sinal teórico de indicadores macro sobre ações domésticas.
MACRO_PRIOR_SIGN = {
    "selic": -1,
    "selic_real": -1,
    "ipca_12m": -1,
    "ipca_m": -1,
    "igpm_12m": -1,
    "pib": +1,
    "desemprego": -1,
    "usdbrl": 0,
    "us10y": -1,
    "fedfunds": -1,
    "us_cpi_yoy": -1,
    "divida_pib": -1,
    "confianca_consumidor": +1,
    "pmi": +1,
    "ibov": +1,
    "dxy_broad": -1,
    "us_breakeven": -1,
    "spread_crediticio": -1,
    "credito": +1,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out
