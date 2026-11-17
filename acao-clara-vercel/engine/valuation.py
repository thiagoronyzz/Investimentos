"""Núcleo 02 — Fundamentos financeiros e valuation.

Dois modelos independentes de valor justo (fluxo de caixa descontado e
valuation relativo por múltiplos de pares) combinados em uma estimativa única.
A diferença entre o valor justo e o preço atual é convertida em retorno
esperado no horizonte assumindo convergência parcial (meia-vida `τ`).
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .context import Context
from .estimate import CoreEstimate

TAX_RATE = 0.34
EQUITY_RISK_PREMIUM = 0.06
COST_OF_DEBT_SPREAD = 0.02
DCF_YEARS = 5
MIN_WACC_GAP = 0.02


def _terminal_growth(context: Context) -> float:
    """Crescimento terminal nominal: inflação corrente + 2 p.p., entre 3% e 9%.

    Os fluxos são nominais e em reais, então a taxa terminal também precisa ser
    nominal; usar um crescimento real com uma taxa de desconto nominal subestima
    sistematicamente o valor da empresa.
    """
    inflation = None
    for key in ("ipca_12m", "ipca_m"):
        series = context.macro.get(key)
        if series is not None and len(series.dropna()):
            inflation = float(series.dropna().iloc[-1]) / 100.0
            break
    if inflation is None:
        inflation = 0.045
    return float(np.clip(inflation + 0.02, 0.03, 0.09))


def risk_free_rate(context: Context) -> tuple[float | None, str]:
    for key in ("selic", "selic_meta"):
        series = context.macro.get(key)
        if series is not None and len(series.dropna()):
            return float(series.dropna().iloc[-1]) / 100.0, f"{key} (fonte de dados)"
    return None, "sem Selic disponível; DCF ignorado"


def cost_of_capital(context: Context, fundamentals: dict) -> tuple[float | None, dict]:
    """WACC por CAPM com estrutura de capital informada pela fonte."""
    rf, source = risk_free_rate(context)
    if rf is None or rf <= 0:
        return None, {"reason": source}
    market_cap = fundamentals.get("market_cap")
    debt = fundamentals.get("total_debt") or 0.0
    beta = fundamentals.get("beta")
    beta_source = "fonte de dados"
    if beta is None:
        beta = 1.0
        beta_source = "beta padrão 1,0 (fonte não informou)"
    equity_weight = 1.0
    if market_cap and debt:
        equity_weight = market_cap / max(market_cap + debt, 1.0)
    ke = rf + beta * EQUITY_RISK_PREMIUM
    kd = rf + COST_OF_DEBT_SPREAD
    wacc = equity_weight * ke + (1 - equity_weight) * kd * (1 - TAX_RATE)
    return float(np.clip(wacc, 0.05, 0.35)), {
        "risk_free": round(rf, 4),
        "beta": round(float(beta), 3),
        "beta_source": beta_source,
        "cost_of_equity": round(ke, 4),
        "cost_of_debt": round(kd, 4),
        "equity_weight": round(equity_weight, 4),
        "wacc": round(float(wacc), 4),
        "risk_free_source": source,
    }


def dcf_value(context: Context) -> tuple[float | None, dict]:
    """Fluxo de caixa descontado em dois estágios (crescimento + terminal)."""
    fundamentals = context.fundamentals or {}
    fcf = fundamentals.get("free_cash_flow")
    basis = "fluxo de caixa livre informado"
    if fcf is None and fundamentals.get("ebitda"):
        fcf = float(fundamentals["ebitda"]) * 0.6
        basis = "60% do EBITDA informado (aproximação)"
    market_cap = fundamentals.get("market_cap")
    price = context.price
    if fcf is None or fcf <= 0 or not market_cap or not price:
        return None, {"reason": "faltam fluxo de caixa livre, valor de mercado ou preço"}
    wacc, capital = cost_of_capital(context, fundamentals)
    if wacc is None:
        return None, {"reason": capital.get("reason", "WACC não calculado")}
    terminal_g = _terminal_growth(context)
    if wacc - terminal_g < MIN_WACC_GAP:
        return None, {"reason": f"WACC ({wacc:.2%}) muito próximo do crescimento terminal ({terminal_g:.2%})"}
    growth = float(np.clip(fundamentals.get("revenue_growth") or 0.0, -0.05, 0.25))
    shares = market_cap / price
    cash = fundamentals.get("total_cash") or 0.0
    debt = fundamentals.get("total_debt") or 0.0
    present, projected = 0.0, []
    flow = float(fcf)
    for year in range(1, DCF_YEARS + 1):
        fade = growth * (1.0 - (year - 1) / DCF_YEARS) + terminal_g * ((year - 1) / DCF_YEARS)
        flow *= 1.0 + fade
        present += flow / (1.0 + wacc) ** year
        projected.append({"year": year, "growth": round(fade, 4), "flow": float(flow)})
    terminal_value = flow * (1.0 + terminal_g) / (wacc - terminal_g)
    present += terminal_value / (1.0 + wacc) ** DCF_YEARS
    equity = present + cash - debt
    if shares <= 0:
        return None, {"reason": "quantidade de ações inválida"}
    value = equity / shares
    if value <= 0:
        return None, {"reason": "valor patrimonial projetado não positivo"}
    return float(value), {
        "basis": basis,
        "wacc": capital,
        "terminal_growth": round(terminal_g, 4),
        "initial_growth": round(growth, 4),
        "fcf_base": float(fcf),
        "projections": projected,
        "terminal_value": float(terminal_value),
        "enterprise_value": float(present),
        "equity_value": float(equity),
        "shares": float(shares),
    }


def relative_value(context: Context) -> tuple[float | None, dict]:
    """Valor justo pelos múltiplos medianos das empresas do mesmo setor."""
    fundamentals = context.fundamentals or {}
    peers = context.peers or {}
    price = context.price
    market_cap = fundamentals.get("market_cap")
    if not peers or not market_cap or not price:
        return None, {"reason": "sem pares do setor ou sem valor de mercado"}
    shares = market_cap / price
    cash = fundamentals.get("total_cash") or 0.0
    debt = fundamentals.get("total_debt") or 0.0
    estimates: dict[str, float] = {}
    detail: dict[str, dict] = {}
    if peers.get("pe") and fundamentals.get("pe"):
        eps = price / fundamentals["pe"]
        estimates["pe"] = eps * peers["pe"]
        detail["pe"] = {"company": fundamentals["pe"], "peers": peers["pe"], "eps": round(eps, 4)}
    if peers.get("pb") and fundamentals.get("pb"):
        bvps = price / fundamentals["pb"]
        estimates["pb"] = bvps * peers["pb"]
        detail["pb"] = {"company": fundamentals["pb"], "peers": peers["pb"], "bvps": round(bvps, 4)}
    if peers.get("ev_ebitda") and fundamentals.get("ev_ebitda") and fundamentals.get("ebitda"):
        enterprise = peers["ev_ebitda"] * fundamentals["ebitda"]
        estimates["ev_ebitda"] = (enterprise + cash - debt) / shares if shares else float("nan")
        detail["ev_ebitda"] = {"company": fundamentals["ev_ebitda"], "peers": peers["ev_ebitda"]}
    estimates = {key: value for key, value in estimates.items() if value and value > 0 and np.isfinite(value)}
    if not estimates:
        return None, {"reason": "nenhum múltiplo comparável disponível"}
    return float(np.mean(list(estimates.values()))), {"estimates": estimates, "detail": detail, "peer_count": peers.get("count"), "sector": peers.get("sector")}


def quality_signals(context: Context) -> list[dict]:
    """Comparação dos indicadores com a mediana dos pares (não define preço)."""
    fundamentals = context.fundamentals or {}
    peers = context.peers or {}
    out = []
    for key, label in (("roe", "ROE"), ("net_margin", "Margem líquida"), ("revenue_growth", "Crescimento da receita"),
                       ("earnings_growth", "Crescimento do lucro"), ("net_debt_ebitda", "Dívida líquida/EBITDA"),
                       ("dividend_yield", "Dividend yield")):
        value = fundamentals.get(key)
        if value is None:
            continue
        peer_value = peers.get(f"{key}_median")
        if peer_value is None:
            out.append({"name": label, "value": round(float(value), 4), "peers": None})
            continue
        better = value >= peer_value
        if key == "net_debt_ebitda":
            better = value <= peer_value
        out.append({"name": label, "value": round(float(value), 4), "peers": round(float(peer_value), 4), "better": bool(better)})
    return out


def run(context: Context) -> CoreEstimate:
    estimate = CoreEstimate(core="fundamentals", label="Fundamentos e valuation")
    if not context.fundamentals:
        return CoreEstimate.skipped("fundamentals", estimate.label, "A fonte não forneceu fundamentos para esta empresa.")
    dcf, dcf_detail = dcf_value(context)
    relative, relative_detail = relative_value(context)
    models = {"dcf": dcf, "relativo": relative}
    available = {key: value for key, value in models.items() if value is not None}
    price = context.price
    convergence = 1.0 - math.exp(-context.horizon_td / config.VALUATION_CONVERGENCE_TAU)
    estimate.data = {
        "dcf": dcf_detail,
        "relative": relative_detail,
        "fair_value": float(np.mean(list(available.values()))) if available else None,
        "models_available": sorted(available),
        "convergence_factor": round(convergence, 4),
        "quality": quality_signals(context),
        "metrics": {key: value for key, value in context.fundamentals.items() if isinstance(value, (int, float))},
    }
    if not available:
        estimate.status = "skip"
        estimate.notes.append("Nenhum modelo de valuation pôde ser calculado com os dados disponíveis.")
        return estimate
    fair = float(np.mean(list(available.values())))
    upside = fair / price - 1.0
    expected = upside * convergence
    estimate.expected_return = float(np.clip(expected, -0.9, 3.0))
    estimate.sigma = abs(float(np.std(list(available.values()))) / price) if len(available) > 1 else abs(upside) * 0.5
    estimate.confidence = round(min(0.8, 0.30 + 0.20 * len(available) + (0.10 if (context.peers or {}).get("count", 0) >= 4 else 0.0)), 3)
    for key, value in available.items():
        estimate.add_signal(
            f"Valor justo ({'DCF' if key == 'dcf' else 'múltiplos de pares'})",
            (value / price - 1.0) * convergence * 100,
            "alta" if value > price else "baixa",
            f"valor estimado por ação equivalente a {value / price - 1.0:+.1%} frente ao preço atual; convergência parcial de {convergence:.0%} no horizonte",
        )
    estimate.notes.append(
        "O valor justo é uma estimativa sensível às premissas (WACC, crescimento, múltiplos dos pares). "
        "A convergência ao valor justo é assumida como parcial e gradual, não imediata."
    )
    estimate.sort_signals()
    estimate.status = "ok"
    return estimate
