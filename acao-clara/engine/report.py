"""Orquestrador do motor: monta o contexto, executa os oito núcleos, combina as
camadas A/B/C, gera a distribuição de cenários, mede o risco e devolve o pacote
completo que a página exibe.

Ordem de execução:
  1. contexto (preços, fatores, macro, fundamentos, pares, eventos) — sem lookahead;
  2. descoberta de variáveis (com correção de múltiplos testes);
  3. Núcleos 01–07 (Camada A);
  4. pesos dinâmicos (Camada B) e ajustes de risco (Camada C);
  5. cenários e probabilidades;
  6. Núcleo 08 (risco e confiabilidade);
  7. registro no livro de previsões, para aprendizado posterior.
"""
from __future__ import annotations

import math
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config, discovery, ensemble, events, flows, interdependence, macro, peers, risk, simulation, timeseries, valuation
from .context import Context, series_from_rows
from .estimate import CoreEstimate
from .learning import ForecastLedger
from .series import MarketData

CORE_RUNNERS = [
    ("fundamentals", valuation.run),
    ("macro", macro.run),
    ("interdependence", interdependence.run),
    ("events", events.run),
    ("flows", flows.run),
    ("statistics", timeseries.run),
    ("simulation", simulation.run),
]

CORE_LABELS = {
    "fundamentals": "Fundamentos e valuation",
    "macro": "Macroeconomia",
    "interdependence": "Interdependência de mercados",
    "events": "Eventos externos",
    "flows": "Fluxo financeiro",
    "statistics": "Modelagem estatística",
    "simulation": "Simulação de cenários",
}

DISCLAIMER = (
    "Projeção quantitativa demonstrativa produzida por modelos matemáticos e estatísticos. "
    "Não é recomendação de investimento, não é garantia de preço e não considera informações "
    "privilegiadas, custos, tributos ou a situação pessoal de cada investidor."
)


class EngineError(Exception):
    """Erro esperado do motor (mensagem exibível ao usuário)."""


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------
def build_context(
    data: MarketData,
    symbol: str,
    horizon_days: int = config.DEFAULT_HORIZON,
    with_discovery: bool = True,
    with_candidates: bool = True,
    with_peers: bool = True,
) -> tuple[Context, dict]:
    symbol = symbol.upper().strip()
    if horizon_days not in config.HORIZONS_DAYS:
        raise EngineError("Horizonte inválido. Use 7, 30, 90, 180 ou 365 dias.")
    history = data.history(symbol, years=10)
    if history is None or history.empty:
        raise EngineError("Não há histórico suficiente para este ativo na fonte de dados.")
    if len(history) < config.MIN_HISTORY_BARS:
        raise EngineError(f"Histórico insuficiente: {len(history)} pregões disponíveis; o motor precisa de pelo menos {config.MIN_HISTORY_BARS}.")
    fundamentals = data.fundamentals(symbol)
    sector = peers.sector_of(symbol, (fundamentals or {}).get("sector"))
    peer_set = (peers.sample_peers(data, symbol) if with_peers else None) or {}
    macro_raw = data.macro()
    macro_series = {key: series_from_rows(payload.get("rows", []), key) for key, payload in macro_raw.items()}
    macro_series = {key: value for key, value in macro_series.items() if not value.empty}
    macro_meta = {key: {field: payload.get(field) for field in ("label", "source", "source_name", "verified", "unit", "sign_prior")} for key, payload in macro_raw.items()}
    driver_keys = interdependence.drivers_for(symbol, sector)
    drivers, kinds = interdependence.resolve_series(data, driver_keys, macro_series)
    all_factors = data.histories([entry for entry in config.FACTOR_SYMBOLS], years=5) if with_candidates else {}
    candidates = {}
    for factor_symbol, frame in all_factors.items():
        key = _factor_key(factor_symbol)
        if key in drivers:
            continue
        candidates[key] = pd.Series(frame["close"].to_numpy(dtype=float), index=pd.DatetimeIndex([pd.Timestamp(d) for d in frame["date"]]))
    context = Context(
        symbol=symbol,
        name=(fundamentals or {}).get("name") or symbol.removesuffix(".SA"),
        as_of=history["date"].iloc[-1],
        history=history,
        drivers=drivers,
        candidates=candidates,
        macro=macro_series,
        macro_meta=macro_meta,
        fundamentals=fundamentals,
        peers=peer_set,
        events=[],
        horizon_days=horizon_days,
        sector=sector,
        demo=data.demo,
    )
    context.events = events.build_events(data, symbol, sector, context.as_of)
    found = {"ok": False, "reason": "descoberta desativada", "discoveries": []}
    if with_discovery and candidates:
        try:
            found = discovery.scan(context)
            extra = discovery.suggested_additions(context, found.get("discoveries", []), limit=3)
            for key in extra:
                series = context.candidates.pop(key, None)
                if series is not None:
                    context.drivers[key] = series
            if extra:
                context.notes.append(f"Descoberta automática acrescentou: {', '.join(interdependence.driver_label(key) for key in extra)}.")
        except Exception as exc:  # descoberta é opcional
            found = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "discoveries": []}
    return context, found


def _factor_key(factor_symbol: str) -> str:
    for key, entry in interdependence.load_driver_spec().get("registry", {}).items():
        if entry.get("symbol") == factor_symbol:
            return key
    return factor_symbol.replace("^", "idx_").replace("=", "_").replace(".", "_")


def macro_overview(data: MarketData) -> dict:
    """Painel macroeconômico isolado (situação, direção e velocidade de cada indicador)."""
    macro_raw = data.macro()
    macro_series = {key: series_from_rows(payload.get("rows", []), key) for key, payload in macro_raw.items()}
    macro_series = {key: value for key, value in macro_series.items() if not value.empty}
    macro_meta = {key: {field: payload.get(field) for field in ("label", "source", "source_name", "verified", "unit", "sign_prior")} for key, payload in macro_raw.items()}
    stub = Context(symbol="MACRO", name="Indicadores macroeconômicos", as_of=datetime.now().date(), history=pd.DataFrame({"date": [], "close": []}), macro=macro_series, macro_meta=macro_meta, demo=data.demo)
    return {
        "indicators": macro.macro_state(stub),
        "sources": data.health_report(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Nível atual, direção (inclinação) e velocidade (aceleração) de cada indicador, com o desvio em relação à própria média histórica.",
    }


# ---------------------------------------------------------------------------
# Projeção
# ---------------------------------------------------------------------------
def projection(
    data: MarketData,
    symbol: str,
    horizon_days: int = config.DEFAULT_HORIZON,
    ledger: ForecastLedger | None = None,
    record: bool = True,
    mode: str = "full",
) -> dict:
    fast = mode == "fast"
    context, found = build_context(data, symbol, horizon_days, with_discovery=not fast, with_candidates=not fast, with_peers=not fast)
    estimates: list[CoreEstimate] = []
    for core, runner in CORE_RUNNERS:
        try:
            estimate = runner(context)
        except Exception as exc:
            traceback.print_exc()
            estimate = CoreEstimate.failed(core, CORE_LABELS[core], f"{type(exc).__name__}: {exc}"[:200])
        estimates.append(estimate)

    _clip_cores(estimates, context)

    accuracy = ledger.accuracy(symbol, horizon_days) if ledger else None
    if ledger is not None:
        _score_symbol(ledger, data, symbol, context.history)
        accuracy = ledger.accuracy(symbol, horizon_days)
    weights = ensemble.dynamic_weights(estimates, horizon_days, accuracy, ledger=ledger, symbol=symbol)
    combined = ensemble.combine(estimates, weights)
    if not combined.get("ok"):
        raise EngineError("Nenhum núcleo conseguiu produzir uma estimativa com os dados disponíveis.")

    annual_vol = _annual_vol(context)
    quality_ratio = len([item for item in estimates if item.status == "ok"]) / max(1, len(estimates))
    adjustments: list[dict] = []
    expected_return = ensemble.apply_adjustments(combined["expected_return"], context, annual_vol, quality_ratio, adjustments)
    expected_price = float(context.price * (1.0 + expected_return))

    sim_estimate = next((item for item in estimates if item.core == "simulation"), None)
    wealth = (sim_estimate.raw.get("wealth") if sim_estimate else None)
    scenarios: dict = {"ok": False, "reason": "simulação indisponível"}
    if wealth is not None:
        target_sigma = _combined_log_sigma(context, combined, annual_vol)
        scenarios = simulation.build_scenarios(np.asarray(wealth, dtype=float), context.price, expected_return, target_sigma)

    statistics_estimate = next((item for item in estimates if item.core == "statistics"), None)
    logistic_probability = (statistics_estimate.data or {}).get("probability_up") if statistics_estimate else None
    probability_up = _blend_probability(logistic_probability, scenarios.get("probability_up"))

    matrix = (next((item for item in estimates if item.core == "interdependence"), CoreEstimate(core="interdependence", label="")).data or {}).get("matrix", {})
    garch = (sim_estimate.data or {}).get("garch") if sim_estimate else None
    risk_report = risk.run(context, estimates, matrix, scenarios if scenarios.get("ok") else None, garch, accuracy)

    for item in estimates:
        item.weight_base = weights.get(item.core, {}).get("weight_base", 0.0)
        item.weight_final = weights.get(item.core, {}).get("weight", 0.0)

    payload = {
        "symbol": context.symbol,
        "name": context.name,
        "sector": context.sector,
        "as_of": str(context.as_of),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "price": round(float(context.price), 4),
        "currency": context.currency,
        "horizon_days": context.horizon_days,
        "horizon_label": config.HORIZON_LABELS.get(context.horizon_days, ""),
        "horizon_trading_days": context.horizon_td,
        "projection": {
            "expected_price": round(expected_price, 4),
            "expected_return": round(expected_return, 6),
            "expected_return_pp": round(expected_return * 100, 3),
            "probability_up": None if probability_up is None else round(probability_up, 4),
            "range": scenarios.get("range"),
            "scenarios": scenarios.get("scenarios", []),
            "potential_gain": scenarios.get("potential_gain"),
            "potential_loss": scenarios.get("potential_loss"),
            "risk_reward": scenarios.get("risk_reward"),
            "var95_pp": scenarios.get("var95_pp"),
            "cvar95_pp": scenarios.get("cvar95_pp"),
        },
        "cores": [item.to_json() for item in estimates],
        "weights": weights,
        "adjustments": adjustments,
        "combined": {key: value for key, value in combined.items() if key != "contributions"} | {"contributions": combined.get("contributions", [])},
        "drivers": {
            "programmed": sorted(context.drivers),
            "matrix": matrix,
            "discovery": found,
        },
        "macro": macro.macro_state(context),
        "events": (next((item for item in estimates if item.core == "events"), CoreEstimate(core="events", label="")).data or {}).get("events", []),
        "flows": (next((item for item in estimates if item.core == "flows"), CoreEstimate(core="flows", label="")).data or {}),
        "risk": risk_report,
        "learning": _learning_block(ledger, symbol, horizon_days, expected_price, context.price, expected_return),
        "context": context.summary(),
        "sources": data.health_report(),
        "mode": mode,
        "demo": context.demo,
        "disclaimer": DISCLAIMER,
    }
    if record and ledger is not None:
        try:
            payload["learning"]["recorded_id"] = ledger.record(
                {
                    "symbol": context.symbol,
                    "horizon_days": horizon_days,
                    "ref_date": str(context.as_of),
                    "price": context.price,
                    "expected_price": expected_price,
                    "expected_return": expected_return,
                    "probability_up": probability_up,
                    "range": scenarios.get("range"),
                    "confidence": risk_report["confidence"],
                    "models": {item.core: item.expected_return for item in estimates if item.usable},
                    "weights": {core: detail.get("weight") for core, detail in weights.items()},
                }
            )
        except Exception as exc:
            payload["learning"]["record_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return payload


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def _clip_cores(estimates: list[CoreEstimate], context: Context) -> None:
    """Limita cada núcleo a 3 desvios-padrão do horizonte (teste de sanidade)."""
    annual = _annual_vol(context)
    if not np.isfinite(annual):
        return
    limit = 3.0 * annual * math.sqrt(context.horizon_td / 252.0)
    for item in estimates:
        if not item.usable or abs(item.expected_return) <= limit:
            continue
        original = item.expected_return
        item.expected_return = float(np.sign(original) * limit)
        item.notes.append(
            f"Estimativa original de {original:+.1%} foi limitada a 3σ do horizonte ({limit:+.1%}) por ser implausível para a volatilidade do ativo."
        )


def _annual_vol(context: Context) -> float:
    returns = context.close.pct_change().to_numpy()
    returns = returns[np.isfinite(returns)]
    if returns.size < 60:
        return float("nan")
    return float(np.std(returns[-252:], ddof=1) * math.sqrt(252))


def _combined_log_sigma(context: Context, combined: dict, annual_vol: float) -> float | None:
    """Dispersão combinada: simulação + divergência entre núcleos."""
    spread = combined.get("model_spread") or 0.0
    if not annual_vol or not np.isfinite(annual_vol):
        return None
    base = annual_vol * math.sqrt(context.horizon_td / 252.0)
    return float(math.sqrt(base**2 + (spread**2)))


def _blend_probability(logistic_probability, scenario_probability) -> float | None:
    values = [value for value in (logistic_probability, scenario_probability) if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.mean(values))


def _score_symbol(ledger: ForecastLedger, data: MarketData, symbol: str, history: pd.DataFrame) -> None:
    """Confere as previsões maduras deste ativo usando o histórico já carregado."""
    try:
        due = [row for row in ledger.due() if row["symbol"].upper() == symbol.upper()][:5]
    except Exception:
        return
    if not due:
        return
    dates = pd.to_datetime(history["date"])

    def lookup(sym: str, target) -> float | None:
        mask = dates >= pd.Timestamp(target)
        if not mask.any():
            return None
        return float(history["close"].to_numpy()[np.flatnonzero(mask.to_numpy())[0]])

    try:
        ledger.score(lookup)
    except Exception:
        traceback.print_exc()


def _learning_block(ledger: ForecastLedger | None, symbol: str, horizon_days: int, expected_price: float, price: float, expected_return: float) -> dict:
    if ledger is None:
        return {"enabled": False, "reason": "livro de previsões indisponível nesta instância"}
    block: dict = {"enabled": True, "stats": ledger.stats(), "accuracy": ledger.accuracy(symbol, horizon_days), "accuracy_all": ledger.accuracy()}
    previous = ledger.previous(symbol, horizon_days)
    if previous and previous.get("previous"):
        older = previous["previous"]
        block["change_since_last"] = {
            "previous_created_at": older["created_at"],
            "previous_expected_price": older["expected_price"],
            "previous_expected_return_pp": None if older["expected_return"] is None else round(float(older["expected_return"]) * 100, 3),
            "current_expected_price": round(expected_price, 4),
            "delta_pp": round((expected_return - float(older["expected_return"] or 0.0)) * 100, 3) if older["expected_return"] is not None else None,
            "note": "Comparação entre a projeção atual e a anterior registrada para o mesmo ativo e horizonte.",
        }
    return block
