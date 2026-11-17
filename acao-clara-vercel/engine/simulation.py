"""Núcleo 07 — Simulação de cenários futuros.

O núcleo responde "e se as condições mudarem?" com simulação de Monte Carlo:
milhares de trajetórias de preço com volatilidade estimada por GARCH(1,1),
caudas gordas (distribuição t de Student) e deriva histórica encolhida. A
simulação fornece a *forma* da distribuição (dispersão, assimetria, drawdown);
o centro é definido pela combinação dos núcleos na Camada C, para evitar que a
simulação use a própria resposta como premissa.
"""
from __future__ import annotations

import math

import numpy as np

from . import config, stats
from .context import Context
from .estimate import CoreEstimate

DRIFT_SHRINKAGE = 0.5
MAX_DAILY_DRIFT = 0.002
VOL_PERSISTENCE = 0.10


def run(context: Context) -> CoreEstimate:
    estimate = CoreEstimate(core="simulation", label="Simulação de cenários")
    close = context.close.to_numpy()
    returns = stats.returns(close)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 250:
        return CoreEstimate.skipped("simulation", estimate.label, "Histórico curto demais para simular trajetórias.")
    drift_daily = float(np.clip(np.mean(returns) * DRIFT_SHRINKAGE, -MAX_DAILY_DRIFT, MAX_DAILY_DRIFT))
    garch = stats.garch11(returns, context.horizon_td)
    if garch.ok:
        daily_vol = float(garch.forecast_vol / math.sqrt(252))
        vol_source = f"GARCH(1,1) projetado (α={garch.alpha:.3f}, β={garch.beta:.3f})"
    else:
        daily_vol = float(np.std(returns[-126:], ddof=1))
        vol_source = "volatilidade histórica de 126 pregões (GARCH indisponível)"
    daily_vol = max(daily_vol, 1e-4)
    result = stats.monte_carlo(
        daily_drift=drift_daily,
        daily_vol=daily_vol,
        steps=context.horizon_td,
        paths=config.MC_PATHS,
        seed=config.MC_SEED,
        tail_df=6.0,
        vol_persistence=VOL_PERSISTENCE,
    )
    wealth = result["final"]
    estimate.raw["wealth"] = wealth
    logs = np.log(np.clip(wealth, 1e-6, None))
    estimate.expected_return = float(np.clip(result["mean"] - 1.0, -0.9, 3.0))
    estimate.sigma = float(np.std(logs))
    estimate.confidence = round(float(np.clip(0.35 + 0.25 * (1.0 if garch.ok else 0.0) + 0.10 * min(1.0, len(returns) / 2000.0), 0.0, 0.8)), 3)
    estimate.data = {
        "paths": result["paths"],
        "steps": result["steps"],
        "drift_daily": round(drift_daily, 6),
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "daily_vol": round(daily_vol, 6),
        "annual_vol": round(daily_vol * math.sqrt(252), 4),
        "vol_source": vol_source,
        "garch": garch.to_json(),
        "mean_wealth": round(result["mean"], 5),
        "median_wealth": round(result["median"], 5),
        "p_up": round(result["p_up"], 4),
        "quantiles": {key: round(value, 5) for key, value in result["quantiles"].items()},
        "var95": round(result["var95"], 5),
        "cvar95": round(result["cvar95"], 5),
        "max_drawdown_mean": round(result["max_drawdown_mean"], 5),
        "max_drawdown_p95": round(result["max_drawdown_p95"], 5),
        "log_sigma": round(float(np.std(logs)), 5),
    }
    estimate.notes.append(
        "A simulação usa deriva histórica encolhida pela metade e volatilidade GARCH; ela descreve a dispersão possível, "
        "não um caminho provável. O preço esperado final é ajustado pela combinação dos demais núcleos."
    )
    estimate.status = "ok"
    return estimate


def build_scenarios(wealth: np.ndarray, price: float, expected_return: float, target_log_sigma: float | None = None) -> dict:
    """Distribuição de cenários com preços e probabilidades.

    As trajetórias simuladas são reescaladas para o centro (retorno esperado da
    combinação de núcleos) e, quando informado, para a dispersão combinada.
    """
    wealth = np.asarray(wealth, dtype=float)
    wealth = wealth[np.isfinite(wealth) & (wealth > 0)]
    if wealth.size < 100:
        return {"ok": False, "reason": "simulação indisponível"}
    logs = np.log(wealth)
    centered = logs - float(np.mean(logs))
    scale = 1.0
    if target_log_sigma and np.isfinite(target_log_sigma) and float(np.std(logs)) > 0:
        scale = float(np.clip(target_log_sigma / float(np.std(logs)), 0.5, 2.0))
    adjusted = centered * scale
    target_mean = max(1.0 + expected_return, 0.02)
    shift = math.log(target_mean) - math.log(float(np.mean(np.exp(adjusted))))
    adjusted = adjusted + shift
    ratios = np.exp(adjusted)
    order = np.argsort(ratios)[::-1]
    sorted_ratios = ratios[order]
    total = sorted_ratios.size
    scenarios = []
    start = 0
    for index, (name, mass) in enumerate(zip(config.SCENARIO_NAMES, config.SCENARIO_MASSES)):
        last_bucket = index == len(config.SCENARIO_MASSES) - 1
        end = total if last_bucket else int(round(start + mass * total))
        end = min(total, max(end, start + 1))
        chunk = sorted_ratios[start:end]
        if chunk.size == 0:
            break
        scenarios.append(
            {
                "name": name,
                "price": round(float(price * np.mean(chunk)), 4),
                "return_pp": round(float((np.mean(chunk) - 1.0) * 100), 3),
                "probability": round(chunk.size / total, 4),
                "from": round(float(price * chunk[-1]), 4),
                "to": round(float(price * chunk[0]), 4),
            }
        )
        start = end
    expected_price = float(price * float(np.mean(ratios)))
    p_up = float(np.mean(ratios > 1.0))
    low = float(price * float(np.quantile(ratios, config.CONFIDENCE_RANGE_QUANTILES[0])))
    high = float(price * float(np.quantile(ratios, config.CONFIDENCE_RANGE_QUANTILES[1])))
    downside = max(price - low, 1e-9)
    upside = max(high - price, 0.0)
    return {
        "ok": True,
        "scenarios": scenarios,
        "expected_price": round(expected_price, 4),
        "expected_return_pp": round((expected_price / price - 1.0) * 100, 3),
        "probability_up": round(p_up, 4),
        "probability_down": round(1.0 - p_up, 4),
        "range": {"low": round(low, 4), "high": round(high, 4), "quantiles": list(config.CONFIDENCE_RANGE_QUANTILES)},
        "potential_gain": round(upside, 4),
        "potential_loss": round(downside, 4),
        "risk_reward": round(upside / downside, 3) if downside > 0 else None,
        "var95_pp": round(float(np.quantile(ratios, 0.05) - 1.0) * 100, 3),
        "cvar95_pp": round(float(np.mean(ratios[ratios <= np.quantile(ratios, 0.05)]) - 1.0) * 100, 3),
        "log_sigma_used": round(float(np.std(np.log(ratios))), 5),
        "scale_applied": round(scale, 4),
    }
