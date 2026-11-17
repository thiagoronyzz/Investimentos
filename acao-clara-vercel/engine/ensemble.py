"""Combinação dos núcleos: Camadas A, B e C.

Camada A — cada núcleo produz sua própria estimativa (ver os módulos `macro`,
`valuation`, `interdependence`, `events`, `flows`, `timeseries` e `simulation`).

Camada B — pesos dinâmicos: o peso-base de cada horizonte é multiplicado por um
fator que depende da precisão histórica registrada no livro de previsões, da
confiança do núcleo na execução atual, da qualidade dos dados e da
redundância com outros núcleos. Limites impedem que um único núcleo domine.

Camada C — ajustes de risco: a estimativa combinada passa por testes de
sanidade (limite absoluto, limite por volatilidade do horizonte e redução por
qualidade de dados) antes de virar projeção.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np

from . import config
from .context import Context
from .estimate import CoreEstimate


def base_weights(horizon_days: int) -> dict[str, float]:
    horizon = horizon_days if horizon_days in config.CORE_WEIGHTS else min(config.CORE_WEIGHTS, key=lambda item: abs(item - horizon_days))
    return dict(config.CORE_WEIGHTS[horizon])


def _blend(accuracy_models: dict, core: str, confidence: float) -> float:
    """Mistura a precisão histórica registrada com a confiança da execução atual."""
    record = (accuracy_models or {}).get(core)
    skill = record.get("skill") if record else None
    if skill is None:
        return float(confidence)
    return float(0.6 * skill + 0.4 * confidence)


def _redundancy_penalty(ledger, symbol: str | None, cores: list[str]) -> dict[str, float]:
    """Reduz o peso de núcleos que erram de forma quase idêntica (mesma informação)."""
    penalty = {core: 1.0 for core in cores}
    if ledger is None or len(cores) < 2:
        return penalty
    rows = ledger.history(symbol, limit=200)
    scored = [row for row in rows if row.get("realized_return") is not None]
    if len(scored) < 8:
        return penalty
    errors: dict[str, list[float]] = {}
    for row in scored:
        try:
            models = json.loads(row.get("models") or "{}")
        except ValueError:
            continue
        actual = float(row["realized_return"])
        for core in cores:
            value = models.get(core)
            if value is None:
                continue
            errors.setdefault(core, []).append(abs(actual - float(value)))
    for index, first in enumerate(cores):
        for second in cores[index + 1:]:
            a, b = errors.get(first, []), errors.get(second, [])
            if len(a) < 8 or len(b) < 8 or len(a) != len(b):
                continue
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            correlation = float(np.corrcoef(a, b)[0, 1])
            if correlation > 0.8:
                penalty[second] = min(penalty[second], 0.7)
    return penalty


def dynamic_weights(estimates: Iterable[CoreEstimate], horizon_days: int, accuracy: dict | None, ledger=None, symbol: str | None = None) -> dict[str, dict]:
    base = base_weights(horizon_days)
    usable = [item for item in estimates if item.usable]
    if not usable:
        return {}
    accuracy_models = (accuracy or {}).get("models", {}) if accuracy else {}
    names = [item.core for item in usable]
    penalty = _redundancy_penalty(ledger, symbol, names)
    raw: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for item in usable:
        weight_base = base.get(item.core, 0.0)
        blend = _blend(accuracy_models, item.core, item.confidence)
        multiplier = float(np.clip(2.0 * blend, 1.0 / config.WEIGHT_ADJUST_CAP, config.WEIGHT_ADJUST_CAP))
        raw[item.core] = weight_base * multiplier * penalty.get(item.core, 1.0)
        detail[item.core] = {
            "weight_base": round(weight_base, 4),
            "confidence": round(float(item.confidence), 3),
            "historical_skill": accuracy_models.get(item.core, {}).get("skill"),
            "blend": round(blend, 3),
            "multiplier": round(multiplier, 3),
            "redundancy_penalty": penalty.get(item.core, 1.0),
        }
    weights = _normalize(raw)
    for core, weight in weights.items():
        detail[core]["weight"] = round(weight, 4)
    return detail


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    """Normaliza os pesos com piso e teto por "preenchimento de água" (water-filling).

    O teto de `MAX_CORE_WEIGHT` só é aplicável quando é viável: com poucos
    núcleos ativos cada um precisa necessariamente de uma fatia maior (com dois
    núcleos, 50% é o mínimo possível), e o teto é relaxado nesse caso. Os núcleos
    que estourariam o teto ficam travados no limite e o restante é redistribuído
    proporcionalmente entre os demais.
    """
    items = sorted(
        [(core, max(value, config.MIN_CORE_WEIGHT)) for core, value in raw.items() if value > 0],
        key=lambda pair: pair[1],
        reverse=True,
    )
    if not items:
        return {}
    cap = max(config.MAX_CORE_WEIGHT, 1.0 / len(items))
    weights: dict[str, float] = {}
    remaining = list(items)
    fixed_count = 0
    while remaining:
        budget = 1.0 - cap * fixed_count
        total = sum(value for _, value in remaining)
        if budget <= 0 or total <= 0:
            break
        candidate = {core: budget * value / total for core, value in remaining}
        violators = [core for core, value in remaining if candidate[core] > cap + 1e-12]
        if not violators:
            weights.update(candidate)
            break
        for core in violators:
            weights[core] = cap
            fixed_count += 1
        remaining = [pair for pair in remaining if pair[0] not in violators]
    total = sum(weights.values())
    if total > 0:
        weights = {core: value / total for core, value in weights.items()}
    return weights


def combine(estimates: list[CoreEstimate], weight_detail: dict[str, dict]) -> dict:
    usable = {item.core: item for item in estimates if item.usable and item.core in weight_detail}
    if not usable:
        return {"ok": False, "reason": "nenhum núcleo utilizável"}
    expected = 0.0
    contributions = []
    for core, item in usable.items():
        weight = weight_detail[core]["weight"]
        expected += weight * item.expected_return
        contributions.append({"core": core, "label": item.label, "weight": round(weight, 4), "return_pp": round(item.expected_return * 100, 3), "contribution_pp": round(weight * item.expected_return * 100, 3)})
    contributions.sort(key=lambda row: abs(row["contribution_pp"]), reverse=True)
    sigmas = [item.sigma for item in usable.values() if item.sigma is not None and np.isfinite(item.sigma)]
    return {
        "ok": True,
        "expected_return": float(expected),
        "contributions": contributions,
        "weighted_sigma": float(np.sqrt(sum((weight_detail[core]["weight"] * usable[core].sigma) ** 2 for core in usable if usable[core].sigma is not None and np.isfinite(usable[core].sigma)))),
        "model_spread": float(np.std([item.expected_return for item in usable.values()])) if len(usable) > 1 else 0.0,
        "sigma_pool": float(np.mean(sigmas)) if sigmas else float("nan"),
    }


def apply_adjustments(expected_return: float, context: Context, annual_vol: float | None, quality_ratio: float, adjustments: list[dict]) -> float:
    """Camada C: limites de sanidade aplicados à estimativa combinada."""
    value = float(expected_return)
    adjustments.append({"step": "combinação ponderada", "value_pp": round(value * 100, 3), "detail": "soma das contribuições dos núcleos"})
    value = float(np.clip(value, -config.MAX_ABS_RETURN, config.MAX_ABS_RETURN))
    if annual_vol and np.isfinite(annual_vol) and annual_vol > 0:
        sigma_horizon = annual_vol * math.sqrt(context.horizon_td / 252.0)
        limit = 2.5 * sigma_horizon
        if abs(value) > limit:
            adjustments.append(
                {
                    "step": "limite por volatilidade",
                    "value_pp": round(value * 100, 3),
                    "detail": f"retorno combinado excedia 2,5σ do horizonte ({limit:.1%}); reduzido para o limite",
                }
            )
            value = float(np.sign(value) * limit)
    quality_ratio = float(np.clip(quality_ratio, 0.0, 1.0))
    shrink = 0.70 + 0.30 * quality_ratio
    if shrink < 1.0:
        adjustments.append({"step": "redução por qualidade de dados", "value_pp": round(value * shrink * 100, 3), "detail": f"fator {shrink:.2f} conforme núcleos ativos e cobertura de dados"})
        value *= shrink
    return float(np.clip(value, -config.MAX_ABS_RETURN, config.MAX_ABS_RETURN))
