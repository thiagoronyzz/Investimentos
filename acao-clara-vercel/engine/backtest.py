"""Validação walk-forward: o motor volta no tempo e avalia o que teria projetado.

Para cada data-origem, o contexto é truncado exatamente naquela data (os mesmos
caminhos de código usados na projeção corrente), a combinação de núcleos é
reexecutada e o resultado é comparado com o retorno realmente observado no
horizonte. Isso responde quais núcleos antecipam movimentos e quais apenas
parecem bons em retrospecto. Observação: os blocos se sobrepõem ao horizonte,
portanto as observações não são independentes.
"""
from __future__ import annotations

import traceback
from datetime import datetime

import numpy as np

from . import config, ensemble, report
from .context import Context
from .estimate import CoreEstimate

CORE_LABELS = report.CORE_LABELS
CORE_RUNNERS = report.CORE_RUNNERS


def choose_origins(context: Context, count: int = 4, step: int = 126) -> list[int]:
    last = len(context.history) - 1 - context.horizon_td
    floor = max(config.MIN_HISTORY_BARS, int(len(context.history) * 0.35))
    origins = []
    index = last
    while index > floor and len(origins) < count:
        origins.append(index)
        index -= step
    return sorted(origins)


def walk_forward(context: Context, count: int = 4, step: int = 126, ledger=None) -> dict:
    origins = choose_origins(context, count=count, step=step)
    if len(origins) < 2:
        return {"ok": False, "reason": "histórico insuficiente para walk-forward", "origins": []}
    close = context.close.to_numpy()
    rows: list[dict] = []
    per_core: dict[str, list[tuple[float, float]]] = {}
    for index in origins:
        as_of = context.history["date"].iloc[index]
        sub = context.truncate(as_of)
        estimates: list[CoreEstimate] = []
        for core, runner in CORE_RUNNERS:
            try:
                estimates.append(runner(sub))
            except Exception as exc:
                estimates.append(CoreEstimate.failed(core, CORE_LABELS[core], f"{type(exc).__name__}: {exc}"[:160]))
        # Reproduz o pipeline completo: limites por núcleo, pesos dinâmicos e Camada C.
        report._clip_cores(estimates, sub)
        accuracy = ledger.accuracy(context.symbol, context.horizon_days) if ledger is not None else None
        weights = ensemble.dynamic_weights(estimates, context.horizon_days, accuracy, ledger=ledger, symbol=context.symbol)
        combined = ensemble.combine(estimates, weights)
        if not combined.get("ok"):
            continue
        annual_vol = report._annual_vol(sub)
        quality_ratio = len([item for item in estimates if item.status == "ok"]) / max(1, len(estimates))
        expected = ensemble.apply_adjustments(combined["expected_return"], sub, annual_vol, quality_ratio, [])
        realized = float(close[index + context.horizon_td] / close[index] - 1.0)
        row = {
            "origin": str(as_of),
            "expected_return_pp": round(expected * 100, 3),
            "realized_return_pp": round(realized * 100, 3),
            "error_pp": round(abs(combined["expected_return"] - realized) * 100, 3),
            "hit": bool((combined["expected_return"] > 0) == (realized > 0)),
            "cores": {item.core: round(item.expected_return * 100, 3) for item in estimates if item.usable},
        }
        rows.append(row)
        for core, value in row["cores"].items():
            per_core.setdefault(core, []).append((value / 100.0, realized))
    if not rows:
        return {"ok": False, "reason": "nenhuma origem pôde ser avaliada", "origins": []}
    # Erros já em pontos percentuais: não converter de novo.
    ensemble_errors = [abs(row["expected_return_pp"] - row["realized_return_pp"]) for row in rows]
    baseline_errors = [abs(row["realized_return_pp"]) for row in rows]
    predictions = [row["expected_return_pp"] for row in rows]
    realized = [row["realized_return_pp"] for row in rows]
    ic = None
    if len(rows) >= 3 and np.std(predictions) > 0 and np.std(realized) > 0:
        ic = float(np.corrcoef(predictions, realized)[0, 1])
    cores_summary = {}
    for core, pairs in per_core.items():
        errors = [abs(predicted - actual) for predicted, actual in pairs]
        hits = [1 if (predicted > 0) == (actual > 0) else 0 for predicted, actual in pairs]
        cores_summary[core] = {
            "label": CORE_LABELS.get(core, core),
            "mae_pp": round(float(np.mean(errors)) * 100, 3),   # erros por núcleo estão em decimal
            "hit_rate": round(float(np.mean(hits)), 4),
            "n": len(pairs),
        }
    mae = float(np.mean(ensemble_errors))
    baseline_mae = float(np.mean(baseline_errors))
    skill = float(np.clip(1.0 - mae / baseline_mae, 0.0, 1.0)) if baseline_mae > 0 else 0.0
    # Converte os erros por núcleo (calculados em decimal) para pontos percentuais.
    return {
        "ok": True,
        "origins": rows,
        "horizon_days": context.horizon_days,
        "horizon_trading_days": context.horizon_td,
        "ensemble": {
            "mae_pp": round(mae, 3),
            "baseline_mae_pp": round(baseline_mae, 3),
            "better_than_baseline": bool(mae < baseline_mae),
            "hit_rate": round(float(np.mean([1 if row["hit"] else 0 for row in rows])), 4),
            "information_coefficient": None if ic is None else round(ic, 4),
            "skill": round(skill, 3),
            "score": round(0.35 + 0.65 * skill, 3),
            "n": len(rows),
        },
        "cores": cores_summary,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Blocos sobrepostos ao horizonte: as observações não são independentes e o erro tende a ser subestimado.",
    }
