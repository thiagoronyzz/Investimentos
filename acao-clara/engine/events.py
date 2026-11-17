"""Núcleo 04 — Eventos externos, políticos e geopolíticos.

Sem IA: a classificação é feita por regras programadas (`data/event_rules.json`)
aplicadas a manchetes e a um livro de eventos revisado manualmente
(`data/events.json`). Cada evento recebe intensidade, direção, setores
afetados, horizonte, certeza e parcela já precificada; o impacto decai com o
tempo e é limitado por saturação para que nenhum acontecimento isolado domine a
projeção. O motor não presume que toda notícia positiva gere alta.
"""
from __future__ import annotations

import json
import math
import unicodedata
from datetime import date, datetime

import numpy as np

from . import config
from .context import Context
from .estimate import CoreEstimate

DECAY_DAYS = config.EVENT_DECAY_DAYS
SATURATION = config.EVENT_MAX_IMPACT


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFD", str(text or "").lower()).encode("ascii", "ignore").decode("ascii")


def load_rules() -> dict:
    try:
        return json.loads((config.DATA_DIR / "event_rules.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"types": {}, "sector_terms": {}, "impact_by_horizon": {}}


def load_ledger() -> list[dict]:
    try:
        data = json.loads((config.DATA_DIR / "events.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(data.get("events", []))


def impact_share(horizon_days: int, rules: dict) -> float:
    table = rules.get("impact_by_horizon", {})
    return float(table.get(str(horizon_days), table.get("90", 0.3)))


def detect_sectors(title: str, rules: dict) -> list[str]:
    text = _normalize(title)
    found = []
    for sector, terms in rules.get("sector_terms", {}).items():
        if any(_normalize(term) in text for term in terms):
            found.append(sector)
    return found


def classify(title: str, rules: dict) -> dict | None:
    """Classifica uma manchete pelo tipo com maior número de palavras-chave."""
    text = _normalize(title)
    best, best_hits = None, 0
    for type_key, rule in rules.get("types", {}).items():
        hits = sum(1 for keyword in rule.get("keywords", []) if _normalize(keyword) in text)
        if hits > best_hits:
            best, best_hits = (type_key, rule), hits
    if best is None:
        return None
    type_key, rule = best
    return {
        "type": type_key,
        "label": rule.get("label", type_key),
        "direction": int(rule.get("direction", 0)),
        "sectors_rule": rule.get("sectors", {}),
        "intensity": float(rule.get("intensity", 5)),
        "horizon_days": int(rule.get("horizon_days", 90)),
        "certainty": float(rule.get("certainty", 0.5)),
        "priced_in": float(rule.get("priced_in", 0.4)),
        "note": rule.get("note", ""),
        "matched_keywords": [keyword for keyword in rule.get("keywords", []) if _normalize(keyword) in text][:4],
        "sectors_mentioned": detect_sectors(title, rules),
    }


def resolve_direction(rule_sectors: dict, default: int, sector: str | None) -> int:
    if sector and sector in rule_sectors:
        return int(rule_sectors[sector])
    return int(rule_sectors.get("*", default))


def build_events(data, symbol: str, sector: str | None, as_of, rules: dict | None = None, limit: int = 25) -> list[dict]:
    """Junta o livro manual e as manchetes classificadas, já com direção resolvida."""
    rules = rules or load_rules()
    ticker = symbol.upper().removesuffix(".SA")
    today = as_of if isinstance(as_of, date) else datetime.now().date()
    out: list[dict] = []
    for event in load_ledger():
        tickers = [str(item).upper() for item in event.get("tickers", [])]
        sectors = event.get("sectors", [])
        if tickers and ticker not in tickers:
            continue
        if not tickers and sectors and "*" not in sectors and (sector not in sectors):
            continue
        rule = rules.get("types", {}).get(event.get("type", ""), {})
        out.append(
            {
                "source": "manual",
                "title": event.get("title", event.get("type", "evento")),
                "date": str(event.get("date", today)),
                "type": event.get("type", "manual"),
                "label": event.get("title", rule.get("label", "evento manual")),
                "direction": int(event.get("direction", rule.get("direction", 0))),
                "intensity": float(event.get("intensity", rule.get("intensity", 5))),
                "horizon_days": int(event.get("horizon_days", rule.get("horizon_days", 90))),
                "certainty": float(event.get("certainty", rule.get("certainty", 0.5))),
                "priced_in": float(event.get("priced_in", rule.get("priced_in", 0.4))),
                "note": event.get("note", rule.get("note", "")),
                "publisher": "livro de eventos",
            }
        )
    for item in data.news(symbol, limit=limit):
        title = item.get("title") or ""
        if not title:
            continue
        classification = classify(title, rules)
        if classification is None:
            continue
        direction = resolve_direction(classification["sectors_rule"], classification["direction"], sector)
        mentioned = ticker.lower() in _normalize(item.get("publisher", "")) or ticker.lower() in _normalize(title)
        if direction == 0 and not mentioned:
            continue
        out.append(
            {
                "source": "regra",
                "title": title,
                "date": str(item.get("date") or today),
                "type": classification["type"],
                "label": classification["label"],
                "direction": direction,
                "intensity": classification["intensity"],
                "horizon_days": classification["horizon_days"],
                "certainty": classification["certainty"],
                "priced_in": classification["priced_in"],
                "note": classification["note"],
                "matched_keywords": classification["matched_keywords"],
                "sectors_mentioned": classification["sectors_mentioned"],
                "company_mentioned": bool(mentioned),
                "publisher": item.get("publisher", ""),
            }
        )
    for event in out:
        try:
            event["age_days"] = max(0, (today - datetime.fromisoformat(str(event["date"])[:10]).date()).days)
        except ValueError:
            event["age_days"] = 0
    out.sort(key=lambda event: event["age_days"])
    return out


def event_impact(event: dict, horizon_days: int, rules: dict) -> dict:
    """Impacto estimado de um evento, em retorno decimal, já com decaimento."""
    direction = float(event.get("direction", 0))
    if direction == 0:
        return {"impact": 0.0, "reason": "direção neutra ou indefinida"}
    base = impact_share(horizon_days, rules)
    horizon_share = min(1.0, horizon_days / max(1, int(event.get("horizon_days", 90))))
    freshness = math.exp(-float(event.get("age_days", 0)) / DECAY_DAYS)
    certainty = float(event.get("certainty", 0.5))
    priced_in = float(event.get("priced_in", 0.4))
    impact = direction * (float(event.get("intensity", 5)) / 10.0) * base * horizon_share * freshness * certainty * (1.0 - 0.6 * priced_in)
    return {
        "impact": float(impact),
        "base": base,
        "horizon_share": round(horizon_share, 3),
        "freshness": round(freshness, 3),
        "certainty": certainty,
        "priced_in_adjustment": round(1.0 - 0.6 * priced_in, 3),
    }


def run(context: Context) -> CoreEstimate:
    estimate = CoreEstimate(core="events", label="Eventos externos")
    rules = load_rules()
    events = context.events or []
    scored = []
    for event in events:
        detail = event_impact(event, context.horizon_days, rules)
        scored.append({**event, **detail})
    scored.sort(key=lambda item: abs(item["impact"]), reverse=True)
    estimate.data["events"] = scored[:20]
    estimate.data["total_events"] = len(scored)
    if not scored:
        return CoreEstimate.skipped("events", estimate.label, "Nenhum evento classificável foi encontrado para esta empresa no período.")
    raw = float(sum(item["impact"] for item in scored))
    total = float(SATURATION * math.tanh(raw / SATURATION))
    for item in scored[:8]:
        if abs(item["impact"]) < 1e-5:
            continue
        estimate.add_signal(
            f"{item['label']} ({item['date']})",
            item["impact"] * 100,
            "alta" if item["impact"] > 0 else "baixa",
            f"intensidade {item['intensity']:.0f}/10 · certeza {item['certainty']:.0%} · {item['age_days']} dias · já precificado {item['priced_in_adjustment']:.0%}",
        )
    estimate.expected_return = float(np.clip(total, -0.9, 3.0))
    estimate.sigma = float(np.sqrt(sum(item["impact"] ** 2 for item in scored)))
    active = [item for item in scored if abs(item["impact"]) > 1e-5]
    mean_certainty = float(np.mean([item["certainty"] for item in active])) if active else 0.0
    estimate.confidence = round(float(np.clip(0.20 + 0.35 * mean_certainty + 0.10 * min(1.0, len(active) / 4.0), 0.0, 0.75)), 3)
    estimate.notes.append(
        "A classificação é por regras de palavras-chave e pode errar sentido e intensidade. "
        "O impacto decai com o tempo e é saturado para evitar que um único evento domine a projeção."
    )
    estimate.sort_signals()
    estimate.status = "ok" if active else "skip"
    if not active:
        estimate.notes.append("Os eventos encontrados foram classificados como neutros para esta empresa.")
    return estimate
