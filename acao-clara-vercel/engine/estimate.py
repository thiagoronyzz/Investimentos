"""Estrutura única de saída de cada núcleo do motor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context import to_jsonable


@dataclass
class CoreEstimate:
    core: str
    label: str
    status: str = "skip"                     # ok | skip | fail
    expected_return: float | None = None     # retorno decimal esperado no horizonte
    sigma: float | None = None               # incerteza da estimativa (decimal)
    confidence: float = 0.0                  # 0..1 qualidade de dados + método
    signals: list[dict] = field(default_factory=list)   # fatores que empurram o preço
    notes: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    error: str | None = None
    weight_base: float = 0.0
    weight_final: float = 0.0
    raw: dict = field(default_factory=dict)   # objetos não serializáveis (arrays) usados internamente

    def add_signal(self, name: str, contribution: float, direction: str = "", detail: str = "") -> None:
        """`contribution` em pontos percentuais do preço de referência."""
        self.signals.append({"name": name, "contribution_pp": round(float(contribution), 3), "direction": direction, "detail": detail})

    def sort_signals(self, limit: int = 8) -> None:
        self.signals.sort(key=lambda item: abs(item["contribution_pp"]), reverse=True)
        self.signals = self.signals[:limit]

    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.expected_return is not None

    def to_json(self) -> dict:
        return {
            "core": self.core,
            "label": self.label,
            "status": self.status,
            "expected_return": to_jsonable(self.expected_return),
            "expected_return_pp": to_jsonable(None if self.expected_return is None else self.expected_return * 100),
            "sigma": to_jsonable(self.sigma),
            "confidence": to_jsonable(round(self.confidence, 4)),
            "signals": self.signals,
            "notes": self.notes,
            "data": to_jsonable(self.data),
            "error": self.error,
            "weight_base": to_jsonable(round(self.weight_base, 4)),
            "weight": to_jsonable(round(self.weight_final, 4)),
        }

    @staticmethod
    def failed(core: str, label: str, error: str) -> "CoreEstimate":
        return CoreEstimate(core=core, label=label, status="fail", error=error)

    @staticmethod
    def skipped(core: str, label: str, reason: str) -> "CoreEstimate":
        return CoreEstimate(core=core, label=label, status="skip", notes=[reason])


def contribution(name: str, value: float, detail: str = "") -> dict:
    return {"name": name, "contribution_pp": round(float(value) * 100, 3), "direction": "alta" if value > 0 else "baixa" if value < 0 else "neutro", "detail": detail}
