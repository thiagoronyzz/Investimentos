"""Motor quantitativo de projeção de ativos — Ação Clara.

Implementação puramente matemática/estatística (sem IA generativa): combina
macroeconomia, fundamentos, interdependência entre mercados, eventos
classificados por regras, fluxo financeiro, modelagem estatística, simulação
de cenários e métricas de risco em uma projeção ponderada com horizontes de
7, 30, 90, 180 e 365 dias.
"""

__version__ = "2.0.0"

CORES = (
    "fundamentals",
    "macro",
    "interdependence",
    "events",
    "flows",
    "statistics",
    "simulation",
)

CORE_LABELS = {
    "fundamentals": "Fundamentos e valuation",
    "macro": "Macroeconomia",
    "interdependence": "Interdependência de mercados",
    "events": "Eventos externos",
    "flows": "Fluxo financeiro",
    "statistics": "Modelagem estatística",
    "simulation": "Simulação de cenários",
}

__all__ = ["__version__", "CORES", "CORE_LABELS"]
