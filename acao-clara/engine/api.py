"""Rotas da API do motor quantitativo, independentes de framework.

Tanto o servidor local (`http.server`) quanto a aplicação Flask usada na Vercel
chamam estas funções; assim as duas versões expõem exatamente os mesmos
endpoints e a mesma lógica.
"""
from __future__ import annotations

import traceback
from datetime import datetime

from . import __version__, backtest, config, report
from .learning import ForecastLedger
from .series import DataSourceError, MarketData

SYMBOL_PATTERN = r"[A-Z0-9]{4,12}\.SA"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _symbol(params: dict) -> str:
    import re

    symbol = str(params.get("symbol", [""])[0]).upper().strip()
    if not symbol:
        raise ApiError("Informe o código da ação (por exemplo, PETR4.SA).")
    if not symbol.endswith(".SA"):
        symbol = f"{symbol}.SA"
    if not re.fullmatch(SYMBOL_PATTERN, symbol):
        raise ApiError("Escolha uma ação brasileira válida na busca.")
    return symbol


def _horizon(params: dict, default: int = config.DEFAULT_HORIZON) -> int:
    raw = params.get("horizon", [str(default)])[0]
    try:
        horizon = int(raw)
    except (TypeError, ValueError):
        raise ApiError("Horizonte inválido. Use 7, 30, 90, 180 ou 365 dias.")
    if horizon not in config.HORIZONS_DAYS:
        raise ApiError("Horizonte inválido. Use 7, 30, 90, 180 ou 365 dias.")
    return horizon


def routes() -> dict:
    """Descrição dos endpoints, usada pelo /api/health e pela documentação."""
    return {
        "/api/quant": "Projeção quantitativa completa (oito núcleos, cenários, riscos e pesos).",
        "/api/backtest": "Validação walk-forward da combinação de núcleos para um ativo e horizonte.",
        "/api/macro": "Painel macroeconômico: nível, direção e velocidade de cada indicador.",
        "/api/ledger": "Livro de previsões: precisão histórica registrada e aprendizado.",
        "/api/health": "Estado das fontes de dados, do livro de previsões e do motor.",
    }


def handle(path: str, params: dict, data: MarketData, ledger: ForecastLedger | None = None) -> dict:
    try:
        if path == "/api/quant":
            symbol = _symbol(params)
            horizon = _horizon(params)
            mode = str(params.get("mode", ["full"])[0]).lower()
            mode = mode if mode in ("full", "fast") else "full"
            return report.projection(data, symbol, horizon, ledger=ledger, record=True, mode=mode)
        if path == "/api/backtest":
            symbol = _symbol(params)
            horizon = _horizon(params)
            try:
                count = int(params.get("origins", ["4"])[0])
            except (TypeError, ValueError):
                count = 4
            count = max(2, min(6, count))
            context, _ = report.build_context(data, symbol, horizon, with_discovery=False, with_candidates=False)
            return backtest.walk_forward(context, count=count, ledger=ledger)
        if path == "/api/macro":
            return report.macro_overview(data)
        if path == "/api/ledger":
            if ledger is None:
                return {"enabled": False, "reason": "livro de previsões indisponível nesta instância"}
            symbol = str(params.get("symbol", [""])[0]).upper().strip() or None
            return {
                "enabled": True,
                "stats": ledger.stats(),
                "accuracy": ledger.accuracy(symbol, _horizon(params) if params.get("horizon") else None),
                "accuracy_all": ledger.accuracy(),
                "history": ledger.history(symbol, limit=25),
            }
        if path == "/api/health":
            return {
                "engine_version": __version__,
                "horizons_days": list(config.HORIZONS_DAYS),
                "demo": data.demo,
                "sources": data.health_report(),
                "ledger": ledger.stats() if ledger is not None else {"enabled": False},
                "routes": routes(),
                "checked_at": datetime.now().isoformat(timespec="seconds"),
            }
    except ApiError:
        raise
    except report.EngineError as exc:
        raise ApiError(str(exc), 422)
    except DataSourceError as exc:
        raise ApiError(str(exc) or "Fonte de dados indisponível.", 502)
    except Exception as exc:
        traceback.print_exc()
        raise ApiError(f"Não foi possível concluir a análise ({type(exc).__name__}).", 500)
    raise ApiError("Não encontrado.", 404)
