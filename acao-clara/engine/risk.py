"""Núcleo 08 — Risco, confiabilidade e validação.

Não é acessório: uma projeção interessante com dados instáveis vale pouco.
Aqui são calculados volatilidade, drawdown esperado, beta, sensibilidade a
choques, divergência entre modelos, qualidade dos dados, detecção de mudança de
regime e o erro histórico das projeções (livro de previsões + walk-forward).
A saída inclui uma nota de confiabilidade de 0 a 100 com os componentes que a
formaram.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import stats
from .context import Context
from .estimate import CoreEstimate


def rolling_beta(close: pd.Series, market: pd.Series, window: int = 126) -> pd.Series:
    frame = pd.DataFrame({"stock": stats.returns(close.to_numpy()), "market": stats.returns(market.to_numpy())}).dropna()
    if len(frame) < window:
        return pd.Series(dtype=float)
    covariance = frame.stock.rolling(window).cov(frame.market)
    variance = frame.market.rolling(window).var()
    return (covariance / variance.replace(0.0, np.nan)).dropna()


def detect_regime(close: pd.Series, market: pd.Series | None = None) -> dict:
    """Mudança de regime por volatilidade e beta móveis (z-score sobre 3 anos)."""
    returns = stats.returns(close.to_numpy())
    vol_series = pd.Series(returns).rolling(21).std()
    z = stats.zscore(vol_series.to_numpy(), window=756)
    current_z = float(z[-1]) if np.isfinite(z[-1]) else 0.0
    out = {
        "vol_zscore": round(current_z, 3),
        "regime": "volatilidade elevada" if current_z > 1.0 else "volatilidade baixa" if current_z < -1.0 else "normal",
        "beta_break": None,
    }
    if market is not None:
        beta = rolling_beta(close, market)
        if len(beta) > 252:
            recent = float(beta.tail(63).mean())
            previous = float(beta.iloc[-252:-63].mean()) if len(beta) > 252 else float("nan")
            spread = float(beta.tail(63).std()) or 0.05
            out["beta_recent"] = round(recent, 3)
            out["beta_previous"] = None if not np.isfinite(previous) else round(previous, 3)
            out["beta_break"] = bool(abs(recent - previous) > 2 * spread) if np.isfinite(previous) else None
    return out


def macro_shocks(context: Context, matrix: dict) -> list[dict]:
    """Impacto de um choque de 1 desvio-padrão em cada variável externa."""
    shocks = []
    for key, item in (matrix or {}).items():
        beta = item.get("beta")
        if beta is None:
            continue
        aligned = context.aligned_driver(key)
        if aligned is None:
            continue
        returns = stats.log_returns(aligned)
        returns = returns[np.isfinite(returns)]
        if returns.size < 60:
            continue
        sigma = float(np.std(returns, ddof=1)) * math.sqrt(context.horizon_td)
        impact = float(beta) * sigma
        shocks.append(
            {
                "variable": item.get("label", key),
                "beta": round(float(beta), 4),
                "sigma_horizon": round(sigma, 5),
                "impact_pp": round(impact * 100, 3),
                "reading": f"um choque de 1σ em {item.get('label', key)} moveria o preço em {impact:+.2%}",
            }
        )
    shocks.sort(key=lambda item: abs(item["impact_pp"]), reverse=True)
    return shocks


def data_quality(context: Context, estimates: list[CoreEstimate]) -> dict:
    ok = [item for item in estimates if item.status == "ok"]
    verified = [item.get("verified", True) for item in context.macro_meta.values()]
    history_days = len(context.history)
    return {
        "cores_ok": len(ok),
        "cores_total": len(estimates),
        "cores_ratio": round(len(ok) / max(1, len(estimates)), 3),
        "macro_verified_ratio": round(float(np.mean(verified)), 3) if verified else None,
        "drivers": len(context.drivers),
        "macro_series": len(context.macro),
        "events": len(context.events),
        "history_bars": history_days,
        "history_years": round(history_days / 252.0, 2),
        "has_fundamentals": context.fundamentals is not None,
        "has_peers": bool(context.peers),
        "stale_days": None,
    }


def run(context: Context, estimates: list[CoreEstimate], matrix: dict | None, scenarios: dict | None, garch: dict | None, accuracy: dict | None) -> dict:
    close = context.close
    returns = stats.returns(close.to_numpy())
    finite = returns[np.isfinite(returns)]
    market = None
    for key in ("ibov", "^BVSP"):
        aligned = context.aligned_driver(key)
        if aligned is not None:
            market = pd.Series(aligned)
            break
    hist_vol = {
        "20": round(stats.annualized_vol(finite, 20), 4),
        "60": round(stats.annualized_vol(finite, 60), 4),
        "252": round(stats.annualized_vol(finite, 252), 4),
    }
    beta = rolling_beta(close, market) if market is not None else pd.Series(dtype=float)
    regime = detect_regime(close, market)
    ok_returns = [item.expected_return for item in estimates if item.usable and item.expected_return is not None]
    divergence = float(np.std(ok_returns)) if len(ok_returns) > 1 else float("nan")
    quality = data_quality(context, estimates)
    expected_drawdown = scenarios.get("max_drawdown_mean") if scenarios else None
    components = {
        "modelos_concordam": round(float(np.clip(1.0 - min(1.0, (divergence or 0.0) / 0.15), 0.0, 1.0)), 3) if np.isfinite(divergence) else 0.3,
        "qualidade_dos_dados": round(float(np.clip(0.4 * quality["cores_ratio"] + 0.3 * min(1.0, quality["drivers"] / 6.0) + 0.3 * min(1.0, quality["history_bars"] / 1500.0), 0.0, 1.0)), 3),
        "estabilidade_volatilidade": round(float(np.clip(1.0 - min(1.0, abs(regime["vol_zscore"]) / 2.5), 0.0, 1.0)), 3),
        "desempenho_historico": round(float(np.clip((accuracy or {}).get("score", 0.5), 0.0, 1.0)), 3),
    }
    if quality.get("macro_verified_ratio") is not None:
        components["fontes_verificadas"] = round(float(quality["macro_verified_ratio"]), 3)
    confidence = 100.0 * float(np.mean(list(components.values())))
    return {
        "volatility": {**hist_vol, "projected": (garch or {}).get("forecast_vol_anual"), "source": (garch or {}).get("ok")},
        "beta": round(float(beta.tail(63).mean()), 3) if len(beta) else None,
        "beta_series_tail": [round(float(value), 3) for value in beta.tail(20).to_numpy()] if len(beta) else [],
        "regime": regime,
        "model_divergence": None if not np.isfinite(divergence) else round(divergence, 5),
        "model_spread": {
            "min": round(min(ok_returns), 5) if ok_returns else None,
            "max": round(max(ok_returns), 5) if ok_returns else None,
        },
        "expected_max_drawdown": expected_drawdown,
        "var95_pp": (scenarios or {}).get("var95_pp"),
        "cvar95_pp": (scenarios or {}).get("cvar95_pp"),
        "macro_shocks": macro_shocks(context, matrix or {}),
        "data_quality": quality,
        "historical_accuracy": accuracy,
        "confidence_components": components,
        "confidence": round(confidence, 1),
        "confidence_label": _label(confidence),
        "warnings": _warnings(context, estimates, quality, regime, divergence, accuracy),
    }


def _label(confidence: float) -> str:
    if confidence >= 70:
        return "alta"
    if confidence >= 50:
        return "média"
    if confidence >= 30:
        return "baixa"
    return "muito baixa"


def _warnings(context: Context, estimates: list[CoreEstimate], quality: dict, regime: dict, divergence: float, accuracy: dict | None) -> list[str]:
    warnings = []
    failed = [item.label for item in estimates if item.status == "fail"]
    if failed:
        warnings.append("Núcleos com falha nesta execução: " + ", ".join(failed) + ". Os pesos foram redistribuídos.")
    if quality["cores_ratio"] < 0.6:
        warnings.append("Menos de 60% dos núcleos produziram estimativa; a projeção depende de poucas fontes.")
    if quality.get("macro_verified_ratio") is not None and quality["macro_verified_ratio"] < 1.0:
        warnings.append("Parte das séries macroeconômicas não teve o nome confirmado na fonte; o motor marcou esses dados como não verificados.")
    if regime.get("regime") == "volatilidade elevada":
        warnings.append("Volatilidade muito acima da média histórica: relações passadas podem não valer no regime atual.")
    if regime.get("beta_break"):
        warnings.append("O beta em relação ao mercado mudou de forma relevante nos últimos meses.")
    if np.isfinite(divergence) and divergence > 0.10:
        warnings.append(f"Os núcleos divergem bastante (desvio-padrão de {divergence:.1%}); trate a faixa como ampla.")
    if accuracy and accuracy.get("n", 0) >= 5 and accuracy.get("better_than_baseline") is False:
        warnings.append("Nas previsões já registradas e conferidas, o motor não superou a referência simples.")
    if context.demo:
        warnings.append("Modo demonstração: todos os dados são sintéticos e servem apenas para exercitar o motor.")
    return warnings
