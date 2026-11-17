"""Núcleo 06 — Modelagem estatística e matemática.

Vários modelos independentes estimam o retorno no horizonte; cada um é avaliado
fora da amostra (walk-forward em três blocos) e a combinação usa pesos
inversamente proporcionais ao erro observado, com limites para que nenhum modelo
domine. São usados: ARIMA (AR com ordem por AIC), Ridge e Lasso sobre variáveis
técnicas, regressão logística convertida em retorno esperado, modelo fatorial
(PCA sobre as variáveis externas), correção de erro por cointegração e a deriva
histórica como referência.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config, interdependence, stats
from .context import Context
from .estimate import CoreEstimate
from .features import technical_features

FOLDS = 3
MAX_MODEL_WEIGHT = 0.5
MIN_MODEL_WEIGHT = 0.05
DRIFT_SHRINKAGE = 0.5


def _format_mae(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "não medido"
    if not np.isfinite(number):
        return "não medido"
    return f"{number * 100:.2f} p.p."


def _lasso_pipeline():
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Lasso(alpha=0.0015, max_iter=8000))


def _walk_forward_mae(predict, x: np.ndarray, y: np.ndarray, folds: int = FOLDS) -> tuple[float, int]:
    """Erro absoluto médio fora da amostra em blocos cronológicos sucessivos."""
    valid = np.isfinite(y)
    indices = np.flatnonzero(valid)
    if len(indices) < folds * 60:
        return float("nan"), 0
    blocks = np.array_split(indices, folds)
    errors, count = [], 0
    for block in blocks[1:]:
        if len(block) == 0:
            continue
        train = indices[indices < block[0]]
        if len(train) < 120:
            continue
        try:
            predictions = predict(x[train], y[train], x[block])
        except Exception:
            continue
        errors.extend(np.abs(np.asarray(predictions, dtype=float) - y[block]).tolist())
        count += len(block)
    if not errors:
        return float("nan"), 0
    return float(np.mean(errors)), int(count)


def run(context: Context) -> CoreEstimate:
    estimate = CoreEstimate(core="statistics", label="Modelagem estatística")
    close = context.close
    close_array = close.to_numpy()
    volume = context.history.get("volume")
    horizon = context.horizon_td
    target = stats.forward_returns(close_array, horizon)
    features = technical_features(close, volume)
    models: dict[str, dict] = {}

    # 1. Deriva histórica (referência estatística) --------------------------------
    known = target[np.isfinite(target)]
    if len(known) > 60:
        drift = float(np.mean(known)) * DRIFT_SHRINKAGE
        models["deriva_histórica"] = {
            "expected_return": drift,
            "oos_mae": float(np.mean(np.abs(known - drift))),
            "detail": f"retorno médio histórico {float(np.mean(known)):+.2%} encolhido por {DRIFT_SHRINKAGE:.0%}",
        }

    # 2. ARIMA (AR com ordem por AIC) ---------------------------------------------
    simple_returns = stats.returns(close_array)
    ar = stats.ar_forecast(simple_returns[np.isfinite(simple_returns)], horizon)
    if ar.get("ok"):
        def predict_ar(train_x, train_y, test_x):
            return np.full(len(test_x), ar["cumulative_return"])

        mae, count = _walk_forward_mae(predict_ar, np.zeros((len(target), 1)), target)
        models["arima"] = {
            "expected_return": float(ar["cumulative_return"]),
            "oos_mae": mae,
            "detail": f"AR({ar['order']}) em retornos diários, ordem escolhida por AIC",
        }

    # 3/4/5. Modelos sobre variáveis técnicas ------------------------------------
    design_frame = features.copy()
    design_frame["_y"] = target
    clean = design_frame.dropna()
    x = np.empty((0, 1))
    y = np.empty(0)
    last_row = None
    if len(clean) > config.REGRESSION_MIN_SAMPLES * 2:
        x = clean.drop(columns=["_y"]).to_numpy()
        y = clean["_y"].to_numpy()
        last_row = features.iloc[-1].to_numpy(dtype=float)
        last_row = np.where(np.isfinite(last_row), last_row, np.nanmean(x, axis=0))

        def predict_ridge(train_x, train_y, test_x):
            model = stats.ridge_regression(train_x, train_y, alpha=10.0)
            return model["model"].predict(test_x) if model.get("ok") else np.zeros(len(test_x))

        def predict_lasso(train_x, train_y, test_x):
            model = _lasso_pipeline()
            model.fit(train_x, train_y)
            return model.predict(test_x)

        for name, predict, label in (
            ("ridge_tecnico", predict_ridge, "Ridge sobre retornos, volatilidade e médias móveis"),
            ("lasso_tecnico", predict_lasso, "Lasso (seleção de variáveis técnicas)"),
        ):
            mae, count = _walk_forward_mae(predict, x, y)
            if name == "ridge_tecnico":
                full = stats.ridge_regression(x, y, alpha=10.0)
                if not full.get("ok"):
                    continue
                expected = float(full["model"].predict(last_row[None, :])[0])
            else:
                model = _lasso_pipeline()
                model.fit(x, y)
                expected = float(model.predict(last_row[None, :])[0])
            models[name] = {
                "expected_return": float(np.clip(expected, -0.95, 3.0)),
                "oos_mae": mae,
                "detail": label,
                "n": int(count),
            }

        logistic = stats.logistic_probability(x, np.where(y > 0, 1, 0))
        if logistic.get("ok"):
            up_mean = float(np.mean(y[y > 0])) if (y > 0).any() else 0.0
            down_mean = float(np.mean(y[y <= 0])) if (y <= 0).any() else 0.0
            probability = float(logistic["model"].predict_proba(last_row[None, :])[0, 1])
            models["logistica"] = {
                "expected_return": float(np.clip(probability * up_mean + (1 - probability) * down_mean, -0.95, 3.0)),
                "oos_mae": float("nan"),
                "detail": f"P(alta) = {probability:.1%}; retorno esperado = P·retorno médio de alta + (1−P)·retorno médio de queda",
                "probability_up": probability,
            }

    # 6. Modelo fatorial (PCA sobre variáveis externas) ---------------------------
    factor_names = list(context.drivers)
    if len(factor_names) >= 3:
        columns = {}
        for key in factor_names:
            aligned = context.aligned_driver(key)
            if aligned is not None:
                columns[key] = stats.log_returns(aligned)
        factor_frame = pd.DataFrame(columns)
        combined = factor_frame.copy()
        combined["_y"] = target
        combined = combined.dropna()
        if len(combined) > config.REGRESSION_MIN_SAMPLES:
            matrix = combined.drop(columns=["_y"]).to_numpy()
            components = min(3, matrix.shape[1])
            pca = PCA(n_components=components).fit(matrix)
            scores = pca.transform(matrix)
            outcome = combined["_y"].to_numpy()
            regression = stats.ridge_regression(scores, outcome, alpha=10.0)
            if regression.get("ok"):
                last_scores = pca.transform(factor_frame.iloc[[-1]].to_numpy(dtype=float))
                expected_factor = np.zeros(components)
                for index in range(components):
                    forecast = stats.ar_forecast(scores[:, index][np.isfinite(scores[:, index])], horizon)
                    if forecast.get("ok"):
                        expected_factor[index] = 0.5 * forecast["cumulative_return"]
                coefficients = np.asarray(regression["coefficients"], dtype=float)
                expected = float(np.clip(np.dot(coefficients, expected_factor), -0.95, 3.0))
                models["fatorial"] = {
                    "expected_return": expected,
                    "oos_mae": float("nan"),
                    "detail": f"PCA com {components} fatores sobre {len(factor_names)} variáveis externas; variância explicada {pca.explained_variance_ratio_.sum():.0%}",
                }

    # 7. Correção de erro por cointegração ---------------------------------------
    for key, series in context.drivers.items():
        aligned = context.aligned_driver(key)
        if aligned is None:
            continue
        test = stats.engle_granger(np.log(close_array), np.log(aligned))
        if not test.get("cointegrated"):
            continue
        gap = math.log(close_array[-1]) - (math.log(aligned[-1]) * test["beta"])
        expected = float(np.clip(-0.05 * gap, -0.95, 3.0))
        models["cointegracao"] = {
            "expected_return": expected,
            "oos_mae": float("nan"),
            "detail": f"relação de longo prazo com {interdependence.driver_label(key)} (ADF p={test.get('pvalue', float('nan')):.3f}); desvio atual {gap:+.3f} em log",
        }
        break

    usable = {name: item for name, item in models.items() if np.isfinite(item.get("expected_return", float("nan")))}
    if not usable:
        estimate.data["models"] = models
        return CoreEstimate.skipped("statistics", estimate.label, "Nenhum modelo estatístico pôde ser estimado com os dados disponíveis.")
    maes = {name: item.get("oos_mae") for name, item in usable.items()}
    finite = {name: value for name, value in maes.items() if value is not None and np.isfinite(value)}
    if finite:
        inverse = {name: 1.0 / max(value, 1e-4) for name, value in finite.items()}
        total = sum(inverse.values())
        weights = {name: value / total for name, value in inverse.items()}
        # Modelos cujo erro não pôde ser medido recebem o piso e são exibidos para comparação.
        reference = max(weights.values()) if weights else 1.0
        for name in usable:
            weights.setdefault(name, MIN_MODEL_WEIGHT * reference)
    else:
        weights = {name: 1.0 for name in usable}
    weights = {name: min(MAX_MODEL_WEIGHT, max(MIN_MODEL_WEIGHT, value)) for name, value in weights.items()}
    weight_sum = sum(weights.values())
    weights = {name: value / weight_sum for name, value in weights.items()}
    expected = float(sum(usable[name]["expected_return"] * weights[name] for name in usable))
    for name, item in usable.items():
        estimate.add_signal(
            name,
            item["expected_return"] * 100,
            "alta" if item["expected_return"] > 0 else "baixa",
            f"peso {weights[name]:.0%} · erro fora da amostra {_format_mae(item.get('oos_mae'))} · {item['detail']}",
        )
    estimate.expected_return = float(np.clip(expected, -0.9, 3.0))
    spread = float(np.std([item["expected_return"] for item in usable.values()]))
    estimate.sigma = spread
    measured = [value for value in maes.values() if value is not None and np.isfinite(value)]
    estimate.confidence = round(float(np.clip(0.20 + 0.20 * len(usable) / 6.0 + 0.15 * (1.0 if measured else 0.0) - 0.10 * min(1.0, spread / 0.10), 0.05, 0.85)), 3)
    estimate.data["models"] = {
        name: {
            "expected_return": round(float(item["expected_return"]), 5),
            "expected_return_pp": round(float(item["expected_return"]) * 100, 3),
            "oos_mae": None if item.get("oos_mae") is None or not np.isfinite(item["oos_mae"]) else round(float(item["oos_mae"]), 5),
            "weight": round(float(weights.get(name, 0.0)), 4),
            "detail": item["detail"],
        }
        for name, item in usable.items()
    }
    estimate.data["probability_up"] = usable.get("logistica", {}).get("probability_up")
    estimate.notes.append(
        "Cada modelo é independente; os pesos vêm do erro medido fora da amostra, com limites mínimos e máximos. "
        "Modelos com erro não medido recebem peso mínimo e são exibidos para comparação."
    )
    estimate.sort_signals()
    estimate.status = "ok"
    return estimate
