"""Testes do motor quantitativo (oito núcleos).

Rodam integralmente sem internet: o modo demonstração gera um universo
sintético coerente (fatores macro, commodities e ações com sensibilidades
conhecidas, incluindo uma variável espúria) que permite verificar o motor de
ponta a ponta, inclusive a descoberta de variáveis e o controle de falsas
descobertas.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ACAO_CLARA_DEMO"] = "1"

import numpy as np
import pandas as pd

from engine import backtest, config, discovery, ensemble, events, interdependence, macro, peers, report, simulation, stats, timeseries, valuation
from engine.api import ApiError, handle
from engine.learning import ForecastLedger
from engine.series import MarketData

DATA = MarketData(demo=True, cache_dir=Path(tempfile.mkdtemp()))


def ledger_path():
    return Path(tempfile.mkdtemp()) / "test.db"


class StatisticsTests(unittest.TestCase):
    def test_window_returns_matches_manual_calculation(self):
        values = np.array([100.0, 110.0, 121.0, 133.1, 146.41])
        out = stats.window_returns(values, 2)
        self.assertTrue(np.isnan(out[0]) and np.isnan(out[1]))
        self.assertAlmostEqual(out[2], 0.21, places=6)
        lagged = stats.window_returns(values, 2, lag=1)
        self.assertTrue(np.isnan(lagged[2]))
        self.assertAlmostEqual(lagged[3], 0.21, places=6)

    def test_benjamini_hochberg_controls_false_discoveries(self):
        rng = np.random.default_rng(7)
        pvalues = rng.uniform(0.0, 1.0, 200)
        self.assertEqual(int(stats.benjamini_hochberg(pvalues, q=0.10).sum()), 0)
        strong = np.concatenate([[0.0001, 0.0002], rng.uniform(0.2, 1.0, 50)])
        rejected = stats.benjamini_hochberg(strong, q=0.10)
        self.assertTrue(rejected[0] and rejected[1])
        self.assertLessEqual(int(rejected.sum()), 5)

    def test_adf_separates_random_walk_from_white_noise(self):
        rng = np.random.default_rng(11)
        walk = np.cumsum(rng.normal(0, 1, 900))
        noise = rng.normal(0, 1, 900)
        self.assertGreater(stats.adf_test(walk)["pvalue"], 0.10)
        self.assertLess(stats.adf_test(noise)["pvalue"], 0.05)

    def test_garch_and_ar_produce_finite_forecasts(self):
        rng = np.random.default_rng(3)
        series = rng.normal(0.0005, 0.02, 1200)
        fit = stats.garch11(series, 60)
        self.assertTrue(fit.ok)
        self.assertGreater(fit.alpha, 0.0)
        self.assertLess(fit.alpha + fit.beta, 1.0)
        self.assertTrue(np.isfinite(fit.forecast_vol) and fit.forecast_vol > 0)
        ar = stats.ar_forecast(series, 60)
        self.assertTrue(ar["ok"])
        self.assertTrue(np.isfinite(ar["cumulative_return"]))
        self.assertEqual(len(ar["path"]), 60)

    def test_purged_split_leaves_a_gap_of_at_least_the_horizon(self):
        """O teste não pode começar antes de o rótulo do treino vencer."""
        for n, horizon in ((1200, 5), (1200, 21), (1200, 62), (1200, 252), (900, 252)):
            train_end, test_start = stats.purged_split(n, horizon)
            self.assertGreaterEqual(test_start - train_end, horizon, f"n={n} h={horizon}")
            self.assertGreater(n - test_start, 0, f"n={n} h={horizon}")
            self.assertGreater(train_end, 0)

    def test_purged_split_falls_back_when_history_is_short(self):
        """Com histórico curto a purga é reduzida, mas treino e teste sobrevivem."""
        train_end, test_start = stats.purged_split(300, 252)
        self.assertLess(test_start, 300)
        self.assertGreater(train_end, 0)
        self.assertGreater(test_start, train_end)
        self.assertEqual(stats.purged_split(30, 62), (30, 30))

    def test_shrink_forecast_dampens_models_without_out_of_sample_skill(self):
        value, detail = stats.shrink_forecast(-0.40, 0.0, 0.30)
        self.assertAlmostEqual(value, -0.10, places=6)
        self.assertTrue(detail["capped_at_sigma"] is False)
        self.assertAlmostEqual(detail["shrink_factor"], 0.25, places=6)

    def test_shrink_forecast_keeps_skilled_models_and_caps_at_one_sigma(self):
        value, detail = stats.shrink_forecast(0.20, 1.0, 0.30)
        self.assertAlmostEqual(value, 0.20, places=6)
        self.assertAlmostEqual(detail["shrink_factor"], 1.0, places=6)
        capped, detail = stats.shrink_forecast(1.20, 1.0, 0.30)
        self.assertAlmostEqual(capped, 0.30, places=6)
        self.assertTrue(detail["capped_at_sigma"])

    def test_regression_cores_report_their_shrinkage(self):
        """Macro, interdependência e fluxo publicam o encolhimento aplicado."""
        context, _ = report.build_context(DATA, "PETR4.SA", 365)
        for core in (macro, interdependence):
            estimate = core.run(context)
            if estimate.status != "ok":
                continue
            detail = estimate.data.get("shrinkage")
            self.assertIsNotNone(detail, core.__name__)
            self.assertLessEqual(abs(detail["final"]), abs(detail["raw"]) + 1e-12, core.__name__)
            self.assertLessEqual(abs(estimate.expected_return), 1.0, core.__name__)

    def test_monte_carlo_distribution_is_consistent(self):
        result = stats.monte_carlo(0.0002, 0.02, 60, paths=2000, seed=5)
        self.assertTrue(0.0 <= result["p_up"] <= 1.0)
        quantiles = [result["quantiles"][key] for key in ("p1", "p5", "p10", "p50", "p90", "p95", "p99")]
        self.assertEqual(quantiles, sorted(quantiles))
        self.assertLess(result["var95"], 0.0)
        self.assertLessEqual(result["cvar95"], result["var95"])
        self.assertLess(result["max_drawdown_mean"], 0.0)


class ScenarioTests(unittest.TestCase):
    def test_scenarios_cover_the_distribution(self):
        rng = np.random.default_rng(9)
        wealth = np.exp(rng.normal(0.05, 0.25, 6000))
        out = simulation.build_scenarios(wealth, price=50.0, expected_return=0.05, target_log_sigma=0.25)
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["scenarios"]), 5)
        prices = [item["price"] for item in out["scenarios"]]
        self.assertEqual(prices, sorted(prices, reverse=True))
        self.assertAlmostEqual(sum(item["probability"] for item in out["scenarios"]), 1.0, places=6)
        self.assertAlmostEqual(out["expected_price"] / 50.0 - 1.0, 0.05, places=6)
        self.assertTrue(out["range"]["low"] < 50.0 < out["range"]["high"])
        self.assertGreater(out["potential_gain"], 0.0)
        self.assertGreater(out["potential_loss"], 0.0)


class ValuationTests(unittest.TestCase):
    def test_dcf_uses_only_available_data_and_reports_premises(self):
        context, _ = report.build_context(DATA, "PETR4.SA", 365, with_discovery=False, with_candidates=False)
        value, detail = valuation.dcf_value(context)
        if value is None:
            self.assertIn("reason", detail)
            return
        self.assertGreater(value, 0.0)
        self.assertIn("wacc", detail)
        self.assertGreater(detail["wacc"]["wacc"], detail["terminal_growth"])
        self.assertEqual(len(detail["projections"]), valuation.DCF_YEARS)

    def test_relative_valuation_requires_peers(self):
        context, _ = report.build_context(DATA, "PETR4.SA", 90, with_discovery=False, with_candidates=False, with_peers=False)
        value, detail = valuation.relative_value(context)
        self.assertIsNone(value)
        self.assertIn("reason", detail)
        sample = peers.sample_peers(DATA, "PETR4.SA")
        self.assertIsNotNone(sample)
        self.assertGreaterEqual(sample["count"], 2)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context, cls.found = report.build_context(DATA, "PETR4.SA", 90)

    def test_macro_reports_level_direction_and_velocity(self):
        state = macro.macro_state(self.context)
        self.assertGreater(len(state), 0)
        for item in state.values():
            self.assertIn("zscore", item)
            self.assertIn("direction_per_period", item)
            self.assertIn("acceleration", item)
            self.assertIn(item["trend"], ("alta", "queda", "estável"))
        estimate = macro.run(self.context)
        self.assertEqual(estimate.status, "ok")
        self.assertTrue(np.isfinite(estimate.expected_return))

    def test_interdependence_betas_are_in_return_units(self):
        estimate = interdependence.run(self.context)
        self.assertEqual(estimate.status, "ok")
        matrix = estimate.data["matrix"]
        self.assertGreater(len(matrix), 0)
        for item in matrix.values():
            if item.get("beta") is not None:
                self.assertLess(abs(item["beta"]), 5.0, "beta fora de escala plausível")
        # A relação plantada no universo sintético (petróleo com defasagem) deve aparecer.
        self.assertGreater(matrix["oil_wti"]["pearson"], 0.3)

    def test_events_are_classified_by_rules_and_decay(self):
        rules = events.load_rules()
        classified = events.classify("Banco central reduz juros básicos em 0,50 ponto", rules)
        self.assertIsNotNone(classified)
        self.assertEqual(classified["type"], "rate_cut")
        event = {"direction": -1, "intensity": 8, "certainty": 0.8, "priced_in": 0.3, "horizon_days": 90, "age_days": 0}
        fresh = events.event_impact(dict(event), 90, rules)["impact"]
        old = events.event_impact(dict(event, age_days=120), 90, rules)["impact"]
        self.assertLess(fresh, 0.0)
        self.assertGreater(abs(fresh), abs(old))
        neutral = events.event_impact(dict(event, direction=0), 90, rules)["impact"]
        self.assertEqual(neutral, 0.0)

    def test_flows_core_uses_only_price_and_volume(self):
        estimate = flows_module_run(self.context)
        self.assertIn(estimate.status, ("ok", "skip"))
        if estimate.status == "ok":
            self.assertTrue(abs(estimate.expected_return) < 0.5)
            self.assertIn("missing_sources", estimate.data)

    def test_statistics_core_combines_models_with_bounds(self):
        estimate = timeseries.run(self.context)
        self.assertEqual(estimate.status, "ok")
        models = estimate.data["models"]
        self.assertGreater(len(models), 1)
        for item in models.values():
            self.assertLess(abs(item["expected_return"]), 3.0)
        self.assertAlmostEqual(sum(item["weight"] for item in models.values()), 1.0, places=3)

    def test_simulation_core_exposes_distribution(self):
        estimate = simulation.run(self.context)
        self.assertEqual(estimate.status, "ok")
        self.assertIn("wealth", estimate.raw)
        self.assertGreater(estimate.data["paths"], 100)
        self.assertGreater(estimate.data["annual_vol"], 0.0)

    def test_discovery_finds_planted_relation_and_rejects_spurious(self):
        self.assertTrue(self.found["ok"])
        variables = [item["variable"] for item in self.found["discoveries"]]
        self.assertTrue(any("oil" in item for item in variables), variables)
        self.assertFalse(any("spurious" in item for item in variables), variables)


def flows_module_run(context):
    from engine import flows

    return flows.run(context)


class EnsembleTests(unittest.TestCase):
    def test_weights_respect_caps_and_cover_only_usable_cores(self):
        estimates = [
            report.CoreEstimate(core="fundamentals", label="F", status="ok", expected_return=0.1, confidence=0.9),
            report.CoreEstimate(core="macro", label="M", status="ok", expected_return=-0.05, confidence=0.4),
            report.CoreEstimate(core="flows", label="Fl", status="skip", expected_return=None, confidence=0.0),
        ]
        detail = ensemble.dynamic_weights(estimates, 90, None)
        self.assertEqual(set(detail), {"fundamentals", "macro"})
        total = sum(item["weight"] for item in detail.values())
        self.assertAlmostEqual(total, 1.0, places=6)
        feasible_cap = max(config.MAX_CORE_WEIGHT, 1.0 / len(detail))
        for item in detail.values():
            self.assertLessEqual(item["weight"], feasible_cap + 1e-9)

    def test_camada_c_limits_projection_by_volatility(self):
        context, _ = report.build_context(DATA, "PETR4.SA", 7, with_discovery=False, with_candidates=False)
        adjustments = []
        adjusted = ensemble.apply_adjustments(2.5, context, 0.35, 1.0, adjustments)
        limit = 2.5 * 0.35 * np.sqrt(context.horizon_td / 252.0)
        self.assertAlmostEqual(adjusted, limit, places=6)
        self.assertTrue(any(item["step"] == "limite por volatilidade" for item in adjustments))

    def test_core_estimates_are_clipped_to_three_sigma(self):
        context, _ = report.build_context(DATA, "PETR4.SA", 90, with_discovery=False, with_candidates=False)
        estimate = report.CoreEstimate(core="macro", label="M", status="ok", expected_return=5.0, confidence=0.5)
        report._clip_cores([estimate], context)
        annual = report._annual_vol(context)
        self.assertAlmostEqual(abs(estimate.expected_return), 3.0 * annual * np.sqrt(context.horizon_td / 252.0), places=6)
        self.assertTrue(any("limitada" in note for note in estimate.notes))


class LookaheadTests(unittest.TestCase):
    def test_truncation_removes_future_information(self):
        context, _ = report.build_context(DATA, "VALE3.SA", 30)
        middle = context.history["date"].iloc[len(context.history) // 2]
        past = context.truncate(middle)
        self.assertEqual(past.history["date"].iloc[-1], middle)
        self.assertLess(len(past.history), len(context.history))
        for key, series in past.drivers.items():
            self.assertLessEqual(series.index.max(), pd.Timestamp(middle))
        estimate_past = macro.run(past)
        estimate_full = macro.run(context)
        self.assertNotEqual(estimate_past.expected_return, estimate_full.expected_return)

    def test_features_do_not_change_when_the_future_changes(self):
        from engine.features import technical_features

        history = DATA.history("ITUB4.SA")
        close = history["close"].copy()
        before = technical_features(close).iloc[400]
        close.iloc[401:] = close.iloc[401:] * 3.5
        after = technical_features(close).iloc[400]
        pd.testing.assert_series_equal(before, after)


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = ForecastLedger(path=ledger_path())
        cls.payload = report.projection(DATA, "PETR4.SA", 90, ledger=cls.ledger)

    def test_payload_is_complete_and_json_serialisable(self):
        payload = self.payload
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.assertGreater(len(text), 5000)
        self.assertEqual(len(payload["cores"]), 7)
        self.assertGreater(payload["projection"]["expected_price"], 0.0)
        self.assertEqual(len(payload["projection"]["scenarios"]), 5)
        self.assertAlmostEqual(sum(item["probability"] for item in payload["projection"]["scenarios"]), 1.0, places=6)
        self.assertAlmostEqual(sum(item["weight"] for item in payload["weights"].values()), 1.0, places=4)
        self.assertTrue(0.0 <= payload["projection"]["probability_up"] <= 1.0)
        self.assertIn(payload["risk"]["confidence_label"], ("alta", "média", "baixa", "muito baixa"))
        self.assertTrue(payload["disclaimer"])

    def test_projection_is_recorded_for_learning(self):
        stats_block = self.ledger.stats()
        self.assertGreaterEqual(stats_block["total"], 1)
        rows = self.ledger.history("PETR4.SA", limit=5)
        self.assertTrue(rows)
        self.assertIn("fundamentals", json.loads(rows[0]["models"]))

    def test_scored_forecast_feeds_accuracy(self):
        ledger = ForecastLedger(path=ledger_path())
        ledger.record(
            {
                "symbol": "TEST3.SA",
                "horizon_days": 7,
                "ref_date": "2020-01-02",
                "price": 100.0,
                "expected_price": 110.0,
                "expected_return": 0.10,
                "probability_up": 0.7,
                "range": {"low": 90.0, "high": 130.0},
                "confidence": 60.0,
                "models": {"fundamentals": 0.05, "macro": -0.02},
                "weights": {"fundamentals": 0.6, "macro": 0.4},
            }
        )
        scored = ledger.score(lambda symbol, target: 121.0)
        self.assertEqual(scored, 1)
        accuracy = ledger.accuracy("TEST3.SA", 7)
        self.assertTrue(accuracy["ok"])
        self.assertEqual(accuracy["n"], 1)
        self.assertAlmostEqual(accuracy["mae_pp"], 11.0, places=6)
        self.assertAlmostEqual(accuracy["baseline_mae_pp"], 21.0, places=6)
        self.assertTrue(accuracy["better_than_baseline"])
        self.assertEqual(accuracy["directional_hit"], 1.0)
        self.assertAlmostEqual(accuracy["models"]["fundamentals"]["mae_pp"], 16.0, places=6)

    def test_walk_forward_reproduces_the_pipeline(self):
        context, _ = report.build_context(DATA, "PETR4.SA", 30, with_discovery=False, with_candidates=False)
        result = backtest.walk_forward(context, count=3, ledger=self.ledger)
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["ensemble"]["n"], 2)
        self.assertGreater(result["ensemble"]["mae_pp"], 0.0)
        self.assertLess(result["ensemble"]["mae_pp"], 500.0)
        self.assertGreater(len(result["cores"]), 1)

    def test_horizons_change_the_weights(self):
        short = ensemble.base_weights(7)
        long = ensemble.base_weights(365)
        for weights in (short, long):
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(long["fundamentals"], short["fundamentals"])
        self.assertGreater(short["flows"], long["flows"])
        self.assertEqual(config.horizon_to_trading(365), 252)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = ForecastLedger(path=ledger_path())

    def test_health_and_macro_routes(self):
        health = handle("/api/health", {}, DATA, self.ledger)
        self.assertEqual(health["horizons_days"], list(config.HORIZONS_DAYS))
        self.assertIn("/api/quant", health["routes"])
        overview = handle("/api/macro", {}, DATA, self.ledger)
        self.assertGreater(len(overview["indicators"]), 0)

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ApiError):
            handle("/api/quant", {"symbol": ["PETR4.SA"], "horizon": ["45"]}, DATA, self.ledger)
        with self.assertRaises(ApiError):
            handle("/api/quant", {"symbol": ["INVALIDO"], "horizon": ["90"]}, DATA, self.ledger)

    def test_unknown_route(self):
        with self.assertRaises(ApiError):
            handle("/api/unknown", {}, DATA, self.ledger)


if __name__ == "__main__":
    unittest.main()
