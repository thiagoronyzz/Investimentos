"""Aprendizado matemático com os próprios erros (sem IA).

Cada projeção é gravada em um livro (SQLite). Quando o prazo amadurece, o preço
realizado é confrontado com o que foi projetado e o erro alimenta dois lugares:

  1. a exibição da precisão histórica de cada núcleo e do conjunto;
  2. os pesos dinâmicos da Camada B — núcleos que erram menos ganham peso,
     respeitando limites mínimos e máximos.

Em hospedagem sem disco persistente (serverless), o livro é recriado a cada
instância; nesse caso o motor volta a usar apenas o backtesting walk-forward.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    ref_date TEXT NOT NULL,
    ref_price REAL,
    expected_price REAL,
    expected_return REAL,
    probability_up REAL,
    low_price REAL,
    high_price REAL,
    confidence REAL,
    models TEXT,
    weights TEXT,
    realized_price REAL,
    realized_return REAL,
    scored_at TEXT,
    error_pp REAL,
    hit INTEGER
);
CREATE INDEX IF NOT EXISTS idx_forecasts_symbol ON forecasts(symbol, horizon_days, created_at);
"""


class ForecastLedger:
    def __init__(self, path: Path | str | None = None) -> None:
        override = os.environ.get("ACAO_CLARA_LEDGER_PATH", "").strip()
        if path is None and override:
            path = override
        candidates = [Path(path)] if path else [
            config.RUNTIME_DIR / "forecasts.db",
            # Hospedagem serverless: o diretório do código é somente leitura.
            Path(tempfile.gettempdir()) / "acao-clara-forecasts.db",
        ]
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                # O servidor atende requisições em várias threads; o SQLite é
                # compartilhado com trava própria para evitar erros de concorrência.
                connection = sqlite3.connect(candidate, check_same_thread=False, timeout=30.0)
                connection.row_factory = sqlite3.Row
                connection.executescript(SCHEMA)
                connection.commit()
            except (OSError, sqlite3.Error) as error:  # pragma: no cover - depende do host
                last_error = error
                continue
            self.path = candidate
            self.connection = connection
            self.lock = threading.RLock()
            self.writable = True
            return
        raise RuntimeError(f"não foi possível abrir o livro de previsões: {last_error}")

    def close(self) -> None:
        try:
            self.connection.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------
    def record(self, projection: dict) -> int:
        with self.lock:
            return self._record(projection)

    def _record(self, projection: dict) -> int:
        try:
            return self._insert(projection)
        except sqlite3.Error:  # pragma: no cover - disco cheio/readonly
            self.writable = False
            return 0

    def _insert(self, projection: dict) -> int:
        cursor = self.connection.execute(
            """INSERT INTO forecasts (created_at, symbol, horizon_days, ref_date, ref_price, expected_price,
               expected_return, probability_up, low_price, high_price, confidence, models, weights)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                projection.get("generated_at", datetime.now().isoformat(timespec="seconds")),
                projection["symbol"],
                int(projection["horizon_days"]),
                str(projection["ref_date"]),
                float(projection["price"]),
                float(projection["expected_price"]),
                float(projection["expected_return"]),
                projection.get("probability_up"),
                (projection.get("range") or {}).get("low"),
                (projection.get("range") or {}).get("high"),
                projection.get("confidence"),
                json.dumps(projection.get("models", {})),
                json.dumps(projection.get("weights", {})),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid or 0)

    # ------------------------------------------------------------------
    def due(self, today: date | None = None) -> list[dict]:
        today = today or date.today()
        with self.lock:
            return self._due(today)

    def _due(self, today: date) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM forecasts WHERE realized_price IS NULL").fetchall()
        out = []
        for row in rows:
            try:
                reference = datetime.fromisoformat(row["ref_date"]).date()
            except ValueError:
                continue
            if reference + timedelta(days=int(row["horizon_days"])) <= today:
                out.append(dict(row))
        return out

    def score(self, price_lookup) -> int:
        """Confronta previsões maduras com o preço realizado. Retorna quantas foram avaliadas."""
        with self.lock:
            return self._score(price_lookup)

    def _score(self, price_lookup) -> int:
        scored = 0
        for row in self._due(date.today()):
            target = datetime.fromisoformat(row["ref_date"]).date() + timedelta(days=int(row["horizon_days"]))
            realized = price_lookup(row["symbol"], target)
            if realized is None or realized <= 0:
                continue
            expected = float(row["expected_return"])
            actual = float(realized) / float(row["ref_price"]) - 1.0
            self.connection.execute(
                """UPDATE forecasts SET realized_price=?, realized_return=?, scored_at=?, error_pp=?, hit=? WHERE id=?""",
                (
                    float(realized),
                    actual,
                    datetime.now().isoformat(timespec="seconds"),
                    abs(actual - expected) * 100.0,
                    1 if (actual > 0) == (expected > 0) else 0,
                    int(row["id"]),
                ),
            )
            scored += 1
        self.connection.commit()
        return scored

    # ------------------------------------------------------------------
    def accuracy(self, symbol: str | None = None, horizon_days: int | None = None) -> dict:
        """Precisão histórica registrada: por núcleo e do conjunto."""
        with self.lock:
            return self._accuracy(symbol, horizon_days)

    def _accuracy(self, symbol: str | None = None, horizon_days: int | None = None) -> dict:
        query = "SELECT * FROM forecasts WHERE realized_return IS NOT NULL"
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        if horizon_days:
            query += " AND horizon_days = ?"
            params.append(int(horizon_days))
        rows = [dict(row) for row in self.connection.execute(query, params).fetchall()]
        if not rows:
            return {"ok": False, "n": 0, "reason": "nenhuma previsão madura registrada até agora"}
        errors, baseline, hits, brier = [], [], [], []
        per_model: dict[str, list[float]] = {}
        for row in rows:
            actual = float(row["realized_return"])
            expected = float(row["expected_return"])
            errors.append(abs(actual - expected))
            baseline.append(abs(actual))
            hits.append(1 if (actual > 0) == (expected > 0) else 0)
            if row.get("probability_up") is not None:
                probability = float(row["probability_up"])
                brier.append((probability - (1.0 if actual > 0 else 0.0)) ** 2)
            try:
                models = json.loads(row.get("models") or "{}")
            except ValueError:
                models = {}
            for name, value in models.items():
                try:
                    per_model.setdefault(name, []).append(abs(actual - float(value)))
                except (TypeError, ValueError):
                    continue
        mae = float(np.mean(errors))
        baseline_mae = float(np.mean(baseline))
        skill = float(np.clip(1.0 - mae / baseline_mae, 0.0, 1.0)) if baseline_mae > 0 else 0.0
        models_out = {name: {"mae_pp": round(float(np.mean(values)) * 100, 3), "n": len(values)} for name, values in per_model.items()}
        for name, item in models_out.items():
            item["skill"] = round(float(np.clip(1.0 - (item["mae_pp"] / 100.0) / baseline_mae, 0.0, 1.0)), 3) if baseline_mae > 0 else 0.0
        return {
            "ok": True,
            "n": len(rows),
            "mae_pp": round(mae * 100, 3),
            "baseline_mae_pp": round(baseline_mae * 100, 3),
            "better_than_baseline": bool(mae < baseline_mae),
            "directional_hit": round(float(np.mean(hits)), 4),
            "brier": round(float(np.mean(brier)), 4) if brier else None,
            "skill": round(skill, 3),
            "score": round(0.35 + 0.65 * skill, 3),
            "models": models_out,
            "last_scored": max((row["scored_at"] or "") for row in rows),
        }

    # ------------------------------------------------------------------
    def previous(self, symbol: str, horizon_days: int) -> dict | None:
        with self.lock:
            return self._previous(symbol, horizon_days)

    def _previous(self, symbol: str, horizon_days: int) -> dict | None:
        rows = self.connection.execute(
            """SELECT * FROM forecasts WHERE symbol = ? AND horizon_days = ?
               ORDER BY created_at DESC LIMIT 2""",
            (symbol.upper(), int(horizon_days)),
        ).fetchall()
        if not rows:
            return None
        current = dict(rows[0])
        previous = dict(rows[1]) if len(rows) > 1 else None
        return {"current": current, "previous": previous}

    def history(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        with self.lock:
            return self._history(symbol, limit)

    def _history(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        query = "SELECT * FROM forecasts"
        params: list = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def stats(self) -> dict:
        with self.lock:
            data = self._stats()
        data["path"] = str(self.path)
        data["writable"] = bool(self.writable)
        return data

    def _stats(self) -> dict:
        total = self.connection.execute("SELECT COUNT(*) AS n FROM forecasts").fetchone()["n"]
        scored = self.connection.execute("SELECT COUNT(*) AS n FROM forecasts WHERE realized_return IS NOT NULL").fetchone()["n"]
        symbols = self.connection.execute("SELECT COUNT(DISTINCT symbol) AS n FROM forecasts").fetchone()["n"]
        return {"total": total, "scored": scored, "symbols": symbols, "path": str(self.path)}
