"""SQLite-backed result store.

Schema
------
results
    id        INTEGER PRIMARY KEY AUTOINCREMENT
    ts        TEXT     ISO-8601 timestamp (UTC)
    instance  TEXT     problem name (e.g. "berlin52")
    solver    TEXT     solver key (e.g. "ortools")
    n         INTEGER  number of cities
    cost      REAL     tour cost found
    time_s    REAL     wall-clock seconds
    gap_pct   REAL     % above optimal (NULL if optimal unknown)
    optimal   REAL     known optimal cost (NULL if unknown)
    path      TEXT     JSON-encoded list of city indices

Usage
-----
    from tsp.store import ResultStore

    store = ResultStore()          # uses results.db in cwd
    store.save("berlin52", "ortools", n=52, cost=7542, time_s=10.0,
               gap_pct=0.0, optimal=7542, path=[0,1,...,0])
    rows = store.query(instance="berlin52")
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_DB = "results.db"


class ResultStore:
    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT    NOT NULL,
                    instance TEXT    NOT NULL,
                    solver   TEXT    NOT NULL,
                    n        INTEGER NOT NULL,
                    cost     REAL,
                    time_s   REAL,
                    gap_pct  REAL,
                    optimal  REAL,
                    path     TEXT
                )
            """)

    def save(
        self,
        instance: str,
        solver: str,
        n: int,
        cost: float | None,
        time_s: float | None,
        gap_pct: float | None = None,
        optimal: float | None = None,
        path: list[int] | None = None,
    ) -> int:
        """Insert one result row. Returns the new row id."""
        ts = datetime.now(timezone.utc).isoformat()
        path_json = json.dumps(path) if path is not None else None
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO results
                   (ts, instance, solver, n, cost, time_s, gap_pct, optimal, path)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ts, instance, solver, n, cost, time_s, gap_pct, optimal, path_json),
            )
            return cur.lastrowid

    def query(
        self,
        instance: str | None = None,
        solver: str | None = None,
        last: int | None = None,
    ) -> list[dict]:
        """Fetch results, optionally filtered. Returns list of dicts."""
        clauses, params = [], []
        if instance:
            clauses.append("instance = ?")
            params.append(instance)
        if solver:
            clauses.append("solver = ?")
            params.append(solver)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit = f"LIMIT {last}" if last else ""
        sql = f"SELECT * FROM results {where} ORDER BY ts DESC {limit}"

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            if d["path"]:
                d["path"] = json.loads(d["path"])
            result.append(d)
        return result

    def instances(self) -> list[str]:
        """Return sorted list of distinct instance names in the store."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT instance FROM results ORDER BY instance"
            ).fetchall()
        return [r[0] for r in rows]
