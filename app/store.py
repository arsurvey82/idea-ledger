"""SQLite persistence for the operator's ledger.

Lives outside the repository, in the operator's own directory. The schema is
created on first use and is idempotent, so a second run never clobbers data.

Two things this store deliberately records that the prior system did not:

  * every score's provenance, and the original value when a human overrode it
  * every rejection's cause, so "why was this cut?" is a query rather than a
    search through prose
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core.assumptions import Assumption, AssumptionGraph, Dependent
from .core.calibration import CalibrationStore, Observation
from .core.rules import ChangeClass, Rule, RuleTarget
from .core.types import (
    DimensionScore,
    Evidence,
    FactBase,
    Idea,
    Provenance,
    ScoreVector,
    Status,
)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS facts (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ideas (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    track          TEXT NOT NULL,
    status         TEXT NOT NULL,
    rubric_version INTEGER,
    fields_json    TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    idea_id        TEXT NOT NULL,
    rubric_version INTEGER NOT NULL,
    dimension      TEXT NOT NULL,
    value          INTEGER NOT NULL,
    confidence     REAL NOT NULL,
    falsifier      TEXT NOT NULL DEFAULT '',
    evidence_json  TEXT NOT NULL DEFAULT '[]',
    provenance     TEXT NOT NULL,
    original_value INTEGER,
    PRIMARY KEY (idea_id, rubric_version, dimension)
);

CREATE TABLE IF NOT EXISTS evidence (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    claim        TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    verbatim     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS assumptions (
    id          TEXT PRIMARY KEY,
    statement   TEXT NOT NULL,
    confidence  REAL NOT NULL,
    source      TEXT NOT NULL,
    verified_at TEXT
);

CREATE TABLE IF NOT EXISTS dependencies (
    assumption_id TEXT NOT NULL,
    dependent_id  TEXT NOT NULL,
    kind          TEXT NOT NULL,
    role          TEXT NOT NULL,
    PRIMARY KEY (assumption_id, dependent_id)
);

CREATE TABLE IF NOT EXISTS rules (
    id           TEXT PRIMARY KEY,
    description  TEXT NOT NULL,
    target       TEXT NOT NULL,
    stage        TEXT NOT NULL,
    predicate    TEXT,
    fragment     TEXT NOT NULL DEFAULT '',
    author       TEXT NOT NULL,
    version      INTEGER NOT NULL DEFAULT 1,
    active       INTEGER NOT NULL DEFAULT 0,
    change_class TEXT NOT NULL DEFAULT 'additive',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    dimension  TEXT NOT NULL,
    first_pass INTEGER NOT NULL,
    verified   INTEGER NOT NULL,
    idea_id    TEXT NOT NULL,
    at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejections (
    idea_id  TEXT NOT NULL,
    outcome  TEXT NOT NULL,
    cause    TEXT NOT NULL,
    at       TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class Store:
    path: Path

    # -- lifecycle -------------------------------------------------------
    @classmethod
    def open(cls, home: Path) -> "Store":
        home.mkdir(parents=True, exist_ok=True)
        store = cls(home / "ledger.db")
        with closing(store._conn()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        return store

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- facts -----------------------------------------------------------
    def save_facts(self, facts: FactBase) -> None:
        stamp = now()
        with closing(self._conn()) as conn:
            conn.executemany(
                "INSERT INTO facts (key, value_json, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
                "updated_at=excluded.updated_at",
                [(k, json.dumps(v), stamp) for k, v in facts.fields.items()],
            )
            conn.commit()

    def load_facts(self) -> FactBase:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT key, value_json FROM facts").fetchall()
        return FactBase({r["key"]: json.loads(r["value_json"]) for r in rows})

    # -- ideas and scores ------------------------------------------------
    def save_idea(self, idea: Idea) -> None:
        with closing(self._conn()) as conn:
            conn.execute(
                "INSERT INTO ideas (id,name,track,status,rubric_version,fields_json,created_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name, track=excluded.track, status=excluded.status, "
                "rubric_version=excluded.rubric_version, fields_json=excluded.fields_json",
                (
                    idea.id,
                    idea.name,
                    idea.track,
                    idea.status.value,
                    idea.vector.rubric_version if idea.vector else None,
                    json.dumps(dict(idea.fields)),
                    now(),
                ),
            )
            if idea.vector:
                conn.executemany(
                    "INSERT INTO scores (idea_id,rubric_version,dimension,value,confidence,"
                    "falsifier,evidence_json,provenance,original_value) VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(idea_id,rubric_version,dimension) DO UPDATE SET "
                    "value=excluded.value, confidence=excluded.confidence, "
                    "falsifier=excluded.falsifier, evidence_json=excluded.evidence_json, "
                    "provenance=excluded.provenance, original_value=excluded.original_value",
                    [
                        (
                            idea.id,
                            idea.vector.rubric_version,
                            s.dimension,
                            s.value,
                            s.confidence,
                            s.falsifier,
                            json.dumps(list(s.evidence_ids)),
                            s.provenance.value,
                            s.original_value,
                        )
                        for s in idea.vector.scores
                    ],
                )
            conn.executemany(
                "INSERT INTO evidence (id,url,claim,retrieved_at,verbatim) VALUES (?,?,?,?,?) "
                "ON CONFLICT(id) DO NOTHING",
                [(e.id, e.url, e.claim, e.retrieved_at, e.verbatim) for e in idea.evidence],
            )
            conn.commit()

    def load_ideas(self) -> list[Idea]:
        with closing(self._conn()) as conn:
            ideas = conn.execute("SELECT * FROM ideas ORDER BY name").fetchall()
            out: list[Idea] = []
            for row in ideas:
                vector = None
                if row["rubric_version"] is not None:
                    scores = conn.execute(
                        "SELECT * FROM scores WHERE idea_id=? AND rubric_version=?",
                        (row["id"], row["rubric_version"]),
                    ).fetchall()
                    if scores:
                        vector = ScoreVector(
                            rubric_version=row["rubric_version"],
                            track=row["track"],
                            scores=tuple(
                                DimensionScore(
                                    dimension=s["dimension"],
                                    value=s["value"],
                                    evidence_ids=tuple(json.loads(s["evidence_json"])),
                                    confidence=s["confidence"],
                                    falsifier=s["falsifier"],
                                    provenance=Provenance(s["provenance"]),
                                    original_value=s["original_value"],
                                )
                                for s in scores
                            ),
                        )
                out.append(
                    Idea(
                        id=row["id"],
                        name=row["name"],
                        track=row["track"],
                        status=Status(row["status"]),
                        vector=vector,
                        fields=json.loads(row["fields_json"]),
                    )
                )
        return out

    def record_rejection(self, idea_id: str, outcome: str, cause: str) -> None:
        """Why an idea was cut, kept as data rather than prose."""
        with closing(self._conn()) as conn:
            conn.execute(
                "INSERT INTO rejections (idea_id,outcome,cause,at) VALUES (?,?,?,?)",
                (idea_id, outcome, cause, now()),
            )
            conn.commit()

    def rejections(self) -> list[Mapping[str, Any]]:
        with closing(self._conn()) as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM rejections ORDER BY at DESC"
            ).fetchall()]

    def rejected_ids(self) -> frozenset[str]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT id FROM ideas WHERE status=?", (Status.REJECTED.value,)
            ).fetchall()
        return frozenset(r["id"] for r in rows)

    # -- rules -----------------------------------------------------------
    def save_rule(self, rule: Rule) -> None:
        with closing(self._conn()) as conn:
            conn.execute(
                "INSERT INTO rules (id,description,target,stage,predicate,fragment,author,"
                "version,active,change_class,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET description=excluded.description, "
                "predicate=excluded.predicate, fragment=excluded.fragment, "
                "active=excluded.active, version=excluded.version",
                (
                    rule.id,
                    rule.description,
                    rule.target.value,
                    rule.stage,
                    json.dumps(rule.predicate) if rule.predicate else None,
                    rule.fragment,
                    rule.author,
                    rule.version,
                    int(rule.active),
                    rule.change_class.value,
                    now(),
                ),
            )
            conn.commit()

    def load_rules(self, *, active_only: bool = False) -> list[Rule]:
        sql = "SELECT * FROM rules" + (" WHERE active=1" if active_only else "")
        with closing(self._conn()) as conn:
            rows = conn.execute(sql + " ORDER BY id").fetchall()
        return [
            Rule(
                id=r["id"],
                description=r["description"],
                target=RuleTarget(r["target"]),
                stage=r["stage"],
                predicate=json.loads(r["predicate"]) if r["predicate"] else None,
                fragment=r["fragment"],
                author=r["author"],
                version=r["version"],
                active=bool(r["active"]),
                change_class=ChangeClass(r["change_class"]),
            )
            for r in rows
        ]

    # -- assumptions -----------------------------------------------------
    def save_graph(self, graph: AssumptionGraph) -> None:
        with closing(self._conn()) as conn:
            conn.executemany(
                "INSERT INTO assumptions (id,statement,confidence,source,verified_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET statement=excluded.statement, "
                "confidence=excluded.confidence, source=excluded.source, "
                "verified_at=excluded.verified_at",
                [
                    (a.id, a.statement, a.confidence, a.source, a.verified_at)
                    for a in graph.assumptions.values()
                ],
            )
            for aid in graph.assumptions:
                conn.executemany(
                    "INSERT INTO dependencies (assumption_id,dependent_id,kind,role) "
                    "VALUES (?,?,?,?) ON CONFLICT(assumption_id,dependent_id) DO NOTHING",
                    [
                        (aid, d.id, d.kind, d.role)
                        for d in graph.direct_dependents(aid)
                    ],
                )
            conn.commit()

    def load_graph(self) -> AssumptionGraph:
        graph = AssumptionGraph()
        with closing(self._conn()) as conn:
            for r in conn.execute("SELECT * FROM assumptions").fetchall():
                graph.add(
                    Assumption(
                        id=r["id"],
                        statement=r["statement"],
                        confidence=r["confidence"],
                        source=r["source"],
                        verified_at=r["verified_at"],
                    )
                )
            for r in conn.execute("SELECT * FROM dependencies").fetchall():
                graph.depends(
                    Dependent(r["dependent_id"], r["kind"], r["role"]),
                    on=r["assumption_id"],
                )
        return graph

    # -- calibration -----------------------------------------------------
    def save_observations(self, observations: Iterable[Observation]) -> None:
        with closing(self._conn()) as conn:
            conn.executemany(
                "INSERT INTO observations (dimension,first_pass,verified,idea_id,at) "
                "VALUES (?,?,?,?,?)",
                [(o.dimension, o.first_pass, o.verified, o.idea_id, o.at) for o in observations],
            )
            conn.commit()

    def load_calibration(self) -> CalibrationStore:
        store = CalibrationStore()
        with closing(self._conn()) as conn:
            for r in conn.execute("SELECT * FROM observations").fetchall():
                store.observations.append(
                    Observation(
                        dimension=r["dimension"],
                        first_pass=r["first_pass"],
                        verified=r["verified"],
                        idea_id=r["idea_id"],
                        at=r["at"],
                    )
                )
        return store
