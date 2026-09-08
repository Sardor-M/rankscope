"""Lanes are the ranked outputs of the systems under test: TREC run files or JSONL. Qrels judge them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .ranks import Judgment, Qrels


@dataclass(frozen=True, slots=True)
class Hit:
    doc: str
    score: float | None = None


@dataclass(slots=True)
class Lane:
    """One system's ranked lists, best first, keyed by query id."""

    name: str
    hits: dict[str, list[Hit]] = field(default_factory=dict)

    def depth(self) -> int:
        return max((len(hits) for hits in self.hits.values()), default=0)

    def has_scores(self) -> bool:
        return all(hit.score is not None for hits in self.hits.values() for hit in hits)


def _by_score(hits: list[Hit]) -> list[Hit]:
    """trec_eval's rule: score descending. Ties break on doc id ascending so runs are reproducible."""
    return sorted(hits, key=lambda hit: (-hit.score, hit.doc))


def _check_unique(path: Path, lineno: int, query: str, hits: list[Hit]) -> None:
    seen: set[str] = set()
    for hit in hits:
        if hit.doc in seen:
            raise ValueError(f"{path}:{lineno}: query {query!r} lists document {hit.doc!r} twice")
        seen.add(hit.doc)


def read_lane(path: str | Path, name: str | None = None) -> Lane:
    """TREC run (`qid Q0 docid rank score tag`, ordered by score) or JSONL (`{"query", "hits": [...]}`, file order)."""
    path = Path(path)
    lane = Lane(name or path.stem)
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".jsonl":
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            query = str(record["query"])
            hits = [Hit(str(h["doc"]), float(h["score"]) if h.get("score") is not None else None)
                    for h in record["hits"]]
            if query in lane.hits:
                raise ValueError(f"{path}:{lineno}: query {query!r} appears twice")
            _check_unique(path, lineno, query, hits)
            lane.hits[query] = hits
        return lane
    per_query: dict[str, list[Hit]] = {}
    for lineno, line in enumerate(lines, 1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) < 6:
            raise ValueError(f"{path}:{lineno}: expected 'qid Q0 docid rank score tag'")
        query, _, doc, _, score = parts[:5]
        per_query.setdefault(query, []).append(Hit(doc, float(score)))
    for query, hits in per_query.items():
        _check_unique(path, 0, query, hits)
        lane.hits[query] = _by_score(hits)
    return lane


def write_lane(path: str | Path, lane: Lane, tag: str = "rankscope") -> None:
    """TREC run format; a hit without a score is written with 1 / rank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query, hits in lane.hits.items():
            for rank, hit in enumerate(hits, start=1):
                score = hit.score if hit.score is not None else 1.0 / rank
                handle.write(f"{query} Q0 {hit.doc} {rank} {score:.6f} {tag}\n")


def load_lanes(specs: Iterable[str]) -> dict[str, Lane]:
    """`name=path` or `path` (name = file stem), in the order given."""
    lanes: dict[str, Lane] = {}
    for spec in specs:
        name, separator, location = spec.partition("=")
        if not separator:
            name, location = "", spec
        lane = read_lane(location, name or None)
        if lane.name in lanes:
            raise ValueError(f"two lanes named {lane.name!r}; give one a name with name=path")
        lanes[lane.name] = lane
    return lanes


def read_qrels(path: str | Path) -> Qrels:
    """TREC qrels (`qid 0 docid rel`) or JSONL (`{"query", "relevant": {doc: grade}, "stratum"?}`)."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".jsonl":
        judgments = []
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            grades = {str(doc): int(grade) for doc, grade in dict(record.get("relevant") or {}).items()}
            judgments.append(Judgment(str(record["query"]), grades, str(record.get("stratum") or "unlabelled")))
        return Qrels(judgments)
    grades: dict[str, dict[str, int]] = {}
    for lineno, line in enumerate(lines, 1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) < 4:
            raise ValueError(f"{path}:{lineno}: expected 'qid 0 docid rel'")
        query, _, doc, grade = parts[:4]
        grades.setdefault(query, {})[doc] = int(grade)
    return Qrels(Judgment(query, docs) for query, docs in grades.items())


def read_mapping(path: str | Path) -> dict[str, str]:
    """`key<TAB>value` lines, or a JSON object."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return {str(k): str(v) for k, v in json.loads(text).items()}
    mapping = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{lineno}: expected 'key<TAB>value'")
        mapping[parts[0]] = parts[1]
    return mapping


def read_ids(path: str | Path) -> set[str]:
    """One id per line; blank lines and # comments ignored."""
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def slice_lanes(lanes: Mapping[str, Lane], query: str) -> dict[str, list[Hit]]:
    return {name: list(lane.hits.get(query, [])) for name, lane in lanes.items()}
