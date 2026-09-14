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

    @classmethod
    def from_dict(cls, name: str, mapping: Mapping[str, Mapping[str, float]]) -> "Lane":
        """`{query: {doc: score}}`, the pytrec_eval and BEIR convention; hits ordered by score."""
        lane = cls(name)
        for query, scores in mapping.items():
            if not isinstance(scores, Mapping):
                raise ValueError(f"lane {name!r}: query {query!r} must map doc ids to scores")
            lane.hits[str(query)] = _by_score([Hit(str(doc), float(score)) for doc, score in scores.items()])
        return lane

    def to_dict(self) -> dict[str, dict[str, float]]:
        """The same convention back; a hit without a score gets 1 / rank."""
        return {query: {hit.doc: hit.score if hit.score is not None else 1.0 / rank
                        for rank, hit in enumerate(hits, start=1)} for query, hits in self.hits.items()}


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
    """TREC run (ordered by score), JSONL (`{"query", "hits": [...]}`, file order), or JSON `{query: {doc: score}}`."""
    path = Path(path)
    lane = Lane(name or path.stem)
    if path.suffix == ".json":
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{path}: a JSON lane is an object of the form {{query: {{doc: score}}}}")
        return Lane.from_dict(lane.name, mapping)
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
    """TREC qrels (`qid 0 docid rel`), JSONL (`{"query", "relevant": {...}, "stratum"?}`), or JSON `{query: {doc: grade}}`."""
    path = Path(path)
    if path.suffix == ".json":
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{path}: JSON qrels are an object of the form {{query: {{doc: grade}}}}")
        return Qrels.from_dict(mapping)
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


def truncate(lanes: Mapping[str, Lane], depth: int) -> dict[str, Lane]:
    """Every lane cut to the same depth, so fusion and floors see comparable lists."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    return {name: Lane(lane.name, {query: hits[:depth] for query, hits in lane.hits.items()})
            for name, lane in lanes.items()}


def slice_lanes(lanes: Mapping[str, Lane], query: str) -> dict[str, list[Hit]]:
    return {name: list(lane.hits.get(query, [])) for name, lane in lanes.items()}
