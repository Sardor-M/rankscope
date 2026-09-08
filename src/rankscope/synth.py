"""Synthetic lanes with known truth: a lexical lane that is blind on one stratum, a dense lane that is not."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .jsonio import dump
from .lanes import Hit, Lane, write_lane

LEXICAL = "lexical"
DENSE = "dense"
STRATA = ("easy", "medium", "hard")
NEGATIVE = "negative"
GROUP_SEPARATOR = "#"


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 0
    groups: int = 12
    docs_per_group: int = 6
    catalogue_docs: int = 40
    queries: int = 60
    negatives: int = 30
    calibration_queries: int = 120
    calibration_negatives: int = 60
    hard_negative_share: float = 0.5
    hard_share: float = 0.35
    medium_share: float = 0.35
    twin_share: float = 0.1
    not_indexed_share: float = 0.05
    weak_dense_share: float = 0.3
    depth: int = 20


@dataclass(frozen=True)
class Doc:
    id: str
    group: str
    weak_dense: bool


@dataclass(frozen=True)
class Corpus:
    docs: tuple[Doc, ...]
    indexed: tuple[Doc, ...]


def build_corpus(config: SynthConfig, rng: random.Random) -> Corpus:
    docs: list[Doc] = []
    for g in range(config.groups):
        group = f"D{g:02d}"
        count = config.catalogue_docs if g == 0 else config.docs_per_group
        for i in range(count):
            docs.append(Doc(f"{group}{GROUP_SEPARATOR}{i:03d}", group, rng.random() < config.weak_dense_share))
    indexed = tuple(doc for doc in docs if rng.random() >= config.not_indexed_share)
    return Corpus(tuple(docs), indexed)


def _stratum(rng: random.Random, config: SynthConfig) -> str:
    draw = rng.random()
    if draw < config.hard_share:
        return "hard"
    if draw < config.hard_share + config.medium_share:
        return "medium"
    return "easy"


def _lexical(rng, docs, truth: set[str], stratum: str, bumped_group: str | None) -> dict[str, float]:
    scores = {}
    for doc in docs:
        score = rng.gammavariate(2.0, 1.5)
        if doc.group == bumped_group:
            score += 1.5
        if doc.id in truth and stratum == "easy":
            score = 9.0 + rng.gauss(0.0, 2.0)
        elif doc.id in truth and stratum == "medium":
            score = 5.0 + rng.gauss(0.0, 2.0)
        elif doc.id in truth:
            score = 0.0
        scores[doc.id] = max(0.0, score)
    return scores


def _dense(rng, docs, truth: set[str], bumped_group: str | None) -> dict[str, float]:
    scores = {}
    for doc in docs:
        score = rng.gauss(0.45, 0.08)
        if doc.group == bumped_group:
            score += 0.08
        if doc.id in truth:
            score = rng.gauss(0.62 if doc.weak_dense else 0.78, 0.07)
        scores[doc.id] = min(0.999, max(0.0, score))
    return scores


def _top(scores: dict[str, float], depth: int) -> list[Hit]:
    order = sorted(scores, key=lambda doc: (-scores[doc], doc))[:depth]
    return [Hit(doc, round(scores[doc], 6)) for doc in order if scores[doc] > 0.0]


def generate(config: SynthConfig, corpus: Corpus, rng: random.Random, *, queries: int, negatives: int, prefix: str):
    """Qrels rows, strata and two lanes for one query set over a shared corpus."""
    indexed = list(corpus.indexed)
    qrels: list[tuple[str, str, int]] = []
    strata: dict[str, str] = {}
    lanes = {LEXICAL: Lane(LEXICAL), DENSE: Lane(DENSE)}
    for q in range(queries):
        query = f"{prefix}q{q:03d}"
        stratum = _stratum(rng, config)
        target = rng.choice(corpus.docs)
        truth = {target.id}
        qrels.append((query, target.id, 2))
        if rng.random() < config.twin_share:
            twin = rng.choice([doc for doc in corpus.docs if doc.group != target.group])
            truth.add(twin.id)
            qrels.append((query, twin.id, 1))
        strata[query] = stratum
        lanes[LEXICAL].hits[query] = _top(_lexical(rng, indexed, truth, stratum, target.group), config.depth)
        lanes[DENSE].hits[query] = _top(_dense(rng, indexed, truth, target.group), config.depth)
    for n in range(negatives):
        query = f"{prefix}n{n:03d}"
        hard = rng.random() < config.hard_negative_share
        bumped = rng.choice(corpus.docs).group if hard else None
        qrels.append((query, indexed[0].id, 0))
        strata[query] = NEGATIVE
        lanes[LEXICAL].hits[query] = _top(_lexical(rng, indexed, set(), "easy", bumped), config.depth)
        lanes[DENSE].hits[query] = _top(_dense(rng, indexed, set(), bumped), config.depth)
    return qrels, strata, lanes


DEFAULT_GATES = {
    "rules": [
        {"name": "hard share of lexical misses is material", "path": "attribution.blind_share_of_misses.lexical",
         "op": ">=", "threshold": 0.2},
        {"name": "dense hit@20 on the hard stratum", "path": "metrics.dense.doc.hard.hit@20",
         "op": ">=", "threshold": 0.6},
        {"name": "rrf keeps hit@5 over all strata", "path": "metrics.rrf.doc.all.hit@5",
         "op": ">=", "threshold": 0.5},
        {"name": "false accepts at the FPIR-calibrated floor", "path": "gates.fpir-0.05.far",
         "op": "<=", "threshold": 0.05},
    ]
}


def _write_qrels(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{query} 0 {doc} {grade}\n" for query, doc, grade in rows), encoding="utf-8")


def _write_strata(path: Path, strata: dict[str, str]) -> None:
    path.write_text("".join(f"{query}\t{stratum}\n" for query, stratum in strata.items()), encoding="utf-8")


def write(config: SynthConfig, out: str | Path) -> dict:
    """qrels.txt, qrels-calibration.txt, strata.tsv, lanes/*.run, corpus.txt, gates.json under `out`."""
    out = Path(out)
    rng = random.Random(config.seed)
    corpus = build_corpus(config, rng)
    qrels, strata, lanes = generate(config, corpus, random.Random(config.seed * 7919 + 1),
                                    queries=config.queries, negatives=config.negatives, prefix="")
    cal_qrels, cal_strata, cal_lanes = generate(config, corpus, random.Random(config.seed * 7919 + 2),
                                                queries=config.calibration_queries,
                                                negatives=config.calibration_negatives, prefix="cal-")
    _write_qrels(out / "qrels.txt", qrels)
    _write_qrels(out / "qrels-calibration.txt", cal_qrels)
    _write_strata(out / "strata.tsv", {**strata, **cal_strata})
    for name in (LEXICAL, DENSE):
        merged = Lane(name, {**lanes[name].hits, **cal_lanes[name].hits})
        write_lane(out / "lanes" / f"{name}.run", merged, tag=name)
    (out / "corpus.txt").write_text("".join(f"{doc.id}\n" for doc in corpus.indexed), encoding="utf-8")
    dump(DEFAULT_GATES, out / "gates.json")
    dump(asdict(config), out / "synth-config.json")
    return {"docs": len(corpus.docs), "indexed": len(corpus.indexed), "groups": config.groups,
            "queries": config.queries + config.negatives,
            "calibration": config.calibration_queries + config.calibration_negatives}
