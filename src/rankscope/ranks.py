"""Judgments and rank semantics: where the first relevant hit sits, at document or group level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

LEVELS = ("doc", "group")
UNLABELLED = "unlabelled"

GroupOf = Callable[[str], str]


@dataclass(frozen=True)
class Judgment:
    """Graded relevance for one query; no positive grade means the answer is not in the corpus."""

    query: str
    grades: Mapping[str, int]
    stratum: str = UNLABELLED

    @property
    def relevant(self) -> frozenset[str]:
        return frozenset(doc for doc, grade in self.grades.items() if grade > 0)

    @property
    def negative(self) -> bool:
        return not self.relevant


class Qrels:
    """Judgments keyed by query id, with helpers for positives, negatives and strata."""

    def __init__(self, judgments: Iterable[Judgment] = ()):
        self.by_query: dict[str, Judgment] = {}
        for judgment in judgments:
            if judgment.query in self.by_query:
                raise ValueError(f"duplicate judgment for query {judgment.query!r}")
            self.by_query[judgment.query] = judgment

    def __len__(self) -> int:
        return len(self.by_query)

    def __iter__(self):
        return iter(self.by_query.values())

    def positives(self) -> list[Judgment]:
        return [j for j in self.by_query.values() if not j.negative]

    def negatives(self) -> list[Judgment]:
        return [j for j in self.by_query.values() if j.negative]

    def with_strata(self, strata: Mapping[str, str]) -> "Qrels":
        return Qrels(
            Judgment(j.query, j.grades, strata.get(j.query, j.stratum)) for j in self.by_query.values()
        )

    def with_negatives(self, queries: Iterable[str], stratum: str = "negative") -> "Qrels":
        """Add queries judged to have no relevant document at all."""
        merged = list(self.by_query.values())
        for query in queries:
            if query not in self.by_query:
                merged.append(Judgment(query, {}, stratum))
        return Qrels(merged)


def group_by_separator(separator: str) -> GroupOf:
    """`D07#003` -> `D07`: the group is the id up to the first separator."""

    def group_of(doc: str) -> str:
        return doc.split(separator, 1)[0]

    return group_of


def group_by_mapping(mapping: Mapping[str, str]) -> GroupOf:
    def group_of(doc: str) -> str:
        return mapping.get(doc, doc)

    return group_of


def rank_of(
    hits: Sequence, relevant: Iterable[str], *, level: str = "doc", group_of: GroupOf | None = None
) -> int:
    """1-based rank of the first hit that is relevant (doc level) or shares a group with one; 0 if none."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
    wanted = set(relevant)
    if level == "group":
        if group_of is None:
            raise ValueError("group level needs a group_of function")
        wanted = {group_of(doc) for doc in wanted}
    for position, hit in enumerate(hits, start=1):
        key = hit.doc if level == "doc" else group_of(hit.doc)
        if key in wanted:
            return position
    return 0


def indexed(corpus: Iterable[str] | None, relevant: Iterable[str]) -> bool:
    """Whether any relevant document exists in the corpus at all; None means unknown, assumed yes."""
    if corpus is None:
        return True
    known = corpus if isinstance(corpus, (set, frozenset)) else set(corpus)
    return any(doc in known for doc in relevant)
