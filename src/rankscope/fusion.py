"""Rank fusion: reciprocal rank fusion, min-max convex fusion, and a grid fit for the weights."""

from __future__ import annotations

import itertools
from typing import Callable, Mapping, Sequence

from .lanes import Hit

RRF_K = 60

Objective = Callable[[list[Hit]], float]


def rrf(lanes: Mapping[str, Sequence[Hit]], k: int = RRF_K) -> list[Hit]:
    """Sum of 1 / (k + rank) over the lanes a document appears in; ties break on doc id."""
    scores: dict[str, float] = {}
    for hits in lanes.values():
        for position, hit in enumerate(hits, start=1):
            scores[hit.doc] = scores.get(hit.doc, 0.0) + 1.0 / (k + position)
    order = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return [Hit(doc, scores[doc]) for doc in order]


def minmax(values: Sequence[float]) -> list[float]:
    """Scale to [0, 1] over the list; a constant list maps to all ones."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def convex(lanes: Mapping[str, Sequence[Hit]], weights: Mapping[str, float]) -> list[Hit]:
    """Weighted sum of min-max normalised lane scores; a lane that lacks a document contributes 0."""
    fused: dict[str, float] = {}
    for name, hits in lanes.items():
        weight = float(weights.get(name, 0.0))
        if any(hit.score is None for hit in hits):
            raise ValueError(f"lane {name!r} has hits without scores; convex fusion needs scores")
        normalised = minmax([hit.score for hit in hits])
        for hit, value in zip(hits, normalised):
            fused[hit.doc] = fused.get(hit.doc, 0.0) + weight * value
    order = sorted(fused, key=lambda doc: (-fused[doc], doc))
    return [Hit(doc, fused[doc]) for doc in order]


def weight_grid(names: Sequence[str], step: float = 0.1) -> list[dict[str, float]]:
    """Every weight vector on the simplex at the given step, in a fixed order."""
    steps = round(1 / step)
    grid = []
    for combo in itertools.product(range(steps + 1), repeat=len(names)):
        if sum(combo) == steps:
            grid.append({name: c / steps for name, c in zip(names, combo)})
    return grid


def fit_weights(
    per_query: Sequence[tuple[Mapping[str, Sequence[Hit]], Objective]],
    names: Sequence[str],
    step: float = 0.1,
) -> tuple[dict[str, float], float, list[tuple[dict[str, float], float]]]:
    """Grid search for the weights maximising the mean objective over calibration queries."""
    if not per_query:
        raise ValueError("no calibration queries to fit weights on")
    table: list[tuple[dict[str, float], float]] = []
    best: tuple[dict[str, float], float] | None = None
    for weights in weight_grid(names, step):
        value = sum(score(convex(lanes, weights)) for lanes, score in per_query) / len(per_query)
        table.append((weights, value))
        if best is None or value > best[1] + 1e-12:
            best = (weights, value)
    return best[0], best[1], table


def parse_weights(text: str) -> dict[str, float]:
    """`lexical=0.3,dense=0.7` -> a weight mapping."""
    weights = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if not name or not value:
            raise ValueError(f"weights need name=value pairs, got {part!r}")
        weights[name.strip()] = float(value)
    return weights
