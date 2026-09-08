"""JSON with infinities spelled out, so a floor of -inf survives a round trip and other tools can read it."""

from __future__ import annotations

import json
import math
from pathlib import Path


def encode(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def dumps(value) -> str:
    return json.dumps(encode(value), indent=2, ensure_ascii=False)


def dump(value, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dumps(value) + "\n", encoding="utf-8")


def load(path: str | Path):
    return decode(json.loads(Path(path).read_text(encoding="utf-8")))
