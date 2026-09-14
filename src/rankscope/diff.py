"""Compare two reports: what moved, and whether the intervals say it is more than noise."""

from __future__ import annotations

from typing import Mapping, Sequence

IMPROVED, REGRESSED, NOISE, NO_INTERVAL = "improved", "regressed", "within noise", "no interval"


def judge(before: float, after: float, before_ci, after_ci, *, higher_is_better: bool = True) -> str:
    """Improved or regressed only when the two intervals do not overlap."""
    if not before_ci or not after_ci:
        return NO_INTERVAL
    if after_ci[0] > before_ci[1]:
        return IMPROVED if higher_is_better else REGRESSED
    if after_ci[1] < before_ci[0]:
        return REGRESSED if higher_is_better else IMPROVED
    return NOISE


def _row(where: str, metric: str, before: float, after: float, verdict: str, lower_is_better: bool = False) -> dict:
    return {"where": where, "metric": metric, "before": before, "after": after, "delta": after - before,
            "verdict": verdict, "lower_is_better": lower_is_better}


def diff(before: Mapping, after: Mapping) -> list[dict]:
    """One row per metric present in both reports, for every ranking, level and stratum, plus attribution and gates."""
    rows: list[dict] = []
    for ranking, levels in before.get("metrics", {}).items():
        for level, strata in levels.items():
            for stratum, b in strata.items():
                a = after.get("metrics", {}).get(ranking, {}).get(level, {}).get(stratum)
                if not a:
                    continue
                for key, value in b.items():
                    if key == "n" or key.endswith("_ci") or key not in a or isinstance(value, bool):
                        continue
                    if not isinstance(value, (int, float)):
                        continue
                    verdict = judge(value, a[key], b.get(f"{key}_ci"), a.get(f"{key}_ci"))
                    rows.append(_row(f"{ranking} · {level} · {stratum}", key, value, a[key], verdict))
    b = before.get("attribution", {}).get("not_surfaced_share")
    a = after.get("attribution", {}).get("not_surfaced_share")
    if b and a and b.get("n") and a.get("n"):
        verdict = judge(b["p"], a["p"], (b["lo"], b["hi"]), (a["lo"], a["hi"]), higher_is_better=False)
        rows.append(_row("attribution", "not_surfaced_share", b["p"], a["p"], verdict, lower_is_better=True))
    for name, gate_before in before.get("gates", {}).items():
        gate_after = after.get("gates", {}).get(name)
        if not gate_after:
            continue
        for key in ("far", "fnir"):
            b, a = gate_before[key], gate_after[key]
            if not b.get("n") or not a.get("n"):
                continue
            verdict = judge(b["p"], a["p"], (b["lo"], b["hi"]), (a["lo"], a["hi"]), higher_is_better=False)
            rows.append(_row(f"gate {name}", key, b["p"], a["p"], verdict, lower_is_better=True))
    return rows


def notable(rows: Sequence[Mapping]) -> list[dict]:
    """The `all` stratum, attribution and gates in full, plus any other row that moved beyond noise."""
    return [dict(row) for row in rows
            if row["where"].endswith("· all") or row["where"].startswith(("gate", "attribution"))
            or row["verdict"] in (IMPROVED, REGRESSED)]


def summary(rows: Sequence[Mapping]) -> dict[str, int]:
    counts = {IMPROVED: 0, REGRESSED: 0, NOISE: 0, NO_INTERVAL: 0}
    for row in rows:
        counts[row["verdict"]] += 1
    return counts
