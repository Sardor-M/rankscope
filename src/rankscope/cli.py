"""rankscope: evaluate | calibrate | compare | verdict | fuse | synth | goldens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import goldens, synth
from .calibrate import calibrate
from .fusion import parse_weights
from .gate import Gate
from .jsonio import dump, dumps, load
from .lanes import Lane, load_lanes, read_ids, read_mapping, read_qrels, write_lane
from .measure import CONVEX, RRF, evaluate, per_query_metrics, rankings_for
from .ranks import group_by_mapping, group_by_separator
from .report import query_lines, render, render_calibration, render_compare, render_verdict
from .stats import paired_bootstrap
from .verdict import judge, overall


def _ks(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(part) for part in text.split(",") if part.strip()}))


def _inputs(args):
    """Lanes, qrels (with strata and extra negatives applied) and the group function."""
    lanes = load_lanes(args.lane)
    qrels = read_qrels(args.qrels)
    if getattr(args, "strata", None):
        qrels = qrels.with_strata(read_mapping(args.strata))
    if getattr(args, "negatives", None):
        qrels = qrels.with_negatives(read_ids(args.negatives))
    group_of = None
    if getattr(args, "group_sep", None):
        group_of = group_by_separator(args.group_sep)
    elif getattr(args, "groups", None):
        group_of = group_by_mapping(read_mapping(args.groups))
    return lanes, qrels, group_of


def _weights(args, thresholds: dict | None) -> dict[str, float]:
    if getattr(args, "weights", None):
        text = args.weights
        return dict(load(text).get("weights", {})) if text.endswith(".json") else parse_weights(text)
    return dict(thresholds.get("weights", {})) if thresholds else {}


def cmd_evaluate(args) -> int:
    lanes, qrels, group_of = _inputs(args)
    thresholds = load(args.thresholds) if args.thresholds else None
    weights = _weights(args, thresholds)
    fuse = tuple(part for part in args.fuse.split(",") if part and part != "none")
    if weights and CONVEX not in fuse:
        fuse = (*fuse, CONVEX)
    gates = [Gate(g["name"], float(g["floor"]), args.margin) for g in (thresholds or {}).get("gates", [])]
    for text in args.gate or []:
        name, _, floor = text.partition("=")
        gates.append(Gate(name, float(floor) if floor else float("-inf"), args.margin))
    gate_on = args.gate_on or (thresholds or {}).get("gate_on") or (list(lanes)[-1] if gates else None)
    corpus = read_ids(args.corpus) if args.corpus else None
    report = evaluate(
        lanes, qrels, ks=_ks(args.k), window=args.window, group_of=group_of, fuse=fuse, rrf_k=args.rrf_k,
        weights=weights or None, corpus=corpus, blind=args.blind, gate_on=gate_on, gates=gates,
        gate_level=args.gate_level, margin=args.margin, with_sweep=args.sweep,
    )
    if args.format == "json":
        print(dumps(report))
    else:
        if not args.quiet and args.format == "text":
            print("\n".join(query_lines(report)))
            print()
        print(render(report, ci_k=args.ci_k, markdown=args.format == "markdown"))
    if args.out:
        dump(report, args.out)
        print(f"\nreport written: {args.out}", file=sys.stderr)
    return 0


def cmd_calibrate(args) -> int:
    lanes, qrels, group_of = _inputs(args)
    gate_on = args.gate_on or list(lanes)[-1]
    result = calibrate(
        lanes, qrels, gate_on=gate_on, alpha=args.alpha, fpir=args.fpir, sigmas=args.sigmas, rrf_k=args.rrf_k,
        objective=args.objective, window=args.window, step=args.step, margin=args.margin, group_of=group_of,
        gate_level=args.gate_level,
    )
    print(render_calibration(result))
    dump(result, args.out)
    print(f"thresholds written: {args.out}", file=sys.stderr)
    return 0


def cmd_compare(args) -> int:
    lanes, qrels, group_of = _inputs(args)
    thresholds = load(args.thresholds) if args.thresholds else None
    weights = _weights(args, thresholds) or None
    names = list(lanes) + [RRF] + ([CONVEX] if weights else [])
    baseline = args.baseline or names[0]
    candidate = args.candidate or names[1]
    for name in (baseline, candidate):
        if name not in names:
            raise SystemExit(f"unknown ranking {name!r}; choose from {names}")
    fuse = (RRF, CONVEX) if weights else (RRF,)
    metric = args.metric
    a, b = [], []
    for judgment in qrels.positives():
        every = rankings_for(lanes, judgment.query, fuse=fuse, rrf_k=args.rrf_k, weights=weights)
        for name, values in ((baseline, a), (candidate, b)):
            metrics = per_query_metrics(every[name], judgment, _ks(args.k), group_of)["doc"]
            if metric == "mrr":
                values.append(1.0 / metrics["rank"] if metrics["rank"] else 0.0)
            elif metric.startswith("hit@"):
                k = int(metric[4:])
                values.append(1.0 if 0 < metrics["rank"] <= k else 0.0)
            elif metric in metrics:
                values.append(float(metrics[metric]))
            else:
                raise SystemExit(f"unknown metric {metric!r}; use mrr, hit@k, recall@k or ndcg@k with k in --k")
    result = paired_bootstrap(a, b, resamples=args.resamples, seed=args.seed)
    print(render_compare(result, metric, baseline, candidate, args.min_delta))
    if args.out:
        dump({"metric": metric, "baseline": baseline, "candidate": candidate, "min_delta": args.min_delta, **result}, args.out)
    return 0 if result["p_better"] >= 0.95 and result["delta"] >= args.min_delta else 1


def cmd_verdict(args) -> int:
    report = load(args.report)
    rows = judge(report, load(args.gates)["rules"])
    print(render_verdict(rows, overall(rows), markdown=args.format == "markdown"))
    return 0 if all(row["point"] for row in rows) else 1


def cmd_fuse(args) -> int:
    lanes = load_lanes(args.lane)
    weights = parse_weights(args.weights) if args.weights and not args.weights.endswith(".json") else (
        dict(load(args.weights).get("weights", {})) if args.weights else None)
    method = CONVEX if weights else RRF
    queries = sorted({query for lane in lanes.values() for query in lane.hits})
    fused = Lane(method)
    for query in queries:
        fused.hits[query] = rankings_for(lanes, query, fuse=(method,), rrf_k=args.rrf_k, weights=weights)[method][: args.depth]
    write_lane(args.out, fused, tag=method)
    print(f"{method} over {', '.join(lanes)}: {len(queries)} queries -> {args.out}")
    return 0


def cmd_synth(args) -> int:
    config = synth.SynthConfig(seed=args.seed, queries=args.queries, negatives=args.negatives,
                               calibration_queries=args.calibration_queries,
                               calibration_negatives=args.calibration_negatives, hard_share=args.hard_share)
    counts = synth.write(config, args.out)
    print(" · ".join(f"{key} {value}" for key, value in counts.items()) + f" -> {args.out}")
    return 0


def cmd_goldens(args) -> int:
    if args.check:
        problems = goldens.check(Path(args.check).read_text(encoding="utf-8"))
        for problem in problems:
            print(problem)
        print(f"{len(problems)} mismatching line(s)" if problems else "every line agrees with this implementation")
        return 1 if problems else 0
    if args.out:
        print(f"goldens written: {goldens.write(args.out)}")
    else:
        sys.stdout.write(goldens.build())
    return 0


def _add_inputs(parser: argparse.ArgumentParser, *, qrels: bool = True) -> None:
    parser.add_argument("--lane", action="append", required=True, metavar="[NAME=]PATH",
                        help="a TREC run file or JSONL lane; repeatable, order kept")
    if qrels:
        parser.add_argument("--qrels", required=True, help="TREC qrels or JSONL judgments")
        parser.add_argument("--strata", help="query<TAB>stratum file, or JSON object")
        parser.add_argument("--negatives", help="ids of extra queries judged to have no relevant document")
        parser.add_argument("--group-sep", help="doc ids are GROUP<sep>REST; enables group-level metrics")
        parser.add_argument("--groups", help="doc<TAB>group mapping file; enables group-level metrics")
    parser.add_argument("--rrf-k", type=int, default=60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rankscope", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evaluate", help="metrics, attribution, fusion and gates over lanes")
    _add_inputs(p)
    p.add_argument("--k", default="1,5,10,20")
    p.add_argument("--window", type=int, default=5, help="the window the next stage sees (attribution)")
    p.add_argument("--fuse", default="rrf", help="comma list of rrf, convex, or none")
    p.add_argument("--weights", help="name=w,name=w or a thresholds JSON from `calibrate`")
    p.add_argument("--corpus", help="one indexed doc id per line; enables not_indexed attribution")
    p.add_argument("--blind", help="stratum treated as the label-less class")
    p.add_argument("--thresholds", help="thresholds JSON from `calibrate`: weights, gate_on, floors")
    p.add_argument("--gate-on", help="ranking the floors apply to: a lane, rrf or convex")
    p.add_argument("--gate", action="append", metavar="NAME=FLOOR", help="extra gate (repeatable)")
    p.add_argument("--gate-level", choices=("doc", "group"), default="doc")
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--sweep", action="store_true", help="FPIR/FNIR at every observed floor")
    p.add_argument("--ci-k", type=int, help="k whose interval the tables show (default: max k)")
    p.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    p.add_argument("--out", help="write the full report JSON here")
    p.add_argument("--quiet", action="store_true", help="skip the per-query lines")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("calibrate", help="fit fusion weights and floors on calibration queries")
    _add_inputs(p)
    p.add_argument("--gate-on", help="ranking the floors apply to (default: the last lane)")
    p.add_argument("--gate-level", choices=("doc", "group"), default="doc")
    p.add_argument("--alpha", type=float, default=0.1, help="conformal miss rate")
    p.add_argument("--fpir", type=float, default=0.05, help="false-positive identification rate to bound")
    p.add_argument("--sigmas", type=float, default=3.0)
    p.add_argument("--objective", choices=("mrr", "hit"), default="mrr")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("compare", help="paired bootstrap: is the candidate better than the baseline?")
    _add_inputs(p)
    p.add_argument("--baseline", help="ranking name (default: first lane)")
    p.add_argument("--candidate", help="ranking name (default: second lane); rrf and convex allowed")
    p.add_argument("--metric", default="mrr", help="mrr, hit@k, recall@k or ndcg@k")
    p.add_argument("--k", default="1,5,10,20")
    p.add_argument("--weights")
    p.add_argument("--thresholds")
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-delta", type=float, default=0.0, help="minimum effect size to call it better")
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("verdict", help="judge a report against rules written before the run")
    p.add_argument("--report", required=True)
    p.add_argument("--gates", required=True)
    p.add_argument("--format", choices=("text", "markdown"), default="text")
    p.set_defaults(func=cmd_verdict)

    p = sub.add_parser("fuse", help="write a fused TREC run from several lanes")
    _add_inputs(p, qrels=False)
    p.add_argument("--weights", help="name=w,... or a thresholds JSON; without it, rrf")
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fuse)

    p = sub.add_parser("synth", help="generate synthetic lanes and qrels with known truth")
    p.add_argument("--out", default="data/synth")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--queries", type=int, default=60)
    p.add_argument("--negatives", type=int, default=30)
    p.add_argument("--calibration-queries", type=int, default=120)
    p.add_argument("--calibration-negatives", type=int, default=60)
    p.add_argument("--hard-share", type=float, default=0.35)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("goldens", help="write or check the cross-implementation reference vectors")
    p.add_argument("--out")
    p.add_argument("--check", help="a goldens file produced by another implementation")
    p.set_defaults(func=cmd_goldens)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
