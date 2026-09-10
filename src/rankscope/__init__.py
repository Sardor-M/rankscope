"""rankscope: certify retrieval results with intervals, attribution, fusion and a calibrated abstain gate."""

from .calibrate import calibrate
from .fusion import convex, fit_weights, rrf
from .gate import Decision, Gate, decide, operating_point, sweep
from .lanes import Hit, Lane, load_lanes, read_lane, read_qrels, write_lane
from .measure import evaluate, rankings_for
from .metrics import hit_rate, miss_attribution, mrr, ndcg_at, recall_at, summarise
from .ranks import Judgment, Qrels, group_by_mapping, group_by_separator, rank_of
from .stats import conformal_floor, conformal_quantile, negatives_for, null_floor, paired_bootstrap, wilson
from .verdict import judge, overall

__version__ = "0.2.0"

__all__ = [
    "Decision", "Gate", "Hit", "Judgment", "Lane", "Qrels", "calibrate", "conformal_floor", "conformal_quantile",
    "convex", "decide", "evaluate", "fit_weights", "group_by_mapping", "group_by_separator", "hit_rate", "judge",
    "load_lanes", "miss_attribution", "mrr", "ndcg_at", "negatives_for", "null_floor", "operating_point", "overall",
    "paired_bootstrap", "rank_of", "rankings_for", "read_lane", "read_qrels", "recall_at", "rrf", "summarise",
    "sweep", "wilson", "write_lane",
]
