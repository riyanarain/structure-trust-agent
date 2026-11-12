"""Metrics and statistics.

Per-residue trust: Spearman correlation with true lDDT-Ca per protein, and AUROC for flagging residues with true lDDT-Ca below a cutoff.
Domain placement: balanced accuracy, accuracy, and F1 for catching misplaced pairs, pooled over proteins. The positive class is "misplaced".
Comparisons between methods: paired bootstrap over proteins and Wilcoxon signed-rank on per-protein values.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .agent import RunResult
from .config import Config


def auroc(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    if len(score_pos) == 0 or len(score_neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([score_pos, score_neg]))
    r_pos = ranks[: len(score_pos)].sum()
    return float((r_pos - len(score_pos) * (len(score_pos) + 1) / 2) / (len(score_pos) * len(score_neg)))


def residue_metrics(p, res: RunResult, cfg: Config) -> dict:
    out = {"spearman": np.nan, "auroc": np.nan, "coverage": np.nan}
    if p.true_lddt is None or res.residue_trust is None:
        return out
    trust = np.array([np.nan if x is None else x for x in res.residue_trust], dtype=float)
    out["coverage"] = float(np.mean(~np.isnan(trust)))
    trust = np.where(np.isnan(trust), cfg.unscored_residue_default, trust)
    ok = ~np.isnan(p.true_lddt)
    if ok.sum() < 10 or np.ptp(trust[ok]) == 0:
        return out
    out["spearman"] = float(stats.spearmanr(trust[ok], p.true_lddt[ok])[0])
    bad = ok & (p.true_lddt < cfg.low_lddt)
    good = ok & ~bad
    out["auroc"] = auroc(-trust[bad], -trust[good])  # high score = flagged as untrustworthy
    return out


def pair_outcomes(p, res: RunResult) -> list[dict]:
    rows = []
    for (a, b), truth in p.true_pairs.items():
        j = res.pair_judgments.get(f"{a}-{b}")
        rows.append(
            {
                "pair": f"{a}-{b}",
                "truth_wrong": truth["wrong"],
                "err": truth["err"],
                "pred_wrong": None if j is None else (not j["reliable"]),
            }
        )
    return rows


def pair_metrics(rows: list[dict]) -> dict:
    """rows from pair_outcomes pooled over proteins. Unanswered pairs count as errors."""
    if not rows:
        return {"n_pairs": 0, "accuracy": np.nan, "balanced_accuracy": np.nan, "f1_wrong": np.nan, "answered": np.nan}
    t = np.array([r["truth_wrong"] for r in rows])
    pr = np.array([(not r["truth_wrong"]) if r["pred_wrong"] is None else r["pred_wrong"] for r in rows])
    tp, tn = (t & pr).sum(), (~t & ~pr).sum()
    fp, fn = (~t & pr).sum(), (t & ~pr).sum()
    tpr = tp / (tp + fn) if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    return {
        "n_pairs": int(len(rows)),
        "n_wrong": int(t.sum()),
        "accuracy": float((tp + tn) / len(rows)),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "f1_wrong": float(f1),
        "answered": float(np.mean([r["pred_wrong"] is not None for r in rows])),
    }


def tool_stats(results: list[RunResult]) -> dict:
    calls = [c for r in results for c in r.tool_calls]
    n = max(len(results), 1)
    return {
        "tool_calls_per_protein": len(calls) / n,
        "tool_error_rate": float(np.mean([not c.get("ok", False) for c in calls])) if calls else 0.0,
        "model_calls_per_protein": sum(r.n_model_calls for r in results) / n,
        "failed_runs": int(sum(r.error is not None for r in results)),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Mean of (a - b) over proteins with a 95% bootstrap interval. NaN pairs are dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = (a - b)[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}


def wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = (a - b)[ok]
    if len(d) < 5 or np.all(d == 0):
        return {"n": int(len(d)), "p": float("nan")}
    return {"n": int(len(d)), "p": float(stats.wilcoxon(d).pvalue)}


def evaluate_method(proteins: dict, results: dict[str, RunResult], cfg: Config) -> dict:
    """proteins: acc -> Protein; results: acc -> RunResult for one method."""
    per_protein = {}
    rows = []
    for acc, res in results.items():
        p = proteins[acc]
        m = residue_metrics(p, res, cfg)
        pr = pair_outcomes(p, res)
        rows += pr
        m["pair_accuracy"] = float(np.mean([(r["pred_wrong"] == r["truth_wrong"]) for r in pr])) if pr else np.nan
        per_protein[acc] = m
    spearman = np.array([m["spearman"] for m in per_protein.values()])
    au = np.array([m["auroc"] for m in per_protein.values()])
    return {
        "n_proteins": len(results),
        "mean_spearman": float(np.nanmean(spearman)) if np.any(~np.isnan(spearman)) else float("nan"),
        "mean_auroc": float(np.nanmean(au)) if np.any(~np.isnan(au)) else float("nan"),
        "pairs": pair_metrics(rows),
        "tools": tool_stats(list(results.values())),
        "per_protein": per_protein,
    }
