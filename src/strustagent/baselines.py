"""Rule-based baselines that need no model.

plddt_threshold: trust = pLDDT / 100. A pair counts as reliable if both domains average at least the threshold.
                 pLDDT carries no information about relative domain placement, so this is the weak baseline for pairs.
pae_threshold:   same per-residue trust. A pair counts as reliable if the mean between-domain PAE is under the threshold.
                 This is an addition beyond the proposal. It is the strong numeric rule the text condition could simply apply.
"""

from __future__ import annotations

import numpy as np

from .agent import RunResult
from .config import Config


def _pair_stats(p, a, b):
    ra, rb = p.domain(a).residues, p.domain(b).residues
    pae = 0.5 * (p.pae[np.ix_(ra, rb)].mean() + p.pae[np.ix_(rb, ra)].mean())
    plddt = min(p.plddt[ra].mean(), p.plddt[rb].mean())
    return pae, plddt


def plddt_threshold(p, cfg: Config) -> RunResult:
    res = RunResult(p.acc, "baseline", "plddt_threshold")
    res.residue_trust = (p.plddt / 100.0).tolist()
    for a, b in p.pairs:
        _, pl = _pair_stats(p, a, b)
        res.pair_judgments[f"{a}-{b}"] = {"reliable": pl >= cfg.plddt_threshold, "confidence": pl / 100.0}
    return res


def pae_threshold(p, cfg: Config) -> RunResult:
    res = RunResult(p.acc, "baseline", "pae_threshold")
    res.residue_trust = (p.plddt / 100.0).tolist()
    for a, b in p.pairs:
        pae, _ = _pair_stats(p, a, b)
        res.pair_judgments[f"{a}-{b}"] = {"reliable": pae < cfg.pae_pair_threshold, "confidence": float(1.0 / (1.0 + pae))}
    return res


BASELINES = {"plddt_threshold": plddt_threshold, "pae_threshold": pae_threshold}
