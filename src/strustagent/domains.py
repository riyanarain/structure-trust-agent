"""Define domains from the PAE matrix.

Residues with low pLDDT are left out, the remaining residues are clustered with average linkage on the
symmetrized PAE, and clusters shorter than `min_len` are dropped. A domain can be discontinuous.
The domain definition uses only the prediction (PAE and pLDDT), never the experimental structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class Domain:
    id: str
    residues: np.ndarray  # 0-based residue indices
    ranges: list[tuple[int, int]]  # 1-based inclusive

    @property
    def n(self) -> int:
        return int(len(self.residues))

    def ranges_str(self) -> str:
        return ", ".join(f"{a}-{b}" for a, b in self.ranges)


def _to_ranges(idx: np.ndarray) -> list[tuple[int, int]]:
    idx = np.sort(idx)
    ranges = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i != prev + 1:
            ranges.append((start + 1, prev + 1))
            start = i
        prev = i
    ranges.append((start + 1, prev + 1))
    return ranges


def domains_from_pae(
    pae: np.ndarray,
    plddt: np.ndarray,
    pae_cut: float = 5.0,
    min_len: int = 40,
    plddt_min: float = 70.0,
) -> list[Domain]:
    sym = (pae + pae.T) / 2.0
    ordered = np.where(plddt >= plddt_min)[0]
    if len(ordered) < min_len:
        return []
    sub = sym[np.ix_(ordered, ordered)].copy()
    np.fill_diagonal(sub, 0.0)
    labels = fcluster(linkage(squareform(sub, checks=False), method="average"), t=pae_cut, criterion="distance")
    groups = [ordered[labels == lab] for lab in np.unique(labels)]
    groups = [g for g in groups if len(g) >= min_len]
    groups.sort(key=lambda g: int(g.min()))
    return [Domain(f"D{i + 1}", g, _to_ranges(g)) for i, g in enumerate(groups)]


def domain_pairs(domains: list[Domain], limit: int | None = None) -> list[tuple[str, str]]:
    ds = sorted(domains, key=lambda d: -d.n)[:limit] if limit else list(domains)
    ds = sorted(ds, key=lambda d: int(d.residues.min()))
    return [(ds[i].id, ds[j].id) for i in range(len(ds)) for j in range(i + 1, len(ds))]
