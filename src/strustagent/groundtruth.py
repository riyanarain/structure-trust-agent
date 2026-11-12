"""Ground truth from the experimental structure.

lDDT here is the C-alpha version (lDDT-Ca), the same quantity AlphaFold's pLDDT estimates (Jumper et al. 2021).
It is superposition-free, so it cannot see misplaced domains. Domain placement is scored separately:
superpose the prediction on domain A and measure the C-alpha RMSD of domain B (and the reverse).
"""

from __future__ import annotations

import numpy as np

THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def pairwise_dist(x: np.ndarray) -> np.ndarray:
    d = x[:, None, :] - x[None, :, :]
    return np.sqrt((d**2).sum(-1))


def lddt_ca(model: np.ndarray, ref: np.ndarray, radius: float = 15.0) -> np.ndarray:
    """Per-residue lDDT-Ca of `model` against `ref`. NaN where the reference residue is missing or has no neighbours."""
    n = len(ref)
    d_ref = pairwise_dist(ref)
    d_mod = pairwise_dist(model)
    valid = ~np.isnan(d_ref) & ~np.isnan(d_mod)
    neighbours = valid & (d_ref < radius) & ~np.eye(n, dtype=bool)
    diff = np.abs(np.where(valid, d_mod - d_ref, np.inf))
    count = neighbours.sum(1)
    score = np.zeros(n)
    for t in THRESHOLDS:
        score += ((diff < t) & neighbours).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = score / (len(THRESHOLDS) * count)
    out[count == 0] = np.nan
    out[np.isnan(ref[:, 0])] = np.nan
    return out


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation r and translation t so that p @ r.T + t best matches q."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - r @ pc


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def _usable(idx: np.ndarray, model: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(model[idx, 0]) & ~np.isnan(ref[idx, 0])
    return idx[ok]


def _directed(model, ref, ia, ib) -> tuple[float, float, float]:
    r, t = kabsch(model[ia], ref[ia])
    rms_a = _rmsd(model[ia] @ r.T + t, ref[ia])
    rb, tb = kabsch(model[ib], ref[ib])
    rms_b_internal = _rmsd(model[ib] @ rb.T + tb, ref[ib])
    placed = _rmsd(model[ib] @ r.T + t, ref[ib])
    return rms_a, rms_b_internal, placed


def pair_placement(
    model: np.ndarray,
    ref: np.ndarray,
    res_a: np.ndarray,
    res_b: np.ndarray,
    wrong_angstrom: float = 5.0,
    internal_max: float = 4.0,
    min_residues: int = 20,
) -> dict | None:
    """Placement error for a domain pair, or None if the pair cannot be scored fairly."""
    ia, ib = _usable(res_a, model, ref), _usable(res_b, model, ref)
    if len(ia) < min_residues or len(ib) < min_residues:
        return None
    rms_a, rms_b_int, err_ab = _directed(model, ref, ia, ib)
    _, _, err_ba = _directed(model, ref, ib, ia)
    if rms_a > internal_max or rms_b_int > internal_max:
        return None  # a domain is itself badly predicted, so this is not a placement question
    err = 0.5 * (err_ab + err_ba)
    return {"err": err, "wrong": bool(err > wrong_angstrom), "n_a": int(len(ia)), "n_b": int(len(ib))}
