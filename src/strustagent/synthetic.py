"""Synthetic proteins for tests and for checking the pipeline offline.

A synthetic protein has two compact domains joined by a flexible linker. The "experimental" structure is the truth.
The "prediction" is the truth with domain B rotated rigidly about the linker, so that each domain is internally
correct but the pair is misplaced. The PAE is built to look like AlphaFold's: low inside domains and, between domains,
low if the pair is placed well and high if it is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_1to3

from .config import Config
from .protein import Protein, compute_truth, make_protein

AAS = "ACDEFGHIKLMNPQRSTVWY"


def _random_walk(rng, n, step=3.8, persistence=0.65):
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    pts = [np.zeros(3)]
    for _ in range(n - 1):
        new = rng.normal(size=3)
        new /= np.linalg.norm(new)
        direction = persistence * direction + (1 - persistence) * new
        direction /= np.linalg.norm(direction)
        pts.append(pts[-1] + step * direction)
    return np.array(pts)


def _rotation(axis, deg):
    axis = axis / np.linalg.norm(axis)
    a = np.deg2rad(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def make_synthetic(seed: int = 0, misplace_deg: float = 40.0, n_dom: int = 90, linker: int = 14, cfg: Config | None = None, acc: str = "SYN0001") -> Protein:
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    a = _random_walk(rng, n_dom)
    b = _random_walk(rng, n_dom)
    link = _random_walk(rng, linker)
    a -= a.mean(0)
    link = link - link[0] + a[-1] + np.array([4.0, 0, 0])
    b = b - b[0] + link[-1] + np.array([4.0, 0, 0])
    truth = np.vstack([a, link, b])
    model = truth.copy()
    start = n_dom + linker
    hinge = truth[start - 1]
    rot = _rotation(rng.normal(size=3), misplace_deg)
    model[start:] = (truth[start:] - hinge) @ rot.T + hinge
    model += rng.normal(scale=0.3, size=model.shape)
    n = len(truth)
    seq = "".join(rng.choice(list(AAS), size=n))
    plddt = np.full(n, 92.0)
    plddt[n_dom : n_dom + linker] = 55.0
    plddt += rng.normal(scale=2.0, size=n)
    pae = np.full((n, n), 2.0) + np.abs(rng.normal(scale=0.5, size=(n, n)))
    between = 22.0 if misplace_deg > 15 else 4.0
    pae[:n_dom, start:] = between
    pae[start:, :n_dom] = between
    pae[n_dom:start, :] = 15.0
    pae[:, n_dom:start] = 15.0
    pae = (pae + pae.T) / 2
    p = make_protein(acc, seq, model, plddt, pae, cfg)
    p.expt_coords = truth
    compute_truth(p, cfg)
    return p


def write_ca_pdb(path: str | Path, seq: str, coords: np.ndarray, bfactor: np.ndarray | None = None, first_resid: int = 1) -> None:
    lines = []
    serial = 1
    for i, (aa, xyz) in enumerate(zip(seq, coords)):
        if np.isnan(xyz[0]):
            continue
        b = 0.0 if bfactor is None else float(bfactor[i])
        res3 = protein_letters_1to3[aa].upper()
        lines.append(f"ATOM  {serial:5d}  CA  {res3:>3s} A{first_resid + i:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{b:6.2f}           C")
        serial += 1
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")
