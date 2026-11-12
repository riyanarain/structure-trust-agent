"""One protein: prediction, confidence, domains, and (when available) experimental ground truth.

On-disk layout, one folder per UniProt accession:
    meta.json     accession, sequence, experimental entry, all PDB ids cross-referenced to the accession
    model.pdb     AlphaFold DB prediction
    pae.json      AlphaFold DB predicted aligned error
    expt.cif      experimental structure (held out from the agent)
    truth.json    cached ground truth (true lDDT-Ca per residue and domain pair placement)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .domains import Domain, domain_pairs, domains_from_pae
from .groundtruth import lddt_ca, pair_placement
from .structure import best_chain, project_to_reference, read_chain


@dataclass
class Protein:
    acc: str
    seq: str
    coords: np.ndarray  # (L, 3) predicted C-alpha
    plddt: np.ndarray  # (L,)
    pae: np.ndarray  # (L, L)
    domains: list[Domain]
    pairs: list[tuple[str, str]]
    model_path: str | None = None
    exclude_pdb_ids: set[str] = field(default_factory=set)
    # ground truth, never shown to the agent
    expt_coords: np.ndarray | None = None
    true_lddt: np.ndarray | None = None
    true_pairs: dict[tuple[str, str], dict] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.seq)

    def domain(self, did: str) -> Domain:
        return next(d for d in self.domains if d.id == did)


def load_pae(path: str | Path) -> np.ndarray:
    """Read an AlphaFold DB PAE file. Handles the matrix format and the older residue-triplet format."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = data[0]
    if "predicted_aligned_error" in data:
        return np.asarray(data["predicted_aligned_error"], dtype=float)
    r1, r2, dist = (np.asarray(data[k]) for k in ("residue1", "residue2", "distance"))
    n = int(max(r1.max(), r2.max()))
    pae = np.zeros((n, n))
    pae[r1 - 1, r2 - 1] = dist
    return pae


def make_protein(acc, seq, coords, plddt, pae, cfg: Config, **kw) -> Protein:
    doms = domains_from_pae(pae, plddt, cfg.domain_pae_cut, cfg.domain_min_len, cfg.domain_plddt_min)
    pairs = domain_pairs(doms, cfg.max_domains_in_prompt)
    keep = {x for p in pairs for x in p}
    doms = [d for d in doms if d.id in keep or len(doms) <= cfg.max_domains_in_prompt]
    return Protein(acc, seq, coords, plddt, pae, doms, pairs, **kw)


def compute_truth(p: Protein, cfg: Config) -> None:
    if p.expt_coords is None:
        return
    p.true_lddt = lddt_ca(p.coords, p.expt_coords, cfg.lddt_radius)
    p.true_pairs = {}
    for a, b in p.pairs:
        res = pair_placement(
            p.coords,
            p.expt_coords,
            p.domain(a).residues,
            p.domain(b).residues,
            cfg.placement_wrong_angstrom,
            cfg.placement_internal_max_angstrom,
            cfg.placement_min_residues,
        )
        if res is not None:
            p.true_pairs[(a, b)] = res


def load_protein(folder: str | Path, cfg: Config, with_truth: bool = True) -> Protein:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    seq = meta["sequence"]
    chain = read_chain(folder / "model.pdb")
    coords, bfac, identity, _ = project_to_reference(chain, seq)
    if identity < 0.95:
        raise ValueError(f"{meta['uniprot']}: predicted model does not match the UniProt sequence (identity {identity:.2f})")
    plddt = np.nan_to_num(bfac, nan=0.0)
    pae = load_pae(folder / "pae.json")
    if pae.shape[0] != len(seq):
        raise ValueError(f"{meta['uniprot']}: PAE size {pae.shape[0]} differs from sequence length {len(seq)}")
    p = make_protein(
        meta["uniprot"],
        seq,
        coords,
        plddt,
        pae,
        cfg,
        model_path=str(folder / "model.pdb"),
        exclude_pdb_ids={x.lower() for x in meta.get("pdb_ids_all", [])},
    )
    if with_truth and (folder / "expt.cif").exists():
        expt = meta["experimental"]
        ech, _ = best_chain(folder / "expt.cif", seq, expt.get("chain"))
        p.expt_coords, _, _, _ = project_to_reference(ech, seq)
        compute_truth(p, cfg)
    return p
