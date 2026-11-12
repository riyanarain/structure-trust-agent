"""Read structures and project them onto a reference (UniProt) sequence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser

try:  # Biopython >= 1.80
    from Bio.Data.PDBData import protein_letters_3to1_extended as _THREE_TO_ONE
except ImportError:  # pragma: no cover
    from Bio.Data.IUPACData import protein_letters_3to1 as _THREE_TO_ONE


@dataclass
class Chain:
    seq: str
    coords: np.ndarray  # (n, 3) C-alpha
    bfactor: np.ndarray  # (n,) for AlphaFold models this is pLDDT


def _one_letter(resname: str) -> str | None:
    return _THREE_TO_ONE.get(resname.upper()) or _THREE_TO_ONE.get(resname.capitalize())


def read_chain(path: str | Path, chain_id: str | None = None, model_index: int = 0) -> Chain:
    path = str(path)
    parser = MMCIFParser(QUIET=True) if path.endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    structure = parser.get_structure("s", path)
    model = structure[model_index]
    if chain_id is not None and chain_id in [c.id for c in model]:
        chain = model[chain_id]
    else:
        chain = next(iter(model))
    seq, xyz, bf = [], [], []
    for res in chain:
        if res.id[0] != " " or "CA" not in res:
            continue
        aa = _one_letter(res.get_resname())
        if aa is None or len(aa) != 1:
            continue
        atom = res["CA"]
        seq.append(aa)
        xyz.append(atom.coord)
        bf.append(atom.get_bfactor())
    if not seq:
        raise ValueError(f"No C-alpha atoms read from {path} chain {chain_id}")
    return Chain("".join(seq), np.asarray(xyz, dtype=float), np.asarray(bf, dtype=float))


def _aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score = 2.0
    al.mismatch_score = -1.0
    al.open_gap_score = -2.0
    al.extend_gap_score = -0.5
    al.end_gap_score = 0.0  # free end gaps, so partial constructs align cleanly
    return al


def align_to_reference(seq: str, ref: str) -> tuple[np.ndarray, float, float]:
    """Return (index map seq->ref with -1 for unaligned, identity over aligned positions, coverage of ref)."""
    aln = _aligner().align(ref, seq)[0]
    ref_blocks, seq_blocks = aln.aligned
    idx = -np.ones(len(seq), dtype=int)
    matches = 0
    aligned = 0
    for (r0, r1), (s0, s1) in zip(ref_blocks, seq_blocks):
        for k in range(r1 - r0):
            idx[s0 + k] = r0 + k
            aligned += 1
            matches += int(ref[r0 + k] == seq[s0 + k])
    identity = matches / aligned if aligned else 0.0
    coverage = aligned / len(ref)
    return idx, identity, coverage


def project_to_reference(chain: Chain, ref: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Place chain coordinates (and B-factors) on the reference numbering. Missing residues are NaN."""
    idx, identity, coverage = align_to_reference(chain.seq, ref)
    coords = np.full((len(ref), 3), np.nan)
    bfac = np.full(len(ref), np.nan)
    ok = idx >= 0
    coords[idx[ok]] = chain.coords[ok]
    bfac[idx[ok]] = chain.bfactor[ok]
    return coords, bfac, identity, coverage


def best_chain(path: str | Path, ref: str, preferred: str | None = None) -> tuple[Chain, str]:
    """Pick the chain in a file that best matches the reference sequence (preferred chain first)."""
    path = str(path)
    parser = MMCIFParser(QUIET=True) if path.endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    model = parser.get_structure("s", path)[0]
    ids = [c.id for c in model]
    order = ([preferred] if preferred in ids else []) + [c for c in ids if c != preferred]
    best, best_score = None, -1.0
    for cid in order:
        try:
            chain = read_chain(path, cid)
        except ValueError:
            continue
        _, identity, coverage = align_to_reference(chain.seq, ref)
        score = identity * coverage
        if score > best_score + 1e-9:
            best, best_score, best_id = chain, score, cid
    if best is None:
        raise ValueError(f"No usable chain in {path}")
    return best, best_id
