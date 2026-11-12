"""Tools the agent can call: Foldseek structural support and ESM-2 sequence plausibility.

Deliberately there is no tool that returns pLDDT or PAE numbers. That would hand the numbers to the image
condition and defeat the image versus text comparison.

Leakage control for Foldseek: hits to any PDB entry cross-referenced to the protein's UniProt accession are removed,
so the held-out experimental structure cannot be found. Homologs deposited after the AlphaFold cutoff can still
appear. Pass `release_dates` (a {pdb_id: "YYYY-MM-DD"} map) to also drop every hit newer than the cutoff.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

TOOL_DOCS = """\
foldseek_support: search experimentally solved structures for matches to regions of the predicted structure.
  args: {"regions": [[start, end], ...]}   (1-based inclusive residue ranges, at most 4 regions)
  returns, per region: top hits with TM-score, query coverage, and sequence identity. A high TM-score over most of a
  region means similar experimental structures exist for that region. It says nothing about how regions sit relative to each other.
esm2_plausibility: score how plausible each region's amino acid sequence is under the ESM-2 protein language model.
  args: {"regions": [[start, end], ...]}   (1-based inclusive, at most 4 regions; use [1, <length>] for the whole protein)
  returns, per region: mean log-probability of the true residues and the mean probability. Low values mean the sequence looks unusual,
  which is common in disordered or poorly conserved regions.
"""


def parse_regions(args: dict, length: int, limit: int = 4) -> list[tuple[int, int]]:
    regions = args.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError('args must contain a non-empty "regions" list')
    out = []
    for r in regions[:limit]:
        a, b = int(r[0]), int(r[1])
        if not (1 <= a <= b <= length):
            raise ValueError(f"region {a}-{b} is outside 1-{length}")
        out.append((a, b))
    return out


@dataclass
class ToolContext:
    protein: object
    cfg: Config
    release_dates: dict[str, str] | None = None
    cache: dict | None = None


def parse_foldseek_m8(text: str, exclude: set[str], top_k: int, release_dates=None, cutoff: str | None = None) -> list[dict]:
    """Columns: query,target,alntmscore,qtmscore,ttmscore,qcov,tcov,fident,evalue,bits"""
    hits = []
    for line in text.strip().splitlines():
        f = line.split("\t")
        if len(f) < 10:
            continue
        pdb = f[1][:4].lower()
        if pdb in exclude:
            continue
        if release_dates and cutoff and release_dates.get(pdb, "0000-00-00") > cutoff:
            continue
        hits.append(
            {
                "target": f[1],
                "tm_score": round(float(f[3]), 3),  # query-normalized TM-score
                "query_coverage": round(float(f[5]), 3),
                "seq_identity": round(float(f[7]), 3),
                "evalue": float(f[8]),
            }
        )
    hits.sort(key=lambda h: -h["tm_score"])
    return hits[:top_k]


def _write_region_pdb(model_path: str, start: int, end: int, out: Path) -> None:
    from Bio.PDB import PDBIO, PDBParser, Select

    class Region(Select):
        def accept_residue(self, residue):  # noqa: N802
            return start <= residue.id[1] <= end

    s = PDBParser(QUIET=True).get_structure("m", model_path)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(out), Region())


def foldseek_support(ctx: ToolContext, args: dict) -> dict:
    cfg, p = ctx.cfg, ctx.protein
    if not cfg.foldseek_db:
        raise RuntimeError("foldseek_db is not set in the config")
    results = []
    for a, b in parse_regions(args, p.length):
        key = ("foldseek", p.acc, a, b)
        if ctx.cache is not None and key in ctx.cache:
            results.append(ctx.cache[key])
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            q, out = tmp / "q.pdb", tmp / "out.m8"
            _write_region_pdb(p.model_path, a, b, q)
            cmd = [
                cfg.foldseek_bin, "easy-search", str(q), cfg.foldseek_db, str(out), str(tmp / "tmp"),
                "--alignment-type", "1", "-e", "0.001", "--max-seqs", "200", "-v", "1",
                "--format-output", "query,target,alntmscore,qtmscore,ttmscore,qcov,tcov,fident,evalue,bits",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            hits = parse_foldseek_m8(out.read_text() if out.exists() else "", p.exclude_pdb_ids, cfg.foldseek_top_k, ctx.release_dates, cfg.pdb_release_after)
        res = {"region": [a, b], "top_hits": hits, "best_tm_score": hits[0]["tm_score"] if hits else None}
        if ctx.cache is not None:
            ctx.cache[key] = res
        results.append(res)
    return {"results": results}


_ESM = {}


def _esm_logprobs(seq: str, model_name: str) -> np.ndarray:
    """Log-probability of the true residue at each position from one unmasked forward pass (a fast approximation to masked marginals)."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    if model_name not in _ESM:
        _ESM[model_name] = (AutoTokenizer.from_pretrained(model_name), EsmForMaskedLM.from_pretrained(model_name).eval())
    tok, model = _ESM[model_name]
    out = np.zeros(len(seq))
    # ESM-2 handles up to 1022 residues, so score long proteins in overlapping windows
    win, step = 1000, 500
    starts = list(range(0, max(1, len(seq) - win + step), step)) or [0]
    acc = np.zeros(len(seq))
    cnt = np.zeros(len(seq))
    for s in starts:
        chunk = seq[s : s + win]
        enc = tok(chunk, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0, 1:-1]
        lp = torch.log_softmax(logits, dim=-1)
        ids = enc["input_ids"][0, 1:-1]
        vals = lp[torch.arange(len(ids)), ids].numpy()
        acc[s : s + len(chunk)] += vals
        cnt[s : s + len(chunk)] += 1
    out = acc / np.maximum(cnt, 1)
    return out


def esm2_plausibility(ctx: ToolContext, args: dict) -> dict:
    p, cfg = ctx.protein, ctx.cfg
    key = ("esm", p.acc)
    cache = ctx.cache if ctx.cache is not None else {}
    if key not in cache:
        cache[key] = _esm_logprobs(p.seq, cfg.esm_model)
    lp = cache[key]
    results = []
    for a, b in parse_regions(args, p.length):
        seg = lp[a - 1 : b]
        results.append({"region": [a, b], "mean_logprob": round(float(seg.mean()), 3), "mean_prob": round(float(np.exp(seg).mean()), 3)})
    return {"results": results, "whole_protein_mean_logprob": round(float(lp.mean()), 3)}


TOOLS = {"foldseek_support": foldseek_support, "esm2_plausibility": esm2_plausibility}


def run_tool(name: str, ctx: ToolContext, args: dict) -> dict:
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(TOOLS)}")
    return TOOLS[name](ctx, args)
