"""Text evidence: the same information as the images, serialized as numbers.

Image condition: structure render + PAE plot (plus the domain id list, which every condition gets).
Text condition: binned pLDDT table + domain table + inter-domain PAE table (this module).
Both: both of the above.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def domain_header(p) -> str:
    """Neutral metadata given in every condition so that domain ids mean the same thing everywhere."""
    lines = [f"Protein {p.acc}, {p.length} residues (numbering starts at 1)."]
    if not p.domains:
        lines.append("No domains were defined for this protein. Judge per-residue trust only.")
        return "\n".join(lines)
    lines.append("Domains (use these ids exactly):")
    for d in p.domains:
        lines.append(f"  {d.id}: residues {d.ranges_str()} ({d.n} residues)")
    if p.pairs:
        lines.append("Domain pairs to judge: " + ", ".join(f"{a}-{b}" for a, b in p.pairs))
    return "\n".join(lines)


def binned_plddt(p, window: int) -> str:
    rows = []
    for s in range(0, p.length, window):
        e = min(s + window, p.length)
        rows.append(f"  {s + 1}-{e}: {p.plddt[s:e].mean():.0f}")
    return "Mean pLDDT per residue block:\n" + "\n".join(rows)


def pae_tables(p) -> str:
    if not p.domains:
        return f"Mean PAE over the whole protein: {p.pae.mean():.1f} A."
    out = ["Per-domain summary (mean pLDDT, mean PAE within the domain in angstroms):"]
    for d in p.domains:
        within = p.pae[np.ix_(d.residues, d.residues)].mean()
        out.append(f"  {d.id}: pLDDT {p.plddt[d.residues].mean():.0f}, within-domain PAE {within:.1f}")
    if p.pairs:
        out.append("Between-domain PAE (angstroms). A->B means: align on A, error on B. High values mean the relative position is uncertain.")
        for a, b in p.pairs:
            ra, rb = p.domain(a).residues, p.domain(b).residues
            ab = p.pae[np.ix_(ra, rb)].mean()
            ba = p.pae[np.ix_(rb, ra)].mean()
            out.append(f"  {a}-{b}: {a}->{b} {ab:.1f}, {b}->{a} {ba:.1f}")
    return "\n".join(out)


def text_evidence(p, cfg: Config) -> str:
    return binned_plddt(p, cfg.text_window) + "\n\n" + pae_tables(p)
