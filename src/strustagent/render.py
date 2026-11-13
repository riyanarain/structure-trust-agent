"""Images shown to the agent: a structure colored by pLDDT, and the PAE plot with domain bars.

The domain definitions and per-residue pLDDT are the same information the text condition receives.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from .domains import Domain  # noqa: E402

# AlphaFold DB confidence bands
PLDDT_COLORS = ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"]
PLDDT_BOUNDS = [0, 50, 70, 90, 100.01]
DOMAIN_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return buf.getvalue()


def render_pae(pae: np.ndarray, domains: list[Domain]) -> bytes:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(pae, cmap="Greens_r", vmin=0, vmax=max(31.75, float(np.nanmax(pae))), interpolation="nearest")
    n = len(pae)
    for k, d in enumerate(domains):
        color = DOMAIN_COLORS[k % len(DOMAIN_COLORS)]
        for a, b in d.ranges:
            ax.plot([a - 1, b - 1], [-0.025 * n, -0.025 * n], color=color, lw=5, clip_on=False, solid_capstyle="butt")
            ax.plot([-0.025 * n, -0.025 * n], [a - 1, b - 1], color=color, lw=5, clip_on=False, solid_capstyle="butt")
        a0, b0 = d.ranges[0]
        ax.text((a0 + b0) / 2, -0.06 * n, d.id, color=color, ha="center", va="bottom", fontsize=9, clip_on=False)
    ax.set_xlabel("Scored residue")
    ax.set_ylabel("Aligned residue")
    ax.set_title("Predicted aligned error (angstroms)", fontsize=10, pad=24)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _png(fig)


def _orient(xyz: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(xyz[:, 0])
    c = xyz[ok] - xyz[ok].mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    out = np.full_like(xyz, np.nan)
    out[ok] = c @ vt.T
    return out


def _trace(ax, xyz, colors, lw=3.0):
    pts = np.stack([xyz[:-1], xyz[1:]], axis=1)
    keep = ~np.isnan(pts).any(axis=(1, 2))
    lc = Line3DCollection(pts[keep], colors=[colors[i] for i in np.where(keep)[0]], linewidths=lw)
    ax.add_collection3d(lc)
    # a cube centered on the bounding box, just large enough for the longest extent
    lo, hi = np.nanmin(xyz, axis=0), np.nanmax(xyz, axis=0)
    mid, half = (lo + hi) / 2, float(np.max(hi - lo)) / 2 * 1.02
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)


def render_structure(coords: np.ndarray, plddt: np.ndarray, domains: list[Domain]) -> bytes:
    xyz = _orient(coords)
    cmap = ListedColormap(PLDDT_COLORS)
    norm = BoundaryNorm(PLDDT_BOUNDS, cmap.N)
    by_plddt = [cmap(norm(v)) for v in plddt]
    dom_of = {}
    for k, d in enumerate(domains):
        for r in d.residues:
            dom_of[int(r)] = DOMAIN_COLORS[k % len(DOMAIN_COLORS)]
    by_domain = [dom_of.get(i, "#BBBBBB") for i in range(len(plddt))]
    fig = plt.figure(figsize=(9.0, 3.7))
    fig.subplots_adjust(left=0.0, right=1.0, wspace=-0.05, top=0.84, bottom=0.04)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _trace(ax1, xyz, by_plddt)
    _trace(ax2, xyz, by_domain)
    ax1.set_title("Colored by pLDDT\n(blue >90, cyan 70-90, yellow 50-70, orange <50)", fontsize=9)
    labels = [f"{d.id}" for d in domains] + ["unassigned"]
    handles = [plt.Line2D([0], [0], color=DOMAIN_COLORS[k % len(DOMAIN_COLORS)], lw=3) for k in range(len(domains))]
    handles.append(plt.Line2D([0], [0], color="#BBBBBB", lw=3))
    ax2.set_title("Colored by domain", fontsize=9)
    ax2.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8, frameon=False)
    return _png(fig)


def render_all(p) -> list[bytes]:
    return [render_structure(p.coords, p.plddt, p.domains), render_pae(p.pae, p.domains)]
