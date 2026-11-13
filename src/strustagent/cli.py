"""Command line: select, fetch, inspect, run, evaluate, demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .agent import RunResult, run_one
from .baselines import BASELINES
from .config import Config
from .evaluate import evaluate_method, paired_bootstrap, wilcoxon
from .llm import CachedClient, GeminiClient, OpenAICompatClient
from .protein import load_protein
from .tools import ToolContext

MODEL_METHODS = {"agent:image", "agent:text", "agent:both", "fixed:image", "fixed:text", "fixed:both", "notools:image", "notools:text", "notools:both"}
COMPARISONS = [
    ("fixed:image", "notools:image"),
    ("fixed:text", "notools:text"),
    ("fixed:both", "notools:both"),
    ("fixed:image", "fixed:text"),
    ("fixed:both", "fixed:text"),
    ("notools:image", "notools:text"),
    ("notools:both", "notools:text"),
    ("notools:image", "pae_threshold"),
    ("notools:text", "pae_threshold"),
    ("notools:text", "plddt_threshold"),
    ("agent:image", "agent:text"),
    ("agent:both", "agent:text"),
    ("agent:image", "pae_threshold"),
    ("agent:text", "pae_threshold"),
    ("agent:image", "plddt_threshold"),
    ("agent:image", "notools:image"),
    ("agent:image", "fixed:image"),
]


def load_all(data_dir: Path, cfg: Config, limit: int | None = None, require_pairs: bool = False) -> dict:
    proteins = {}
    for folder in sorted(p for p in data_dir.iterdir() if (p / "meta.json").exists()):
        try:
            p = load_protein(folder, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {folder.name}: {exc}")
            continue
        if require_pairs and not p.true_pairs:
            continue
        proteins[p.acc] = p
        if limit and len(proteins) >= limit:
            break
    return proteins


def cmd_select(a, cfg):
    from .data import select_candidates

    select_candidates(cfg, a.out)


def cmd_fetch(a, cfg):
    from .data import fetch_all

    fetch_all(a.manifest, a.data_dir, cfg, a.limit)


def cmd_inspect(a, cfg):
    proteins = load_all(Path(a.data_dir), cfg)
    rows = []
    for p in proteins.values():
        wrong = sum(v["wrong"] for v in p.true_pairs.values())
        rows.append((p.acc, p.length, len(p.domains), len(p.true_pairs), wrong, float(np.nanmean(p.true_lddt)) if p.true_lddt is not None else float("nan")))
    print(f"{'acc':<10}{'len':>6}{'domains':>9}{'scored pairs':>14}{'misplaced':>11}{'mean lDDT':>11}")
    for r in rows:
        print(f"{r[0]:<10}{r[1]:>6}{r[2]:>9}{r[3]:>14}{r[4]:>11}{r[5]:>11.2f}")
    print(f"\n{len(rows)} proteins, {sum(r[2] >= 2 for r in rows)} multi-domain, {sum(r[3] for r in rows)} scored pairs, {sum(r[4] for r in rows)} misplaced")


def make_client(a, cfg):
    model = a.model or cfg.model
    if not model:
        raise SystemExit("Pass --model (no default on purpose)")
    if a.provider == "gemini":
        inner = GeminiClient(model)
    else:
        inner = OpenAICompatClient(model, a.base_url, a.api_key)
    return CachedClient(inner, a.cache_dir)


def cmd_run(a, cfg):
    cfg.model = a.model or cfg.model
    proteins = load_all(Path(a.data_dir), cfg, a.limit, a.require_pairs)
    out = Path(a.out)
    methods = a.methods
    needs_model = any(m in MODEL_METHODS for m in methods)
    client = make_client(a, cfg) if needs_model else None
    ctx_cache: dict = {}
    images_cache: dict = {}
    release_dates = json.loads(Path(a.release_dates).read_text()) if a.release_dates else None
    out.mkdir(parents=True, exist_ok=True)
    cfg.dump(out / "config_used.json")
    for method in methods:
        mdir = out / method.replace(":", "_")
        mdir.mkdir(exist_ok=True)
        for acc, p in proteins.items():
            f = mdir / f"{acc}.json"
            if f.exists():
                continue
            if method in BASELINES:
                res = BASELINES[method](p, cfg)
            else:
                mode, cond = method.split(":")
                ctx = ToolContext(p, cfg, release_dates, ctx_cache)
                res = run_one(client, p, mode, cond, cfg, ctx, images_cache)
            text = json.dumps(res.to_json())  # serialize first so a failure never leaves an empty file behind
            f.write_text(text)
        print(f"{method}: done")
    if client is not None:
        print(f"model calls made: {client.calls}, cache hits: {client.hits}")


def load_results(runs: Path, method: str) -> dict[str, RunResult]:
    out = {}
    d = runs / method.replace(":", "_")
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        r = RunResult(j["acc"], j["mode"], j["condition"])
        r.residue_trust = None if j["residue_trust"] is None else [math.nan if x is None else x for x in j["residue_trust"]]
        r.pair_judgments, r.tool_calls = j["pair_judgments"], j["tool_calls"]
        r.n_model_calls, r.error, r.final = j["n_model_calls"], j["error"], j["final"]
        out[r.acc] = r
    return out


def cmd_evaluate(a, cfg):
    runs = Path(a.runs)
    methods = sorted(d.name.replace("_", ":", 1) if d.name.split("_")[0] in ("agent", "fixed", "notools") else d.name for d in runs.iterdir() if d.is_dir())
    proteins = load_all(Path(a.data_dir), cfg, require_pairs=False)
    report = {"methods": {}, "comparisons": {}}
    for m in methods:
        res = load_results(runs, m)
        res = {k: v for k, v in res.items() if k in proteins}
        report["methods"][m] = evaluate_method(proteins, res, cfg)
    print(f"{'method':<18}{'n':>5}{'Spearman':>10}{'AUROC':>8}{'pair bal.acc':>14}{'pair acc':>10}{'tools/prot':>12}{'failed':>8}")
    for m, r in report["methods"].items():
        pm = r["pairs"]
        print(f"{m:<18}{r['n_proteins']:>5}{r['mean_spearman']:>10.3f}{r['mean_auroc']:>8.3f}{pm['balanced_accuracy']:>14.3f}{pm['accuracy']:>10.3f}{r['tools']['tool_calls_per_protein']:>12.2f}{r['tools']['failed_runs']:>8}")
    for x, y in COMPARISONS:
        if x in report["methods"] and y in report["methods"]:
            px, py = report["methods"][x]["per_protein"], report["methods"][y]["per_protein"]
            accs = sorted(set(px) & set(py))
            entry = {}
            for metric in ("spearman", "pair_accuracy"):
                va = np.array([px[k][metric] for k in accs])
                vb = np.array([py[k][metric] for k in accs])
                entry[metric] = {"bootstrap": paired_bootstrap(va, vb), "wilcoxon": wilcoxon(va, vb)}
            report["comparisons"][f"{x} vs {y}"] = entry
    if report["comparisons"]:
        print("\nPaired comparisons (mean difference, 95% bootstrap interval, Wilcoxon p):")
        for name, e in report["comparisons"].items():
            for metric, v in e.items():
                b = v["bootstrap"]
                print(f"  {name:<34}{metric:<15}{b['mean_diff']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]  n={b['n']}  p={v['wilcoxon']['p']:.3g}")
    (runs / "report.json").write_text(json.dumps(report, indent=1, default=float))


def cmd_demo(a, cfg):
    from .agent import build_user_message
    from .render import render_all
    from .synthetic import make_synthetic

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = make_synthetic(seed=a.seed, misplace_deg=a.misplace_deg, cfg=cfg)
    imgs = render_all(p)
    (out / "structure.png").write_bytes(imgs[0])
    (out / "pae.png").write_bytes(imgs[1])
    (out / "text_evidence.txt").write_text(build_user_message(p, "text", cfg).text)
    print(f"wrote {out}/structure.png, pae.png, text_evidence.txt")
    print("true domain pair placement:", p.true_pairs)


def cmd_figure(a, cfg):
    from .render import render_all

    p = load_protein(Path(a.data_dir) / a.acc, cfg, with_truth=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = render_all(p)
    (out / "example_structure.png").write_bytes(imgs[0])
    (out / "example_pae.png").write_bytes(imgs[1])
    print(f"wrote {out}/example_structure.png and example_pae.png for {a.acc}; domains: {[(d.id, d.ranges) for d in p.domains]}; true pairs: {p.true_pairs}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="strustagent")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--out", default="data/manifest.json")
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("fetch")
    s.add_argument("--manifest", default="data/manifest.json")
    s.add_argument("--data-dir", default="data/proteins")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("inspect")
    s.add_argument("--data-dir", default="data/proteins")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("run")
    s.add_argument("--data-dir", default="data/proteins")
    s.add_argument("--out", default="runs/pilot")
    s.add_argument("--methods", nargs="+", default=["plddt_threshold", "pae_threshold", "agent:image", "agent:text", "agent:both"])
    s.add_argument("--model", default="")
    s.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    s.add_argument("--base-url", default="http://localhost:8000/v1")
    s.add_argument("--api-key", default="EMPTY")
    s.add_argument("--cache-dir", default=".llm_cache")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--require-pairs", action="store_true", help="only proteins with at least one scoreable domain pair")
    s.add_argument("--release-dates", default=None, help="JSON {pdb_id: YYYY-MM-DD} to filter Foldseek hits newer than the cutoff")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("evaluate")
    s.add_argument("--data-dir", default="data/proteins")
    s.add_argument("--runs", default="runs/pilot")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("figure", help="write the structure and PAE images for one fetched protein")
    s.add_argument("acc")
    s.add_argument("--data-dir", default="data/proteins")
    s.add_argument("--out", default="figs")
    s.set_defaults(fn=cmd_figure)

    s = sub.add_parser("demo")
    s.add_argument("--out", default="demo_out")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--misplace-deg", type=float, default=40.0)
    s.set_defaults(fn=cmd_demo)

    a = ap.parse_args(argv)
    a.fn(a, Config.load(a.config))


if __name__ == "__main__":
    main()
