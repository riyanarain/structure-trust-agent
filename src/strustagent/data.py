"""Build the evaluation set: select proteins, then download predictions and experimental structures.

NOT TESTED AGAINST THE LIVE SERVICES. This code was written without network access to UniProt, RCSB, PDBe, or the
AlphaFold DB. Run `strustagent select` and `strustagent fetch` on a handful of proteins first, and expect to adjust field
names if an API has changed. Known change: the AlphaFold DB API renamed `uniprotSequence` to `sequence` (old name removed after 25 June 2026).
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import requests

from .config import Config
from .structure import best_chain, project_to_reference, read_chain

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
RCSB_GQL = "https://data.rcsb.org/graphql"
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
RCSB_FILE = "https://files.rcsb.org/download/{pdb}.cif"

_session = requests.Session()


def _get(url, **kw):
    for attempt in range(5):
        r = _session.get(url, timeout=60, **kw)
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def select_candidates(cfg: Config, out_path: str | Path, page_size: int = 500, log=print) -> list[dict]:
    """Reviewed human proteins with a disease annotation and PDB structures, then keep those with a post-cutoff experimental entry."""
    query = f"(reviewed:true) AND (organism_id:9606) AND (cc_disease:*) AND (structure_3d:true) AND (length:[{cfg.min_len} TO {cfg.max_len}])"
    params = {"query": query, "fields": "accession,gene_primary,length,xref_pdb", "format": "tsv", "size": page_size}
    rows, url = [], UNIPROT
    while url and len(rows) < 6 * cfg.max_proteins:
        r = _get(url, params=params if url == UNIPROT else None)
        reader = csv.DictReader(io.StringIO(r.text), delimiter="\t")
        rows += list(reader)
        url = r.links.get("next", {}).get("url")
    log(f"UniProt: {len(rows)} candidate proteins")
    out = []
    for row in rows:
        pdbs = [x for x in row.get("PDB", "").split(";") if x]
        if not pdbs:
            continue
        out.append({"uniprot": row["Entry"], "gene": row.get("Gene Names (primary)", ""), "length": int(row["Length"]), "pdb_ids_all": pdbs})
    log(f"Looking up release dates for {sum(len(o['pdb_ids_all']) for o in out)} PDB entries")
    info = _rcsb_entry_info(sorted({p for o in out for p in o["pdb_ids_all"]}))
    selected = []
    for o in out:
        eligible = []
        for pid in o["pdb_ids_all"]:
            e = info.get(pid.upper())
            if not e or e["release_date"] <= cfg.pdb_release_after:
                continue
            if e["method"] not in ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"):
                continue
            if e["resolution"] is None or e["resolution"] > cfg.max_resolution:
                continue
            eligible.append({"pdb_id": pid.lower(), **e})
        if eligible:
            eligible.sort(key=lambda x: x["resolution"])
            o["eligible"] = eligible[:5]
            selected.append(o)
    selected = selected[: cfg.max_proteins * 2]  # headroom for proteins that fail the later coverage checks
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(selected, indent=1))
    log(f"{len(selected)} proteins have an eligible post-cutoff experimental structure -> {out_path}")
    return selected


def _rcsb_entry_info(ids: list[str], batch: int = 100) -> dict[str, dict]:
    q = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id
      rcsb_accession_info{initial_release_date}
      rcsb_entry_info{resolution_combined experimental_method}}}"""
    out = {}
    for i in range(0, len(ids), batch):
        r = _session.post(RCSB_GQL, json={"query": q, "variables": {"ids": ids[i : i + batch]}}, timeout=60)
        r.raise_for_status()
        for e in (r.json().get("data") or {}).get("entries") or []:
            if not e:
                continue
            res = (e["rcsb_entry_info"].get("resolution_combined") or [None])[0]
            out[e["rcsb_id"].upper()] = {
                "release_date": e["rcsb_accession_info"]["initial_release_date"][:10],
                "resolution": res,
                "method": e["rcsb_entry_info"].get("experimental_method"),
            }
    return out


def fetch_protein(entry: dict, data_dir: str | Path, cfg: Config, log=print) -> bool:
    acc = entry["uniprot"]
    folder = Path(data_dir) / acc
    if (folder / "meta.json").exists():
        return True
    folder.mkdir(parents=True, exist_ok=True)
    pred = _get(AFDB.format(acc=acc)).json()
    pred = pred[0] if isinstance(pred, list) else pred
    seq = pred.get("sequence") or pred.get("uniprotSequence")
    pdb_url, pae_url = pred.get("pdbUrl"), pred.get("paeDocUrl")
    if not (seq and pdb_url and pae_url):
        log(f"{acc}: AlphaFold DB response is missing sequence, pdbUrl or paeDocUrl (keys: {sorted(pred)})")
        return False
    (folder / "model.pdb").write_bytes(_get(pdb_url).content)
    (folder / "pae.json").write_bytes(_get(pae_url).content)
    mchain = read_chain(folder / "model.pdb")
    _, _, ident, _ = project_to_reference(mchain, seq)
    if ident < 0.95:
        log(f"{acc}: prediction does not match the sequence")
        return False
    for cand in entry["eligible"]:
        path = folder / "expt.cif"
        path.write_bytes(_get(RCSB_FILE.format(pdb=cand["pdb_id"])).content)
        try:
            chain, cid = best_chain(path, seq)
        except ValueError:
            continue
        _, _, ident, cov = project_to_reference(chain, seq)
        if ident >= cfg.min_identity and cov >= cfg.min_coverage:
            meta = {
                "uniprot": acc,
                "gene": entry.get("gene", ""),
                "sequence": seq,
                "pdb_ids_all": entry["pdb_ids_all"],
                "experimental": {**cand, "chain": cid, "identity": ident, "coverage": cov},
            }
            (folder / "meta.json").write_text(json.dumps(meta, indent=1))
            return True
    log(f"{acc}: no eligible experimental entry passed the coverage and identity checks")
    for f in folder.iterdir():
        f.unlink()
    folder.rmdir()
    return False


def fetch_all(manifest: str | Path, data_dir: str | Path, cfg: Config, limit: int | None = None, log=print) -> int:
    entries = json.loads(Path(manifest).read_text())
    ok = 0
    for e in entries:
        if ok >= (limit or cfg.max_proteins):
            break
        try:
            ok += int(fetch_protein(e, data_dir, cfg, log))
        except Exception as exc:  # noqa: BLE001
            log(f"{e['uniprot']}: failed ({exc})")
    log(f"{ok} proteins ready in {data_dir}")
    return ok
