"""All tunable choices in one place. Values marked CHOICE are judgment calls you should set and justify in the report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- data selection ----
    # AlphaFold 2 trained on PDB structures released up to 30 April 2018 (Jumper et al. 2021, Methods).
    pdb_release_after: str = "2018-04-30"
    max_resolution: float = 3.0
    min_len: int = 150
    max_len: int = 1000
    min_coverage: float = 0.5  # fraction of the UniProt sequence the experimental chain must cover
    min_identity: float = 0.9  # identity between the experimental chain and the UniProt sequence
    max_proteins: int = 300

    # ---- domain definition (CHOICE) ----
    domain_source: str = "pae"  # "pae" (cluster the PAE matrix). Annotation-based domains are not implemented yet.
    domain_pae_cut: float = 5.0  # average-linkage cut on symmetrized PAE, in angstroms
    domain_min_len: int = 40
    domain_plddt_min: float = 70.0  # residues below this are left out of domain clustering
    max_domains_in_prompt: int = 6

    # ---- ground truth (CHOICE) ----
    lddt_radius: float = 15.0
    low_lddt: float = 0.6  # residues with true lDDT-Ca below this count as "untrustworthy" for AUROC
    placement_wrong_angstrom: float = 5.0  # domain pair is "misplaced" if the mean B-after-superposing-on-A and A-after-superposing-on-B RMSD exceeds this
    placement_internal_max_angstrom: float = 4.0  # both domains must be internally accurate, so the error is about placement and not folding
    placement_min_residues: int = 20

    # ---- baselines ----
    plddt_threshold: float = 70.0  # AlphaFold DB labels residues below 70 as low confidence
    pae_pair_threshold: float = 8.0  # CHOICE: mean inter-domain PAE below this counts as "reliable placement"

    # ---- agent ----
    model: str = ""  # set via --model; deliberately no default so a wrong model name never runs silently
    max_steps: int = 6
    text_window: int = 10  # residues per row in the binned pLDDT table
    unscored_residue_default: float = 0.5  # trust assigned to residues the agent did not mention

    # ---- tools ----
    tools: list = field(default_factory=lambda: ["foldseek_support", "esm2_plausibility"])  # tools the agent may use; drop foldseek_support if it is not installed
    foldseek_bin: str = "foldseek"
    foldseek_db: str = ""  # path to a Foldseek PDB database, e.g. made with `foldseek databases PDB pdb tmp`
    foldseek_top_k: int = 5
    esm_model: str = "facebook/esm2_t12_35M_UR50D"

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            for key, value in data.items():
                if not hasattr(cfg, key):
                    raise KeyError(f"Unknown config key: {key}")
                setattr(cfg, key, value)
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
