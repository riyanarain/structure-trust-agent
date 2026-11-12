# structure-trust-agent

Code for the CS 286 / BIODS 276 project **"Does Vision Help? Image Versus Text Evidence in an Agent That Judges Predicted Protein Structures."**

A vision-language model is given an AlphaFold-predicted structure and decides which regions, and which domain placements, can be trusted. The central experiment gives the same underlying evidence as **images** (structure render, PAE plot), as **text** (binned pLDDT, PAE tables), or **both**, and compares the judgments to experimental PDB structures that were released after AlphaFold 2's training cutoff.

## Install

```
pip install -e '.[dev]'          # core + tests
pip install -e '.[gemini]'       # Gemini client (google-genai)
pip install -e '.[openai]'       # OpenAI-compatible client (e.g. Qwen2.5-VL served by vLLM)
pip install -e '.[esm]'          # ESM-2 plausibility tool (torch, transformers)
```

Foldseek must be on your PATH, with a PDB database built for it, for the structural-support tool (`foldseek_bin` / `foldseek_db` in the config).

## Check the offline pieces

```
pytest
strustagent demo --out demo_out    # synthetic two-domain protein: images + text evidence + ground truth
```

## Run order

```
strustagent select                                   # candidate proteins -> data/manifest.json
strustagent fetch --limit 20                         # AlphaFold DB model + PAE and the experimental entry
strustagent inspect                                  # counts, scoreable domain pairs, misplaced pairs
strustagent run --model <model-name> --limit 20 --require-pairs     # pilot
strustagent evaluate                                 # table + paired comparisons -> runs/pilot/report.json
```

Methods: `agent:{image,text,both}`, `fixed:{...}` (all tools always run), `notools:{...}`, and the baselines `plddt_threshold`, `pae_threshold`. Model replies are cached in `.llm_cache`, so reruns are free and reproducible.

## Design notes

- **Ground truth.** Per-residue lDDT-Ca against the experimental structure (superposition-free). lDDT cannot see a rigidly misplaced domain, so domain-pair placement is scored separately: superpose on one domain, measure the RMSD of the other (both directions averaged), and only score a pair when both domains are internally accurate.
- **Domains** come from clustering the PAE matrix (prediction only) and are given to the model in every condition.
- **No tool returns pLDDT or PAE numbers**, otherwise the image-versus-text comparison would be meaningless.
- **Leakage controls.** Only experimental entries released after 2018-04-30 are used; Foldseek hits to the protein's own PDB entries are excluded (and, with `--release-dates`, hits released after the cutoff). Homologs deposited later can still appear as hits.
- Thresholds in `config.py` marked CHOICE are judgment calls and should be justified or varied in the report.

## Status

Tested offline: ground truth, domain detection, rendering, evidence construction, agent loop (with a scripted model), baselines, evaluation statistics, caching, and structure alignment/IO.
**Not tested against live services:** `data.py` (UniProt, RCSB, AlphaFold DB), the Gemini and OpenAI-compatible clients, Foldseek, and ESM-2. Try each on one protein first; API field names may need adjusting.
