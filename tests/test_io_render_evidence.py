import json

import numpy as np

from strustagent.agent import build_user_message
from strustagent.config import Config
from strustagent.protein import load_protein
from strustagent.render import render_all
from strustagent.structure import align_to_reference, project_to_reference, read_chain
from strustagent.synthetic import make_synthetic, write_ca_pdb


def test_render_outputs_pngs():
    p = make_synthetic(seed=0)
    imgs = render_all(p)
    assert len(imgs) == 2
    assert all(i[:8] == b"\x89PNG\r\n\x1a\n" for i in imgs)


def test_conditions_give_the_right_evidence():
    cfg = Config()
    p = make_synthetic(seed=0)
    image = build_user_message(p, "image", cfg)
    text = build_user_message(p, "text", cfg)
    both = build_user_message(p, "both", cfg)
    assert len(image.images) == 2 and not text.images and len(both.images) == 2
    # the image condition gets no numbers from the PAE or per-residue tables
    assert "Mean pLDDT per residue block" not in image.text
    assert "Between-domain PAE" not in image.text
    # every condition sees the same domain ids
    for m in (image, text, both):
        assert "D1: residues" in m.text and "D1-D2" in m.text
    assert "Between-domain PAE" in text.text and "Mean pLDDT per residue block" in text.text
    # nothing about the experimental structure leaks into any prompt
    assert "lDDT" not in text.text and "experimental" not in image.text.lower()


def test_alignment_handles_partial_constructs_and_mismatches(tmp_path):
    p = make_synthetic(seed=5)
    seq = p.seq
    sub_seq = seq[20:170]
    sub_xyz = p.expt_coords[20:170]
    f = tmp_path / "c.pdb"
    write_ca_pdb(f, sub_seq, sub_xyz, first_resid=500)  # different numbering on purpose
    chain = read_chain(f)
    coords, _, ident, cov = project_to_reference(chain, seq)
    assert ident == 1.0 and abs(cov - 150 / len(seq)) < 1e-9
    assert np.isnan(coords[:20, 0]).all() and np.allclose(coords[20:170], sub_xyz, atol=1e-2)
    mutated = sub_seq[:50] + ("A" if sub_seq[50] != "A" else "C") + sub_seq[51:]
    _, ident2, _ = align_to_reference(mutated, seq)
    assert 1.0 > ident2 > 0.95


def test_load_protein_round_trip(tmp_path):
    cfg = Config()
    p = make_synthetic(seed=6, misplace_deg=50.0, acc="P00001")
    folder = tmp_path / "P00001"
    folder.mkdir()
    write_ca_pdb(folder / "model.pdb", p.seq, p.coords, p.plddt)
    (folder / "pae.json").write_text(json.dumps([{"predicted_aligned_error": p.pae.tolist(), "max_predicted_aligned_error": 31.75}]))
    (folder / "meta.json").write_text(json.dumps({"uniprot": "P00001", "sequence": p.seq, "pdb_ids_all": ["1ABC", "2XYZ"], "experimental": {"chain": "A"}}))
    # the experimental structure is read from expt.cif, so convert a Ca PDB into mmCIF with Biopython
    from Bio.PDB import MMCIFIO, PDBParser

    write_ca_pdb(tmp_path / "e.pdb", p.seq, p.expt_coords)
    io = MMCIFIO()
    io.set_structure(PDBParser(QUIET=True).get_structure("e", tmp_path / "e.pdb"))
    io.save(str(folder / "expt.cif"))
    q = load_protein(folder, cfg)
    assert q.exclude_pdb_ids == {"1abc", "2xyz"}
    assert [d.id for d in q.domains] == ["D1", "D2"]
    assert q.true_pairs[("D1", "D2")]["wrong"] is True
    assert np.nanmean(q.true_lddt) > 0.5
