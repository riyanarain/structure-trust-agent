import numpy as np

from strustagent.config import Config
from strustagent.domains import domains_from_pae
from strustagent.groundtruth import kabsch, lddt_ca, pair_placement
from strustagent.synthetic import _rotation, make_synthetic


def test_lddt_identical_is_one_and_perturbation_lowers_it():
    p = make_synthetic(seed=1, misplace_deg=0.0)
    ref = p.expt_coords
    same = lddt_ca(ref, ref)
    assert np.allclose(same[~np.isnan(same)], 1.0)
    rng = np.random.default_rng(0)
    noisy = ref + rng.normal(scale=2.0, size=ref.shape)
    assert np.nanmean(lddt_ca(noisy, ref)) < 0.8


def test_lddt_is_blind_to_domain_rotation_but_placement_is_not():
    """The reason domain placement is scored separately."""
    p = make_synthetic(seed=2, misplace_deg=60.0)
    cfg = Config()
    # within each domain lDDT stays high even though the pair is misplaced
    dom_a = p.domain("D1").residues
    assert np.nanmean(p.true_lddt[dom_a]) > 0.9
    assert p.true_pairs[("D1", "D2")]["wrong"] is True
    assert p.true_pairs[("D1", "D2")]["err"] > cfg.placement_wrong_angstrom


def test_well_placed_pair_is_not_flagged_and_bad_domain_is_unscorable():
    p = make_synthetic(seed=3, misplace_deg=1.0)
    # a confident, well-placed protein is a single PAE domain, so there is no pair to score
    assert len(p.domains) == 1 and p.true_pairs == {}
    a = np.arange(0, 90)
    b = np.arange(104, 194)
    ok = pair_placement(p.coords, p.expt_coords, a, b)
    assert ok is not None and ok["wrong"] is False
    scrambled = p.coords.copy()
    rng = np.random.default_rng(1)
    scrambled[a] += rng.normal(scale=6.0, size=(len(a), 3))
    assert pair_placement(scrambled, p.expt_coords, a, b) is None


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3))
    r_true = _rotation(np.array([1.0, 2.0, 3.0]), 70)
    q = pts @ r_true.T + np.array([5.0, -2.0, 1.0])
    r, t = kabsch(pts, q)
    assert np.allclose(pts @ r.T + t, q, atol=1e-6)


def test_domains_from_pae_find_two_domains_and_skip_linker():
    p = make_synthetic(seed=4)
    doms = domains_from_pae(p.pae, p.plddt, 5.0, 40, 70.0)
    assert [d.id for d in doms] == ["D1", "D2"]
    assert doms[0].ranges[0][0] == 1
    assert doms[1].ranges[-1][1] == p.length
    covered = set(doms[0].residues) | set(doms[1].residues)
    assert not covered & set(range(90, 104))  # linker residues have low pLDDT
