import pytest

from adaptive_diffusionguard.governance.cosref import StaticCOSREFPolicy


def test_static_cosref_formula_for_intra_and_inter() -> None:
    policy = StaticCOSREFPolicy(omega_intra=0.8, omega_inter=0.2, seed=7)
    assert policy.keep_probability("a", "a", 0.5) == pytest.approx(0.9)
    assert policy.keep_probability("a", "b", 0.5) == pytest.approx(0.6)
    assert policy.keep_probability("a", "b", 0.0) == 1.0


def test_static_cosref_rejects_out_of_range_data() -> None:
    with pytest.raises(ValueError):
        StaticCOSREFPolicy(1.1, 0.5, seed=1)
    policy = StaticCOSREFPolicy(1.0, 1.0, seed=1)
    with pytest.raises(ValueError):
        policy.keep_probability("a", "b", -0.1)
