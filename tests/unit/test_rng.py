import numpy as np

from radarsim.core.rng import SeededRNG


def test_same_seed_and_name_reproducible_across_instances() -> None:
    a = SeededRNG(seed=42).spawn("targets")
    b = SeededRNG(seed=42).spawn("targets")
    np.testing.assert_array_equal(a.random(10), b.random(10))


def test_different_names_are_decorrelated() -> None:
    rng = SeededRNG(seed=42)
    a = rng.spawn("targets").random(10)
    b = rng.spawn("radar").random(10)
    assert not np.array_equal(a, b)


def test_different_seeds_produce_different_streams() -> None:
    a = SeededRNG(seed=1).spawn("targets").random(10)
    b = SeededRNG(seed=2).spawn("targets").random(10)
    assert not np.array_equal(a, b)


def test_repeated_spawn_returns_same_generator() -> None:
    rng = SeededRNG(seed=42)
    stream = rng.spawn("targets")
    assert rng.spawn("targets") is stream
