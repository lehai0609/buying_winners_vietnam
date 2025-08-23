"""
Tests for deterministic seeding utilities.
"""
import random
import pytest

from bwv.utils import deterministic_sample, set_global_seed


def test_deterministic_sample_same_seed_same_output():
    s1 = deterministic_sample(12345)
    s2 = deterministic_sample(12345)
    assert s1 == s2


def test_different_seeds_different_output():
    a = deterministic_sample(12345)
    b = deterministic_sample(54321)
    assert a != b


def test_set_global_seed_affects_random():
    set_global_seed(42)
    # random.random() first value for seed=42 is stable across stdlib implementations
    assert random.random() == 0.6394267984578837


def test_negative_seed_raises_value_error():
    with pytest.raises(ValueError):
        deterministic_sample(-1)
    with pytest.raises(ValueError):
        set_global_seed(-1)
