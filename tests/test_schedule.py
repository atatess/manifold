"""Unmasking schedule: how many tokens to commit per step, and which ones.

Mirrors Stable-DiffCoder's `get_num_transfer_tokens` (even split, remainder
front-loaded) and `get_transfer_index` (greedy argmax + softmax-confidence top-k),
but in fp32 (MLX/MPS have no float64) and with deterministic tie-breaking.
"""
import numpy as np

from manifold.schedule import (
    confidence_of_argmax,
    num_transfer_per_step,
    select_transfer,
)


# --- num_transfer_per_step ------------------------------------------------

def test_even_split():
    assert num_transfer_per_step(4, 4) == [1, 1, 1, 1]


def test_remainder_front_loaded():
    assert num_transfer_per_step(10, 4) == [3, 3, 2, 2]


def test_single_step_commits_all():
    assert num_transfer_per_step(5, 1) == [5]


def test_more_steps_than_tokens():
    assert num_transfer_per_step(3, 5) == [1, 1, 1, 0, 0]


def test_zero_masked():
    assert num_transfer_per_step(0, 3) == [0, 0, 0]


def test_counts_sum_to_mask_count():
    for mask_count, steps in [(7, 3), (16, 5), (1, 1), (100, 7)]:
        assert sum(num_transfer_per_step(mask_count, steps)) == mask_count


# --- confidence_of_argmax -------------------------------------------------

def test_argmax_tokens_and_confidence_range():
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    x0, conf = confidence_of_argmax(logits)
    assert x0.tolist() == [0, 2]
    assert conf.dtype == np.float32
    assert np.all((conf > 0.0) & (conf <= 1.0))
    # bigger logit gap -> higher confidence
    assert conf[0] > conf[1]


# --- select_transfer ------------------------------------------------------

def test_topk_returns_highest_confidence_first():
    conf = np.array([0.1, 0.9, 0.5, 0.8], dtype=np.float32)
    masked = np.array([True, True, True, True])
    assert select_transfer(conf, masked, 2) == [1, 3]


def test_only_masked_positions_eligible():
    conf = np.array([0.9, 0.95, 0.5], dtype=np.float32)
    masked = np.array([True, False, True])
    assert select_transfer(conf, masked, 1) == [0]


def test_ties_break_to_lower_index():
    conf = np.array([0.5, 0.5, 0.1], dtype=np.float32)
    masked = np.array([True, True, True])
    assert select_transfer(conf, masked, 1) == [0]


def test_k_exceeds_available():
    conf = np.array([0.9, 0.8], dtype=np.float32)
    masked = np.array([True, True])
    assert select_transfer(conf, masked, 5) == [0, 1]


def test_k_zero():
    conf = np.array([0.9, 0.8], dtype=np.float32)
    masked = np.array([True, True])
    assert select_transfer(conf, masked, 0) == []
