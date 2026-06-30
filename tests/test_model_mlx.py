"""Validate that we can thread a custom additive attention mask through mlx-lm's
Llama forward — the load-bearing mechanism of the MLX runtime path.

Uses a tiny random-weight model so it runs without the 4.6 GB checkpoint.
"""
import mlx.core as mx
import numpy as np
import pytest

from manifold.masks import build_block_causal_mask

llama = pytest.importorskip("mlx_lm.models.llama")
from manifold.model_mlx import forward_with_mask  # noqa: E402


def tiny_model():
    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=40,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


IDS = mx.array([[1, 2, 3, 4, 5, 6]])


def test_block_size_one_reproduces_native_causal():
    model = tiny_model()
    native = np.array(model(IDS))  # mlx-lm's built-in causal mask
    L = IDS.shape[1]
    ours = np.array(forward_with_mask(model, IDS, mx.array(build_block_causal_mask(L, 1))))
    assert np.allclose(native, ours, atol=2e-3)


def test_full_mask_changes_early_positions():
    model = tiny_model()
    L = IDS.shape[1]
    causal = np.array(model(IDS))[0]
    full = np.array(forward_with_mask(model, IDS, mx.array(build_block_causal_mask(L, L))))[0]
    # position 0: causal sees only itself; full sees the whole sequence -> differs.
    # (Later positions also differ in a multi-layer model, since earlier hidden
    # states change under bidirectional attention and propagate forward.)
    assert not np.allclose(causal[0], full[0], atol=1e-3)


def test_block_causal_differs_from_full():
    model = tiny_model()
    full = np.array(forward_with_mask(model, IDS, mx.array(build_block_causal_mask(6, 6))))[0]
    blk = np.array(forward_with_mask(model, IDS, mx.array(build_block_causal_mask(6, 3))))[0]
    # pos 0 (block 0) cannot see pos 3+ (later block) -> differs from full attention
    assert not np.allclose(full[0], blk[0], atol=1e-3)
