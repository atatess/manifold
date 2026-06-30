"""Block-causal attention mask: bidirectional within a block, causal across blocks.

Mirrors Stable-DiffCoder's `make_block_causal_mask` = kron(tril(num_blocks), ones(block)).
"""
import numpy as np

from manifold.masks import NEG, build_block_causal_mask


def test_shape_and_dtype():
    m = build_block_causal_mask(8, block_size=4)
    assert m.shape == (8, 8)
    assert m.dtype == np.float32


def test_allowed_is_zero_blocked_is_neg_no_inf():
    m = build_block_causal_mask(8, block_size=4)
    assert np.isfinite(m).all()  # NEG must be finite (no -inf -> NaN in bf16 softmax)
    vals = set(np.unique(m).tolist())
    assert vals <= {0.0, float(NEG)}


def test_diagonal_always_allowed():
    m = build_block_causal_mask(7, block_size=4)
    assert (np.diag(m) == 0.0).all()


def test_bidirectional_within_block():
    # block_size=4, positions 0..3 are one block -> attend both directions
    m = build_block_causal_mask(8, block_size=4)
    assert m[0, 3] == 0.0
    assert m[3, 0] == 0.0


def test_causal_across_blocks():
    # block0 = 0..3, block1 = 4..7
    m = build_block_causal_mask(8, block_size=4)
    assert m[0, 4] == NEG   # earlier block cannot see a later block
    assert m[4, 0] == 0.0   # later block sees earlier block


def test_partial_final_block():
    # L=6, block_size=4 -> block0=0..3, block1=4..5
    m = build_block_causal_mask(6, block_size=4)
    assert m[4, 5] == 0.0   # bidirectional inside partial block
    assert m[5, 4] == 0.0
    assert m[3, 4] == NEG   # block0 cannot see block1
    assert m[4, 3] == 0.0   # block1 sees block0


def test_block_size_one_is_plain_causal():
    m = build_block_causal_mask(4, block_size=1)
    assert m[2, 1] == 0.0   # can see strictly earlier
    assert m[1, 2] == NEG   # cannot see later
    assert m[2, 2] == 0.0


def test_custom_neg_value():
    m = build_block_causal_mask(8, block_size=4, neg=-1e4)
    assert m[0, 4] == -1e4
