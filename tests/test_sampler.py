"""The controllable block-diffusion denoising loop.

These tests use a *fake* forward function (dependency injection) so the loop's
control surface — pause, inspect the partial sequence, and intervene between
steps — is validated without the real model. This is the Phase 1 "done when".
"""
import numpy as np

from manifold.masks import build_block_causal_mask
from manifold.sampler import Intervention, denoise

VOCAB = 1000
MASK = 5


def fake_forward(target, gen_start, confidences=None, record_masks=None):
    """Forward that points each gen position at target[i] with a chosen magnitude.

    Higher magnitude -> higher softmax confidence. Ignores the attention mask
    except to record its shape when `record_masks` is given.
    """
    def forward(seq, mask):
        if record_masks is not None:
            record_masks.append(np.asarray(mask).shape)
        L = len(seq)
        logits = np.full((L, VOCAB), -10.0, dtype=np.float32)
        for i, tok in enumerate(target):
            mag = 10.0 if confidences is None else float(confidences[i])
            logits[gen_start + i, tok] = mag
        return logits
    return forward


def committed_gen_positions(seq, gen_start, gen_len):
    return {i for i in range(gen_len) if seq[gen_start + i] != MASK}


# --- basic fill -----------------------------------------------------------

def test_fills_masked_region_with_predictions():
    prompt = [1, 2]
    target = [100, 101, 102, 103]
    out = denoise(
        fake_forward(target, gen_start=len(prompt)),
        prompt, gen_length=4, block_length=4, steps_per_block=4, mask_id=MASK,
    )
    assert out[len(prompt):].tolist() == target
    assert out[:len(prompt)].tolist() == prompt  # prompt untouched


def test_no_mask_tokens_remain_after_full_denoise():
    prompt = [7]
    target = [10, 11, 12, 13, 14, 15, 16, 17]
    out = denoise(
        fake_forward(target, gen_start=1),
        prompt, gen_length=8, block_length=4, steps_per_block=4, mask_id=MASK,
    )
    assert MASK not in out.tolist()


# --- forward is called with the block-causal mask -------------------------

def test_forward_receives_block_causal_mask_of_full_length():
    prompt = [1, 2]
    target = [100, 101, 102, 103]
    seen = []
    denoise(
        fake_forward(target, gen_start=2, record_masks=seen),
        prompt, gen_length=4, block_length=4, steps_per_block=4, mask_id=MASK,
    )
    L = len(prompt) + 4
    assert seen and all(shape == (L, L) for shape in seen)


# --- pause + inspect partial state each step ------------------------------

def test_on_step_observes_growing_partial():
    prompt = [1]
    target = [100, 101, 102, 103]
    # confidences pick commit order: idx1 > idx3 > idx2 > idx0
    snapshots = []

    def on_step(ctx):
        snapshots.append(committed_gen_positions(ctx.seq, 1, 4))
        return None

    denoise(
        fake_forward(target, gen_start=1, confidences=[1, 4, 2, 3]),
        prompt, gen_length=4, block_length=4, steps_per_block=4, mask_id=MASK,
        on_step=on_step,
    )
    # pre-commit snapshots reflect prior commits, in confidence order
    assert snapshots == [set(), {1}, {1, 3}, {1, 3, 2}]


# --- intervention: forbid unmasking a position this step ------------------

def test_forbid_unmask_defers_a_position():
    prompt = [1]
    target = [100, 101]
    order = []

    def on_step(ctx):
        order.append(committed_gen_positions(ctx.seq, 1, 2))
        if ctx.step == 0:
            return Intervention(forbid_unmask=[1])  # absolute pos of gen idx 0
        return None

    out = denoise(
        fake_forward(target, gen_start=1, confidences=[5, 1]),
        prompt, gen_length=2, block_length=2, steps_per_block=2, mask_id=MASK,
        on_step=on_step,
    )
    # idx0 (higher conf) would normally commit first; forbidden, so idx1 goes first
    assert order == [set(), {1}]
    assert out[1:].tolist() == target


# --- intervention: bias logits to change the committed token --------------

def test_logit_bias_changes_committed_token():
    prompt = [1]
    target = [100, 101]

    def on_step(ctx):
        # persistent steer: whenever block-local idx 0 is predicted, force token 999
        bias = np.zeros(VOCAB, dtype=np.float32)
        bias[999] = 20.0  # outweighs target's magnitude -> 999 wins at idx0
        return Intervention(logit_bias={0: bias})

    out = denoise(
        fake_forward(target, gen_start=1, confidences=[5, 5]),
        prompt, gen_length=2, block_length=2, steps_per_block=2, mask_id=MASK,
        on_step=on_step,
    )
    assert out[1] == 999
    assert out[2] == 101


# --- intervention: re-mask an already-committed position ------------------

def test_remask_uncommits_a_position():
    prompt = [1]
    target = [100, 101]

    def on_step(ctx):
        # last step commits nothing (counts = [1,1,0]); re-mask gen idx 0
        if ctx.step == 2:
            return Intervention(remask=[1])  # absolute pos of gen idx 0
        return None

    out = denoise(
        fake_forward(target, gen_start=1, confidences=[5, 5]),
        prompt, gen_length=2, block_length=2, steps_per_block=3, mask_id=MASK,
        on_step=on_step,
    )
    assert out[1] == MASK   # un-committed by the hook
    assert out[2] == 101
