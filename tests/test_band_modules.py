"""Standalone acceptance checks for band-attention-v2.

Run with: ``python3 tests/test_band_modules.py``.
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# The repository's SWT implementation normally imports PyWavelets. Keep these
# acceptance checks runnable in lightweight CI environments that omit it.
try:
    import pywt  # noqa: F401
except ModuleNotFoundError:
    fallback = types.ModuleType('pywt')

    class Wavelet:
        def __init__(self, _name):
            value = 2 ** -0.5
            self.dec_lo = [value, value]
            self.dec_hi = [-value, value]
            self.rec_lo = [value, value]
            self.rec_hi = [value, -value]

    fallback.Wavelet = Wavelet
    sys.modules['pywt'] = fallback

from layers.band_modules import BandAttention, BandMixing
from layers.SWTAttention_Family import GeomAttention, GeomAttentionLayer


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def randomize_final(module):
    with torch.no_grad():
        module.mlp[-1].weight.normal_()
        module.mlp[-1].bias.normal_()


def make_layer(**kwargs):
    return GeomAttentionLayer(
        GeomAttention(attention_dropout=0.0, alpha=1.0), d_model=8,
        d_channel=3, m=3, geomattn_dropout=0.0, **kwargs,
    )


def test_band_attention():
    coeffs = torch.randn(2, 3, 4, 8)
    attention = BandAttention(4)
    weighted, weights = attention(coeffs)
    check(weighted.shape == coeffs.shape and weights.shape == (2, 4), 'shared shapes')
    check(torch.equal(weights, torch.ones_like(weights)), 'identity weights at init')
    check(torch.equal(weighted, coeffs), 'identity output at init')

    per_variable = BandAttention(4, per_variable=True)
    _, weights = per_variable(coeffs)
    check(weights.shape == (2, 3, 4), 'per-variable weight shape')

    zero_mean = coeffs - coeffs.mean(dim=(1, 3), keepdim=True)
    mean_pooled = BandAttention(4, pooling='mean')._pool(zero_mean)
    energy_pooled = BandAttention(4, pooling='energy')._pool(zero_mean)
    check(mean_pooled.abs().max().item() < 1e-5, 'signed mean should collapse')
    check(energy_pooled.min().item() > 0.5, 'energy pooling should retain details')

    softmax = BandAttention(4, activation='softmax')
    identity_softmax = BandAttention(4, activation='identity_softmax')
    randomize_final(softmax)
    randomize_final(identity_softmax)
    softmax_weights = softmax.compute_weights(coeffs)
    identity_weights = identity_softmax.compute_weights(coeffs)
    check(torch.allclose(softmax_weights.sum(-1), torch.ones(2)), 'softmax sums to one')
    check(torch.allclose(softmax_weights.mean(-1), torch.full((2,), .25)), 'softmax mean')
    check(torch.allclose(identity_weights.mean(-1), torch.ones(2)), 'identity_softmax mean')
    try:
        BandAttention(4, activation='invalid')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid activation must fail')


def test_band_mixing():
    coeffs = torch.randn(2, 3, 4, 8)
    mixer = BandMixing(4)
    check(torch.equal(mixer(coeffs), coeffs), 'mixing identity at init')
    check(sum(p.numel() for p in mixer.parameters()) < 200, 'mixing parameter budget')
    with torch.no_grad():
        mixer.mix_c2f[-1].weight.normal_()
    changed = mixer(coeffs)
    perturbed = coeffs.clone()
    perturbed[:, :, 0] += 10
    perturbed_changed = mixer(perturbed)
    check((perturbed_changed[:, :, 1:] - changed[:, :, 1:]).abs().max().item() > 1e-3,
          'bands must exchange information')


def test_geom_attention_layer():
    torch.manual_seed(7)
    inputs = torch.randn(2, 3, 8)
    baseline = make_layer()
    disabled = make_layer(use_band_attention=False, use_band_mixing=False)
    disabled.load_state_dict(baseline.state_dict())
    baseline.eval()
    disabled.eval()
    reference, _ = baseline(inputs, inputs, inputs)
    actual, _ = disabled(inputs, inputs, inputs)
    check(torch.equal(actual, reference), 'disabled feature path must be exactly unchanged')

    enabled = make_layer(use_band_attention=True, use_band_mixing=True)
    enabled.load_state_dict(baseline.state_dict(), strict=False)
    enabled.eval()
    actual, _ = enabled(inputs, inputs, inputs)
    check(torch.allclose(actual, reference), 'enabled modules must be identity at init')

    for apply_to in ('v', 'qk', 'qkv'):
        layer = make_layer(use_band_attention=True, band_attention_apply_to=apply_to)
        layer(inputs, inputs, inputs)
    make_layer(use_band_attention=True, band_attention_per_variable=True)(inputs, inputs, inputs)
    make_layer(use_band_attention=True, band_attention_horizon=True, pred_len=24)(inputs, inputs, inputs)
    try:
        make_layer(band_attention_apply_to='invalid')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid apply_to must fail')

    gradients = make_layer(use_band_attention=True)
    output, _ = gradients(inputs, inputs, inputs)
    output.square().mean().backward()
    band_grads = [p.grad for p in gradients.band_attention.parameters()]
    check(any(g is not None and g.abs().sum() > 0 for g in band_grads), 'band gradients')


if __name__ == '__main__':
    test_band_attention()
    test_band_mixing()
    test_geom_attention_layer()
    print('12 band-attention-v2 acceptance checks passed')
