"""Learnable SWT-band modules used by :mod:`layers.SWTAttention_Family`.

Fact A: after SWT, each band is an independent attention head: the ``h`` band
index is shared by Q and K in ``einsum('blhe,bshe->bhls')`` and remains fixed
when values are aggregated. Bands can only exchange information before that
attention or through the fixed ISWT filter bank.

Fact B: a scalar band multiplier on Q/K is a temperature, not an importance
weight. It rescales every score in one band's softmax; only a multiplier on V
directly changes that band's contribution. Keep this distinction when adding
new band controls.
"""

import torch
import torch.nn as nn


class BandAttention(nn.Module):
    """Compute optional, scalar importance weights for SWT coefficient bands."""

    _ACTIVATIONS = {'identity_softmax', 'tanh', 'softmax', 'sigmoid'}
    _POOLING = {'energy', 'abs', 'mean'}

    def __init__(self, num_bands, hidden_dim=None, activation='identity_softmax',
                 pooling='energy', per_variable=False, horizon_conditioned=False,
                 pred_len=None):
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(f'Unsupported band attention activation: {activation}')
        if pooling not in self._POOLING:
            raise ValueError(f'Unsupported band attention pooling: {pooling}')

        self.num_bands = num_bands
        self.activation = activation
        self.pooling = pooling
        self.per_variable = per_variable
        self.horizon_conditioned = horizon_conditioned
        self.pred_len = pred_len
        hidden_dim = hidden_dim or max(4, 2 * num_bands)

        self.mlp = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bands),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.horizon_projection = (
            nn.Linear(1, hidden_dim) if horizon_conditioned else None
        )
        self.last_weights = None

    def _pool(self, coeffs):
        dims = (3,) if self.per_variable else (1, 3)
        if self.pooling == 'energy':
            return coeffs.pow(2).mean(dim=dims).add(1e-8).sqrt()
        if self.pooling == 'abs':
            return coeffs.abs().mean(dim=dims)
        return coeffs.mean(dim=dims)

    def compute_weights(self, coeffs, pred_len=None):
        """Return band weights, without changing ``coeffs``."""
        pooled = self._pool(coeffs)
        hidden = self.mlp[1](self.mlp[0](pooled))
        if self.horizon_projection is not None:
            horizon = self.pred_len if pred_len is None else pred_len
            if horizon is None:
                raise ValueError('pred_len is required when horizon_conditioned=True')
            horizon = torch.as_tensor(horizon, dtype=hidden.dtype, device=hidden.device)
            if torch.any(horizon <= 0):
                raise ValueError('pred_len must be positive when horizon_conditioned=True')
            horizon = horizon.log().reshape(1, 1)
            hidden = hidden + self.horizon_projection(horizon)
        logits = self.mlp[2](hidden)

        if self.activation == 'identity_softmax':
            weights = torch.softmax(logits, dim=-1) * self.num_bands
        elif self.activation == 'tanh':
            weights = 1 + torch.tanh(logits)
        elif self.activation == 'softmax':
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = torch.sigmoid(logits)
        self.last_weights = weights.detach()
        return weights

    @staticmethod
    def apply_weights(coeffs, weights):
        """Broadcast shared ``[B, M]`` or per-variable ``[B, N, M]`` weights."""
        if weights.ndim == 2:
            return coeffs * weights[:, None, :, None]
        if weights.ndim == 3:
            return coeffs * weights[..., None]
        raise ValueError('weights must have shape [B, M] or [B, N, M]')

    def forward(self, coeffs, pred_len=None):
        weights = self.compute_weights(coeffs, pred_len=pred_len)
        return self.apply_weights(coeffs, weights), weights


class BandMixing(nn.Module):
    """Residual learned mixing across SWT bands, ordered coarse-to-fine."""

    def __init__(self, num_bands, hidden_dim=None, dropout=0.0, bidirectional=True):
        super().__init__()
        hidden_dim = hidden_dim or max(4, 2 * num_bands)
        self.bidirectional = bidirectional
        self.mix_c2f = self._branch(num_bands, hidden_dim, dropout)
        self.mix_f2c = self._branch(num_bands, hidden_dim, dropout) if bidirectional else None

    @staticmethod
    def _branch(num_bands, hidden_dim, dropout):
        layers = [nn.Linear(num_bands, hidden_dim), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_bands))
        branch = nn.Sequential(*layers)
        nn.init.zeros_(branch[-1].weight)
        nn.init.zeros_(branch[-1].bias)
        return branch

    def forward(self, coeffs):
        x = coeffs.transpose(-1, -2)
        delta = self.mix_c2f(x)
        if self.bidirectional:
            reversed_x = torch.flip(x, dims=(-1,))
            delta = delta + torch.flip(self.mix_f2c(reversed_x), dims=(-1,))
        return (x + delta).transpose(-1, -2)
