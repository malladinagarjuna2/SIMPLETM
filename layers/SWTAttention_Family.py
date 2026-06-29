import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import pywt


class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients
        self.debug_stagewise = False
        self.debug_preview_len = 5
        self.debug_prefix = 'Wavelet'

        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)

            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)

    def _log_tensor(self, name, tensor):
        if not self.debug_stagewise:
            return

        detached = tensor.detach().cpu()
        flat = detached.reshape(-1)
        preview = flat[:max(1, int(self.debug_preview_len))].tolist()
        print(
            f'[{self.debug_prefix}] {name}: shape={tuple(detached.shape)}, '
            f'mean={detached.mean().item():.6f}, std={detached.std(unbiased=False).item():.6f}, '
            f'min={detached.min().item():.6f}, max={detached.max().item():.6f}, '
            f'preview={preview}'
        )

    def forward(self, x):
        self._log_tensor('input', x)
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        self._log_tensor('output', coeffs)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for level in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, 'circular')
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            self._log_tensor(f'level_{level + 1}_detail_coeff', detail_coeff)
            self._log_tensor(f'level_{level + 1}_approx_coeff', approx_coeffs)
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:, :, 0, :]
        detail_coeffs = coeffs[:, :, 1:, :]

        for i in range(m):
            detail_coeff = detail_coeffs[:, :, i, :]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, 'circular')
            detail_coeff_pad = F.pad(detail_coeff, pad, 'circular')

            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            self._log_tensor(f'reconstruction_level_{i + 1}', approx_coeff)
            dilation //= 2

        return approx_coeff


class GeomAttentionLayer(nn.Module):
    def __init__(self, attention, d_model,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, geomattn_dropout=0.5):
        super(GeomAttentionLayer, self).__init__()

        self.d_channel = d_channel
        self.inner_attention = attention
        self.debug_stagewise = False
        self.debug_preview_len = 5

        self.swt = WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size)
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(d_channel=self.d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size),
        )

    def _log_tensor(self, name, tensor):
        if not self.debug_stagewise:
            return

        detached = tensor.detach().cpu()
        flat = detached.reshape(-1)
        preview = flat[:max(1, int(self.debug_preview_len))].tolist()
        print(
            f'[GeomAttentionLayer] {name}: shape={tuple(detached.shape)}, '
            f'mean={detached.mean().item():.6f}, std={detached.std(unbiased=False).item():.6f}, '
            f'min={detached.min().item():.6f}, max={detached.max().item():.6f}, '
            f'preview={preview}'
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        self.swt.debug_stagewise = self.debug_stagewise
        self.swt.debug_preview_len = self.debug_preview_len
        self.swt.debug_prefix = 'StationaryWaveletTransform'
        self.out_projection[1].debug_stagewise = self.debug_stagewise
        self.out_projection[1].debug_preview_len = self.debug_preview_len
        self.out_projection[1].debug_prefix = 'InverseWaveletUpdate'

        self._log_tensor('queries_input', queries)
        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)
        self._log_tensor('queries_after_swt', queries)
        self._log_tensor('keys_after_swt', keys)
        self._log_tensor('values_after_swt', values)

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        self._log_tensor('queries_after_linear_projection', queries)
        self._log_tensor('keys_after_linear_projection', keys)
        self._log_tensor('values_after_linear_projection', values)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
        )
        self._log_tensor('attention_weighted_values', out)

        out = self.out_projection(out)
        self._log_tensor('updated_multivariate_coefficients', out)

        return out, attn


class GeomAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False,
                 alpha=1., score_mode='dot_wedge', cross_weight=0.5, cross_dim=None):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.alpha = alpha
        self.score_mode = score_mode
        self.cross_weight = cross_weight
        self.cross_projection = nn.Linear(cross_dim, 3, bias=False) if cross_dim is not None else None
        self.cross_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.debug_stagewise = False
        self.debug_preview_len = 5

    def _log_tensor(self, name, tensor):
        if not self.debug_stagewise:
            return

        detached = tensor.detach().cpu()
        flat = detached.reshape(-1)
        preview = flat[:max(1, int(self.debug_preview_len))].tolist()
        print(
            f'[GeomAttention] {name}: shape={tuple(detached.shape)}, '
            f'mean={detached.mean().item():.6f}, std={detached.std(unbiased=False).item():.6f}, '
            f'min={detached.min().item():.6f}, max={detached.max().item():.6f}, '
            f'preview={preview}'
        )

    def _normalize_scores(self, scores):
        mean = scores.mean(dim=-1, keepdim=True)
        stdev = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (scores - mean) / stdev

    def _projected_cross_scores(self, queries, keys):
        if self.cross_projection is None:
            raise ValueError('cross_dim must be set when score_mode=cross3d')
        queries_3d = self.cross_projection(queries)
        keys_3d = self.cross_projection(keys)
        cross = torch.cross(queries_3d.unsqueeze(2), keys_3d.unsqueeze(1), dim=-1)
        return torch.linalg.norm(cross, dim=-1).permute(0, 3, 1, 2)

    def _mix_dot_and_cross(self, dot_scores, cross_scores):
        dot_scores = self._normalize_scores(dot_scores)
        cross_scores = self._normalize_scores(cross_scores)
        gate = torch.sigmoid(self.cross_gate_logit)
        return (1 - gate) * dot_scores + gate * cross_scores, gate

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        dot_product = torch.einsum('blhe,bshe->bhls', queries, keys)
        self._log_tensor('dot_product_scores', dot_product)

        queries_norm2 = torch.sum(queries ** 2, dim=-1)
        keys_norm2 = torch.sum(keys ** 2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2
        wedge_norm2 = F.relu(wedge_norm2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)
        self._log_tensor('wedge_product_norm', wedge_norm)

        if self.score_mode == 'dot':
            scores = dot_product
        elif self.score_mode == 'wedge':
            scores = wedge_norm
        elif self.score_mode == 'normalized_dot_wedge':
            scores = (1 - self.alpha) * self._normalize_scores(dot_product) + self.alpha * self._normalize_scores(wedge_norm)
        elif self.score_mode == 'cross3d':
            cross_scores = self._projected_cross_scores(queries, keys)
            self._log_tensor('projected_cross3d_scores', cross_scores)
            scores = (1 - self.cross_weight) * self._normalize_scores(dot_product) + self.cross_weight * self._normalize_scores(cross_scores)
        elif self.score_mode == 'cross3d_gate':
            cross_scores = self._projected_cross_scores(queries, keys)
            self._log_tensor('projected_cross3d_scores', cross_scores)
            scores, gate = self._mix_dot_and_cross(dot_product, cross_scores)
            self._log_tensor('cross3d_gate_value', gate)
        else:
            scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale
        self._log_tensor('combined_attention_scores', scores)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(L, S)).to(scores.device)
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        A = self.dropout(torch.softmax(scores, dim=-1))
        self._log_tensor('softmax_attention_weights', A)

        V = torch.einsum('bhls,bshd->blhd', A, values)
        self._log_tensor('updated_value_tensor', V)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), scores.abs().mean()
