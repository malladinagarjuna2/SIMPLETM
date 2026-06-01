import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", dec_in=866):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.debug_stagewise = False
        self.debug_preview_len = 5

    def _log_tensor(self, name, tensor):
        if not self.debug_stagewise:
            return

        detached = tensor.detach().cpu()
        flat = detached.reshape(-1)
        preview = flat[:max(1, int(self.debug_preview_len))].tolist()
        print(
            f'[EncoderLayer] {name}: shape={tuple(detached.shape)}, '
            f'mean={detached.mean().item():.6f}, std={detached.std(unbiased=False).item():.6f}, '
            f'min={detached.min().item():.6f}, max={detached.max().item():.6f}, '
            f'preview={preview}'
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        self.attention.debug_stagewise = self.debug_stagewise
        self.attention.debug_preview_len = self.debug_preview_len
        self.attention.inner_attention.debug_stagewise = self.debug_stagewise
        self.attention.inner_attention.debug_preview_len = self.debug_preview_len

        self._log_tensor('input_tokens', x)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        self._log_tensor('attention_output', new_x)
        x = x + self.dropout(new_x)
        self._log_tensor('after_residual_add', x)
        y = x = self.norm1(x)
        self._log_tensor('after_norm1', x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        self._log_tensor('after_feedforward_conv1', y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        self._log_tensor('after_feedforward_conv2', y)
        out = self.norm2(x + y)
        self._log_tensor('after_norm2_updated_tokens', out)
        return out, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                if hasattr(attn_layer, 'debug_stagewise'):
                    attn_layer.debug_stagewise = getattr(self, 'debug_stagewise', False)
                    attn_layer.debug_preview_len = getattr(self, 'debug_preview_len', 5)
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
