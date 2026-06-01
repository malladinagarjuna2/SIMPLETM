import torch
import torch.nn as nn

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.debug_stagewise = False
        self.debug_preview_len = 5

    def _log_tensor(self, name, tensor):
        if not self.debug_stagewise:
            return

        detached = tensor.detach().cpu()
        flat = detached.reshape(-1)
        preview = flat[:max(1, int(self.debug_preview_len))].tolist()
        print(
            f'[Tokenization] {name}: shape={tuple(detached.shape)}, '
            f'mean={detached.mean().item():.6f}, std={detached.std(unbiased=False).item():.6f}, '
            f'min={detached.min().item():.6f}, max={detached.max().item():.6f}, '
            f'preview={preview}'
        )

    def forward(self, x, x_mark):
        self._log_tensor('input_before_permute', x)
        x = x.permute(0, 2, 1)
        self._log_tensor('after_permute_variable_tokens', x)
        if x_mark is None:
            x = self.value_embedding(x)
            self._log_tensor('after_linear_projection', x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
            self._log_tensor('after_concat_and_linear_projection', x)
        x = self.dropout(x)
        self._log_tensor('after_dropout', x)
        return x
