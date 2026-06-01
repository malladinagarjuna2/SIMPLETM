import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.SWTAttention_Family import GeomAttentionLayer, GeomAttention
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.kernel_size = configs.kernel_size
        self.debug_stagewise = getattr(configs, 'debug_stagewise', False)
        self.debug_preview_len = getattr(configs, 'debug_preview_len', 5)
        self.debug_test_only = getattr(configs, 'debug_test_only', True)
        self._debug_force_enabled = False
        self.latest_debug = None

        enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, 
                                               configs.embed, configs.freq, configs.dropout)
        self.enc_embedding = enc_embedding
        self.enc_embedding.debug_stagewise = self.debug_stagewise and not self.debug_test_only
        self.enc_embedding.debug_preview_len = self.debug_preview_len

        encoder = Encoder(
            [  
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention, alpha=self.alpha
                        ),
                        configs.d_model, 
                        requires_grad=configs.requires_grad, 
                        wv=configs.wv, 
                        m=configs.m, 
                        d_channel=configs.dec_in, 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder = encoder
        self.encoder.debug_stagewise = self.debug_stagewise and not self.debug_test_only
        self.encoder.debug_preview_len = self.debug_preview_len

        projector = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.projector = projector

    def set_debug(self, enabled: bool):
        """
        Turn verbose stagewise logging on/off at runtime.
        We use this to print deep traces only for a few test batches.
        """
        enabled = bool(enabled)
        self._debug_force_enabled = enabled
        # Propagate into submodules that own print statements
        self.enc_embedding.debug_stagewise = enabled
        self.enc_embedding.debug_preview_len = self.debug_preview_len
        self.encoder.debug_stagewise = enabled
        self.encoder.debug_preview_len = self.debug_preview_len

    def _tensor_preview(self, tensor):
        preview_len = max(1, int(self.debug_preview_len))
        detached = tensor.detach().cpu()
        flat = detached.reshape(-1)
        preview = flat[:preview_len].tolist()
        return {
            'shape': tuple(detached.shape),
            'mean': float(detached.mean().item()),
            'std': float(detached.std(unbiased=False).item()),
            'min': float(detached.min().item()),
            'max': float(detached.max().item()),
            'preview': preview,
        }


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        debug_info = {}
        debug_enabled = self.debug_stagewise and (self._debug_force_enabled or not self.debug_test_only)
        if debug_enabled:
            debug_info['input_x_enc'] = self._tensor_preview(x_enc)
            debug_info['input_x_mark_enc'] = None if x_mark_enc is None else self._tensor_preview(x_mark_enc)

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x_enc /= stdev
            x_enc = x_enc / stdev
            if debug_enabled:
                debug_info['norm_means'] = self._tensor_preview(means)
                debug_info['norm_stdev'] = self._tensor_preview(stdev)
                debug_info['normalized_x_enc'] = self._tensor_preview(x_enc)

        _, _, N = x_enc.shape

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector
        # Linear Projection             B L N -> B L' (pseudo temporal tokens) N 
        enc_out = enc_embedding(x_enc, x_mark_enc) 
        if debug_enabled:
            debug_info['embedding_output'] = self._tensor_preview(enc_out)

        # SimpleTM Layer                B L' N -> B L' N 
        enc_out, attns = encoder(enc_out, attn_mask=None)
        if debug_enabled:
            debug_info['encoder_output'] = self._tensor_preview(enc_out)
            debug_info['attention_regularizer'] = [
                float(attn.detach().cpu().item()) if torch.is_tensor(attn) else float(attn)
                for attn in attns
            ]

        # Output Projection             B L' N -> B H (Horizon) N
        dec_out = projector(enc_out).permute(0, 2, 1)[:, :, :N] 
        if debug_enabled:
            debug_info['projector_output_pre_denorm'] = self._tensor_preview(dec_out)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            if debug_enabled:
                debug_info['forecast_output'] = self._tensor_preview(dec_out)
        elif debug_enabled:
            debug_info['forecast_output'] = self._tensor_preview(dec_out)

        if debug_enabled:
            debug_info['notes'] = [
                'Input arrives as [batch, seq_len, num_variables].',
                'DataEmbedding_inverted permutes it to [batch, num_variables, seq_len], so each variable becomes a token.',
                'The test loop passes time marks, but Model.forward currently calls forecast(x_enc, None, None, None), so time features are ignored in this architecture.'
            ]
            self.latest_debug = debug_info
        else:
            self.latest_debug = None

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, None, None, None)
        return dec_out, attns 
