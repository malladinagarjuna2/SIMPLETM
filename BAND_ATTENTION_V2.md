# Band Attention v2

SWT produces coefficient stacks ordered `[a_m, d_m, ..., d_1]`. In geometric
attention, the stack becomes `[B, D, m+1, N]`: the band index is the attention
head. The score einsum, `blhe,bshe->bhls`, holds that index fixed, as does the
value einsum. Thus bands do not exchange information inside attention (Fact A);
they only meet in the fixed ISWT filter bank.

A Q/K band scalar rescales every score in one band's softmax and acts as a
temperature rather than a direct importance weight (Fact B). V scalars directly
change the reconstruction contribution. V2 therefore defaults to V-only.

The fixes are: energy pooling rather than signed means that erase zero-mean
detail coefficients; `identity_softmax`, whose weights average exactly one;
and residual `BandMixing`, which learns coarse-to-fine/fine-to-coarse exchange
before projection. Both modules are exact identities at initialization.

`--use_band_attention` enables energy-pooled V-only weights. Its controls are
`--band_attention_activation`, `--band_attention_pooling`,
`--band_attention_per_variable`, `--band_attention_apply_to`,
`--band_attention_separate_qkv`, and `--band_attention_horizon`. Mixing is
enabled with `--use_band_mixing`, with optional `--band_mixing_hidden_dim`,
`--band_mixing_dropout`, and `--band_mixing_unidirectional`.

Run the acceptance checks with `python3 tests/test_band_modules.py`. For the
multi-seed ablation, set the paths and dimensions at the top of
`scripts/band_attention_v2/ablation.sh`, then run it from the repository root.
