# SimpleTM Local Changes

This file summarizes the local changes made to support model inspection, scalability experiments, and Colab compatibility.

## 1. Stagewise Debug Logging

Added optional stagewise logging so the model can print tensor shapes and previews at important internal stages.

Enable it with:

```bash
DEBUG_STAGEWISE=1 BATCH_SIZE=1 DEBUG_MAX_BATCHES=1 DEBUG_PREVIEW_LEN=5 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

Main files changed:

- `run.py`
- `model/SimpleTM.py`
- `experiments/exp_long_term_forecasting.py`
- `layers/Embed.py`
- `layers/SWTAttention_Family.py`
- `layers/Transformer_Encoder.py`

The debug trace includes:

- dataset window shapes
- tokenization and linear projection
- stationary wavelet transform coefficients
- dot-product attention scores
- wedge-product scores
- sparse top-k scores, when enabled
- softmax attention weights
- inverse wavelet reconstruction
- updated multivariate coefficients
- encoder residual and feed-forward updates
- final forecast output

## 2. Sparse Attention Scalability Experiment

Added an optional sparse top-k mode inside `GeomAttention`.

New command-line flags:

```bash
--attention_mode full
--attention_mode sparse
--sparse_top_k 32
```

Default behavior remains full attention:

```bash
ATTENTION_MODE=full SPARSE_TOP_K=0 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

Sparse experiment example:

```bash
ATTENTION_MODE=sparse SPARSE_TOP_K=32 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

Implementation detail:

- Full attention computes all scores as before.
- Sparse attention computes scores, keeps only the top-k values per query, masks the rest with `-inf`, then applies softmax.
- This is an experimental scalability step. It reduces effective attention connectivity, but it does not yet reduce full score-matrix computation.

## 3. ECL Script Runtime Options

Updated `scripts/multivariate_forecasting/ECL/SimpleTM.sh` to accept environment variables.

Available options:

```bash
ATTENTION_MODE=full|sparse
SPARSE_TOP_K=32
BATCH_SIZE=1
DEBUG_STAGEWISE=1
DEBUG_MAX_BATCHES=1
DEBUG_PREVIEW_LEN=5
```

Examples:

```bash
ATTENTION_MODE=sparse SPARSE_TOP_K=32 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

```bash
ATTENTION_MODE=sparse SPARSE_TOP_K=32 BATCH_SIZE=1 DEBUG_STAGEWISE=1 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

## 4. NumPy 2 Compatibility Fix

Updated `utils/tools.py`:

```python
np.Inf
```

to:

```python
np.inf
```

Reason:

- `np.Inf` was removed in NumPy 2.0.
- Colab currently uses a newer NumPy version, so training failed during `EarlyStopping` initialization.

## 5. Colab Notes

If the shell script fails with errors such as:

```text
--data_path: command not found
```

convert Windows line endings to Linux line endings:

```bash
sed -i 's/\r$//' scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

If the dataset is missing, the ECL script expects:

```text
dataset/electricity/electricity.csv
```

On Linux/Colab, `Electricity` and `electricity` are different folders.

## 6. Recommended Experiment Comparison

Run and compare:

```bash
ATTENTION_MODE=full SPARSE_TOP_K=0 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

```bash
ATTENTION_MODE=sparse SPARSE_TOP_K=32 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
```

Compare:

- MSE
- MAE
- runtime
- memory usage
- stagewise attention behavior

This gives a first ablation for the scalability improvement.
