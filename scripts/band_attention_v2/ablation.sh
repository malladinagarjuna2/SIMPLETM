#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-./dataset/ETT-small}
DATA_PATH=${DATA_PATH:-ETTh1.csv}
ENC_IN=${ENC_IN:-7}
PRED_LEN=${PRED_LEN:-96}
SEEDS=${SEEDS:-"2025 2026 2027"}
LOGDIR=${LOGDIR:-./logs/band_attention_v2}
mkdir -p "${LOGDIR}"

COMMON=(--is_training 1 --model_id band_attention_v2 --model SimpleTM --data ETTh1
  --root_path "${ROOT_PATH}" --data_path "${DATA_PATH}" --features M
  --enc_in "${ENC_IN}" --dec_in "${ENC_IN}" --c_out "${ENC_IN}" --pred_len "${PRED_LEN}")

run_tag() {
  local tag=$1
  shift
  for seed in ${SEEDS}; do
    python3 run.py "${COMMON[@]}" --fix_seed "${seed}" "$@" \
      2>&1 | tee "${LOGDIR}/${tag}_p${PRED_LEN}_s${seed}.log"
  done
}

run_tag baseline
run_tag v1_band --use_band_attention --band_attention_activation softmax --band_attention_pooling mean --band_attention_apply_to qkv --band_attention_separate_qkv
run_tag fix_pool --use_band_attention --band_attention_activation softmax --band_attention_apply_to qkv --band_attention_separate_qkv
run_tag fix_scale --use_band_attention --band_attention_apply_to qkv --band_attention_separate_qkv
run_tag fix_apply --use_band_attention
run_tag apply_qk --use_band_attention --band_attention_apply_to qk
run_tag apply_qkv --use_band_attention --band_attention_apply_to qkv
run_tag per_var --use_band_attention --band_attention_per_variable
run_tag mixing --use_band_mixing
run_tag band_plus_mixing --use_band_attention --use_band_mixing
run_tag horizon --use_band_attention --band_attention_horizon
run_tag v2_full --use_band_attention --band_attention_per_variable --band_attention_horizon --use_band_mixing
