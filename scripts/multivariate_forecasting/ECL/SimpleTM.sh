export CUDA_VISIBLE_DEVICES=0
model_name=SimpleTM

# Set DEBUG_STAGEWISE=1 to print internal tensors stage-by-stage during the *test* phase.
# Example:
#   DEBUG_STAGEWISE=1 bash scripts/multivariate_forecasting/ECL/SimpleTM.sh
DEBUG_STAGEWISE=${DEBUG_STAGEWISE:-0}
DEBUG_MAX_BATCHES=${DEBUG_MAX_BATCHES:-1}
DEBUG_PREVIEW_LEN=${DEBUG_PREVIEW_LEN:-5}
ATTENTION_MODE=${ATTENTION_MODE:-full}
SPARSE_TOP_K=${SPARSE_TOP_K:-0}

EXTRA_ARGS="--attention_mode ${ATTENTION_MODE} --sparse_top_k ${SPARSE_TOP_K}"
if [ "$DEBUG_STAGEWISE" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --debug_stagewise --debug_max_batches ${DEBUG_MAX_BATCHES} --debug_preview_len ${DEBUG_PREVIEW_LEN}"
  echo "[Debug] Stagewise logging enabled for test phase."
fi
echo "[Attention] mode=${ATTENTION_MODE}, sparse_top_k=${SPARSE_TOP_K}"

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.01 \
  --batch_size ${BATCH_SIZE:-256} \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0 \
  ${EXTRA_ARGS}

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.006 \
  --batch_size ${BATCH_SIZE:-256} \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0 \
  ${EXTRA_ARGS}

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.006 \
  --batch_size ${BATCH_SIZE:-256} \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 5e-5 \
  ${EXTRA_ARGS}

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.006 \
  --batch_size ${BATCH_SIZE:-256} \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 5e-5 \
  ${EXTRA_ARGS}
