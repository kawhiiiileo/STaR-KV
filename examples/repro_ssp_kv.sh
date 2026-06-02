#!/usr/bin/env bash
# ScreenSpot-Pro + UI-TARS + STaR-KV (see README).
set -euo pipefail

GPU="${GPU:-0}"

_EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=starkv_local_paths.sh
source "${_EXAMPLES_DIR}/starkv_local_paths.sh"
starkv_setup_conda || starkv_resolve_python

MODEL_PATH="${MODEL_PATH:-${UITARS_MODEL_PATH}}"
RESULTS_BASE="${RESULTS_BASE:-${STARKV_RESULTS_DIR}/repro_ssp_kv}"

if [[ -n "${BUDGETS:-}" ]]; then
  read -r -a _BUDGET_LIST <<< "${BUDGETS}"
else
  _BUDGET_LIST=("${BUDGET:-20}")
fi

STARKV_FLAGS=(
  --kv_cache starkv
  --kv_group_soft_prior_lambda 0.5
  --kv_group_online_profile_steps 5
  --kv_group_online_profile_decay 0.9
  --kv_group_online_profile_tau 1.0
  --kv_group_online_profile_lambda_ramp_steps 10
  --alpha 2 --temperature 3.5 --window_size 8
  --kv_entropy_budget_enable
  --kv_entropy_budget_min_scale 0.75
  --kv_entropy_budget_max_scale 1.25
  --kv_entropy_budget_smooth 0.0
  --kv_group_temporal_enable
  --kv_group_temporal_delta 0.1
  --kv_group_temporal_rho 0.9
  --kv_group_temporal_eps 0.0
  --kv_group_temporal_discount_min 0.0
  --kv_group_temporal_warmup_steps 0
)

for BUDGET in "${_BUDGET_LIST[@]}"; do
  OUT="${RESULTS_BASE}/starkv_b${BUDGET}"
  mkdir -p "${OUT}"
  echo "[ssp_kv] STaR-KV budget=${BUDGET}% GPU=${GPU} -> ${OUT}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u "${STARKV_EVAL_DIR}/screenspotpro_eval.py" \
    --model_path "${MODEL_PATH}" \
    --screenspot_imgs "${SCREENSPOTPRO_IMGS}" \
    --screenspot_test "${SCREENSPOTPRO_TEST}" \
    --task all \
    --attention_implementation flash_attention_2 \
    --model_dtype bfloat16 \
    --max_new_tokens 400 \
    --kv_cache_budget "${BUDGET}" \
    --results_dir "${OUT}" \
    "${STARKV_FLAGS[@]}"
done

echo "[ssp_kv] Done under ${RESULTS_BASE}/"
