#!/usr/bin/env bash
# UI-TARS-1.5-7B + ScreenSpot-Pro with the canonical STaR-KV settings.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/examples/starkv_local_paths.sh"
GPU="${GPU:-0}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/outputs/repro_ssp_$(date -u +%Y%m%d_%H%M%S)}"
read -r -a BUDGET_LIST <<< "${BUDGETS:-${BUDGET:-20}}"
for path in "${UITARS_MODEL_PATH}" "${SCREENSPOTPRO_IMGS}" "${SCREENSPOTPRO_TEST}"; do
  [[ -d "${path}" ]] || { echo "Missing model or dataset directory: ${path}" >&2; exit 2; }
done
FLAGS=(
  --kv_cache starkv
  --baseline_gqa_mode group_mean
  --kv_budget_semantics total_including_window
  --window_size 8 --alpha 2.0 --temperature 3.5
      --kv_group_soft_prior_lambda 0.5
      --kv_group_online_profile_steps 5
      --kv_group_online_profile_decay 0.9
      --kv_group_online_profile_tau 1.0
      --kv_group_online_profile_lambda_ramp_steps 10
      --kv_group_temporal_enable
      --kv_group_temporal_delta 0.2
      --kv_group_temporal_rho 0.9
      --kv_group_temporal_discount_min 0.1
      --kv_group_temporal_eps 0.0
      --kv_group_temporal_warmup_steps 0
      --kv_group_temporal_decay_mode exponential
      --kv_group_mi_granularity group
      --kv_group_mi_prior_mode online
      --kv_group_sfr_enable
      --kv_group_sfr_lambda 0.3
  --kv_group_single_image_mi_enable
)
for budget in "${BUDGET_LIST[@]}"; do
  case "${budget}" in
    5|10|20|40) ;;
    *) echo "Supported budgets: 5 10 20 40" >&2; exit 2 ;;
  esac
  OUT="${RESULTS_BASE}/starkv_b${budget}"
  mkdir -p "${OUT}"
  CMD=("${STARKV_PYTHON}" -u "${ROOT}/eval/screenspotpro_eval.py"
    --model_path "${UITARS_MODEL_PATH}"
    --screenspot_imgs "${SCREENSPOTPRO_IMGS}"
    --screenspot_test "${SCREENSPOTPRO_TEST}"
    --task all --max_new_tokens 400
    --attention_implementation flash_attention_2 --model_dtype bfloat16
    --kv_cache_budget "${budget}" --results_dir "${OUT}"
    "${FLAGS[@]}")
  printf '%q ' "${CMD[@]}" > "${OUT}/command.sh"
  printf '\n' >> "${OUT}/command.sh"
  echo "[ssp] budget=${budget}% GPU=${GPU} -> ${OUT}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" 2>&1 | tee "${OUT}/eval.log"
  if grep -qE 'CUDA error|device-side assert|index out of bounds|Error in generation|Generation failed' "${OUT}/eval.log"; then
    echo "Evaluation reported a generation error; results are incomplete." >&2
    exit 1
  fi
  [[ -s "${OUT}/screenspotpro_summary_results.json" ]] || { echo "Missing summary" >&2; exit 1; }
done
