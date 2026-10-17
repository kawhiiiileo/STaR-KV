#!/usr/bin/env bash
# Portable local path loader for the canonical reproduction runner.

_starkv_examples_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STARKV_ROOT="${STARKV_ROOT:-$(cd "${_starkv_examples_dir}/.." && pwd)}"

_starkv_config="${STARKV_CONFIG:-${_starkv_examples_dir}/starkv_local_paths.env}"
if [[ -f "${_starkv_config}" ]]; then
  # shellcheck source=/dev/null
  set -a
  source "${_starkv_config}"
  set +a
fi

export UITARS_MODEL_PATH="${UITARS_MODEL_PATH:-}"
export OPENCUA_MODEL_PATH="${OPENCUA_MODEL_PATH:-}"
export SCREENSPOTPRO_IMGS="${SCREENSPOTPRO_IMGS:-}"
export SCREENSPOTPRO_TEST="${SCREENSPOTPRO_TEST:-}"
export SCREENSPOTV2_IMGS="${SCREENSPOTV2_IMGS:-}"
export SCREENSPOTV2_TEST="${SCREENSPOTV2_TEST:-}"
export ANDROIDCONTROL_IMGS="${ANDROIDCONTROL_IMGS:-}"
export ANDROIDCONTROL_TEST="${ANDROIDCONTROL_TEST:-}"
export ANB_DATA="${ANB_DATA:-}"
export ANB_IMGS="${ANB_IMGS:-}"

if [[ -n "${STARKV_PYTHON:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  export STARKV_PYTHON="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  export STARKV_PYTHON="$(command -v python3)"
else
  export STARKV_PYTHON="$(command -v python)"
fi

export PYTHONPATH="${STARKV_ROOT}:${STARKV_ROOT}/eval${PYTHONPATH:+:${PYTHONPATH}}"

unset _starkv_examples_dir _starkv_config
