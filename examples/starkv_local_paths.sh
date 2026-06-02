#!/usr/bin/env bash
# Portable path setup: env vars + PYTHONPATH (no machine-specific paths).
#
# Quick start (from this repo root):
#   export STARKV_MODEL_DIR STARKV_DATASETS_DIR  # or create examples/starkv_local_paths.env
#   source examples/starkv_local_paths.sh
#   GPU=0 BUDGET=20 bash examples/repro_ssp_kv.sh
#
# Override file location: export STARKV_CONFIG=/path/to/my.env before source.
#
# Defaults: STARKV_* override everything. If unset, use REPO/archive/{model,datasets,results}
# when that tree exists; REPO is this repo root, or the parent dir when ../archive exists.

_starkv_paths_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_starkv_pkg_root="$(cd "${_starkv_paths_dir}/.." && pwd)"
if [[ -d "${_starkv_pkg_root}/../archive" ]]; then
  _starkv_repo_default="$(cd "${_starkv_pkg_root}/.." && pwd)"
else
  _starkv_repo_default="${_starkv_pkg_root}"
fi

REPO="${STARKV_LOCAL_ROOT:-${_starkv_repo_default}}"
export REPO
export STARKV_ROOT="${STARKV_ROOT:-${_starkv_pkg_root}}"
export STARKV_EXAMPLES_DIR="${STARKV_EXAMPLES_DIR:-${STARKV_ROOT}/examples}"

_starkv_load_user_config() {
  local f
  if [[ -n "${STARKV_CONFIG:-}" && -f "${STARKV_CONFIG}" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${STARKV_CONFIG}"; set +a
    return 0
  fi
  for f in \
    "${_starkv_paths_dir}/starkv_local_paths.env" \
    "${STARKV_EXAMPLES_DIR}/starkv_local_paths.env" \
    "${STARKV_ROOT}/starkv_local_paths.env" \
    "${REPO}/starkv_local_paths.env" \
    "${_starkv_paths_dir}/starkv_local_paths.local.sh" \
    "${STARKV_ROOT}/starkv_local_paths.local.sh"
  do
    if [[ -f "${f}" ]]; then
      # shellcheck source=/dev/null
      set -a; source "${f}"; set +a
      echo "[starkv_local] loaded config: ${f}" >&2
      return 0
    fi
  done
  return 0
}

_starkv_load_user_config

export STARKV_RESULTS_DIR="${STARKV_RESULTS_DIR:-${REPO}/archive/results}"
export STARKV_MODEL_DIR="${STARKV_MODEL_DIR:-${REPO}/archive/model}"
export STARKV_DATASETS_DIR="${STARKV_DATASETS_DIR:-${REPO}/archive/datasets}"
export EVAL_EXTRAS_DIR="${EVAL_EXTRAS_DIR:-${REPO}/archive/STaR-KV_eval_extras}"
export MM_MIND2WEB_EVAL="${MM_MIND2WEB_EVAL:-${EVAL_EXTRAS_DIR}/multimodal_mind2web_eval.py}"
export OSWORLD_EVAL="${OSWORLD_EVAL:-${EVAL_EXTRAS_DIR}/osworld_eval.py}"

export UITARS_MODEL_PATH="${UITARS_MODEL_PATH:-${STARKV_MODEL_DIR}/UI-TARS-1.5-7B}"
export OPENCUA_MODEL_PATH="${OPENCUA_MODEL_PATH:-${STARKV_MODEL_DIR}/OpenCUA-7B}"
export MODEL_PATH="${MODEL_PATH:-${UITARS_MODEL_PATH}}"

export ANDROIDCONTROL_IMGS="${ANDROIDCONTROL_IMGS:-${STARKV_DATASETS_DIR}/androidcontrol}"
export ANDROIDCONTROL_TEST="${ANDROIDCONTROL_TEST:-${STARKV_DATASETS_DIR}/androidcontrol/data}"
export SCREENSPOTPRO_IMGS="${SCREENSPOTPRO_IMGS:-${STARKV_DATASETS_DIR}/ScreenSpot-Pro/images}"
export SCREENSPOTPRO_TEST="${SCREENSPOTPRO_TEST:-${STARKV_DATASETS_DIR}/ScreenSpot-Pro/annotations}"
export ANB_DATA="${ANB_DATA:-${STARKV_DATASETS_DIR}/agentnetbench/test_data}"
export ANB_IMGS="${ANB_IMGS:-${STARKV_DATASETS_DIR}/agentnetbench/test_data/images}"
export MM_MIND2WEB_IMGS="${MM_MIND2WEB_IMGS:-${STARKV_DATASETS_DIR}/multimodal-mind2web/release_images}"
export MM_MIND2WEB_TEST="${MM_MIND2WEB_TEST:-${STARKV_DATASETS_DIR}/multimodal-mind2web/data/samples}"
export SCREENSPOTV2_IMGS="${SCREENSPOTV2_IMGS:-${STARKV_DATASETS_DIR}/screenspot-v2-prepared/images}"
export SCREENSPOTV2_TEST="${SCREENSPOTV2_TEST:-${STARKV_DATASETS_DIR}/screenspot-v2-prepared/annotations}"
export OSWORLD_DATA="${OSWORLD_DATA:-${STARKV_DATASETS_DIR}/osworld-verified/evaluation_examples}"

_starkv_tmp_root="${TMPDIR:-/tmp}"

# Resolve PYTHON_BIN: explicit > active conda env > python3 > python
starkv_resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    export PYTHON_BIN
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
    export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
    return 0
  fi
  echo "[starkv_local] ERROR: no python found; set PYTHON_BIN or activate your conda env" >&2
  return 1
}

# Optional conda: set STARKV_AUTO_CONDA=1 and STARKV_CONDA_ENV=<name> in starkv_local_paths.env
starkv_setup_conda() {
  if [[ -z "${CONDA_BASE:-}" ]] && [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
    export CONDA_BASE
  fi
  if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if [[ "${STARKV_AUTO_CONDA:-0}" == "1" && -n "${STARKV_CONDA_ENV:-}" ]]; then
      conda activate "${STARKV_CONDA_ENV}" 2>/dev/null || true
    fi
  fi
  starkv_resolve_python
}

starkv_export_pythonpath() {
  export STARKV_EVAL_DIR="${STARKV_ROOT}/eval"
  export EVAL_DIR="${STARKV_EVAL_DIR}"
  export PYTHONPATH="${EVAL_DIR}:${STARKV_ROOT}/starkv${PYTHONPATH:+:${PYTHONPATH}}"
}

starkv_export_pythonpath

_starkv_model_complete() {
  local d="$1"
  [[ -f "${d}/config.json" ]] || return 1
  [[ -f "${d}/model.safetensors.index.json" ]] || ls "${d}"/*.safetensors >/dev/null 2>&1 || return 1
  return 0
}

_starkv_sync_model_to_local() {
  local src="$1"
  local cache_root="${STARKV_MODEL_CACHE:-${_starkv_tmp_root}/starkv_model_cache}"
  local name
  name="$(basename "${src}")"
  local dst="${cache_root}/${name}"
  local stamp="${dst}/.sync_stamp"
  local lock="${cache_root}/.sync_${name}.lock"

  mkdir -p "${cache_root}"
  exec 200>"${lock}"
  if ! flock -x 200 2>/dev/null; then
    echo "[starkv_local] WARN: could not lock ${lock}, using source model path" >&2
    return 1
  fi

  local need=0
  if [[ "${STARKV_MODEL_FORCE_SYNC:-0}" == "1" ]]; then
    need=1
  elif ! _starkv_model_complete "${dst}"; then
    need=1
  elif [[ -f "${src}/config.json" ]] && [[ -f "${stamp}" ]] && [[ "${src}/config.json" -nt "${stamp}" ]]; then
    need=1
  fi

  if [[ "${need}" == "1" ]]; then
    echo "[starkv_local] rsync model ${src} -> ${dst}" >&2
    mkdir -p "${dst}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "${src}/" "${dst}/"
    else
      rm -rf "${dst}"
      cp -a "${src}" "${dst}"
    fi
    touch "${stamp}"
    echo "[starkv_local] model cache ready: ${dst}" >&2
  else
    echo "[starkv_local] model cache hit: ${dst}" >&2
  fi

  export MODEL_PATH="${dst}"
  flock -u 200 2>/dev/null || true
}

starkv_ensure_opencua_runtime() {
  local repo_root="${1:-${REPO}}"
  local starkv_src="${STARKV_ROOT}"
  local model_src="${MODEL_PATH}"

  if [[ ! -f "${model_src}/config.json" ]] && [[ ! -f "${model_src}/model.safetensors.index.json" ]]; then
    echo "[starkv_local] ERROR: model not found at ${model_src}" >&2
    echo "[starkv_local] Set MODEL_PATH or OPENCUA_MODEL_PATH" >&2
    return 1
  fi

  if [[ "${STARKV_MODEL_RUNTIME:-1}" == "1" ]] && [[ "${model_src}" != "${_starkv_tmp_root}"/* ]]; then
    _starkv_sync_model_to_local "${model_src}" || export MODEL_PATH="${model_src}"
  fi

  if [[ ! -d "${starkv_src}/eval" ]] || [[ ! -d "${starkv_src}/starkv" ]]; then
    echo "[starkv_local] ERROR: ${starkv_src}/eval or starkv missing." >&2
    return 1
  fi

  if [[ "${OPENCUA_CODE_RUNTIME:-0}" == "1" ]]; then
    local local_runtime="${OPENCUA_LOCAL_RUNTIME:-${_starkv_tmp_root}/starkv_opencua_runtime}"
    local sync_stamp="${local_runtime}/.sync_stamp"
    if [[ "${OPENCUA_FORCE_SYNC:-0}" == "1" ]] || [[ ! -f "${sync_stamp}" ]] \
        || [[ "${starkv_src}/eval/attention_helpers.py" -nt "${sync_stamp}" ]]; then
      echo "[starkv_local] sync code ${starkv_src} -> ${local_runtime}" >&2
      mkdir -p "${local_runtime}"
      if command -v rsync >/dev/null 2>&1; then
        rsync -a "${starkv_src}/eval/" "${local_runtime}/eval/"
        rsync -a "${starkv_src}/starkv/" "${local_runtime}/starkv/"
      else
        rm -rf "${local_runtime}/eval" "${local_runtime}/starkv"
        cp -a "${starkv_src}/eval" "${local_runtime}/"
        cp -a "${starkv_src}/starkv" "${local_runtime}/"
      fi
      touch "${sync_stamp}"
    fi
    export STARKV_ROOT="${local_runtime}"
  else
    export STARKV_ROOT="${starkv_src}"
  fi

  starkv_export_pythonpath
  export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
  starkv_resolve_python || return 1

  echo "[starkv_local] REPO=${REPO}" >&2
  echo "[starkv_local] MODEL_PATH=${MODEL_PATH}" >&2
  echo "[starkv_local] EVAL_DIR=${EVAL_DIR}" >&2
  echo "[starkv_local] PYTHON_BIN=${PYTHON_BIN}" >&2
  return 0
}

# Default python for scripts that only source this file (no conda block)
starkv_resolve_python 2>/dev/null || true
