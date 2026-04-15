#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

gpus="${GPUS:-0,1,2,3}"
free_threshold_mib="${FREE_THRESHOLD_MIB:-500}"
IFS=',' read -r -a gpu_ids <<< "${gpus}"

if ((${#gpu_ids[@]} == 0 || ${#gpu_ids[@]} > 4)); then
  echo "GPUS must select between 1 and 4 GPUs, got: ${gpus}" >&2
  exit 1
fi

for gpu_id in "${gpu_ids[@]}"; do
  used_mib="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if ((used_mib >= free_threshold_mib)); then
    echo "GPU ${gpu_id} is not free: ${used_mib} MiB used, threshold ${free_threshold_mib} MiB." >&2
    exit 1
  fi
done

if [[ -f /data2/smilehan/app/miniconda3/etc/profile.d/conda.sh ]]; then
  source /data2/smilehan/app/miniconda3/etc/profile.d/conda.sh
fi
conda activate easytsf

export CUDA_VISIBLE_DEVICES="${gpus}"
export PYTHONUNBUFFERED=1

log_dir="${repo_dir}/logs/grid3d_risk_v1_0414"
mkdir -p "${log_dir}"
run_id="$(date +%Y%m%d_%H%M%S)"

configs=(
  "config/experiments/unet3d/windfield4cast_risk_v1_0414.yaml"
  "config/experiments/unet3d_patchcat/windfield4cast_risk_v1_0414.yaml"
  "config/experiments/patchstg_flat3d/windfield4cast_risk_v1_0414.yaml"
  "config/experiments/fredn_multivariate3d/windfield4cast_risk_v1_0414.yaml"
)

for config in "${configs[@]}"; do
  model_dir="$(basename "$(dirname "${config}")")"
  log_file="${log_dir}/${run_id}_${model_dir}.log"
  exp_dir="${repo_dir}/checkpoint/grid3d_risk_v1_0414_runs/${run_id}/${model_dir}_seed42"
  echo "[$(date --iso-8601=seconds)] starting ${config} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  python -m easytsf.workflow.experiment "${config}" --set exp_dir="${exp_dir}" 2>&1 | tee "${log_file}"
  echo "[$(date --iso-8601=seconds)] finished ${config}; log=${log_file}"
done
