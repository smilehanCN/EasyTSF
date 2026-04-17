#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

free_threshold_mib="${FREE_THRESHOLD_MIB:-500}"
required_gpu_count="${REQUIRED_GPU_COUNT:-4}"
memory_headroom_gb="${MEMORY_HEADROOM_GB:-34}"
run_tag="${WSH_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
cpus_per_trial="${BENCH_CPUS_PER_TRIAL:-4}"
precision="${WSH_PRECISION:-32-true}"

if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${GPUS}"
else
  mapfile -t gpu_ids < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v threshold="${free_threshold_mib}" '$2 + 0 < threshold {gsub(/ /, "", $1); print $1}' \
      | head -n "${required_gpu_count}"
  )
fi

if ((${#gpu_ids[@]} < required_gpu_count)); then
  echo "Need ${required_gpu_count} free GPUs below ${free_threshold_mib} MiB, found ${#gpu_ids[@]}." >&2
  exit 1
fi

gpu_ids=("${gpu_ids[@]:0:${required_gpu_count}}")

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

export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

log_root="${repo_dir}/logs/windshear_v1_0416_h10_p10_tuning/${run_tag}"
calibration_dir="${log_root}/calibration"
benchmark_log_dir="${log_root}/benchmark_logs"
report_dir="${log_root}/reports"
results_dir="${log_root}/results"
mkdir -p "${calibration_dir}" "${benchmark_log_dir}" "${report_dir}"
mkdir -p "${results_dir}"

calibration_csv="${calibration_dir}/calibration_results.csv"
selection_json="${calibration_dir}/selected_config.json"

cat > "${calibration_csv}" <<'EOF'
model,devices,local_batch,effective_batch,peak_allocated_gb,peak_reserved_gb,peak_nvidia_smi_gb,success,error
EOF

models=(unet3d fno3d afno3d)

declare -A experiment_path=(
  [unet3d]="config/experiments/unet3d/windshear_v1_0416_h10_p10.yaml"
  [fno3d]="config/experiments/fno3d/windshear_v1_0416_h10_p10.yaml"
  [afno3d]="config/experiments/afno3d/windshear_v1_0416_h10_p10.yaml"
)
declare -A benchmark_path=(
  [unet3d]="config/benchmarks/unet3d/windshear_v1_0416_h10_p10_tune.py"
  [fno3d]="config/benchmarks/fno3d/windshear_v1_0416_h10_p10_tune.py"
  [afno3d]="config/benchmarks/afno3d/windshear_v1_0416_h10_p10_tune.py"
)
declare -A heavy_override_key=(
  [unet3d]="base_channels"
  [fno3d]="fno_width"
  [afno3d]="afno_embed_dim"
)
declare -A heavy_override_value=(
  [unet3d]="20"
  [fno3d]="24"
  [afno3d]="80"
)

declare -A chosen_devices=()
declare -A chosen_batch_size=()
declare -A chosen_effective_batch=()

join_first_gpus() {
  local count="$1"
  local selected=()
  local index=0
  while ((index < count)); do
    selected+=("${gpu_ids[$index]}")
    index=$((index + 1))
  done
  local joined=""
  local item
  for item in "${selected[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${item}"
  done
  printf '%s\n' "${joined}"
}

append_failure_row() {
  local model="$1"
  local devices="$2"
  local batch_size="$3"
  local effective_batch="$4"
  local message="$5"
  printf '%s,%s,%s,%s,,,,false,"%s"\n' \
    "${model}" \
    "${devices}" \
    "${batch_size}" \
    "${effective_batch}" \
    "${message//\"/\'}" >> "${calibration_csv}"
}

append_success_row() {
  local json_path="$1"
  python - "$json_path" "${calibration_csv}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
with csv_path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            summary["model"],
            summary["devices"],
            summary["local_batch"],
            summary["effective_batch"],
            "{:.6f}".format(summary["peak_allocated_gb"]),
            "{:.6f}".format(summary["peak_reserved_gb"]),
            "{:.6f}".format(summary["peak_nvidia_smi_gb"]),
            "true",
            "",
        ]
    )
PY
}

run_calibration_attempt() {
  local model="$1"
  local devices="$2"
  local batch_size="$3"
  local effective_batch="$4"

  if ((devices > required_gpu_count)); then
    append_failure_row "${model}" "${devices}" "${batch_size}" "${effective_batch}" "insufficient free GPUs"
    return 1
  fi

  local gpu_subset
  gpu_subset="$(join_first_gpus "${devices}")"
  local strategy="auto"
  if ((devices > 1)); then
    strategy="ddp"
  fi

  local json_path="${calibration_dir}/${model}_d${devices}_b${batch_size}.json"
  local log_path="${calibration_dir}/${model}_d${devices}_b${batch_size}.log"

  local -a script_args=(
    scripts/calibrate_windshear_h10p10_batch.py
    "${experiment_path[$model]}"
    --output-json "${json_path}"
    --set "batch_size=${batch_size}"
    --set "devices=${devices}"
    --set "strategy=${strategy}"
    --set "precision=${precision}"
    --set "max_epochs=1"
    --set "check_val_every_n_epoch=1"
    --set "num_workers=2"
    --set "${heavy_override_key[$model]}=${heavy_override_value[$model]}"
  )

  local -a cmd=()

  if ((devices > 1)); then
    cmd=(torchrun --standalone --nproc_per_node="${devices}" "${script_args[@]}")
  else
    cmd=(python "${script_args[@]}")
  fi

  echo "[$(date --iso-8601=seconds)] Calibrating ${model} with devices=${devices}, batch_size=${batch_size}, effective_batch=${effective_batch}, CUDA_VISIBLE_DEVICES=${gpu_subset}" | tee -a "${log_root}/launcher.log"
  if CUDA_VISIBLE_DEVICES="${gpu_subset}" "${cmd[@]}" > "${log_path}" 2>&1; then
    if python - "$json_path" "${memory_headroom_gb}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
threshold = float(sys.argv[2])
if summary["peak_reserved_gb"] >= threshold or summary["peak_nvidia_smi_gb"] >= threshold:
    raise SystemExit(1)
PY
    then
      append_success_row "${json_path}"
      return 0
    fi

    append_failure_row "${model}" "${devices}" "${batch_size}" "${effective_batch}" "memory threshold exceeded; see $(basename "${log_path}")"
    return 1
  fi

  append_failure_row "${model}" "${devices}" "${batch_size}" "${effective_batch}" "failed; see $(basename "${log_path}")"
  return 1
}

candidate_pairs_for_target() {
  local target="$1"
  case "${target}" in
    8)
      echo "2:4 4:2"
      ;;
    4)
      echo "1:4 2:2 4:1"
      ;;
    2)
      echo "1:2 2:1"
      ;;
    *)
      return 1
      ;;
  esac
}

common_target=""
for target in 8 4 2; do
  target_ok=1
  for model in "${models[@]}"; do
    model_ok=0
    for pair in $(candidate_pairs_for_target "${target}"); do
      devices="${pair%%:*}"
      batch_size="${pair##*:}"
      if run_calibration_attempt "${model}" "${devices}" "${batch_size}" "${target}"; then
        chosen_devices["${model}"]="${devices}"
        chosen_batch_size["${model}"]="${batch_size}"
        chosen_effective_batch["${model}"]="${target}"
        model_ok=1
        break
      fi
    done
    if ((model_ok == 0)); then
      target_ok=0
    fi
  done
  if ((target_ok == 1)); then
    common_target="${target}"
    break
  fi
done

if [[ -z "${common_target}" ]]; then
  echo "No common effective batch succeeded across all models." >&2
  exit 1
fi

python - "${selection_json}" "${common_target}" \
  "${chosen_devices[unet3d]}" "${chosen_batch_size[unet3d]}" \
  "${chosen_devices[fno3d]}" "${chosen_batch_size[fno3d]}" \
  "${chosen_devices[afno3d]}" "${chosen_batch_size[afno3d]}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
summary = {
    "effective_batch": int(sys.argv[2]),
    "unet3d": {"devices": int(sys.argv[3]), "batch_size": int(sys.argv[4])},
    "fno3d": {"devices": int(sys.argv[5]), "batch_size": int(sys.argv[6])},
    "afno3d": {"devices": int(sys.argv[7]), "batch_size": int(sys.argv[8])},
}
output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY

sorted_models=($(python - "${chosen_devices[unet3d]}" "${chosen_devices[fno3d]}" "${chosen_devices[afno3d]}" <<'PY'
entries = [
    ("unet3d", int(__import__("sys").argv[1])),
    ("fno3d", int(__import__("sys").argv[2])),
    ("afno3d", int(__import__("sys").argv[3])),
]
entries.sort(key=lambda item: (-item[1], item[0]))
print(" ".join(name for name, _ in entries))
PY
))

remaining_models=("${sorted_models[@]}")
batch_index=0
while ((${#remaining_models[@]} > 0)); do
  batch_index=$((batch_index + 1))
  current_batch=()
  batch_devices=()
  capacity="${required_gpu_count}"
  next_remaining=()

  for model in "${remaining_models[@]}"; do
    devices="${chosen_devices[$model]}"
    if ((devices <= capacity)); then
      current_batch+=("${model}")
      batch_devices+=("${devices}")
      capacity=$((capacity - devices))
    else
      next_remaining+=("${model}")
    fi
  done

  if ((${#current_batch[@]} == 0)); then
    echo "Unable to schedule any benchmark batch with the selected device allocations." >&2
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] Launching direct sweep batch ${batch_index}: ${current_batch[*]}" | tee -a "${log_root}/launcher.log"

  offset=0
  pids=()
  for model in "${current_batch[@]}"; do
    devices="${chosen_devices[$model]}"
    batch_size="${chosen_batch_size[$model]}"
    subset=()
    for ((index = offset; index < offset + devices; index++)); do
      subset+=("${gpu_ids[$index]}")
    done
    offset=$((offset + devices))

    gpu_subset=""
    for gpu_id in "${subset[@]}"; do
      if [[ -n "${gpu_subset}" ]]; then
        gpu_subset+=","
      fi
      gpu_subset+="${gpu_id}"
    done

    benchmark_log="${benchmark_log_dir}/${model}.log"
    report_path="${report_dir}/${model}.csv"
    model_results_dir="${results_dir}/${model}"

    (
      export CUDA_VISIBLE_DEVICES="${gpu_subset}"
      export WSH_RUN_TAG="${run_tag}"
      python scripts/run_windshear_h10p10_direct_sweep.py \
        --model "${model}" \
        --experiment "${experiment_path[$model]}" \
        --devices "${devices}" \
        --batch-size "${batch_size}" \
        --precision "${precision}" \
        --run-tag "${run_tag}" \
        --results-root "${model_results_dir}" \
        --report-out "${report_path}" 2>&1 | tee "${benchmark_log}"
    ) &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  remaining_models=("${next_remaining[@]}")
done

echo "[$(date --iso-8601=seconds)] Finished WindShear h10_p10 tuning. selection=${selection_json}" | tee -a "${log_root}/launcher.log"
