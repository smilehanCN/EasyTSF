#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easytsf.workflow.config import finalize_runtime_conf, load_experiment_config, parse_config_overrides
from easytsf.workflow.experiment import prepare_runtime_conf_for_task


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single-step WindShear h10_p10 memory calibration.")
    parser.add_argument("experiment", help="Experiment preset ref or yaml path.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Runtime override. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the calibration summary JSON.",
    )
    return parser


def _world_info() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world_size


def _init_process_group(world_size: int) -> bool:
    if world_size <= 1:
        return False
    dist.init_process_group(backend="nccl")
    return True


def _resolve_device(local_rank: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WindShear batch calibration.")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _visible_gpu_id(local_rank: int) -> str:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw.strip() == "":
        return str(local_rank)
    visible_ids = [item.strip() for item in raw.split(",") if item.strip() != ""]
    if local_rank >= len(visible_ids):
        raise RuntimeError(
            "LOCAL_RANK {} is out of range for CUDA_VISIBLE_DEVICES={!r}".format(local_rank, raw)
        )
    return visible_ids[local_rank]


def _query_gpu_memory_gib(physical_gpu_id: str) -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id={}".format(physical_gpu_id),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used_mib = float(result.stdout.strip())
    return used_mib / 1024.0


def _gather_metric(local_value: float, device: torch.device, world_size: int) -> tuple[list[float], float]:
    if world_size <= 1:
        values = [float(local_value)]
        return values, values[0]
    tensor = torch.tensor([float(local_value)], dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    values = [float(item.item()) for item in gathered]
    return values, max(values)


def _extract_optimizer(task) -> torch.optim.Optimizer:
    optimizer_conf = task.configure_optimizers()
    if isinstance(optimizer_conf, dict):
        return optimizer_conf["optimizer"]
    if isinstance(optimizer_conf, (list, tuple)) and len(optimizer_conf) > 0:
        first_item = optimizer_conf[0]
        if isinstance(first_item, dict):
            return first_item["optimizer"]
        return first_item
    raise TypeError("Unsupported optimizer config type: {}".format(type(optimizer_conf).__name__))


def main() -> int:
    args = build_cli_parser().parse_args()
    overrides = parse_config_overrides(args.overrides)
    runtime_conf = finalize_runtime_conf(load_experiment_config(args.experiment), overrides)

    local_rank, rank, world_size = _world_info()
    process_group_initialized = False
    try:
        process_group_initialized = _init_process_group(world_size)
        device = _resolve_device(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        task_spec, datamodule, runtime_conf = prepare_runtime_conf_for_task(runtime_conf)
        batch = next(iter(datamodule.train_dataloader()))
        batch = _move_batch_to_device(batch, device)

        task = task_spec.task_cls(**runtime_conf)
        task = task.to(device)
        task.train()

        if world_size > 1:
            task.model = DDP(task.model, device_ids=[local_rank], output_device=local_rank)

        optimizer = _extract_optimizer(task)
        optimizer.zero_grad(set_to_none=True)

        prediction, var_y = task._forward(batch)
        loss = task.loss_function(prediction, var_y)

        model_ref = task.model.module if isinstance(task.model, DDP) else task.model
        aux_loss = model_ref.get_aux_loss() if hasattr(model_ref, "get_aux_loss") else None
        if aux_loss is not None:
            loss = loss + getattr(task.hparams, "aux_loss_weight", 1.0) * aux_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)

        peak_allocated_gb = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / float(1024 ** 3)
        peak_nvidia_smi_gb = _query_gpu_memory_gib(_visible_gpu_id(local_rank))

        allocated_per_rank, allocated_max = _gather_metric(peak_allocated_gb, device, world_size)
        reserved_per_rank, reserved_max = _gather_metric(peak_reserved_gb, device, world_size)
        smi_per_rank, smi_max = _gather_metric(peak_nvidia_smi_gb, device, world_size)

        if rank == 0:
            output_path = Path(args.output_json).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "model": str(runtime_conf["model"]),
                "devices": int(world_size),
                "local_batch": int(runtime_conf["batch_size"]),
                "effective_batch": int(world_size) * int(runtime_conf["batch_size"]),
                "peak_allocated_gb": allocated_max,
                "peak_reserved_gb": reserved_max,
                "peak_nvidia_smi_gb": smi_max,
                "success": True,
                "error": "",
                "per_rank": {
                    "peak_allocated_gb": allocated_per_rank,
                    "peak_reserved_gb": reserved_per_rank,
                    "peak_nvidia_smi_gb": smi_per_rank,
                },
            }
            output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        if process_group_initialized:
            dist.barrier()
        return 0
    finally:
        if process_group_initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
