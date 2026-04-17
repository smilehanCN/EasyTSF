#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CASE_SPACES = {
    "unet3d": [
        {"case_id": "lr1e-04_c12", "lr": 1e-4, "base_channels": 12},
        {"case_id": "lr1e-04_c16", "lr": 1e-4, "base_channels": 16},
        {"case_id": "lr1e-04_c20", "lr": 1e-4, "base_channels": 20},
        {"case_id": "lr5e-05_c12", "lr": 5e-5, "base_channels": 12},
        {"case_id": "lr5e-05_c16", "lr": 5e-5, "base_channels": 16},
        {"case_id": "lr5e-05_c20", "lr": 5e-5, "base_channels": 20},
    ],
    "fno3d": [
        {"case_id": "lr1e-04_w16", "lr": 1e-4, "fno_width": 16},
        {"case_id": "lr1e-04_w20", "lr": 1e-4, "fno_width": 20},
        {"case_id": "lr1e-04_w24", "lr": 1e-4, "fno_width": 24},
        {"case_id": "lr5e-05_w16", "lr": 5e-5, "fno_width": 16},
        {"case_id": "lr5e-05_w20", "lr": 5e-5, "fno_width": 20},
        {"case_id": "lr5e-05_w24", "lr": 5e-5, "fno_width": 24},
    ],
    "afno3d": [
        {"case_id": "lr1e-04_e48", "lr": 1e-4, "afno_embed_dim": 48},
        {"case_id": "lr1e-04_e64", "lr": 1e-4, "afno_embed_dim": 64},
        {"case_id": "lr1e-04_e80", "lr": 1e-4, "afno_embed_dim": 80},
        {"case_id": "lr5e-05_e48", "lr": 5e-5, "afno_embed_dim": 48},
        {"case_id": "lr5e-05_e64", "lr": 5e-5, "afno_embed_dim": 64},
        {"case_id": "lr5e-05_e80", "lr": 5e-5, "afno_embed_dim": 80},
    ],
}


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a direct WindShear h10_p10 case sweep without Ray.")
    parser.add_argument("--model", required=True, choices=sorted(CASE_SPACES))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--devices", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--report-out", required=True)
    return parser


def _read_metrics(metrics_path: Path) -> tuple[float | None, float | None, float | None]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    val_values = [float(row["val/loss"]) for row in rows if row.get("val/loss") not in {"", None}]
    test_mae = next((float(row["test/mae"]) for row in reversed(rows) if row.get("test/mae") not in {"", None}), None)
    test_mse = next((float(row["test/mse"]) for row in reversed(rows) if row.get("test/mse") not in {"", None}), None)
    return (min(val_values) if val_values else None, test_mae, test_mse)


def _read_hparams(hparams_path: Path) -> dict[str, object]:
    with hparams_path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def main() -> int:
    args = build_cli_parser().parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    strategy = "auto" if args.devices <= 1 else "ddp"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    records: list[dict[str, object]] = []
    for case in CASE_SPACES[args.model]:
        exp_dir = results_root / case["case_id"]
        command = []
        if args.devices > 1:
            command.extend(
                [
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node={}".format(args.devices),
                    "-m",
                    "easytsf.workflow.experiment",
                ]
            )
        else:
            command.extend([sys.executable, "-m", "easytsf.workflow.experiment"])

        command.extend(
            [
                args.experiment,
                "--set",
                "exp_dir='{}'".format(str(exp_dir)),
                "--set",
                "seed=42",
                "--set",
                "devices={}".format(args.devices),
                "--set",
                "batch_size={}".format(args.batch_size),
                "--set",
                "strategy={}".format(strategy),
                "--set",
                "precision={}".format(args.precision),
            ]
        )
        for key, value in case.items():
            if key == "case_id":
                continue
            command.extend(["--set", "{}={}".format(key, value)])

        subprocess.run(command, check=True, cwd=str(REPO_ROOT), env=env)

        metrics_path = exp_dir / "metrics.csv"
        hparams_path = exp_dir / "hparams.yaml"
        val_loss, test_mae, test_mse = _read_metrics(metrics_path)
        hparams = _read_hparams(hparams_path)
        record = {
            "model": hparams.get("model", args.model),
            "dataset": hparams.get("dataset"),
            "hist_len": hparams.get("hist_len"),
            "pred_len": hparams.get("pred_len"),
            "case_id": case["case_id"],
            "seed": hparams.get("seed"),
            "devices": hparams.get("devices"),
            "batch_size": hparams.get("batch_size"),
            "precision": hparams.get("precision", args.precision),
            "val/loss": val_loss,
            "test/mae": test_mae,
            "test/mse": test_mse,
        }
        for key, value in case.items():
            if key != "case_id":
                record[key] = value
        records.append(record)

    fieldnames = list(records[0].keys()) if records else []
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_path = report_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "run_tag": args.run_tag,
                "model": args.model,
                "devices": args.devices,
                "batch_size": args.batch_size,
                "effective_batch": args.devices * args.batch_size,
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
