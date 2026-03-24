import csv
import statistics
from pathlib import Path

import yaml

from .experiment import (
    CONFIG_SECTIONS,
    STUDY_CONFIG_DIR,
    finalize_runtime_conf,
    load_config,
    load_saved_metrics,
    run_training,
    save_metrics,
    save_resolved_config,
)


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("study config must be a mapping: {}".format(path))
    return data


def _resolve_study_path(study_ref):
    ref_path = Path(study_ref).expanduser()
    if ref_path.exists():
        return ref_path.resolve()

    relative_ref = Path(study_ref)
    if relative_ref.suffix not in {".yaml", ".yml"}:
        relative_ref = relative_ref.with_suffix(".yaml")
    resolved_path = (STUDY_CONFIG_DIR / relative_ref).resolve()
    if resolved_path.exists():
        return resolved_path
    raise FileNotFoundError("study config not found: {}".format(study_ref))


def _empty_section_map():
    return {section: {} for section in CONFIG_SECTIONS}


def _normalize_case_overrides(data, source_name):
    if data is None:
        return _empty_section_map()
    if not isinstance(data, dict):
        raise ValueError("{} overrides must be a mapping".format(source_name))

    unknown_sections = set(data.keys()) - set(CONFIG_SECTIONS)
    if unknown_sections:
        raise ValueError("unsupported override sections in {}: {}".format(source_name, sorted(unknown_sections)))

    normalized = _empty_section_map()
    for section in CONFIG_SECTIONS:
        value = data.get(section, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("{} section '{}' must be a mapping".format(source_name, section))
        normalized[section] = dict(value)
    return normalized


def load_study(study_ref):
    study_path = _resolve_study_path(study_ref)
    raw_conf = _load_yaml(study_path)

    study_name = raw_conf.get("name") or study_path.stem
    seeds = raw_conf.get("seeds", [0])
    cases = raw_conf.get("cases", [])

    if not isinstance(study_name, str) or study_name.strip() == "":
        raise ValueError("study name must be a non-empty string: {}".format(study_path))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("study seeds must be a non-empty list: {}".format(study_path))
    if not isinstance(cases, list) or not cases:
        raise ValueError("study cases must be a non-empty list: {}".format(study_path))

    normalized_cases = []
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("study case {} must be a mapping".format(case_index))
        experiment_ref = case.get("experiment")
        if not experiment_ref:
            raise ValueError("study case {} missing experiment".format(case_index))
        normalized_cases.append(
            {
                "name": case.get("name"),
                "experiment": experiment_ref,
                "overrides": _normalize_case_overrides(case.get("overrides"), "study case {}".format(case_index)),
            }
        )

    return {
        "path": study_path,
        "name": study_name,
        "seeds": [int(seed) for seed in seeds],
        "cases": normalized_cases,
    }


def _default_case_name(conf):
    return "{}_{}for{}".format(conf["dataset_name"], conf["hist_len"], conf["pred_len"])


def _build_failure_record(conf, error):
    return {
        "task_name": conf.get("task_name", "mtsf"),
        "model_name": conf["model_name"],
        "dataset_name": conf["dataset_name"],
        "hist_len": int(conf["hist_len"]),
        "pred_len": int(conf["pred_len"]),
        "seed": int(conf["seed"]),
        "conf_hash": conf["conf_hash"],
        "exp_dir": str(Path(conf["exp_dir"]).resolve()),
        "ckpt_path": None,
        "status": "failed",
        "error": str(error),
        "val_metric_name": conf.get("val_metric"),
        "val_metric_value": None,
        "mae": None,
        "mse": None,
    }


def _build_study_row(study_name, case_index, case_name, experiment_ref, metrics, resume_hit):
    row = {
        "study_name": study_name,
        "case_index": case_index,
        "case_name": case_name,
        "experiment": experiment_ref,
        "resume_hit": bool(resume_hit),
    }
    row.update(metrics)
    return row


def _write_runs_report(study_dir, rows):
    runs_path = Path(study_dir) / "runs.csv"
    columns = [
        "study_name",
        "case_index",
        "case_name",
        "experiment",
        "resume_hit",
        "task_name",
        "model_name",
        "dataset_name",
        "hist_len",
        "pred_len",
        "seed",
        "status",
        "mae",
        "mse",
        "val_metric_name",
        "val_metric_value",
        "conf_hash",
        "ckpt_path",
        "exp_dir",
        "error",
    ]
    normalized_rows = []
    for row in rows:
        normalized = {column: row.get(column) for column in columns}
        normalized_rows.append(normalized)
    normalized_rows.sort(key=lambda row: (row["case_index"], row["seed"]))

    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return runs_path, normalized_rows


def _write_summary_report(study_dir, runs_rows):
    summary_path = Path(study_dir) / "summary.csv"
    summary_columns = [
        "task_name",
        "model_name",
        "dataset_name",
        "hist_len",
        "pred_len",
        "mae_mean",
        "mae_std",
        "mse_mean",
        "mse_std",
        "num_seeds",
    ]
    grouped = {}
    for row in runs_rows:
        if row.get("status") != "success":
            continue
        group_key = (row["task_name"], row["model_name"], row["dataset_name"], row["hist_len"], row["pred_len"])
        grouped.setdefault(group_key, {"mae": [], "mse": []})
        if row.get("mae") is not None:
            grouped[group_key]["mae"].append(float(row["mae"]))
        if row.get("mse") is not None:
            grouped[group_key]["mse"].append(float(row["mse"]))

    summary_rows = []
    for group_key in sorted(grouped):
        metric_group = grouped[group_key]
        mae_values = metric_group["mae"]
        mse_values = metric_group["mse"]
        summary_rows.append(
            {
                "task_name": group_key[0],
                "model_name": group_key[1],
                "dataset_name": group_key[2],
                "hist_len": group_key[3],
                "pred_len": group_key[4],
                "mae_mean": statistics.mean(mae_values) if mae_values else None,
                "mae_std": statistics.pstdev(mae_values) if len(mae_values) > 1 else 0.0 if mae_values else None,
                "mse_mean": statistics.mean(mse_values) if mse_values else None,
                "mse_std": statistics.pstdev(mse_values) if len(mse_values) > 1 else 0.0 if mse_values else None,
                "num_seeds": len(mae_values),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def run_study(
    study_ref,
    runtime_overrides=None,
    dry_run=False,
    resume=True,
    fail_fast=False,
):
    study_conf = load_study(study_ref)
    runtime_overrides = dict(runtime_overrides or {})
    study_dir = Path(runtime_overrides.get("save_root", "save")) / "studies" / study_conf["name"]
    study_dir.mkdir(parents=True, exist_ok=True)

    run_specs = []
    for case_index, case in enumerate(study_conf["cases"]):
        for seed in study_conf["seeds"]:
            run_conf = load_config(case["experiment"], overrides=case["overrides"])
            finalized_conf = finalize_runtime_conf(
                run_conf,
                overrides={**runtime_overrides, "seed": seed},
            )
            case_name = case["name"] or _default_case_name(finalized_conf)
            run_specs.append(
                {
                    "case_index": case_index,
                    "case_name": case_name,
                    "experiment": case["experiment"],
                    "conf": finalized_conf,
                }
            )

    if dry_run:
        for spec in run_specs:
            conf = spec["conf"]
            print(
                "[dry-run] case={} seed={} task={} experiment={} dataset={} hist_len={} pred_len={} exp_dir={}".format(
                    spec["case_name"],
                    conf["seed"],
                    conf.get("task_name", "mtsf"),
                    spec["experiment"],
                    conf["dataset_name"],
                    conf["hist_len"],
                    conf["pred_len"],
                    conf["exp_dir"],
                )
            )
        return {
            "study_name": study_conf["name"],
            "study_dir": str(study_dir.resolve()),
            "run_count": len(run_specs),
            "rows": [],
            "runs_path": None,
            "summary_path": None,
        }

    rows = []
    stop_error = None
    for spec in run_specs:
        conf = spec["conf"]
        existing_metrics = load_saved_metrics(conf) if resume else None
        if existing_metrics and existing_metrics.get("status") == "success":
            print("[resume] {} seed={} -> {}".format(spec["case_name"], conf["seed"], conf["exp_dir"]))
            rows.append(
                _build_study_row(
                    study_conf["name"],
                    spec["case_index"],
                    spec["case_name"],
                    spec["experiment"],
                    existing_metrics,
                    resume_hit=True,
                )
            )
            continue

        print("[run] {} seed={} -> {}".format(spec["case_name"], conf["seed"], conf["exp_dir"]))
        try:
            metrics = run_training(conf)
        except Exception as error:
            save_resolved_config(conf)
            metrics = _build_failure_record(conf, error)
            save_metrics(conf, metrics)
            print("[failed] {} seed={} error={}".format(spec["case_name"], conf["seed"], error))
            rows.append(
                _build_study_row(
                    study_conf["name"],
                    spec["case_index"],
                    spec["case_name"],
                    spec["experiment"],
                    metrics,
                    resume_hit=False,
                )
            )
            if fail_fast:
                stop_error = error
                break
            continue

        rows.append(
            _build_study_row(
                study_conf["name"],
                spec["case_index"],
                spec["case_name"],
                spec["experiment"],
                metrics,
                resume_hit=False,
            )
        )

    runs_path, runs_rows = _write_runs_report(study_dir, rows)
    summary_path = _write_summary_report(study_dir, runs_rows)
    result = {
        "study_name": study_conf["name"],
        "study_dir": str(study_dir.resolve()),
        "run_count": len(rows),
        "rows": rows,
        "runs_path": str(runs_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }
    if stop_error is not None:
        raise RuntimeError("study stopped because fail_fast=1") from stop_error
    return result
