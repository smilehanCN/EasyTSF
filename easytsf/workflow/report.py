import csv
import math
from pathlib import Path


def _extract_scalar_metric(value):
    if value in {None, ""}:
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        value = float(value)
        return None if math.isnan(value) else value
    except (TypeError, ValueError):
        return value


def find_checkpoint_path(exp_dir, name="best"):
    ckpt_dir = Path(exp_dir) / "checkpoints"
    if name == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
        return str(ckpt_path.resolve()) if ckpt_path.exists() else None
    candidates = sorted(path for path in ckpt_dir.glob("*.ckpt") if path.name != "last.ckpt")
    return str(candidates[0].resolve()) if candidates else None


def load_saved_metrics(conf):
    path = Path(conf["exp_dir"]) / "metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def last_logged_value(name):
        for row in reversed(rows):
            value = row.get(name)
            if value not in {None, ""}:
                return _extract_scalar_metric(value)
        return None

    return {
        conf["val_metric"]: last_logged_value(conf["val_metric"]),
        "ckpt_path": find_checkpoint_path(conf["exp_dir"], "best"),
        "exp_dir": str(Path(conf["exp_dir"]).resolve()),
        "mae": last_logged_value("test/mae"),
        "mse": last_logged_value("test/mse"),
        "rmse": last_logged_value("test/rmse"),
    }
