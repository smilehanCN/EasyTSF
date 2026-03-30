from .config import finalize_runtime_conf, load_experiment_config

__all__ = [
    "load_experiment_config",
    "finalize_runtime_conf",
    "run_experiment",
    "run_benchmark",
]


def __getattr__(name):
    if name == "run_experiment":
        from .experiment import run_experiment

        return run_experiment
    if name == "run_benchmark":
        from .benchmark import run_benchmark

        return run_benchmark
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
