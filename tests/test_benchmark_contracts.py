import textwrap

import pytest

from benchmark import load_benchmark_config
from tests.helpers import SMOKE_EXPERIMENT_PATH


def _write_benchmark(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_benchmark_module_must_define_benchmark(tmp_path):
    benchmark_path = _write_benchmark(tmp_path / "missing_benchmark.py", "not_benchmark = {}")

    with pytest.raises(ValueError, match="must define benchmark"):
        load_benchmark_config(str(benchmark_path))


def test_benchmark_requires_param_space(tmp_path):
    benchmark_path = _write_benchmark(
        tmp_path / "missing_param_space.py",
        """
        benchmark = {
            "name": "missing_param_space",
            "experiment": %r,
            "seeds": [0, 1],
        }
        """ % str(SMOKE_EXPERIMENT_PATH),
    )

    with pytest.raises(ValueError, match="param_space"):
        load_benchmark_config(str(benchmark_path))


def test_benchmark_param_space_must_not_define_seed(tmp_path):
    benchmark_path = _write_benchmark(
        tmp_path / "seed_in_param_space.py",
        """
        benchmark = {
            "name": "seed_in_param_space",
            "experiment": %r,
            "seeds": [0, 1],
            "param_space": {"seed": 0},
        }
        """ % str(SMOKE_EXPERIMENT_PATH),
    )

    with pytest.raises(ValueError, match="must not define seed"):
        load_benchmark_config(str(benchmark_path))


def test_benchmark_requires_non_empty_seeds(tmp_path):
    benchmark_path = _write_benchmark(
        tmp_path / "empty_seeds.py",
        """
        benchmark = {
            "name": "empty_seeds",
            "experiment": %r,
            "seeds": [],
            "param_space": {},
        }
        """ % str(SMOKE_EXPERIMENT_PATH),
    )

    with pytest.raises(ValueError, match="non-empty list"):
        load_benchmark_config(str(benchmark_path))
