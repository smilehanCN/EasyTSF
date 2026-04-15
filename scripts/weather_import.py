from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - script-only dependency guard
    xr = None
    XR_IMPORT_ERROR = exc
else:
    XR_IMPORT_ERROR = None


DEFAULT_WEATHER_STORAGE_FORMAT = "weather_sharded_npy_v1"
DEFAULT_WEATHER_ARTIFACTS_VERSION = "v1"

SUPPORTED_SOURCE_FORMATS = {
    "weatherbench_netcdf",
    "xarray_netcdf",
    "xarray_zarr",
}


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    base_variable: str
    level: int | float | str | None
    source_variable: str
    role: str
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_variable": self.base_variable,
            "level": self.level,
            "source_variable": self.source_variable,
            "role": self.role,
            "index": self.index,
        }


def _require_xarray() -> None:
    if xr is None:
        raise ImportError(
            "xarray is required for scripts/weather_import.py. Install the optional weather-preprocessing dependencies first."
        ) from XR_IMPORT_ERROR


def dump_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=False)
        handle.write("\n")


def format_channel_level(level: int | float | str | None) -> str | None:
    if level is None:
        return None
    if isinstance(level, str):
        return level
    if isinstance(level, float) and level.is_integer():
        return str(int(level))
    return str(level)


def build_channel_name(base_variable: str, level: int | float | str | None = None) -> str:
    formatted_level = format_channel_level(level)
    if formatted_level is None:
        return str(base_variable)
    return "{}_{}".format(base_variable, formatted_level)


def _parse_structured_arg(raw_value: str | None, *, default: Any) -> Any:
    if raw_value is None:
        return default
    parsed = yaml.safe_load(raw_value)
    return default if parsed is None else parsed


def _normalize_variable_list(raw_variables: list[str] | tuple[str, ...] | None, field_name: str) -> list[str]:
    if raw_variables is None:
        raise ValueError("{} is required".format(field_name))
    if not isinstance(raw_variables, (list, tuple)):
        raise ValueError("{} must be a list of variable names".format(field_name))
    variables = [str(item) for item in raw_variables]
    if len(variables) == 0:
        raise ValueError("{} must not be empty".format(field_name))
    return variables


def _normalize_levels(raw_levels: dict[str, Any] | None) -> dict[str, list[Any]]:
    if raw_levels is None:
        return {}
    if not isinstance(raw_levels, dict):
        raise ValueError("levels must be a mapping from variable name to list of levels")
    normalized = {}
    for variable_name, level_values in raw_levels.items():
        if level_values is None:
            normalized[str(variable_name)] = []
        elif isinstance(level_values, (list, tuple)):
            normalized[str(variable_name)] = list(level_values)
        else:
            normalized[str(variable_name)] = [level_values]
    return normalized


def _normalize_split_spec(split_spec: dict[str, Any]) -> dict[str, tuple[str | None, str | None]]:
    if not isinstance(split_spec, dict):
        raise ValueError("split_spec must be a mapping")

    normalized = {}
    for split_name in ("train", "val", "test"):
        if split_name not in split_spec:
            raise ValueError("split_spec must define '{}'".format(split_name))
        split_value = split_spec[split_name]
        if isinstance(split_value, dict):
            start = split_value.get("start")
            end = split_value.get("end")
        elif isinstance(split_value, (list, tuple)) and len(split_value) == 2:
            start, end = split_value
        else:
            raise ValueError(
                "split_spec['{}'] must be a dict with start/end or a 2-item list".format(
                    split_name,
                )
            )
        normalized[split_name] = (
            None if start is None else str(start),
            None if end is None else str(end),
        )
    return normalized


def _normalize_regrid_shape(raw_shape: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if raw_shape is None:
        return None
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        raise ValueError("regrid_shape must be a 2-item list like [lat_size, lon_size]")
    lat_size = int(raw_shape[0])
    lon_size = int(raw_shape[1])
    if lat_size <= 0 or lon_size <= 0:
        raise ValueError("regrid_shape values must be > 0")
    return lat_size, lon_size


def _open_weatherbench_variable_dir(source_dir: Path, variable_name: str) -> "xr.Dataset":
    _require_xarray()
    variable_dir = source_dir / variable_name
    if not variable_dir.is_dir():
        raise ValueError("missing variable directory '{}' under '{}'".format(variable_name, source_dir))

    paths = sorted(variable_dir.glob("*.nc"))
    if len(paths) == 0:
        raise ValueError("no NetCDF files found for '{}' under '{}'".format(variable_name, variable_dir))

    datasets = [xr.open_dataset(path) for path in paths]
    ds = xr.combine_by_coords(datasets)
    if variable_name not in ds.data_vars:
        if len(ds.data_vars) != 1:
            raise ValueError(
                "expected '{}' in {}, but found data variables {}".format(
                    variable_name,
                    variable_dir,
                    list(ds.data_vars),
                )
            )
        only_variable = next(iter(ds.data_vars))
        ds = ds.rename({only_variable: variable_name})
    return ds[[variable_name]]


def open_weather_source_dataset(source_format: str, source_path: str | Path, variables: list[str]) -> "xr.Dataset":
    _require_xarray()
    source_format = str(source_format)
    source_path = Path(source_path).expanduser()
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            "unsupported source_format '{}'; supported formats are {}".format(
                source_format,
                sorted(SUPPORTED_SOURCE_FORMATS),
            )
        )

    unique_variables = list(dict.fromkeys(str(item) for item in variables))
    if source_format == "weatherbench_netcdf":
        datasets = [
            _open_weatherbench_variable_dir(source_path, variable_name)
            for variable_name in unique_variables
        ]
        return xr.merge(datasets, compat="override")
    if source_format == "xarray_zarr":
        return xr.open_zarr(source_path)
    return xr.open_dataset(source_path)


def open_weather_static_dataset(
    source_format: str,
    source_path: str | Path,
    variables: list[str],
) -> "xr.Dataset":
    _require_xarray()
    if len(variables) == 0:
        raise ValueError("variables must not be empty when loading static fields")

    source_format = str(source_format)
    source_path = Path(source_path).expanduser()
    unique_variables = list(dict.fromkeys(str(item) for item in variables))

    if source_format == "weatherbench_netcdf":
        candidate_paths = []
        constants_path = source_path / "constants.nc"
        if constants_path.is_file():
            candidate_paths.append(constants_path)
        candidate_paths.extend(
            path
            for path in sorted(source_path.glob("*.nc"))
            if path.name != "constants.nc"
        )
        if len(candidate_paths) == 0:
            raise ValueError("static_variables were requested, but no static NetCDF files were found under '{}'".format(source_path))
        datasets = [xr.open_dataset(path) for path in candidate_paths]
        ds = xr.merge(datasets, compat="override")
    elif source_format == "xarray_zarr":
        ds = xr.open_zarr(source_path)
    else:
        ds = xr.open_dataset(source_path)

    missing_variables = sorted(variable_name for variable_name in unique_variables if variable_name not in ds.data_vars)
    if missing_variables:
        raise ValueError("static dataset is missing variables {}".format(missing_variables))
    return ds[unique_variables]


def _standardize_grid_dataset(ds: "xr.Dataset", *, require_valid_time: bool) -> "xr.Dataset":
    rename_map = {}
    if "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    if require_valid_time and "time" in ds.coords and "valid_time" not in ds.coords:
        if "prediction_timedelta" in ds.coords:
            raise ValueError(
                "forecast-style datasets with prediction_timedelta are not supported as training sources"
            )
        rename_map["time"] = "valid_time"
    if rename_map:
        ds = ds.rename(rename_map)

    required_coords = {"latitude", "longitude"}
    if require_valid_time:
        required_coords.add("valid_time")
    missing = sorted(coord_name for coord_name in required_coords if coord_name not in ds.coords)
    if missing:
        raise ValueError("dataset is missing required coordinates {}".format(missing))

    if require_valid_time:
        ds = ds.sortby("valid_time")
    ds = ds.sortby("latitude")
    ds = ds.sortby("longitude")
    return ds


def _standardize_dataset(ds: "xr.Dataset") -> "xr.Dataset":
    ds = _standardize_grid_dataset(ds, require_valid_time=True)
    return ds.assign_coords(valid_time=np.asarray(ds["valid_time"].values).astype("datetime64[ns]"))


def _standardize_static_dataset(ds: "xr.Dataset") -> "xr.Dataset":
    return _standardize_grid_dataset(ds, require_valid_time=False)


def _maybe_regrid_dataset(
    ds: "xr.Dataset",
    regrid_shape: tuple[int, int] | None,
) -> "xr.Dataset":
    if regrid_shape is None:
        return ds
    lat_size, lon_size = regrid_shape
    target_latitude = np.linspace(
        float(ds["latitude"].values[0]),
        float(ds["latitude"].values[-1]),
        lat_size,
        dtype=np.float32,
    )
    target_longitude = np.linspace(
        float(ds["longitude"].values[0]),
        float(ds["longitude"].values[-1]),
        lon_size,
        dtype=np.float32,
    )
    source_latitude = np.asarray(ds["latitude"].values, dtype=np.float32)
    source_longitude = np.asarray(ds["longitude"].values, dtype=np.float32)
    latitude_indices = np.abs(source_latitude[:, None] - target_latitude[None, :]).argmin(axis=0)
    longitude_indices = np.abs(source_longitude[:, None] - target_longitude[None, :]).argmin(axis=0)
    regridded = ds.isel(
        latitude=xr.DataArray(latitude_indices, dims="latitude"),
        longitude=xr.DataArray(longitude_indices, dims="longitude"),
    )
    return regridded.assign_coords(latitude=target_latitude, longitude=target_longitude)


def _build_channel_specs(
    ds: "xr.Dataset",
    input_variables: list[str],
    target_variables: list[str],
    levels: dict[str, list[Any]],
) -> tuple[list[ChannelSpec], list[str]]:
    missing_input = sorted(variable_name for variable_name in input_variables if variable_name not in ds.data_vars)
    if missing_input:
        raise ValueError("dataset is missing input variables {}".format(missing_input))

    missing_target = sorted(variable_name for variable_name in target_variables if variable_name not in input_variables)
    if missing_target:
        raise ValueError(
            "target_variables must be a subset of input_variables; missing {}".format(missing_target)
        )

    channels = []
    target_channel_names = []
    next_index = 0
    target_variable_names = set(target_variables)

    for variable_name in input_variables:
        data_array = ds[variable_name]
        requested_levels = levels.get(variable_name)
        if "level" in data_array.dims:
            available_levels = [level.item() if hasattr(level, "item") else level for level in data_array["level"].values]
            selected_levels = available_levels if not requested_levels else requested_levels
            missing_levels = sorted(level for level in selected_levels if level not in available_levels)
            if missing_levels:
                raise ValueError(
                    "variable '{}' is missing requested levels {}; available levels are {}".format(
                        variable_name,
                        missing_levels,
                        available_levels,
                    )
                )
            for level in selected_levels:
                channel_name = build_channel_name(variable_name, level)
                role = "input_target" if variable_name in target_variable_names else "input_only"
                channels.append(
                    ChannelSpec(
                        name=channel_name,
                        base_variable=variable_name,
                        level=level.item() if hasattr(level, "item") else level,
                        source_variable=variable_name,
                        role=role,
                        index=next_index,
                    )
                )
                if role == "input_target":
                    target_channel_names.append(channel_name)
                next_index += 1
        else:
            if requested_levels:
                raise ValueError(
                    "variable '{}' does not have a level dimension but levels were requested".format(
                        variable_name,
                    )
                )
            channel_name = build_channel_name(variable_name)
            role = "input_target" if variable_name in target_variable_names else "input_only"
            channels.append(
                ChannelSpec(
                    name=channel_name,
                    base_variable=variable_name,
                    level=None,
                    source_variable=variable_name,
                    role=role,
                    index=next_index,
                )
            )
            if role == "input_target":
                target_channel_names.append(channel_name)
            next_index += 1

    return channels, target_channel_names


def _build_static_artifacts(
    static_ds: "xr.Dataset | None",
    static_variables: list[str],
    grid_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if len(static_variables) == 0:
        return [], np.zeros((0, grid_shape[0], grid_shape[1]), dtype=np.float32)
    if static_ds is None:
        raise ValueError("static_variables were configured, but no static dataset was loaded")

    static_channels = []
    static_arrays = []
    for channel_index, variable_name in enumerate(static_variables):
        if variable_name not in static_ds.data_vars:
            raise ValueError("static dataset is missing variable '{}'".format(variable_name))
        data_array = static_ds[variable_name]
        time_dims = [dim_name for dim_name in data_array.dims if dim_name in {"valid_time", "time"}]
        if time_dims:
            raise ValueError("static variable '{}' must not have time dimensions".format(variable_name))
        extra_dims = [dim_name for dim_name in data_array.dims if dim_name not in {"latitude", "longitude"}]
        if extra_dims:
            raise ValueError(
                "static variable '{}' has unsupported dimensions {}; only latitude/longitude are allowed".format(
                    variable_name,
                    extra_dims,
                )
            )
        static_arrays.append(
            np.asarray(
                data_array.transpose("latitude", "longitude").to_numpy(),
                dtype=np.float32,
            )[None, :, :]
        )
        static_channels.append(
            {
                "name": str(variable_name),
                "source_variable": str(variable_name),
                "index": channel_index,
            }
        )

    static_array = np.concatenate(static_arrays, axis=0)
    if tuple(static_array.shape[1:]) != tuple(grid_shape):
        raise ValueError(
            "static grid shape {} does not match dynamic grid shape {}".format(
                tuple(static_array.shape[1:]),
                tuple(grid_shape),
            )
        )
    return static_channels, static_array


def _select_split_dataset(ds: "xr.Dataset", split_range: tuple[str | None, str | None]) -> "xr.Dataset":
    start, end = split_range
    if start is not None:
        start = np.datetime64(start, "ns")
    if end is not None:
        end = np.datetime64(end, "ns")
    return ds.sel(valid_time=slice(start, end))


def _validate_disjoint_split_timestamps(
    split_name: str,
    timestamps: np.ndarray,
    previous_split_timestamps: dict[str, np.ndarray],
) -> None:
    timestamp_ns = np.asarray(timestamps, dtype="datetime64[ns]").view(np.int64)
    for previous_split_name, previous_timestamp_ns in previous_split_timestamps.items():
        overlap = np.intersect1d(previous_timestamp_ns, timestamp_ns)
        if overlap.size == 0:
            continue
        first_overlap = np.asarray(overlap[:1], dtype=np.int64).view("datetime64[ns]")[0]
        raise ValueError(
            "split '{}' overlaps with split '{}' at {}; split_spec ranges must not share timestamps".format(
                split_name,
                previous_split_name,
                str(first_overlap),
            )
        )


def _stack_shard(
    split_ds: "xr.Dataset",
    channels: list[ChannelSpec],
    start_index: int,
    stop_index: int,
) -> np.ndarray:
    shard_channels = []
    time_slice = slice(start_index, stop_index)
    for channel in channels:
        data_array = split_ds[channel.base_variable].isel(valid_time=time_slice)
        if channel.level is not None:
            data_array = data_array.sel(level=channel.level)
        data_array = data_array.transpose("valid_time", "latitude", "longitude")
        array = np.asarray(data_array.to_numpy(), dtype=np.float32)
        shard_channels.append(array[:, None, :, :])
    return np.concatenate(shard_channels, axis=1)


def _compute_split_climatology(
    split_ds: "xr.Dataset",
    channels: list[ChannelSpec],
) -> dict[str, np.ndarray]:
    climatology = {}
    for channel in channels:
        data_array = split_ds[channel.base_variable]
        if channel.level is not None:
            data_array = data_array.sel(level=channel.level)
        climatology[channel.name] = np.asarray(
            data_array.transpose("valid_time", "latitude", "longitude").mean(dim="valid_time").to_numpy(),
            dtype=np.float32,
        )[None, :, :]
    return climatology


def _infer_source_resolution(ds: "xr.Dataset") -> str:
    return "{}x{}".format(ds.sizes["latitude"], ds.sizes["longitude"])


def _infer_frequency_minutes(valid_times: np.ndarray) -> int:
    timestamps = np.asarray(valid_times, dtype="datetime64[ns]")
    if timestamps.size < 2:
        raise ValueError("at least two timestamps are required to infer frequency_minutes")
    deltas = np.diff(timestamps.astype("datetime64[m]").astype(np.int64))
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        raise ValueError("failed to infer frequency_minutes from timestamps")
    return int(positive_deltas[0])


def write_canonical_weather_dataset(
    ds: "xr.Dataset",
    out_dir: str | Path,
    input_variables: list[str],
    target_variables: list[str],
    *,
    static_ds: "xr.Dataset | None" = None,
    static_variables: list[str] | None = None,
    levels: dict[str, list[Any]] | None = None,
    resample_freq: str | None = None,
    split_spec: dict[str, Any],
    shard_len: int,
    source_format: str,
    source_resolution: str | None = None,
    regrid_shape: list[int] | tuple[int, int] | None = None,
    artifacts_version: str = DEFAULT_WEATHER_ARTIFACTS_VERSION,
    storage_format: str = DEFAULT_WEATHER_STORAGE_FORMAT,
) -> dict[str, Any]:
    if shard_len <= 0:
        raise ValueError("shard_len must be > 0")

    levels = _normalize_levels(levels)
    split_spec = _normalize_split_spec(split_spec)
    regrid_shape = _normalize_regrid_shape(regrid_shape)
    if static_variables is None:
        static_variables = []
    elif len(static_variables) > 0:
        static_variables = _normalize_variable_list(static_variables, "static_variables")
    else:
        static_variables = []

    ds = _standardize_dataset(ds)
    inferred_source_resolution = str(source_resolution or _infer_source_resolution(ds))
    if resample_freq is not None:
        ds = ds.resample(valid_time=str(resample_freq)).first()
        ds = ds.dropna(dim="valid_time", how="all")
    ds = _maybe_regrid_dataset(ds, regrid_shape)

    if static_ds is not None:
        static_ds = _standardize_static_dataset(static_ds)
        static_ds = _maybe_regrid_dataset(static_ds, regrid_shape)

    channels, target_channel_names = _build_channel_specs(
        ds=ds,
        input_variables=input_variables,
        target_variables=target_variables,
        levels=levels,
    )
    target_channels = [channel for channel in channels if channel.name in set(target_channel_names)]
    grid_shape = (int(ds.sizes["latitude"]), int(ds.sizes["longitude"]))
    static_channels, static_array = _build_static_artifacts(static_ds, static_variables, grid_shape)

    dataset_dir = Path(out_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "latitude.npy", np.asarray(ds["latitude"].values, dtype=np.float32))
    np.save(dataset_dir / "longitude.npy", np.asarray(ds["longitude"].values, dtype=np.float32))
    np.save(dataset_dir / "static.npy", static_array.astype(np.float32, copy=False))
    dump_json(dataset_dir / "channels.json", [channel.to_dict() for channel in channels])
    dump_json(dataset_dir / "static_channels.json", static_channels)

    channel_sums = np.zeros(len(channels), dtype=np.float64)
    channel_sum_squares = np.zeros(len(channels), dtype=np.float64)
    total_value_count = 0
    split_lengths = {}
    split_timestamps_ns = {}

    for split_name in ("train", "val", "test"):
        split_dir = dataset_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_ds = _select_split_dataset(ds, split_spec[split_name])
        timestamps = np.asarray(split_ds["valid_time"].values, dtype="datetime64[ns]")
        if timestamps.size == 0:
            raise ValueError("split '{}' is empty after applying split_spec".format(split_name))
        _validate_disjoint_split_timestamps(split_name, timestamps, split_timestamps_ns)
        split_timestamps_ns[split_name] = timestamps.view(np.int64)

        np.save(split_dir / "timestamps.npy", timestamps)
        np.savez(split_dir / "climatology.npz", **_compute_split_climatology(split_ds, target_channels))
        split_lengths[split_name] = int(timestamps.size)

        files = []
        shard_start_global = 0
        for shard_id, start_index in enumerate(range(0, timestamps.size, shard_len)):
            stop_index = min(start_index + shard_len, timestamps.size)
            shard = _stack_shard(split_ds, channels, start_index, stop_index)
            shard_path = "{:06d}.npy".format(shard_id)
            np.save(split_dir / shard_path, shard)
            files.append(
                {
                    "path": shard_path,
                    "length": int(stop_index - start_index),
                    "start": int(shard_start_global),
                    "stop": int(shard_start_global + (stop_index - start_index)),
                }
            )
            shard_start_global += stop_index - start_index

            if split_name == "train":
                shard_float64 = shard.astype(np.float64, copy=False)
                channel_sums += shard_float64.sum(axis=(0, 2, 3))
                channel_sum_squares += np.square(shard_float64).sum(axis=(0, 2, 3))
                total_value_count += shard.shape[0] * shard.shape[2] * shard.shape[3]

        dump_json(
            split_dir / "manifest.json",
            {
                "split": split_name,
                "num_steps": int(timestamps.size),
                "num_shards": len(files),
                "shard_len": int(shard_len),
                "files": files,
            },
        )

    if total_value_count <= 0:
        raise ValueError("training split is empty; cannot compute channel statistics")

    channel_means = channel_sums / float(total_value_count)
    channel_variances = (channel_sum_squares / float(total_value_count)) - np.square(channel_means)
    channel_variances = np.maximum(channel_variances, 1e-12)
    channel_stds = np.sqrt(channel_variances)
    np.savez(
        dataset_dir / "stats.npz",
        mean=channel_means.astype(np.float32)[None, :, None, None],
        std=channel_stds.astype(np.float32)[None, :, None, None],
    )

    meta = {
        "storage_format": storage_format,
        "dtype": "float32",
        "shape": [
            int(sum(split_lengths.values())),
            int(len(channels)),
            int(grid_shape[0]),
            int(grid_shape[1]),
        ],
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "frequency_minutes": _infer_frequency_minutes(ds["valid_time"].values),
        "input_channels": [channel.name for channel in channels],
        "target_channels": target_channel_names,
        "static_channels": [channel["name"] for channel in static_channels],
        "split_lengths": split_lengths,
        "shard_len": int(shard_len),
        "source_format": str(source_format),
        "source_resolution": inferred_source_resolution,
        "artifacts_version": str(artifacts_version),
    }
    dump_json(dataset_dir / "meta.json", meta)
    return meta


def import_weather_dataset(
    *,
    source_format: str,
    source: str | Path,
    out_dir: str | Path,
    input_variables: list[str],
    target_variables: list[str] | None,
    static_variables: list[str] | None,
    levels: dict[str, list[Any]] | None,
    resample_freq: str | None,
    split_spec: dict[str, Any],
    shard_len: int,
    source_resolution: str | None = None,
    regrid_shape: list[int] | tuple[int, int] | None = None,
) -> dict[str, Any]:
    input_variables = _normalize_variable_list(input_variables, "input_variables")
    if target_variables is None:
        target_variables = list(input_variables)
    target_variables = _normalize_variable_list(target_variables, "target_variables")
    if static_variables is None:
        static_variables = []
    elif len(static_variables) > 0:
        static_variables = _normalize_variable_list(static_variables, "static_variables")
    else:
        static_variables = []
    levels = _normalize_levels(levels)

    source_dataset = open_weather_source_dataset(
        source_format=source_format,
        source_path=source,
        variables=input_variables,
    )
    static_dataset = None
    if len(static_variables) > 0:
        static_dataset = open_weather_static_dataset(
            source_format=source_format,
            source_path=source,
            variables=static_variables,
        )
    return write_canonical_weather_dataset(
        ds=source_dataset,
        static_ds=static_dataset,
        out_dir=out_dir,
        input_variables=input_variables,
        target_variables=target_variables,
        static_variables=static_variables,
        levels=levels,
        resample_freq=resample_freq,
        split_spec=split_spec,
        shard_len=shard_len,
        source_format=source_format,
        source_resolution=source_resolution,
        regrid_shape=regrid_shape,
    )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import WeatherBench-style data into EasyTSF weather shards.")
    parser.add_argument(
        "--source-format",
        required=True,
        choices=sorted(SUPPORTED_SOURCE_FORMATS),
        help="Input format. Use weatherbench_netcdf for WB1-style variable/year NetCDF directories.",
    )
    parser.add_argument("--source", required=True, help="Path to the source directory, NetCDF file, or Zarr store.")
    parser.add_argument("--out-dir", required=True, help="Output directory for the EasyTSF weather dataset.")
    parser.add_argument(
        "--input-variables",
        required=True,
        help="YAML/JSON list of input variable names.",
    )
    parser.add_argument(
        "--target-variables",
        default=None,
        help="Optional YAML/JSON list of target variable names. Defaults to input_variables.",
    )
    parser.add_argument(
        "--static-variables",
        default="[]",
        help="Optional YAML/JSON list of static variable names to load from constants.nc or the source dataset.",
    )
    parser.add_argument(
        "--levels",
        default="{}",
        help="Optional YAML/JSON mapping from variable name to a list of levels.",
    )
    parser.add_argument(
        "--resample-freq",
        default=None,
        help="Optional xarray resample frequency such as '6h'.",
    )
    parser.add_argument(
        "--split-spec",
        required=True,
        help="YAML/JSON mapping for train/val/test. Each split must define [start, end] or {start, end}.",
    )
    parser.add_argument("--shard-len", type=int, required=True, help="Number of time steps per shard.")
    parser.add_argument(
        "--source-resolution",
        default=None,
        help="Optional source resolution label to record in meta.json.",
    )
    parser.add_argument(
        "--regrid-shape",
        default=None,
        help="Optional YAML/JSON [latitude_size, longitude_size] grid to interpolate onto.",
    )
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    import_weather_dataset(
        source_format=args.source_format,
        source=args.source,
        out_dir=args.out_dir,
        input_variables=_parse_structured_arg(args.input_variables, default=[]),
        target_variables=_parse_structured_arg(args.target_variables, default=None),
        static_variables=_parse_structured_arg(args.static_variables, default=[]),
        levels=_parse_structured_arg(args.levels, default={}),
        resample_freq=args.resample_freq,
        split_spec=_parse_structured_arg(args.split_spec, default={}),
        shard_len=args.shard_len,
        source_resolution=args.source_resolution,
        regrid_shape=_parse_structured_arg(args.regrid_shape, default=None),
    )


if __name__ == "__main__":
    main()
