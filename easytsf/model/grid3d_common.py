from __future__ import annotations


def as_tuple3(value: object, field_name: str, *, positive: bool = True) -> tuple[int, int, int]:
    if value is None:
        raise ValueError("Expected {} to be a length-3 tuple/list, got None".format(field_name))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        parsed = tuple(int(v) for v in value)
        if positive and any(v <= 0 for v in parsed):
            raise ValueError("Expected {} entries to be positive, got {!r}".format(field_name, value))
        return parsed
    raise ValueError("Expected {} to be a length-3 tuple/list, got {!r}".format(field_name, value))


def resolve_grid3d_output_channels(
    *,
    in_channels: int,
    out_channels: int | None,
    output_mode: str,
    risk_num_classes: int,
    risk_num_heads: int,
    model_name: str,
) -> tuple[int, int, int]:
    output_mode = str(output_mode)
    if output_mode not in {"regression", "classification"}:
        raise ValueError(
            "{} output_mode must be one of ['regression', 'classification'], got {}".format(
                model_name,
                output_mode,
            )
        )
    regression_channels = int(in_channels) if out_channels is None else int(out_channels)
    risk_num_classes = int(risk_num_classes)
    risk_num_heads = int(risk_num_heads)
    if regression_channels <= 0:
        raise ValueError("out_channels must be > 0")
    if risk_num_classes <= 0 or risk_num_heads <= 0:
        raise ValueError("risk_num_classes and risk_num_heads must be > 0")
    classification_channels = risk_num_classes * risk_num_heads
    output_channels = regression_channels if output_mode == "regression" else classification_channels
    return regression_channels, classification_channels, output_channels
