"""Small, dependency-free checks for the released per-seed figure records."""

from __future__ import annotations

from collections import Counter
import math
from itertools import product
from typing import Iterable, Mapping, Sequence


def require_exact_keys(
    rows: Sequence[Mapping[str, str]],
    fields: Sequence[str],
    expected: Iterable[tuple[str, ...]],
    label: str,
) -> None:
    """Require exactly one record for every specified experiment/seed cell."""
    expected_rows = list(expected)
    expected_keys = set(expected_rows)
    if len(expected_keys) != len(expected_rows):
        raise ValueError(f"{label}: duplicate keys in expected grid")
    observed = Counter(tuple(row[field] for field in fields) for row in rows)
    missing = expected_keys - observed.keys()
    extra = observed.keys() - expected_keys
    duplicates = [key for key, count in observed.items() if count != 1]
    if missing or extra or duplicates:
        details = []
        for name, keys in (("missing", missing), ("unexpected", extra),
                           ("duplicate", duplicates)):
            if keys:
                details.append(f"{name}={len(keys)} example={min(keys)!r}")
        raise ValueError(f"{label}: incomplete or repeated per-seed grid: "
                         + "; ".join(details))


def require_percentages(
    rows: Sequence[Mapping[str, str]],
    fields: Sequence[str],
    label: str,
) -> None:
    for index, row in enumerate(rows, start=1):
        for field in fields:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{label}: row {index} has invalid {field}") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"{label}: row {index} has out-of-range {field}={value}")


def require_unique_sources(rows: Sequence[Mapping[str, str]], label: str) -> None:
    sources = [row.get("source_file", "") for row in rows]
    if any(not source for source in sources) or len(set(sources)) != len(sources):
        raise ValueError(f"{label}: source_file must be nonempty and unique per run")


def validate_cnn_rows(rows: Sequence[Mapping[str, str]], gc: Sequence[int],
                      spatial_splits: Sequence[int]) -> None:
    require_exact_keys(
        rows, ("dataset", "seed", "channel_groups", "spatial_splits"),
        product(("clean50k", "random50k"), ("0", "1", "2"),
                map(str, gc), map(str, spatial_splits)), "CNN",
    )
    require_unique_sources(rows, "CNN")
    require_percentages(rows, ("final_train_pct",), "CNN")
    require_percentages([row for row in rows if row["dataset"] == "clean50k"],
                        ("final_test_pct",), "CNN clean-label")
    if any(row["final_test_pct"] != "" for row in rows
           if row["dataset"] == "random50k"):
        raise ValueError("CNN random-label test accuracy must be blank")


def validate_vit_rows(rows: Sequence[Mapping[str, str]], gt: Sequence[int],
                      vit_gc: Mapping[str, Sequence[int]]) -> None:
    require_exact_keys(
        rows, ("dataset", "mode", "seed", "token_groups", "channel_groups"),
        (key for dataset in ("clean50k", "random6k", "random20k")
         for mode in ("allnorm", "activation")
         for key in product((dataset,), (mode,), ("40", "41", "42"),
                            map(str, gt), map(str, vit_gc[mode]))), "ViT",
    )
    require_unique_sources(rows, "ViT")
    require_percentages(rows, ("best_fit_pct", "final_train_pct", "final_test_pct"),
                        "ViT")
    if any(row["complete"] != "True" for row in rows):
        raise ValueError("ViT records include an incomplete run")
