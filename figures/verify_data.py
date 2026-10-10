"""Check completeness and manuscript-level checkpoints for bundled data."""

from __future__ import annotations

from plot_from_csv import GC, GT, K, VIT_GC, grid, records, values_by_cell
from validate_grid import validate_cnn_rows, validate_vit_rows


# Printed Table 3 cells: highest three-seed mean / highest individual run,
# selected separately within each family. These are manuscript checkpoints,
# independent of the bundled CSV and its source-file names.
PAPER_VIT_SUMMARY = {
    "baseline": ((48.2, 72.8), (100.0, 100.0), (71.6, 72.3)),
    "allnorm_pln": ((75.3, 92.1), (100.0, 100.0), (72.3, 72.9)),
    "allnorm_mix": ((89.5, 91.2), (100.0, 100.0), (73.5, 74.5)),
    "activation_pln": ((73.7, 97.3), (99.4, 100.0), (69.7, 71.0)),
    "activation_mix": ((91.7, 98.2), (100.0, 100.0), (70.2, 70.7)),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(label: str, observed: float, expected: float, digits: int = 1) -> None:
    if round(observed, digits) != expected:
        raise AssertionError(f"{label}: {observed:.6f}, expected {expected}")


def verify_vit_summary(vit: list[dict[str, str]]) -> None:
    metrics = (("random20k", "best_fit_pct"),
               ("clean50k", "final_train_pct"),
               ("clean50k", "final_test_pct"))
    for family, printed_cells in PAPER_VIT_SUMMARY.items():
        mode = "activation" if family.startswith("activation") else "allnorm"
        for (dataset, metric), (expected_mean, expected_run) in zip(metrics, printed_cells):
            rows = [r for r in vit if r["dataset"] == dataset and r["mode"] == mode]
            cells = values_by_cell(rows, ("token_groups", "channel_groups"), metric)
            if family == "baseline":
                selected = [values for (gt, gc), values in cells.items()
                            if gt == "197" and gc == "1"]
            elif family.endswith("_pln"):
                selected = [values for (gt, gc), values in cells.items()
                            if gt == "197" and gc != "1"]
            else:
                selected = [values for (gt, _), values in cells.items()
                            if gt != "197"]
            require(bool(selected), f"ViT table family {family} is empty")
            best_mean = max(sum(values) / len(values) for values in selected)
            best_run = max(value for values in selected for value in values)
            close(f"ViT table {family} {dataset} {metric} mean",
                  best_mean, expected_mean)
            close(f"ViT table {family} {dataset} {metric} run",
                  best_run, expected_run)


def verify_cnn_summary(cnn: list[dict[str, str]]) -> None:
    for dataset, metric in (("random50k", "final_train_pct"),
                            ("clean50k", "final_train_pct"),
                            ("clean50k", "final_test_pct")):
        rows = [r for r in cnn if r["dataset"] == dataset]
        matrix, _ = grid(rows, ("channel_groups", "spatial_splits"),
                         metric, GC, K)
        require(matrix.shape == (8, 6),
                f"CNN {dataset} grid has shape {matrix.shape}, expected (8, 6)")
        if dataset == "random50k":
            close("CNN random-label LN", matrix[0, 0], 24.6)
            close("CNN random-label best GN", max(matrix[:, 0]), 95.8)
            close("CNN random-label PLN[8]", matrix[GC.index(16), K.index(32)], 100.0)
            close("CNN random-label PLN[4]", matrix[GC.index(32), K.index(32)], 100.0)


def main() -> None:
    cnn = records("cnn_runs.csv")
    vit = records("vit_runs.csv")
    require(len(cnn) == 288, f"expected 288 CNN records, got {len(cnn)}")
    require(len(vit) == 1377, f"expected 1377 ViT records, got {len(vit)}")
    validate_cnn_rows(cnn, GC, K)
    validate_vit_rows(vit, GT, VIT_GC)
    verify_cnn_summary(cnn)

    checks = (
        ("random6k", "allnorm", "best_fit_pct", 85.9, 99.4, 99.5),
        ("random20k", "allnorm", "best_fit_pct", 48.2, 75.3, 89.5),
        # The underlying mean is exactly 73.55; the manuscript prints 73.5.
        ("clean50k", "allnorm", "final_test_pct", 71.6, 72.3, 73.55),
    )
    for dataset, mode, metric, ln, pln, cross in checks:
        rows = [r for r in vit if r["dataset"] == dataset and r["mode"] == mode]
        matrix, _ = grid(rows, ("token_groups", "channel_groups"),
                         metric, GT, VIT_GC[mode])
        close(f"{dataset} LN", matrix[0, 0], ln)
        close(f"{dataset} PLN", max(matrix[0]), pln)
        close(f"{dataset} Mix-Norm", max(matrix[1:].flat), cross,
              digits=2 if dataset == "clean50k" else 1)
    verify_vit_summary(vit)
    print("Verified complete three-seed grids and paper-level checkpoints")


if __name__ == "__main__":
    main()
