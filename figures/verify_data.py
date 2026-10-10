"""Check completeness and manuscript-level checkpoints for bundled data."""

from __future__ import annotations

from plot_from_csv import GC, GT, K, VIT_GC, grid, records


def close(label: str, observed: float, expected: float, digits: int = 1) -> None:
    if round(observed, digits) != expected:
        raise AssertionError(f"{label}: {observed:.6f}, expected {expected}")


def main() -> None:
    cnn = records("cnn_runs.csv")
    vit = records("vit_runs.csv")
    assert len(cnn) == 288
    assert len(vit) == 1377
    assert all(row["complete"] == "True" for row in vit)
    for dataset, metric in (("random50k", "final_train_pct"),
                            ("clean50k", "final_train_pct"),
                            ("clean50k", "final_test_pct")):
        rows = [r for r in cnn if r["dataset"] == dataset]
        matrix, _ = grid(rows, ("channel_groups", "spatial_splits"),
                         metric, GC, K)
        assert matrix.shape == (8, 6)
        if dataset == "random50k":
            close("CNN random-label LN", matrix[0, 0], 24.6)
            close("CNN random-label best GN", max(matrix[:, 0]), 95.8)
            close("CNN random-label PLN[8]", matrix[GC.index(16), K.index(32)], 100.0)

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
    print("Verified complete three-seed grids and paper-level checkpoints")


if __name__ == "__main__":
    main()
