"""Check the sanitized PINN/NTK rows used by the manuscript tables."""

import csv
from collections import Counter
import math
from pathlib import Path
from statistics import median

DATA = Path(__file__).resolve().parent / "data"
TABLE_METHODS = ("ReLU", "PLN4", "PLN8", "PLN16", "PLN32", "Tanh", "SAT")
NTK_METHODS = ("ReLU", "Tanh", "SAT", "PLN4", "LN")
# Appendix Table II of the arXiv manuscript, in TABLE_METHODS order.
# Pinning these printed cells independently of the CSV prevents a coherent
# edit to both CSV metric columns from masquerading as manuscript agreement.
PAPER_TABLE2 = {
    ("128", "1,10"): ("0.999601", "0.000035", "0.001120", "0.001176", "0.007878", "0.000250", "0.000104"),
    ("128", "1,1"): ("0.995612", "0.000941", "0.002284", "0.000970", "0.004231", "0.003733", "0.000979"),
    ("128", "3,1"): ("0.993593", "0.009553", "0.003912", "0.004856", "0.007465", "0.000440", "0.000525"),
    ("256", "1,10"): ("0.998639", "0.000036", "0.004040", "0.001631", "0.000567", "0.000188", "0.000052"),
    ("256", "1,1"): ("0.998485", "0.001416", "0.005552", "0.001839", "0.002643", "0.002653", "0.001219"),
    ("256", "3,1"): ("0.999727", "0.018451", "0.002966", "0.002875", "0.005476", "0.009267", "0.000410"),
}
# The five-method appendix diagnostic table, rounded to five significant
# figures. It is a different protocol from PAPER_TABLE2.
PAPER_NTK = {
    "ReLU": (1, 0.99999, 1.00000),
    "Tanh": (1, 0.0018235, 2.2481e-6),
    "SAT": (1, 0.00067438, 1.2747e-6),
    "PLN4": (12, 0.00067266, 4.8682e-6),
    "LN": (2, 0.64416, 0.62834),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def verify_table2():
    rows = read_rows("table2_selected_runs.csv")
    expected = {(str(width), weights, method)
                for width in (128, 256)
                for weights in ("1,10", "1,1", "3,1")
                for method in TABLE_METHODS}
    keys = [(row["width"], row["loss_weights"], row["method"]) for row in rows]
    require(len(rows) == 42 and len(set(keys)) == 42 and set(keys) == expected,
            "Table II must contain exactly the 42 selected method/width/weight cells")
    for row in rows:
        cell = (row["width"], row["loss_weights"], row["method"])
        require(row["depth"] == "6" and row["epochs"] == "6000",
                f"invalid Table II depth/epochs: {cell}")
        require(row["seed"] in {"1", "2", "4", "8", "42"},
                f"invalid Table II seed: {cell}")
        require((row["optimizer"], float(row["learning_rate"])) in {
            ("adam", 3e-4), ("sgd", 1e-4)},
            f"invalid Table II optimizer/learning rate: {cell}")
        printed = PAPER_TABLE2[(row["width"], row["loss_weights"])][TABLE_METHODS.index(row["method"])]
        require(row["paper_l2_6dp"] == printed,
                f"Table II CSV cell differs from manuscript: {cell}")
        require(f"{float(row['wandb_final_relative_l2']):.6f}" == row["paper_l2_6dp"],
                f"Table II metric does not round to the manuscript cell: {cell}")
    require(Counter(row["optimizer"] for row in rows) == {"adam": 33, "sgd": 9},
            "Table II selected-run optimizer counts differ from the archive")
    return rows


def verify_ntk():
    rows = read_rows("ntk_matched_240.csv")
    expected = {(method, stage, str(width), str(lr), str(seed))
                for method in NTK_METHODS
                for stage in ("adam_only", "adam_lbfgs")
                for width in (128, 256)
                for lr in (1e-5, 3e-5, 1e-4, 3e-4)
                for seed in (1, 2, 4)}
    keys = [(row["method"], row["stage"], row["width"],
             str(float(row["lr"])), row["seed"]) for row in rows]
    require(len(rows) == 240 and len(set(keys)) == 240 and set(keys) == expected,
            "NTK records must contain exactly the 240 matched grid cells")
    paired = {}
    initial_fields = ("init_domain_eff_rank_90", "init_domain_stable_rank",
                      "init_domain_cond", "init_boundary_numerical_rank")
    for row in rows:
        pair_key = (row["method"], row["width"], str(float(row["lr"])), row["seed"])
        for field in ("l2_relative_error", *initial_fields):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid NTK numeric field {field} for {pair_key}")
        if pair_key in paired:
            if any(row[field] != paired[pair_key][field] for field in initial_fields):
                raise ValueError(f"paired NTK stages have different initial metrics: {pair_key}")
        else:
            paired[pair_key] = row
    require(len(paired) == 120, "NTK records must contain 120 paired configurations")
    output = {}
    for method in NTK_METHODS:
        subset = [row for row in rows if row["method"] == method]
        adam = [row for row in subset if row["stage"] == "adam_only"]
        lbfgs = [row for row in subset if row["stage"] == "adam_lbfgs"]
        require(len(adam) == len(lbfgs) == 24,
                f"NTK method {method} must have 24 records per optimizer stage")
        output[method] = {
            "initial_eff_rank_90_median": median(int(row["init_domain_eff_rank_90"]) for row in adam),
            "adam_relative_l2_median": median(float(row["l2_relative_error"]) for row in adam),
            "adam_lbfgs_relative_l2_median": median(float(row["l2_relative_error"]) for row in lbfgs),
        }
        rank, adam_printed, lbfgs_printed = PAPER_NTK[method]
        require(output[method]["initial_eff_rank_90_median"] == rank,
                f"NTK rank differs from manuscript: {method}")
        for field, printed in (("adam_relative_l2_median", adam_printed),
                               ("adam_lbfgs_relative_l2_median", lbfgs_printed)):
            require(f"{output[method][field]:.5g}" == f"{printed:.5g}",
                    f"NTK {field} differs from manuscript: {method}")
    return output


def main():
    table = verify_table2()
    ntk = verify_ntk()
    print(f"Table II: {len(table)} selected runs; Adam 3e-4, SGD 1e-4; all values round to manuscript cells")
    print("NTK: 240 matched numeric rows (five methods, two stages)")
    for method, values in ntk.items():
        print(f"  {method}: rank={values['initial_eff_rank_90_median']:g}, "
              f"Adam L2={values['adam_relative_l2_median']:.8g}, "
              f"Adam+L-BFGS L2={values['adam_lbfgs_relative_l2_median']:.8g}")


if __name__ == "__main__":
    main()
