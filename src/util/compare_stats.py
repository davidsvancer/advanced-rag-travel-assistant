# author: David Švancer
# xlogin: xsvancd00
# vutmail: xsvancd00@vutbr.cz
#
# Declaration: The primary structure of this visualization script 
# was generated using artificial intelligence(ChatGPT) and subsequently modified 
# and extended for exact evaluation of experimental data.

import argparse
import sys
from typing import Dict, Any, List
import numpy as np
import pandas as pd


# injects domain category mappings based on dataset topology constraints
def add_query_categories(df: pd.DataFrame) -> pd.DataFrame:

    categories = (
        ["vague"] * 60 +
        ["negation"] * 60 +
        ["multihop"] * 60 +
        ["false_premise"] * 40 +
        ["basic"] * 50 +
        ["vague"] * 10 +
        ["negation"] * 10 +
        ["multihop"] * 10 +
        ["false_premise"] * 10
    )

    if len(df) != len(categories):
        raise ValueError(
            f"Expected {len(categories)} rows, but got {len(df)}. "
            "Category assignment bounds would be violated."
        )

    df_labeled = df.copy()
    df_labeled["category"] = categories
    return df_labeled


# performs paired bootstrap hypothesis testing calculating empirical confidence intervals
def paired_bootstrap_test(a: np.ndarray, b: np.ndarray, n_bootstrap: int = 10000, seed: int = 42) -> Dict[str, Any]:

    rng = np.random.default_rng(seed)

    a_clean = np.asarray(a, dtype=float)
    b_clean = np.asarray(b, dtype=float)

    # isolate strictly intersecting paired samples omitting missing records
    mask = ~np.isnan(a_clean) & ~np.isnan(b_clean)
    a_clean = a_clean[mask]
    b_clean = b_clean[mask]

    n = len(a_clean)
    if n == 0:
        raise ValueError("Insufficient paired samples remaining after validation filtration.")

    observed_diff = float(np.mean(b_clean - a_clean))

    # execute iterative index resampling array constructs
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_diffs = np.mean(b_clean[indices] - a_clean[indices], axis=1)

    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

    # resolve two-sided empirical probability verifying exact intersection threshold
    p_value = 2 * min(
        float(np.mean(bootstrap_diffs <= 0)),
        float(np.mean(bootstrap_diffs >= 0))
    )
    p_value = min(p_value, 1.0)

    mean_a = float(np.mean(a_clean))
    mean_b = float(np.mean(b_clean))

    return {
        "n": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": observed_diff,
        "diff_pp": observed_diff * 100,
        "relative_change_percent": (observed_diff / mean_a) * 100 if mean_a != 0 else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_pp": ci_low * 100,
        "ci_high_pp": ci_high * 100,
        "bootstrap_p_value": p_value,
    }


# aggregates category-level statistical verification passes
def compare_by_category(df_a: pd.DataFrame, df_b: pd.DataFrame, metrics: List[str], name_a: str, name_b: str, n_bootstrap: int, seed: int) -> pd.DataFrame:

    df_a_cat = add_query_categories(df_a)
    df_b_cat = add_query_categories(df_b)

    results = []

    for category in sorted(df_a_cat["category"].unique()):
        cat_slice_a = df_a_cat[df_a_cat["category"] == category]
        cat_slice_b = df_b_cat[df_b_cat["category"] == category]

        for metric in metrics:
            if metric not in cat_slice_a.columns or metric not in cat_slice_b.columns:
                continue

            boot = paired_bootstrap_test(
                cat_slice_a[metric].values,
                cat_slice_b[metric].values,
                n_bootstrap=n_bootstrap,
                seed=seed
            )

            # statistical significance established if zero boundary is not contained within CI
            is_significant = not (boot["ci_low"] <= 0 <= boot["ci_high"])

            results.append({
                "category": category,
                "metric": metric,
                "n": boot["n"],
                f"mean_{name_a}": boot["mean_a"],
                f"mean_{name_b}": boot["mean_b"],
                "difference_b_minus_a": boot["diff"],
                "difference_percentage_points": boot["diff_pp"],
                "relative_change_percent": boot["relative_change_percent"],
                "bootstrap_ci_95_low": boot["ci_low"],
                "bootstrap_ci_95_high": boot["ci_high"],
                "bootstrap_ci_95_low_pp": boot["ci_low_pp"],
                "bootstrap_ci_95_high_pp": boot["ci_high_pp"],
                "bootstrap_p_value": boot["bootstrap_p_value"],
                "significant_by_bootstrap_ci": is_significant,
            })

    return pd.DataFrame(results)


# orchestrates end-to-end multi-metric comparative bootstrap analysis
def main(args: argparse.Namespace) -> None:

    print("=" * 70)
    print(f"Executing Statistical Verification Suite | Bootstrap passes: {args.n_bootstrap}")
    print(f"Baseline (A): {args.name_a} -> {args.file_a}")
    print(f"Candidate (B): {args.name_b} -> {args.file_b}")
    print("=" * 70)

    df_a = pd.read_csv(args.file_a)
    df_b = pd.read_csv(args.file_b)

    # strict structural verification enforcing index correlation integrity
    if len(df_a) != len(df_b):
        raise ValueError(
            f"Dataset dimensionality mismatch: {len(df_a)} vs {len(df_b)} rows. "
            "Paired inference execution requires absolutely correlated sample blocks."
        )

    overall_results = []

    # process top-level aggregate performance differentials
    for metric in args.metrics:
        if metric not in df_a.columns or metric not in df_b.columns:
            print(f"Warning: Target metric '{metric}' missing in source mappings. Skipping.", file=sys.stderr)
            continue

        boot = paired_bootstrap_test(
            df_a[metric].values,
            df_b[metric].values,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed
        )

        is_significant = not (boot["ci_low"] <= 0 <= boot["ci_high"])

        overall_results.append({
            "metric": metric,
            "n": boot["n"],
            f"mean_{args.name_a}": boot["mean_a"],
            f"mean_{args.name_b}": boot["mean_b"],
            "difference_b_minus_a": boot["diff"],
            "difference_percentage_points": boot["diff_pp"],
            "relative_change_percent": boot["relative_change_percent"],
            "bootstrap_ci_95_low": boot["ci_low"],
            "bootstrap_ci_95_high": boot["ci_high"],
            "bootstrap_ci_95_low_pp": boot["ci_low_pp"],
            "bootstrap_ci_95_high_pp": boot["ci_high_pp"],
            "bootstrap_p_value": boot["bootstrap_p_value"],
            "significant_by_bootstrap_ci": is_significant,
        })

    result_df = pd.DataFrame(overall_results)
    result_df.to_csv(args.output, index=False)

    print("\nOverall statistical comparison finalized:")
    print(result_df.to_string(index=False))
    print(f"Persisted artifacts target: {args.output}\n")

    # execute conditional granular category slice validation execution
    if args.by_category:
        category_df = compare_by_category(
            df_a=df_a,
            df_b=df_b,
            metrics=args.metrics,
            name_a=args.name_a,
            name_b=args.name_b,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed
        )

        category_df.to_csv(args.category_output, index=False)
        print("-" * 70)
        print("Category-level breakdown verification summary:")
        print(category_df.to_string(index=False))
        print(f"Persisted category breakdown target: {args.category_output}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compares paired framework summary outputs applying empirical bootstrapping verification routines.")

    parser.add_argument("--file_a", type=str, required=True, help="Source baseline evaluation CSV target file.")
    parser.add_argument("--file_b", type=str, required=True, help="Source candidate comparison model CSV target file.")
    parser.add_argument("--name_a", type=str, default="Model_A", help="Logical identifier mapped representing base set.")
    parser.add_argument("--name_b", type=str, default="Model_B", help="Logical identifier mapped representing candidate set.")
    
    parser.add_argument("--by_category", action="store_true", help="Triggers isolated verification execution split by dataset domain category.")
    parser.add_argument("--category_output", type=str, default="category_stats_comparison.csv", help="Persistence path mapping granular verification summary outputs.")
    parser.add_argument("--output", type=str, default="overall_stats_comparison.csv", help="Persistence path mapping top-level verification output logs.")
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"],
        help="Space-separated array mapping specific metric columns evaluated."
    )
    
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Iteration parameter boundary defining bootstrap pool bounds.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generator seed input integer.")

    args_parsed = parser.parse_args()
    
    main(args_parsed)