#!/usr/bin/env python3
"""
Paired Statistical Analysis & Visualization for QPSO Delivery Routing.

Performs rigorous statistical comparison of va_qpso vs fixed_beta_qpso:
1. Shapiro-Wilk test for normality on paired differences.
2. Wilcoxon signed-rank test as primary non-parametric significance test.
3. Vargha-Delaney A12 effect size directly implemented.
4. Generates publication-ready annotated comparison bar chart saved as PNG.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_CSV = PROJECT_ROOT / "results" / "experiments.csv"
DEFAULT_PLOT = PROJECT_ROOT / "results" / "route_completion_comparison.png"


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """
    Vargha and Delaney (2000) A12 non-parametric effect size.
    Calculates probability that a randomly selected observation from x
    is strictly lower (better for minimization) than one from y, plus
    half the probability of a tie:
        A12 = (P(X < Y) + 0.5 * P(X == Y))
    
    Interpretation:
        A12 = 0.5  -> No effect (equal performance)
        A12 > 0.5  -> x tends to be better (lower) than y
        A12 >= 0.56 -> Small effect
        A12 >= 0.64 -> Medium effect
        A12 >= 0.71 -> Large effect
    """
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0.5
    wins = 0.0
    for xi in x:
        for yj in y:
            if xi < yj:
                wins += 1.0
            elif xi == yj:
                wins += 0.5
    return wins / (m * n)


def interpret_a12(a12: float) -> str:
    diff = abs(a12 - 0.5)
    if diff < 0.06:
        return "negligible"
    elif diff < 0.14:
        return "small"
    elif diff < 0.21:
        return "medium"
    else:
        return "large"


def analyze_experiments(csv_path: str, plot_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Experiment results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"tier", "seed", "algorithm", "total_route_completion_time"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

    tiers = [t for t in ["low", "medium", "high"] if t in df["tier"].unique()]
    if not tiers:
        tiers = sorted(df["tier"].unique())

    print("=" * 80)
    print("PAIRED STATISTICAL ANALYSIS: va_qpso VS fixed_beta_qpso")
    print(f"Data source: {csv_path}")
    print("=" * 80)

    summary_data = []

    for tier in tiers:
        sub_df = df[df["tier"] == tier]
        va_runs = sub_df[sub_df["algorithm"] == "va_qpso"].sort_values("seed")
        fb_runs = sub_df[sub_df["algorithm"] == "fixed_beta_qpso"].sort_values("seed")

        common_seeds = sorted(list(set(va_runs["seed"]).intersection(set(fb_runs["seed"]))))
        n_pairs = len(common_seeds)

        if n_pairs == 0:
            print(f"\n[Tier: {tier.upper()}] No matched pairs found.")
            continue

        va_matched = va_runs[va_runs["seed"].isin(common_seeds)]
        fb_matched = fb_runs[fb_runs["seed"].isin(common_seeds)]

        t_va = va_matched["total_route_completion_time"].values
        t_fb = fb_matched["total_route_completion_time"].values
        diff = t_va - t_fb  # Negative diff means va_qpso is faster (better)

        # 1. Shapiro-Wilk normality test on paired differences
        if n_pairs >= 3 and np.std(diff) > 1e-8:
            shapiro_stat, shapiro_p = stats.shapiro(diff)
            is_normal = shapiro_p >= 0.05
        else:
            shapiro_stat, shapiro_p = 1.0, 1.0
            is_normal = True

        # 2. Primary test: Wilcoxon signed-rank test
        # Tests if median of paired differences is significantly different from 0
        if np.all(diff == 0):
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        else:
            try:
                # Use two-sided or less depending on hypothesis
                res_w = stats.wilcoxon(diff, alternative="two-sided")
                wilcoxon_stat = res_w.statistic
                wilcoxon_p = res_w.pvalue
            except Exception as e:
                wilcoxon_stat, wilcoxon_p = 0.0, 1.0

        # Secondary check: Paired t-test if normal
        if n_pairs >= 2 and np.std(diff) > 1e-8:
            ttest_stat, ttest_p = stats.ttest_rel(t_va, t_fb)
        else:
            ttest_stat, ttest_p = 0.0, 1.0

        # 3. Vargha-Delaney A12 effect size (lower is better for travel time)
        a12 = vargha_delaney_a12(t_va, t_fb)
        a12_mag = interpret_a12(a12)

        mean_va, std_va = float(np.mean(t_va)), float(np.std(t_va, ddof=1)) if n_pairs > 1 else 0.0
        mean_fb, std_fb = float(np.mean(t_fb)), float(np.std(t_fb, ddof=1)) if n_pairs > 1 else 0.0
        mean_diff = float(np.mean(diff))
        pct_improvement = ((mean_fb - mean_va) / mean_fb * 100.0) if mean_fb > 0 else 0.0

        summary_data.append({
            "tier": tier,
            "n_pairs": n_pairs,
            "mean_va": mean_va,
            "std_va": std_va,
            "mean_fb": mean_fb,
            "std_fb": std_fb,
            "mean_diff": mean_diff,
            "pct_improvement": pct_improvement,
            "shapiro_stat": shapiro_stat,
            "shapiro_p": shapiro_p,
            "is_normal": is_normal,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
            "ttest_stat": ttest_stat,
            "ttest_p": ttest_p,
            "a12": a12,
            "a12_mag": a12_mag,
        })

        print(f"\n>>> TIER: {tier.upper()} ({n_pairs} Paired Seeds) <<<")
        print(f"  Route Completion Time:")
        print(f"    va_qpso        : {mean_va:.2f} +/- {std_va:.2f} s")
        print(f"    fixed_beta_qpso: {mean_fb:.2f} +/- {std_fb:.2f} s")
        print(f"    Mean Difference (Delta = va - fb): {mean_diff:+.2f} s ({pct_improvement:+.2f}%)")
        print(f"  1. Normality Test (Shapiro-Wilk on paired differences):")
        print(f"     W = {shapiro_stat:.4f}, p = {shapiro_p:.4e}")
        if is_normal:
            print(f"     -> Normality NOT rejected (p >= 0.05).")
        else:
            print(f"     -> Normality REJECTED (p < 0.05). Distribution is non-normal.")
        print(f"  2. Primary Test (Wilcoxon signed-rank test):")
        print(f"     W = {wilcoxon_stat:.1f}, p-value = {wilcoxon_p:.4e}")
        sig_str = "Statistically Significant (p < 0.05)" if wilcoxon_p < 0.05 else "Not Statistically Significant (p >= 0.05)"
        print(f"     -> {sig_str}")
        if is_normal:
            print(f"     (Supplementary Paired t-test: t = {ttest_stat:.3f}, p = {ttest_p:.4e})")
        print(f"  3. Effect Size (Vargha-Delaney A12):")
        print(f"     A12 = {a12:.4f} ({a12_mag.upper()} effect size)")
        print(f"     (Probability that a random va_qpso run outperforms fixed_beta_qpso: {a12*100:.1f}%)")

    print("\n" + "=" * 80)

    # 4. Generate annotated bar chart
    generate_bar_chart(summary_data, plot_path)


def generate_bar_chart(summary_data: List[Dict[str, Any]], plot_path: str):
    """
    Generate bar chart of mean completion time per tier per algorithm with
    error bars, annotated with Wilcoxon p-value and Vargha-Delaney A12.
    """
    if not summary_data:
        return

    tiers = [s["tier"].upper() for s in summary_data]
    n_tiers = len(tiers)
    x = np.arange(n_tiers)
    width = 0.35

    means_va = [s["mean_va"] for s in summary_data]
    stds_va = [s["std_va"] for s in summary_data]

    means_fb = [s["mean_fb"] for s in summary_data]
    stds_fb = [s["std_fb"] for s in summary_data]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Clean color palette
    color_va = "#1f77b4"  # Blue
    color_fb = "#ff7f0e"  # Orange

    rects1 = ax.bar(x - width / 2, means_va, width, yerr=stds_va, label="va_qpso (Volatility-Adaptive)",
                    color=color_va, capsize=5, edgecolor="black", alpha=0.9, ecolor="black")
    rects2 = ax.bar(x + width / 2, means_fb, width, yerr=stds_fb, label="fixed_beta_qpso (Linear Anneal)",
                    color=color_fb, capsize=5, edgecolor="black", alpha=0.9, ecolor="black")

    ax.set_ylabel("Mean Route Completion Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Route Completion Time by Traffic Volatility Tier (Paired N=10)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\nVolatility" for t in tiers], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=11, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars with p-value and A12
    max_y = max(max(m + s for m, s in zip(means_va, stds_va)), max(m + s for m, s in zip(means_fb, stds_fb)))
    ax.set_ylim(0, max_y * 1.25)

    for i, s in enumerate(summary_data):
        h1 = means_va[i] + stds_va[i]
        h2 = means_fb[i] + stds_fb[i]
        bracket_y = max(h1, h2) + max_y * 0.05
        
        # Draw bracket
        ax.plot([x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
                [bracket_y, bracket_y + max_y * 0.02, bracket_y + max_y * 0.02, bracket_y],
                color="black", lw=1.2)

        # Text label
        p_str = f"p = {s['wilcoxon_p']:.4f}" if s['wilcoxon_p'] >= 0.0001 else "p < 0.0001"
        a12_str = f"A12 = {s['a12']:.3f} ({s['a12_mag']})"
        diff_str = f"Δ = {s['mean_diff']:+.1f}s ({s['pct_improvement']:+.1f}%)"
        
        text_content = f"{p_str}\n{a12_str}\n{diff_str}"
        ax.text(x[i], bracket_y + max_y * 0.03, text_content, ha="center", va="bottom",
                fontsize=9, fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"\nBar chart successfully saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze paired QPSO experiment results")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to input experiments.csv")
    parser.add_argument("--plot", type=str, default=str(DEFAULT_PLOT), help="Path to output PNG plot")
    args = parser.parse_args()

    analyze_experiments(args.csv, args.plot)


if __name__ == "__main__":
    main()
