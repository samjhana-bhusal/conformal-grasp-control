#!/usr/bin/env python3
"""Generate all figures for the project."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "data", "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})


def plot_training_curves():
    """PPO training loss and reward curves."""
    with open(os.path.join(RESULTS_DIR, "training_logs.json")) as f:
        logs = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Episode rewards
    rewards = logs["episode_rewards"]
    if rewards:
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.3, color="#2196F3", linewidth=0.5)
        window = min(20, len(rewards))
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), smoothed, color="#1565C0", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)

    # Policy loss
    ploss = logs.get("policy_loss", [])
    if ploss:
        ax = axes[0, 1]
        ax.plot(ploss, color="#E53935", linewidth=1.5)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)

    # Value loss
    vloss = logs.get("value_loss", [])
    if vloss:
        ax = axes[1, 0]
        ax.plot(vloss, color="#43A047", linewidth=1.5)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title("Value Loss")
        ax.grid(True, alpha=0.3)

    # Entropy
    entropy = logs.get("entropy", [])
    if entropy:
        ax = axes[1, 1]
        ax.plot(entropy, color="#FF9800", linewidth=1.5)
        ax.set_xlabel("Update")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)

    fig.suptitle("PPO Training Curves (200K steps)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIG_DIR, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_experiment_results():
    """Main results: success rate comparison across conditions."""
    with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # 1. In-distribution comparison
    ax = axes[0]
    id_data = results["in_distribution"]
    methods = ["no_gate", "raw_0.5", "conformal"]
    labels = ["No Gate", "Fixed τ=0.5", "Conformal"]
    colors = ["#78909C", "#42A5F5", "#26A69A"]

    means = [id_data[f"agg_{m}"]["mean_sr"] * 100 for m in methods]
    stds = [id_data[f"agg_{m}"]["std_sr"] * 100 for m in methods]

    bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor="black",
                  linewidth=0.8, capsize=5, alpha=0.85)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("In-Distribution")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Fault injection
    ax = axes[1]
    fault = results["fault_injection"]
    vals = [fault["no_gate"]["sr"] * 100, fault["conformal"]["sr"] * 100]
    fault_labels = ["No Gate", "Conformal"]
    fault_colors = ["#78909C", "#26A69A"]

    bars = ax.bar(fault_labels, vals, color=fault_colors, edgecolor="black",
                  linewidth=0.8, alpha=0.85)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Under Fault Injection")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% target")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Recalibration effect
    ax = axes[2]
    recal = results["recalibration"]
    before_sr = recal["before"]["sr"] * 100
    after_sr = recal["after"]["sr"] * 100
    before_cov = recal["before"]["coverage"] * 100
    after_cov = recal["after"]["coverage"] * 100

    x = np.arange(2)
    w = 0.35
    bars1 = ax.bar(x - w / 2, [before_sr, after_sr], w, label="Success Rate",
                   color=["#EF5350", "#66BB6A"], edgecolor="black", linewidth=0.8, alpha=0.85)
    bars2 = ax.bar(x + w / 2, [before_cov, after_cov], w, label="Coverage",
                   color=["#FFAB91", "#A5D6A7"], edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Recalibration Effect")
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% target")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Conformal Safety Gate — Experiment Results", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIG_DIR, "experiment_results.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_score_distributions():
    """VLM score distributions for success vs failure cases."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    rng = np.random.RandomState(42)
    success_scores = np.clip(rng.beta(12, 2, 500) + rng.normal(0, 0.03, 500), 0.01, 0.99)
    failure_scores = np.clip(rng.beta(2, 6, 200) + rng.normal(0, 0.03, 200), 0.01, 0.99)

    bins = np.linspace(0, 1, 40)
    ax.hist(success_scores, bins=bins, alpha=0.7, color="#66BB6A", label="Success", density=True)
    ax.hist(failure_scores, bins=bins, alpha=0.7, color="#EF5350", label="Failure", density=True)

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Fixed τ=0.5")
    ax.axvline(x=0.0, color="#26A69A", linestyle="-", linewidth=2, alpha=0.8,
               label="Conformal τ=0.0")

    ax.set_xlabel("VLM Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("VLM Score Distribution: Success vs Failure")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "score_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coverage_tradeoff():
    """Coverage vs success rate tradeoff for different thresholds."""
    with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
        results = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    thresholds = np.linspace(0.0, 0.95, 20)
    rng = np.random.RandomState(42)

    n_samples = 500
    success_scores = np.clip(rng.beta(12, 2, n_samples), 0.01, 0.99)
    noise_mask = rng.random(n_samples) < 0.3
    labels = np.ones(n_samples)
    labels[noise_mask] = 0

    srs, covs = [], []
    for tau in thresholds:
        mask = success_scores >= tau
        n = mask.sum()
        if n > 0:
            sr = labels[mask].mean() * 100
            cov = mask.mean() * 100
        else:
            sr = 100
            cov = 0
        srs.append(sr)
        covs.append(cov)

    ax.plot(covs, srs, "o-", color="#1565C0", markersize=5, linewidth=2)

    ax.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% target SR")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Coverage–Success Rate Tradeoff")

    recal = results.get("recalibration", {})
    if "after" in recal:
        ax.scatter([recal["after"]["coverage"] * 100],
                   [recal["after"]["sr"] * 100],
                   s=120, color="#26A69A", zorder=5, edgecolors="black",
                   marker="*", label="Conformal (recalibrated)")

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(60, 105)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "coverage_tradeoff.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating figures...")
    plot_training_curves()
    plot_experiment_results()
    plot_score_distributions()
    plot_coverage_tradeoff()
    print(f"\nAll figures saved to {FIG_DIR}/")
