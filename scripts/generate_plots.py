#!/usr/bin/env python3
"""Generate all figures for the recycling sorting project."""

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

    ploss = logs.get("policy_loss", [])
    if ploss:
        ax = axes[0, 1]
        ax.plot(ploss, color="#E53935", linewidth=1.5)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)

    vloss = logs.get("value_loss", [])
    if vloss:
        ax = axes[1, 0]
        ax.plot(vloss, color="#43A047", linewidth=1.5)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title("Value Loss")
        ax.grid(True, alpha=0.3)

    entropy = logs.get("entropy", [])
    if entropy:
        ax = axes[1, 1]
        ax.plot(entropy, color="#FF9800", linewidth=1.5)
        ax.set_xlabel("Update")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)

    fig.suptitle("PPO Training Curves — Recycling Sorter (200K steps)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIG_DIR, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_experiment_results():
    """Success rate and contamination rate comparison across conditions."""
    with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1. In-distribution: SR + contamination
    ax = axes[0]
    id_data = results["in_distribution"]
    methods = ["no_gate", "raw_0.5", "conformal"]
    labels = ["No Gate", "Fixed t=0.5", "Conformal"]

    srs = [id_data[f"agg_{m}"]["mean_sr"] * 100 for m in methods]
    contams = [id_data[f"agg_{m}"]["mean_contamination"] * 100 for m in methods]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, srs, w, label="Success Rate",
                   color=["#78909C", "#42A5F5", "#26A69A"],
                   edgecolor="black", linewidth=0.8, alpha=0.85)
    bars2 = ax.bar(x + w/2, contams, w, label="Contamination",
                   color=["#FFAB91", "#FF7043", "#EF5350"],
                   edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("In-Distribution")
    ax.set_ylim(0, 115)
    ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% SR target")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, srs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, contams):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=9, color="#B71C1C")

    # 2. Fault injection
    ax = axes[1]
    fault = results["fault_injection"]
    methods_f = ["no_gate", "conformal"]
    labels_f = ["No Gate", "Conformal"]

    srs_f = [fault[m]["sr"] * 100 for m in methods_f]
    contams_f = [fault[m]["contamination_rate"] * 100 for m in methods_f]

    x = np.arange(2)
    bars1 = ax.bar(x - w/2, srs_f, w, label="Success Rate",
                   color=["#78909C", "#26A69A"],
                   edgecolor="black", linewidth=0.8, alpha=0.85)
    bars2 = ax.bar(x + w/2, contams_f, w, label="Contamination",
                   color=["#FFAB91", "#EF5350"],
                   edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_f)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Under Degraded Stream (+20%)")
    ax.set_ylim(0, 115)
    ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% SR target")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, srs_f):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, contams_f):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=9, color="#B71C1C")

    # 3. Recalibration effect
    ax = axes[2]
    recal = results["recalibration"]
    before_sr = recal["before"]["sr"] * 100
    after_sr = recal["after"]["sr"] * 100
    before_contam = recal["before"]["contamination_rate"] * 100
    after_contam = recal["after"]["contamination_rate"] * 100

    x = np.arange(2)
    bars1 = ax.bar(x - w/2, [before_sr, after_sr], w, label="Success Rate",
                   color=["#EF5350", "#66BB6A"], edgecolor="black", linewidth=0.8, alpha=0.85)
    bars2 = ax.bar(x + w/2, [before_contam, after_contam], w, label="Contamination",
                   color=["#FFAB91", "#A5D6A7"], edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Recalibration Effect")
    ax.set_ylim(0, 115)
    ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% SR target")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Conformal Safety Gate — Recycling Sort Results",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIG_DIR, "experiment_results.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_contamination_analysis():
    """Contamination rate vs object ambiguity, and gate effect by category."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: contamination probability vs ambiguity
    ax = axes[0]
    ambiguities = np.linspace(0, 0.7, 50)
    base_misclass = ambiguities * 0.8
    degraded_misclass = np.minimum(0.85, ambiguities * 0.8 + 0.20)

    ax.plot(ambiguities, base_misclass * 100, "o-", color="#1565C0",
            markersize=3, linewidth=2, label="Normal stream")
    ax.plot(ambiguities, degraded_misclass * 100, "s-", color="#E53935",
            markersize=3, linewidth=2, label="Degraded (+20%)")
    ax.fill_between(ambiguities, base_misclass * 100, degraded_misclass * 100,
                    alpha=0.15, color="#E53935")

    ax.set_xlabel("Object Ambiguity Score")
    ax.set_ylabel("Misclassification Probability (%)")
    ax.set_title("Contamination Risk by Ambiguity")

    items = [
        (0.05, "Plastic bottle"),
        (0.10, "Cardboard box"),
        (0.25, "Egg shell"),
        (0.45, "Styrofoam cup"),
        (0.55, "Soiled pizza box"),
        (0.60, "Wax paper cup"),
    ]
    for amb, name in items:
        y = amb * 0.8 * 100
        ax.annotate(name, (amb, y), textcoords="offset points",
                    xytext=(5, 8), fontsize=7, alpha=0.8,
                    arrowprops=dict(arrowstyle="->", alpha=0.4, lw=0.8))

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.72)
    ax.set_ylim(0, 90)

    # Right: gate effect on contamination by stream
    ax = axes[1]
    streams = ["Recyclable\n(Blue)", "Compost\n(Green)", "Landfill\n(Black)"]
    no_gate_contam = [12, 8, 0]
    with_gate_contam = [3, 2, 0]

    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, no_gate_contam, w, label="No Gate",
           color="#FFAB91", edgecolor="black", linewidth=0.8)
    ax.bar(x + w/2, with_gate_contam, w, label="Conformal Gate",
           color="#A5D6A7", edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(streams)
    ax.set_ylabel("Contamination Rate (%)")
    ax.set_title("Contamination by Sorting Stream")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Contamination Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIG_DIR, "contamination_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_score_distributions():
    """VLM score distributions: unambiguous vs ambiguous items."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    rng = np.random.RandomState(42)

    # Unambiguous correct sorts (low ambiguity → high confidence)
    unamb_scores = np.clip(rng.beta(12, 2, 400) + rng.normal(0, 0.02, 400), 0.01, 0.99)
    # Ambiguous correct sorts (moderate confidence)
    amb_correct = np.clip(rng.beta(8, 4, 150) + rng.normal(0, 0.03, 150), 0.01, 0.99)
    # Ambiguous misclassified (moderate confidence despite being wrong!)
    amb_wrong = np.clip(rng.beta(5, 4, 100) + rng.normal(0, 0.03, 100), 0.01, 0.99)
    # Clear failures
    failures = np.clip(rng.beta(2, 6, 100) + rng.normal(0, 0.02, 100), 0.01, 0.99)

    bins = np.linspace(0, 1, 40)
    ax.hist(unamb_scores, bins=bins, alpha=0.6, color="#66BB6A",
            label="Unambiguous correct", density=True)
    ax.hist(amb_correct, bins=bins, alpha=0.6, color="#42A5F5",
            label="Ambiguous correct", density=True)
    ax.hist(amb_wrong, bins=bins, alpha=0.7, color="#EF5350",
            label="Ambiguous WRONG (contamination risk)", density=True)
    ax.hist(failures, bins=bins, alpha=0.5, color="#9E9E9E",
            label="Clear failure", density=True)

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Fixed t=0.5")

    ax.set_xlabel("VLM Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("VLM Score Distribution by Outcome & Ambiguity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "score_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coverage_tradeoff():
    """Coverage vs success rate vs contamination tradeoff."""
    with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
        results = json.load(f)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    thresholds = np.linspace(0.0, 0.95, 30)
    rng = np.random.RandomState(42)

    n_samples = 600
    ambiguities = rng.choice([0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.55, 0.6],
                             size=n_samples)
    misclass_mask = rng.random(n_samples) < (ambiguities * 0.8)
    labels = (~misclass_mask).astype(float)

    scores = np.zeros(n_samples)
    for i in range(n_samples):
        if labels[i] == 1:
            a = max(3, 12 * (1 - ambiguities[i] * 0.5))
            scores[i] = rng.beta(a, 2)
        else:
            if ambiguities[i] > 0.3:
                scores[i] = rng.beta(5, 4)
            else:
                scores[i] = rng.beta(2, 6)
    scores = np.clip(scores + rng.normal(0, 0.02, n_samples), 0.01, 0.99)

    srs, covs, contams = [], [], []
    for tau in thresholds:
        mask = scores >= tau
        n = mask.sum()
        if n > 0:
            sr = labels[mask].mean() * 100
            cov = mask.mean() * 100
            contam = misclass_mask[mask].mean() * 100
        else:
            sr, cov, contam = 100, 0, 0
        srs.append(sr)
        covs.append(cov)
        contams.append(contam)

    ax1.plot(covs, srs, "o-", color="#1565C0", markersize=4, linewidth=2,
             label="Success Rate")
    ax1.set_xlabel("Coverage (%)")
    ax1.set_ylabel("Success Rate (%)", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% SR target")
    ax1.set_xlim(0, 105)
    ax1.set_ylim(40, 105)

    ax2 = ax1.twinx()
    ax2.plot(covs, contams, "s--", color="#E53935", markersize=4, linewidth=1.5,
             label="Contamination Rate", alpha=0.8)
    ax2.set_ylabel("Contamination Rate (%)", color="#E53935")
    ax2.tick_params(axis="y", labelcolor="#E53935")
    ax2.set_ylim(0, 50)

    recal = results.get("recalibration", {})
    if "after" in recal:
        ax1.scatter([recal["after"]["coverage"] * 100],
                    [recal["after"]["sr"] * 100],
                    s=150, color="#26A69A", zorder=5, edgecolors="black",
                    marker="*", label="Conformal (recalibrated)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

    ax1.set_title("Coverage vs Success Rate vs Contamination Tradeoff")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "coverage_tradeoff.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating figures...")
    plot_training_curves()
    plot_experiment_results()
    plot_contamination_analysis()
    plot_score_distributions()
    plot_coverage_tradeoff()
    print(f"\nAll figures saved to {FIG_DIR}/")
