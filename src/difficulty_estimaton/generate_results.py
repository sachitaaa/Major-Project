import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

DIFFICULTY_COLORS = {"Easy": "#0072B2", "Medium": "#E69F00", "Hard": "#D55E00"}
DIFFICULTY_MARKERS = {"Easy": "o", "Medium": "s", "Hard": "^"}
POS_COLOR = "#0072B2"
NEG_COLOR = "#D55E00"
CATEGORICAL_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "grid.color": "#999999",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LABELS = ["Easy", "Medium", "Hard"]

SUBJECT_DISPLAY = {
    "ai": "AI",
    "computernetworks": "Computer Networks",
    "datamining": "Data Mining",
    "softengineering": "Software Engineering",
}

FEATURE_DISPLAY = {
    "stem_word_count": "Stem word count",
    "avg_option_word_count": "Avg. option word count",
    "avg_distractor_jaccard_to_correct": "Avg. distractor Jaccard to correct",
    "explanation_word_count": "Explanation word count",
    "expl_new_info_ratio": "New information ratio (explanation)",
    "flesch_kincaid_grade": "Flesch-Kincaid grade",
}

FEATURE_WEIGHTS_ORDER = [
    "stem_word_count",
    "avg_option_word_count",
    "avg_distractor_jaccard_to_correct",
    "explanation_word_count",
    "expl_new_info_ratio",
    "flesch_kincaid_grade",
]


def compute_per_subject_stats(df, ref_col, comp_col):
    rows = []
    for subj in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subj].dropna(subset=[ref_col, comp_col])
        if len(sub) == 0:
            continue
        y_true = sub[ref_col].tolist()
        y_pred = sub[comp_col].tolist()
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, labels=LABELS)
        rows.append({
            "subject": SUBJECT_DISPLAY.get(subj, subj),
            "n": len(sub),
            "accuracy": acc,
            "kappa": kappa,
        })
    return pd.DataFrame(rows)


def compute_feature_correlations(df):
    corrs = []
    for col in FEATURE_WEIGHTS_ORDER:
        r, p = pearsonr(df[col].astype(float), df["feature_score_scaled"].astype(float))
        corrs.append((FEATURE_DISPLAY[col], r, p))
    corrs.sort(key=lambda x: -abs(x[1]))
    return corrs


def fig_new_info_ratio(df, outdir):
    order = ["computernetworks", "ai", "softengineering", "datamining"]
    means = [df[df["subject"] == s]["expl_new_info_ratio"].mean() for s in order]
    labels = [SUBJECT_DISPLAY[s] for s in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = CATEGORICAL_PALETTE[:len(order)]
    x_pos = np.arange(len(order))
    bars = ax.bar(x_pos, means, color=colors, width=0.6)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + max(means) * 0.02,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Average New Information Ratio")
    ax.set_xlabel("Subject")
    ax.set_title("Explanation New Information Ratio by Subject")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_new_info_ratio.png"), dpi=300)
    plt.close()


def fig_bloom_distribution(df, outdir):
    order = ["ai", "computernetworks", "datamining", "softengineering"]
    bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    colors = CATEGORICAL_PALETTE[:len(bloom_order)]

    data = []
    for s in order:
        sub = df[df["subject"] == s]
        pct = sub["bloom_level_provided"].value_counts(normalize=True) * 100
        data.append([pct.get(b, 0) for b in bloom_order])
    data = np.array(data).T

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x_labels = [SUBJECT_DISPLAY[s] for s in order]
    x_pos = np.arange(len(order))
    bottom = np.zeros(len(order))

    for i, b in enumerate(bloom_order):
        ax.bar(x_pos, data[i], bottom=bottom, label=b, color=colors[i], width=0.65)
        bottom += data[i]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Bloom's Taxonomy Distribution by Subject")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig3_bloom_distribution.png"), dpi=300)
    plt.close()


def fig_confusion_matrix(conf, title, filename, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        cbar=True,
        linewidths=0.8,
        linecolor="white",
        square=True,
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        cbar_kws={"label": "Number of Questions"},
        ax=ax,
    )

    ax.set_xlabel("Comparison Difficulty")
    ax.set_ylabel("Self-Reported Difficulty")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xticklabels(LABELS, rotation=0)
    ax.set_yticklabels(LABELS, rotation=0)
    ax.grid(False)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Number of Questions", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def fig_feature_correlations(corrs, outdir):
    corrs_sorted = sorted(corrs, key=lambda x: x[1])
    labels = [c[0] for c in corrs_sorted]
    values = [c[1] for c in corrs_sorted]
    colors = [NEG_COLOR if v < 0 else POS_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=colors)

    xmin = min(values)
    xmax = max(values)
    span = xmax - xmin
    pad = span * 0.12

    for bar, v in zip(bars, values):
        offset = span * 0.02
        if v >= 0:
            x_pos = v + offset
            ha = "left"
        else:
            x_pos = v - offset
            ha = "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                 va="center", ha=ha, fontsize=9)

    ax.set_xlim(xmin - pad - span * 0.15, xmax + pad + span * 0.15)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Correlation with Feature-Derived Difficulty")
    ax.set_title("Feature Correlations with Feature-Derived Difficulty Score")
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig5_feature_correlations.png"), dpi=300)
    plt.close()


def fig_subject_performance(stats_feat_df, stats_llm_df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

    comparisons = [
        (axes[0], stats_feat_df, "Self-Reported vs. Feature-Derived"),
        (axes[1], stats_llm_df, "Self-Reported vs. LLM-Response-Augmented"),
    ]

    for ax, stats_df, title in comparisons:
        x = np.arange(len(stats_df))
        width = 0.35

        ax.bar(x - width / 2, stats_df["accuracy"], width, label="Accuracy", color=POS_COLOR)
        ax.bar(x + width / 2, stats_df["kappa"], width, label="Cohen's Kappa",
               color=CATEGORICAL_PALETTE[1])

        ax.set_xticks(x)
        ax.set_xticklabels(stats_df["subject"], rotation=20, ha="right")

        max_value = max(stats_df["accuracy"].max(), stats_df["kappa"].max())
        ax.set_ylim(0, max_value * 1.3)
        ax.set_title(title, fontsize=11)
        ax.legend()
        ax.grid(axis="x", visible=False)

        for i, (acc, k) in enumerate(zip(stats_df["accuracy"], stats_df["kappa"])):
            ax.text(i - width / 2, acc + 0.012, f"{acc:.3f}", ha="center", fontsize=8)
            ax.text(i + width / 2, k + 0.012, f"{k:.3f}", ha="center", fontsize=8)

    fig.suptitle("Subject Performance Metrics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig9_subject_performance.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    conf_feat = confusion_matrix(
        df["self_reported_difficulty"], df["feature_derived_difficulty"], labels=LABELS
    )
    conf_llm = confusion_matrix(
        df["self_reported_difficulty"], df["llm_response_augmented_difficulty"], labels=LABELS
    )

    stats_feat_df = compute_per_subject_stats(
        df, "self_reported_difficulty", "feature_derived_difficulty"
    )
    stats_llm_df = compute_per_subject_stats(
        df, "self_reported_difficulty", "llm_response_augmented_difficulty"
    )

    corrs = compute_feature_correlations(df)

    fig_new_info_ratio(df, args.outdir)
    fig_bloom_distribution(df, args.outdir)

    fig_confusion_matrix(
        conf_feat,
        "Confusion Matrix: Self-Reported vs. Feature-Derived Difficulty (n=211)",
        "fig4_confusion_feature.png",
        args.outdir,
    )

    fig_confusion_matrix(
        conf_llm,
        "Confusion Matrix: Self-Reported vs. LLM-Response-Augmented Difficulty (n=211)",
        "fig4b_confusion_llm.png",
        args.outdir,
    )

    fig_feature_correlations(corrs, args.outdir)
    fig_subject_performance(stats_feat_df, stats_llm_df, args.outdir)

    print(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()