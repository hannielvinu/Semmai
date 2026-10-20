"""Render the result figures used in the README and report (results/figures/*.png).

Usage: python scripts/make_figures.py   (after the evaluation scripts)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def fig_methods():
    s = json.loads((RES / "sql_results.json").read_text())
    names = {"syntax_permission": "Syntax + permission", "execution_nonempty": "Execution sanity",
             "similarity_sql": "Similarity (SQL text)", "similarity_text": "Similarity (SQL meaning)",
             "semmai": "Semmai"}
    order = list(names)
    sets = [("generated_test", "Generated test"), ("blind", "Blind test")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, title in ((axes[0], "f1", "F1-score (higher is better)"),
                              (axes[1], "far", "False acceptance rate (lower is better)")):
        y = np.arange(len(order))
        h = 0.36
        for j, (key, label) in enumerate(sets):
            vals = [s["sets"][key]["methods"][m][metric] * 100 for m in order]
            cols = [BLUE if m == "semmai" else GRAY for m in order]
            alpha = 1.0 if j == 0 else 0.55
            bars = ax.barh(y + (j - 0.5) * h, vals, height=h - 0.04, color=cols, alpha=alpha,
                           label=label, edgecolor=SURFACE, linewidth=1)
            for b, v, m in zip(bars, vals, order):
                if m == "semmai" or metric == "far":
                    ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [names[m] for m in order])
        ax.invert_yaxis()
        ax.set_xlim(0, 118)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("SQL verification: Semmai vs. baselines (darker = generated test, lighter = blind test)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "sql_methods.png")


def fig_error_types():
    s = json.loads((RES / "sql_results.json").read_text())
    pt = {k: v for k, v in s["per_type"].items() if v["label"] == 0}
    kinds = sorted(pt, key=lambda k: pt[k]["semmai"])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(kinds))
    series = [("similarity_text", "Similarity (SQL meaning)", ORANGE, "s"),
              ("exec_match_oracle", "Execution-match oracle (needs gold)", AQUA, "D"),
              ("semmai", "Semmai", BLUE, "o")]
    for key, label, col, mk in series:
        ax.scatter([pt[k][key] * 100 for k in kinds], y, s=46, color=col, marker=mk, label=label, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(y, [k.replace("_", " ").lower() for k in kinds])
    ax.set_xlim(0, 104)
    ax.set_xlabel("Detection rate (%)")
    ax.set_title("Detection rate per semantic error type (all SQL test sets)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "sql_error_types.png")


def fig_similarity():
    rows = list(csv.DictReader(open(RES / "similarity_scores.csv")))
    pos = [float(r["sim_text"]) for r in rows if r["label"] == "1"]
    neg = [float(r["sim_text"]) for r in rows if r["label"] == "0"]
    s = json.loads((RES / "sql_results.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(pos, bins=bins, histtype="step", linewidth=2, color=BLUE, label="Correct SQL", density=True)
    ax.hist(neg, bins=bins, histtype="step", linewidth=2, color=ORANGE, label="Semantically wrong SQL", density=True)
    ax.axvline(s["thresholds"]["similarity_text"], color=INK2, linewidth=1, linestyle="--")
    ax.text(s["thresholds"]["similarity_text"] + 0.015, ax.get_ylim()[1] * 0.6, "tuned\nthreshold", fontsize=8,
            color=INK2)
    ax.set_xlabel("Similarity between question and SQL meaning")
    ax.set_ylabel("Density")
    ax.set_title("Similarity cannot separate correct from wrong SQL (distributions overlap)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "similarity_overlap.png")


def fig_multilingual():
    t = json.loads((RES / "tools_results.json").read_text())
    locs = list(t["per_locale"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(locs))
    w = 0.38
    a = [t["per_locale"][l]["similarity"]["f1"] * 100 for l in locs]
    b = [t["per_locale"][l]["semmai"]["f1"] * 100 for l in locs]
    ax.bar(x - w / 2, a, w - 0.04, color=ORANGE, label="Similarity baseline")
    bars = ax.bar(x + w / 2, b, w - 0.04, color=BLUE, label="Semmai")
    for bb, v in zip(bars, b):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, locs)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("F1-score (%)")
    ax.set_title("Multilingual tool-action verification (MASSIVE format), per language")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    save(fig, "multilingual_f1.png")


def fig_pipeline():
    p = json.loads((RES / "pipeline_results.json").read_text())
    er = [r["error_rate"] * 100 for r in p]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    series = [("baseline", "Generate → Execute", GRAY, "o"), ("semmai_block", "+ Semmai (reject)", ORANGE, "s"),
              ("semmai_regen", "+ Semmai (reject & regenerate)", BLUE, "D")]
    for ax, metric, title in ((axes[0], "wrong_executed", "Wrong answers executed (lower is better)"),
                              (axes[1], "correct", "Correct answers (higher is better)")):
        for key, label, col, mk in series:
            ax.plot(er, [r[key][metric] * 100 for r in p], color=col, marker=mk, linewidth=2, markersize=7,
                    label=label, markeredgecolor=SURFACE)
        ax.set_xlabel("LLM error rate (%)")
        ax.set_ylabel("% of questions")
        ax.set_xticks(er)
        ax.set_title(title)
        ax.set_ylim(-2, 102)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].annotate("both Semmai pipelines ≈ 0%", xy=(er[1], 0.5), xytext=(er[1] + 1, 12), fontsize=8,
                     color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.tight_layout()
    save(fig, "pipeline.png")


if __name__ == "__main__":
    fig_methods()
    fig_error_types()
    fig_similarity()
    fig_multilingual()
    fig_pipeline()
