"""Draw the architecture diagram (docs/architecture.png)."""
from pathlib import Path

if hasattr(__import__("sys").stdout, "reconfigure"):
    __import__("sys").stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURFACE, INK, INK2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#a9a8a2"
BLUE, BLUE_BG, ORANGE_BG, AQUA_BG, GRAY_BG = "#2a78d6", "#e3eefb", "#fde6dc", "#dcf3ea", "#efeeea"

fig, ax = plt.subplots(figsize=(12, 5.2))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.2)
ax.axis("off")


def box(x, y, w, h, title, sub, bg, edge=LINE, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=bg, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, color=INK,
            fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h - 0.62, sub, ha="center", va="top", fontsize=8.2, color=INK2, linespacing=1.35)


def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=INK2, lw=1.3))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=8, color=INK2)


box(0.2, 3.3, 1.9, 1.4, "User request", "any language\n\"older than 60\"", GRAY_BG)
box(0.2, 0.5, 1.9, 1.4, "LLM action", "SQL or tool call\nWHERE age >= 60", GRAY_BG)
box(2.7, 3.3, 2.3, 1.4, "Intent extraction", "value + schema linking\ncomparators, dates,\nnegation, aggregates", BLUE_BG)
box(2.7, 0.5, 2.3, 1.4, "Action compiler", "SQL parser / tool-call\nreader -> graph", BLUE_BG)
box(5.6, 3.3, 1.8, 1.4, "Intent USIG", "entities, constraints,\nshape", AQUA_BG)
box(5.6, 0.5, 1.8, 1.4, "Action USIG", "entities, constraints,\nshape", AQUA_BG)
box(8.0, 1.75, 2.0, 1.7, "Proof engine", "canonical form +\n8 proof obligations\n+ similarity gate", BLUE_BG, edge=BLUE)
box(10.5, 3.2, 1.35, 1.0, "Execute", "match", AQUA_BG)
box(10.5, 0.85, 1.35, 1.2, "Reject", "explain,\nregenerate", ORANGE_BG)

arrow(2.1, 4.0, 2.7, 4.0)
arrow(2.1, 1.2, 2.7, 1.2)
arrow(5.0, 4.0, 5.6, 4.0)
arrow(5.0, 1.2, 5.6, 1.2)
arrow(7.4, 4.0, 8.0, 3.1)
arrow(7.4, 1.2, 8.0, 2.1)
arrow(10.0, 2.9, 10.5, 3.6)
arrow(10.0, 2.3, 10.5, 1.5)
ax.text(6.0, 5.0, "Semmai: Generate -> Verify -> Execute / Reject", ha="center", fontsize=12, color=INK,
        fontweight="bold")
out = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
