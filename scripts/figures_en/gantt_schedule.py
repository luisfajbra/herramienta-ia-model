"""English version of the Thesis 2 general schedule (Gantt chart, base deck slide 2).

The original picture is hand-made; the task bars, milestones and deadlines
below were transcribed from it (bar ends measured against the week grid).
Edit the data block to move dates; the drawing code adapts.

Usage (from the repository root):
    venv/Scripts/python.exe scripts/figures_en/gantt_schedule.py

Writes only to ``docs/figures_en/slide02_gantt_schedule_en.png``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "docs" / "figures_en" / "slide02_gantt_schedule_en.png"

# =============================================================================
# DATA (edit here)
# =============================================================================
TITLE = "GENERAL SCHEDULE — THESIS 2"
SUBTITLE = "Incremental strategy · August 2026–January 2027"
FOOTER = ("Each batch is cumulative and balanced. The validation set is kept fixed; "
          "the full Optuna run is reserved for the selected dataset.")

FIRST_MONDAY = date(2026, 8, 10)   # start of week W1
N_WEEKS = 22

# key -> (header / legend label, section colour)
SECTIONS = {
    "consolidation": ("Conceptual consolidation", "#5b8db8"),
    "design": ("Experimental design", "#2a9d8f"),
    "iterative": ("Iterative experimentation", "#178f78"),
    "modeling": ("Modeling and validation", "#8b6fb3"),
    "document": ("Document and closure", "#c85a5a"),
    "complementary": ("Complementary", "#d4a300"),
}

# (section, task label, first week, last week (inclusive), bar colour)
TASKS = [
    ("consolidation", "Conceptual clarity and adjustments to the observations", 1, 9, "#5b8db8"),
    ("consolidation", "Literature review and methodological justification", 1, 9, "#5b8db8"),
    ("consolidation", "Research on new models and spatio-temporal variables", 4, 7, "#4f7fa8"),
    ("design", "Design of scenarios, new variables and automation", 2, 4, "#2f9f92"),
    ("design", "Design and preliminary tests of neural networks", 6, 11, "#298e82"),
    ("iterative", "Pilot batch · quality control and baseline", 3, 4, "#36a394"),
    ("iterative", "Cycle 1 · +1,000 hydrographs → retraining", 4, 6, "#20a486"),
    ("iterative", "Cycle 2 · +1,000 hydrographs → retraining", 6, 8, "#178f78"),
    ("iterative", "Cycle 3 · +1,000 hydrographs → sufficiency analysis", 8, 10, "#107563"),
    ("modeling", "Final optimization and architecture comparison", 10, 13, "#e9a440"),
    ("modeling", "External validation, interpretation and technical discussion", 12, 15, "#8b6fb3"),
    ("document", "Progressive writing and updating of the document", 1, 15, "#c86060"),
    ("document", "Comprehensive review and final corrections", 14, 18, "#aa4949"),
    ("document", "Thesis defense preparation and rehearsal", 19, 22, "#923838"),
    ("complementary", "Evaluation of an uncalibrated network, subject to progress", 14, 15, "#d6a800"),
]

# Curved arrows linking consecutive cycle bars (task index -> task index).
CYCLE_ARROWS = [(6, 7), (7, 8)]

# Dashed milestones (label, date) -- drawn at the end of the corresponding week.
MILESTONES = [
    ("Retraining 1", date(2026, 9, 20)),
    ("Retraining 2", date(2026, 10, 4)),
    ("Data sufficiency", date(2026, 10, 18)),
    ("Final model", date(2026, 11, 8)),
]

# Red deadline lines (two-line label, date).
DEADLINES = [
    # Original label is cut off by the header band ("ENTREGA SOLICITU..."); it is
    # split over three lines here so that it fits below the header.
    ("REQUEST\nSUBMISSION\n20 NOV 2026", date(2026, 11, 20)),
    ("DEFENSE\n8 JAN 2027", date(2027, 1, 8)),
]

# =============================================================================
# LAYOUT (pixel coordinates measured on the original 6734 x 3118 picture)
# =============================================================================
W, H, DPI = 6734, 3118, 300
PT = DPI / 72.0                       # pixels per point
BG = "#f5f8fb"
HEADER_BG = "#12304a"
INK = "#233246"
WEEK_INK = "#2b4a66"
GRID = "#d7e0e8"
SHADE = "#f0f5f8"
ROW_LINE = "#e6ebf0"
MILESTONE = "#5c6d86"
MILESTONE_MARK = "#1f3352"
RED = "#c62828"

X_W1 = 2968.3          # x of the start of W1
DX = 161.1             # pixels per week
X_RIGHT = 6541         # right edge of the chart area
X_PANEL_ORIG = 2049    # left edge of the label panel in the original
Y_TOP, Y_BOTTOM = 472, 2814
Y_MARK = 534           # milestone diamonds
Y_FIRST_BAND = 607
BAND_H, ROW_PITCH, BAR_H = 70, 100, 56
FIRST_TASK_OFFSET, SECTION_GAP = 75, 142

TASK_FONT = 10
SECTION_FONT = 10
WEEK_FONT = 7.4
MILESTONE_FONT = 7.4
DEADLINE_FONT = 8.0


MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def x_of(d: date) -> float:
    return X_W1 + (d - FIRST_MONDAY).days / 7.0 * DX


def x_week(week: int) -> float:
    return X_W1 + (week - 1) * DX


def text_width_px(fig, ax, s: str, **kw) -> float:
    t = ax.text(0, 0, s, **kw)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width
    t.remove()
    return w


def build() -> Path:
    fig = plt.figure(figsize=((W + 0.5) / DPI, (H + 0.5) / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    plt.rcParams["hatch.linewidth"] = 1.2

    # The English task labels are longer than the Spanish ones: widen the label
    # panel to the left (into the empty margin of the original) so no label
    # runs into a bar.
    longest = max(text_width_px(fig, ax, t[1], fontsize=TASK_FONT) for t in TASKS)
    x_panel = min(X_PANEL_ORIG, X_W1 - longest - 140)
    x_label = x_panel + 71
    x_section = x_panel + 26

    # Header band.
    ax.add_patch(Rectangle((34, 34), 6606, 206, color=HEADER_BG, zorder=1))
    ax.text(200, 137, TITLE, color="white", fontsize=20, fontweight="bold",
            va="center", ha="left", zorder=2)
    ax.text(6474, 138, SUBTITLE, color="#d2dce6", fontsize=10, va="center",
            ha="right", zorder=2)

    # Chart background, alternate week shading and week grid.
    ax.add_patch(Rectangle((x_panel, Y_TOP), X_RIGHT - x_panel, Y_BOTTOM - Y_TOP,
                           color="white", zorder=0.5))
    for k in range(1, N_WEEKS + 1):
        if k % 2 == 1:
            ax.add_patch(Rectangle((x_week(k), Y_TOP), DX, Y_BOTTOM - Y_TOP,
                                   color=SHADE, zorder=0.6))
        ax.plot([x_week(k)] * 2, [Y_TOP, Y_BOTTOM], color=GRID, lw=0.9, zorder=0.7)
        monday = FIRST_MONDAY + timedelta(weeks=k - 1)
        ax.text(x_week(k), 411, f"W{k}\n{monday.day:02d}-{MONTHS[monday.month - 1]}",
                ha="center", va="center", fontsize=WEEK_FONT, fontweight="bold",
                color=WEEK_INK, linespacing=1.15, zorder=3)

    # Sections and task rows.
    y = Y_FIRST_BAND
    current = None
    rows = []
    for i, (sec, label, w0, w1, color) in enumerate(TASKS):
        if sec != current:
            if current is not None:
                y = rows[-1] + SECTION_GAP
            name, sec_color = SECTIONS[sec]
            ax.add_patch(Rectangle((x_panel, y - BAND_H / 2), X_RIGHT - x_panel, BAND_H,
                                   color=sec_color, alpha=0.15, lw=0, zorder=1))
            ax.plot([x_panel, X_RIGHT], [y + BAND_H / 2] * 2, color=GRID, lw=0.6, zorder=1)
            ax.text(x_section, y, name.upper(), color=sec_color, fontsize=SECTION_FONT,
                    fontweight="bold", va="center", ha="left", zorder=3)
            current = sec
            yc = y + FIRST_TASK_OFFSET
        else:
            yc = rows[-1] + ROW_PITCH
        rows.append(yc)
        if sec == "complementary":
            ax.add_patch(Rectangle((x_panel, yc - 50), X_RIGHT - x_panel, 100,
                                   color="#fff7d6", lw=0, zorder=0.8))
        ax.plot([x_panel, X_RIGHT], [yc + 50] * 2, color=ROW_LINE, lw=0.6, zorder=0.9)
        ax.text(x_label, yc, label, fontsize=TASK_FONT, color=INK, va="center",
                ha="left", zorder=3)
        x0, x1 = x_week(w0), x_week(w1 + 1)
        if sec == "complementary":
            ax.add_patch(Rectangle((x0, yc - 30), x1 - x0, 60, facecolor=color,
                                   edgecolor="#9a7300", hatch="///", lw=1.2, zorder=2))
        else:
            ax.add_patch(Rectangle((x0, yc - BAR_H / 2), x1 - x0, BAR_H, facecolor=color,
                                   edgecolor="white", lw=1.2, zorder=2))

    # Curved arrows between the cycle bars.
    for a, b in CYCLE_ARROWS:
        xa = x_week(TASKS[a][3] + 1) - 0.32 * DX
        start = (xa, rows[a] + 6)
        end = (xa - 0.45 * DX, rows[b] - 4)
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=-0.3",
                                     arrowstyle="-|>", mutation_scale=11, lw=2.2,
                                     color="#107563", zorder=4))

    # Milestones (dashed) and deadlines (solid red).
    for label, d in MILESTONES:
        x = x_of(d)
        ax.plot([x, x], [Y_MARK, Y_BOTTOM], color=MILESTONE, lw=1.1, ls=(0, (4, 3)), zorder=2.5)
        ax.add_patch(Polygon([(x, Y_MARK - 15), (x + 12, Y_MARK), (x, Y_MARK + 15),
                              (x - 12, Y_MARK)], color=MILESTONE_MARK, zorder=3))
        ax.text(x + 4, Y_TOP + 26, label, rotation=40, rotation_mode="anchor", ha="left",
                va="bottom", fontsize=MILESTONE_FONT, fontweight="bold",
                color=MILESTONE_MARK, zorder=3)
    for label, d in DEADLINES:
        x = x_of(d)
        ax.plot([x, x], [Y_TOP, Y_BOTTOM], color=RED, lw=1.9, zorder=2.6)
        ax.add_patch(Polygon([(x, Y_MARK - 16), (x + 13, Y_MARK), (x, Y_MARK + 16),
                              (x - 13, Y_MARK)], facecolor=RED, edgecolor="white",
                             lw=0.8, zorder=3))
        t = ax.text(x + 22, Y_TOP + 8, label, rotation=40, rotation_mode="anchor", ha="left",
                va="bottom", fontsize=DEADLINE_FONT, fontweight="bold", color="#b71c1c",
                linespacing=1.1, zorder=3)
        top = H - t.get_window_extent(renderer=fig.canvas.get_renderer()).y1
        if top < 240:  # bottom edge of the header band
            print(f"warning: deadline label {label!r} reaches the header band (y={top:.0f})")

    # Legend and footer.
    handles = [Patch(facecolor=SECTIONS[k][1], label=SECTIONS[k][0])
               for k in ("consolidation", "design", "iterative", "modeling", "document")]
    handles.append(Patch(facecolor="#d6a800", edgecolor="#9a7300", hatch="///",
                         label=SECTIONS["complementary"][0]))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(3340 / W, 1 - 2979 / H),
               ncol=6, frameon=False, fontsize=8.6, handlelength=2.6, handleheight=0.8,
               columnspacing=2.2)
    ax.text(200, 3067, FOOTER, fontsize=8.6, color="#506274", va="center", ha="left")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return OUT_FILE


if __name__ == "__main__":
    print("wrote", build().relative_to(ROOT))
