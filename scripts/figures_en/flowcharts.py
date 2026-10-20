"""English versions of the two methodology flowcharts of the base deck (slide 15).

Both charts are redrawn with matplotlib patches on a pixel-based canvas whose
size and node positions were measured on the original pictures, so the layout
and proportions match them.

Usage (from the repository root):
    venv/Scripts/python.exe scripts/figures_en/flowcharts.py

Writes only to ``docs/figures_en/``:
    slide15_flowchart_training_en.png     (simulation + training loop)
    slide15_flowchart_validation_en.png   (validation against SWMM)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "figures_en"

DPI = 100
FONT = "Arial"
INK = "#1f1f1f"
SUB_INK = "#6b6b6b"
LINE_W = 2.6           # points (about 3.6 px at 100 dpi)
TITLE_SIZE = 26
SUB_SIZE = 23
ARROW_SCALE = 42


class Canvas:
    """Figure whose data coordinates are pixels with the origin at the top-left."""

    def __init__(self, width: int, height: int) -> None:
        self.fig = plt.figure(figsize=((width + 0.5) / DPI, (height + 0.5) / DPI), dpi=DPI)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.fig.patch.set_facecolor("#f4f4f4")
        self.ax.add_patch(FancyBboxPatch(
            (4, 4), width - 8, height - 8, boxstyle="round,pad=0,rounding_size=14",
            facecolor="white", edgecolor="#d4d4d4", lw=1.6,
        ))

    # -- shapes ---------------------------------------------------------------
    def box(self, x0, y0, x1, y1, title, sub=None, title_y=None, sub_y=None):
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white",
                                    edgecolor=INK, lw=LINE_W, zorder=2))
        cx = (x0 + x1) / 2
        if title_y is None:
            title_y = (y0 + y1) / 2 if sub is None else (y0 + y1) / 2 - 30
        self.text(cx, title_y, title, bold=True, linespacing=1.55)
        if sub is not None:
            self.text(cx, sub_y if sub_y is not None else title_y + 58, sub, bold=False)

    def ellipse(self, cx, cy, w, h, label):
        self.ax.add_patch(Ellipse((cx, cy), w, h, facecolor="white", edgecolor=INK,
                                  lw=LINE_W, zorder=2))
        self.text(cx, cy + 3, label, bold=True, size=TITLE_SIZE + 2)

    def diamond(self, cx, cy, half_w, half_h, label):
        pts = [(cx, cy - half_h), (cx + half_w, cy), (cx, cy + half_h), (cx - half_w, cy)]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=INK,
                                  lw=LINE_W, zorder=2))
        self.text(cx, cy, label, bold=True, size=TITLE_SIZE - 2, linespacing=1.45)

    def text(self, x, y, s, bold=True, size=None, color=None, rotation=0, linespacing=1.25):
        self.ax.text(x, y, s, ha="center", va="center", fontfamily=FONT, rotation=rotation,
                     fontsize=size or (TITLE_SIZE if bold else SUB_SIZE),
                     fontweight="bold" if bold else "normal", linespacing=linespacing,
                     color=color or (INK if bold else SUB_INK), zorder=4)

    # -- connectors -----------------------------------------------------------
    def line(self, *pts):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=INK, lw=LINE_W, solid_capstyle="butt",
                     solid_joinstyle="miter", zorder=1)

    def arrow(self, start, end):
        self.ax.annotate("", xy=end, xytext=start, zorder=3, arrowprops=dict(
            arrowstyle="-|>", color=INK, lw=LINE_W, mutation_scale=ARROW_SCALE,
            shrinkA=0, shrinkB=0))

    def save(self, name: str) -> Path:
        out = OUT_DIR / name
        self.fig.savefig(out, dpi=DPI)
        plt.close(self.fig)
        return out


def training_flowchart() -> Path:
    """Slide 15, left picture: simulation loop over load factors + training."""
    c = Canvas(2136, 2721)
    cx = 1068
    c.ellipse(cx, 170, 492, 171, "Start")
    c.line((cx, 253), (602, 367))
    c.line((cx, 253), (1531, 367))
    c.box(224, 367, 981, 571, "Study network", "Chicó Sur (.inp)", 442, 500)
    c.box(1153, 367, 1909, 571, "Design hydrograph", "and other hydrographs", 442, 500)
    c.line((602, 571), (602, 628), (1531, 628), (1531, 571))
    c.arrow((cx, 628), (cx, 721))
    c.box(690, 721, 1444, 902, "EPA SWMM simulation", "with the current load factor", 787, 845)
    c.arrow((cx, 902), (cx, 976))
    c.diamond(cx, 1104, 296, 128, "Load factor\nreached 5.0?")
    # "No" branch: multiply the hydrograph and loop back to the simulation.
    c.arrow((cx - 296, 1104), (597, 1104))
    c.text(677, 1065, "No", bold=False)
    c.box(102, 1017, 597, 1193, "Multiply hydrograph", "next factor (+0.2)", 1077, 1132)
    c.line((350, 1017), (350, 794), (420, 794))
    c.arrow((410, 794), (690, 794))
    c.text(302, 800, "repeats: factor 0.2 → 5.0", bold=False, rotation=90)
    # "Yes" branch.
    c.arrow((cx, 1232), (cx, 1326))
    c.text(1130, 1278, "Yes", bold=False)
    c.box(777, 1326, 1357, 1471, "Save data")
    c.arrow((cx, 1471), (cx, 1564))
    c.box(690, 1564, 1444, 1738, "Model training")
    c.arrow((cx, 1738), (cx, 1832))
    c.box(690, 1832, 1444, 2011, "LOSO & cross-validation",
          "leave-one-scenario-out + K-Fold", 1897, 1953)
    c.arrow((cx, 2011), (cx, 2105))
    c.box(690, 2105, 1444, 2278, "Performance metrics")
    c.arrow((cx, 2278), (cx, 2370))
    c.ellipse(cx, 2455, 405, 171, "End")
    return c.save("slide15_flowchart_training_en.png")


def validation_flowchart() -> Path:
    """Slide 15, right picture: prediction vs. real SWMM run on unseen inputs."""
    c = Canvas(2136, 1965)
    cx = 1068
    c.ellipse(cx, 171, 494, 173, "Start")
    c.line((cx, 257), (602, 367))
    c.line((cx, 257), (1532, 367))
    c.box(224, 367, 980, 571, "Study network", "Chicó Sur (.inp)", 442, 501)
    c.box(1154, 367, 1911, 571, "New hydrograph", "unseen factor or shape", 442, 501)
    c.line((602, 571), (602, 629), (1532, 629), (1532, 571))
    c.arrow((cx, 629), (cx, 705))
    c.arrow((cx, 705), (661, 786))
    c.arrow((cx, 705), (1590, 786))
    c.box(254, 786, 1068, 995, "Prediction with the\ntrained model",
          "instant, without running SWMM", 886, 960)
    c.box(1183, 786, 1997, 995, "EPA SWMM simulation", "real run, for verification", 886, 938)
    c.line((661, 995), (661, 1053), (1590, 1053), (1590, 995))
    c.arrow((cx, 1053), (cx, 1146))
    c.box(603, 1146, 1533, 1321, "Compare prediction\nvs. real simulation")
    c.arrow((cx, 1321), (cx, 1414))
    c.box(603, 1414, 1533, 1588, "Validation metrics", "prediction error vs. SWMM", 1476, 1530)
    c.arrow((cx, 1588), (cx, 1676))
    c.ellipse(cx, 1762, 407, 173, "End")
    return c.save("slide15_flowchart_validation_en.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for build in (training_flowchart, validation_flowchart):
        print("wrote", build().relative_to(ROOT))


if __name__ == "__main__":
    main()
