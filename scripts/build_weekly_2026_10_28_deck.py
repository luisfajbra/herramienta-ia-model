"""Build the 2026-10-28 weekly project update from the 2026-10-20 deck style."""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".claude/worktrees/optuna-tuning-familia-ganadora/Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx"
EXPERIMENT = ROOT / "outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z"
PREP_ID = "813323e3ef75b752"
STUDY_PATH = EXPERIMENT / "A/model_selection_Aonly_813323e3ef75b752/studies/xgboost/813323e3ef75b752_20261028T224207Z/study.json"
QUALITY_PATH = EXPERIMENT / f"A/prepared_a_transfer_v1/{PREP_ID}/quality_report.json"
PREREG_PATH = EXPERIMENT / "A/model_selection_Aonly_813323e3ef75b752/selection_preregistration_v2.json"
B_PROGRESS_PATH = EXPERIMENT / "B/batch_progress.json"
OUTPUT = ROOT / "Presentacion_MLSMFP_28-10_avances_auditoria_transferencia.pptx"

NAVY = RGBColor(31, 55, 85)
BLUE = RGBColor(54, 96, 146)
PALE_BLUE = RGBColor(226, 235, 245)
ORANGE = RGBColor(222, 126, 43)
PALE_ORANGE = RGBColor(251, 238, 222)
GREEN = RGBColor(79, 129, 95)
PALE_GREEN = RGBColor(228, 239, 231)
RED = RGBColor(176, 74, 69)
PALE_RED = RGBColor(248, 230, 228)
GRAY = RGBColor(95, 101, 110)
PALE_GRAY = RGBColor(242, 244, 246)
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(45, 48, 52)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hex_rgb(value: RGBColor) -> str:
    return "#%02x%02x%02x" % (value[0], value[1], value[2])


def add_text(slide, x, y, w, h, text, *, size=12, color=BLACK, bold=False,
             font="Montserrat", align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE,
             margin=0.06):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    for idx, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(3)
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return shape


def add_box(slide, x, y, w, h, *, fill=PALE_BLUE, line=None, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line or fill
    sh.line.width = Pt(0.8)
    if radius:
        try:
            sh.adjustments[0] = 0.08
        except (IndexError, ValueError):
            pass
    return sh


def add_card(slide, x, y, w, h, eyebrow, value, detail, *, fill=PALE_BLUE,
             accent=BLUE, value_size=22):
    add_box(slide, x, y, w, h, fill=fill)
    add_text(slide, x + 0.12, y + 0.07, w - 0.24, 0.26, eyebrow.upper(),
             size=8.5, color=accent, bold=True)
    add_text(slide, x + 0.12, y + 0.35, w - 0.24, 0.48, value,
             size=value_size, color=NAVY, bold=True)
    add_text(slide, x + 0.12, y + 0.84, w - 0.24, h - 0.90, detail,
             size=9.4, color=GRAY, valign=MSO_ANCHOR.TOP)


def add_header(slide, title, subtitle, page):
    add_text(slide, 0.44, 0.18, 9.12, 0.48, title, size=22,
             color=NAVY, bold=True, font="Libre Baskerville")
    add_text(slide, 0.47, 0.67, 9.05, 0.36, subtitle, size=10.3,
             color=GRAY)
    add_text(slide, 0.34, 5.32, 8.6, 0.17,
             "MLSMFP | Progress: audit, tuning and network-transfer validation | 2026-10-28",
             size=7.6, color=GRAY)
    add_text(slide, 9.1, 5.25, 0.55, 0.25, str(page), size=11,
             color=GRAY, align=PP_ALIGN.CENTER)


def new_content_slide(prs, title, subtitle, page):
    # The original deck's title-only layout uses the same white canvas and
    # master/theme as the weekly slides; content is drawn in their coordinates.
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_header(slide, title, subtitle, page)
    return slide


def chart_bytes(study: dict) -> io.BytesIO:
    vals = [t.get("value") for t in study["trials"]]
    best = []
    running = float("inf")
    for value in vals:
        if value is not None:
            running = min(running, float(value))
        best.append(running)
    fig, ax = plt.subplots(figsize=(8.5, 4.1), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = list(range(1, len(best) + 1))
    ax.plot(x, vals, color="#9fb7d0", marker="o", markersize=3,
            linewidth=1.2, label="Trial score")
    ax.plot(x, best, color=hex_rgb(BLUE), linewidth=2.6,
            label="Best score so far")
    best_value = float(study["best_value"])
    best_trial = min(i for i, value in enumerate(vals) if value == best_value) + 1
    ax.scatter([best_trial], [best_value], s=55, color=hex_rgb(ORANGE), zorder=5)
    ax.annotate(f"Best: {best_value:.2f} m³\nOptuna trial {best_trial - 1}",
                (best_trial, best_value), xytext=(22, 22),
                textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": hex_rgb(GRAY)})
    ax.set_xlabel("Optuna trial (chronological order)", fontsize=10)
    ax.set_ylabel("Inner GroupKFold(5) RMSE (m³)", fontsize=10)
    ax.grid(axis="y", color="#e6e9ed", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1, len(vals))
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def clean_to_cover_only(prs: Presentation):
    ids = prs.slides._sldIdLst
    keep_id = ids[43]
    for item in list(ids):
        if item is not keep_id:
            rid = item.rId
            ids.remove(item)
            prs.part.drop_rel(rid)


def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Presentation template not found: {TEMPLATE}")
    study = load_json(STUDY_PATH)
    quality = load_json(QUALITY_PATH)
    b_progress = load_json(B_PROGRESS_PATH)
    prereg = load_json(PREREG_PATH)

    prs = Presentation(str(TEMPLATE))
    clean_to_cover_only(prs)
    cover = prs.slides[0]
    for shape in cover.shapes:
        if not shape.has_text_frame:
            continue
        text = shape.text.strip()
        replacement = "2026-10-28" if text == "2026-10-20" else None
        if replacement:
            shape.text = replacement

    # Slide 45: audit and preparation work.
    s = new_content_slide(prs, "Audit closure and reproducible A dataset",
                          "The main dataset is now rebuilt from source artifacts with explicit QA and provenance.", 45)
    cards = [
        (0.35, 1.22, "Frozen source", "1,000", "SWMM runs in network A; original records preserved.", PALE_BLUE, BLUE),
        (2.77, 1.22, "QA eligible", "866", "Runs admitted after continuity, drainage and coverage checks.", PALE_GREEN, GREEN),
        (5.19, 1.22, "Training rows", "138,560", "Exact feature/target rows; 160 nodes and 21 inputs.", PALE_BLUE, BLUE),
        (7.61, 1.22, "Shape groups", "36 / 40", "Four groups have no eligible A run; no targets were filled manually.", PALE_ORANGE, ORANGE),
    ]
    for x, y, eyebrow, value, detail, fill, accent in cards:
        add_card(s, x, y, 2.22, 1.47, eyebrow, value, detail, fill=fill, accent=accent,
                 value_size=19 if len(value) > 6 else 23)
    excluded = 1000 - int(quality["n_runs"])
    add_box(s, 0.35, 3.05, 9.48, 1.40, fill=PALE_GRAY)
    add_text(s, 0.53, 3.19, 2.4, 0.34, f"{excluded} runs excluded by QA", size=15, color=NAVY, bold=True)
    add_text(s, 3.1, 3.17, 6.4, 0.84,
             "100 drainage-not-drained · 35 continuity > 5% · 1 run appears in both groups. "
             "The 5% continuity threshold and configured drainage tolerances remain provisional.",
             size=11, color=BLACK)
    add_text(s, 0.46, 4.63, 9.2, 0.35,
             "Class balance: 41,497 flooded rows (29.9%) · 97,063 not flooded · flood threshold 1 m³.",
             size=10.2, color=GRAY)

    # Slide 46: evaluation boundary.
    s = new_content_slide(prs, "Model selection keeps the external network sealed",
                          "A is used for training and tuning; B remains a one-network external transfer test.", 46)
    add_card(s, 0.48, 1.32, 2.35, 1.45, "Training network A", "Chico Sur", "866 QA-eligible runs; frozen preparation ID 813323e3ef75b752.", fill=PALE_BLUE, accent=BLUE, value_size=17)
    add_card(s, 3.05, 1.32, 2.35, 1.45, "Inner selection", "GroupKFold(5)", "Groups are hydrograph shape IDs; optimize macro_network_rmse on A only.", fill=PALE_GREEN, accent=GREEN, value_size=17)
    add_card(s, 5.62, 1.32, 2.35, 1.45, "Frozen test B", "Chico Norte", "No B labels enter Optuna, candidate fitting, or the selection rule.", fill=PALE_ORANGE, accent=ORANGE, value_size=17)
    add_card(s, 8.19, 1.32, 1.55, 1.45, "Claim scope", "1 network", "This tests transfer to B; it cannot establish population-wide generalization.", fill=PALE_RED, accent=RED, value_size=16)
    add_box(s, 0.50, 3.12, 9.22, 1.34, fill=PALE_GRAY)
    add_text(s, 0.72, 3.27, 8.78, 0.37,
             "Leakage controls now verify source manifests, run-level artifacts, keys and feature rows.",
             size=14, color=NAVY, bold=True)
    add_text(s, 0.72, 3.73, 8.70, 0.51,
             "B predictions must be sealed and replay-verified before exact RPT hashes are opened for scoring. "
             "B labels remain out of SQLite and out of tuning.", size=11, color=BLACK)
    add_text(s, 0.52, 4.62, 9.15, 0.32,
             "Interpretation: grouped internal selection by hydrograph shape; external evaluation by a single held-out network.",
             size=10, color=GRAY)

    # Slide 47: XGBoost result and convergence plot.
    s = new_content_slide(prs, "XGBoost search completed: best inner RMSE 73.78 m³",
                          "50 Optuna trials · GroupKFold(5) by shape · seed 42 · A-only selection", 47)
    s.shapes.add_picture(chart_bytes(study), Inches(0.34), Inches(1.10), width=Inches(6.15))
    add_card(s, 6.72, 1.24, 2.95, 1.18, "Best Optuna trial", "#13", "Zero-based Optuna trial ID; score is the inner CV objective.", fill=PALE_BLUE, accent=BLUE, value_size=24)
    add_card(s, 6.72, 2.64, 2.95, 1.18, "Objective", "73.782 m³", "macro_network_rmse; with one training network this equals A's RMSE.", fill=PALE_GREEN, accent=GREEN, value_size=19)
    add_box(s, 6.72, 4.05, 2.95, 0.76, fill=PALE_ORANGE)
    add_text(s, 6.88, 4.13, 2.62, 0.60,
             "Candidate fitted on A. This is not the transfer score on B.",
             size=10.3, color=BLACK, bold=True)
    add_text(s, 0.48, 4.80, 5.96, 0.21,
             "The plot shows trial values and the best score seen so far; the inner score is a model-selection estimate.",
             size=8.8, color=GRAY)

    # Slide 48: Candidate details and MLP progress.
    s = new_content_slide(prs, "Two finalist families; MLP search remains in progress",
                          "The selection protocol and final ensemble rule were fixed before the B test is opened.", 48)
    add_box(s, 0.45, 1.22, 4.42, 2.82, fill=PALE_BLUE)
    add_text(s, 0.66, 1.38, 4.00, 0.34, "XGBOOST · SEARCH COMPLETE", size=11, color=BLUE, bold=True)
    add_text(s, 0.66, 1.84, 3.95, 0.62, "50 / 50 trials", size=22, color=NAVY, bold=True)
    add_text(s, 0.66, 2.53, 3.94, 1.14,
             "A-only grouped CV selected the candidate. Classifier and regressor artifacts were fitted and their provenance reload was verified.",
             size=11, color=BLACK, valign=MSO_ANCHOR.TOP)
    add_box(s, 5.10, 1.22, 4.42, 2.82, fill=PALE_ORANGE)
    add_text(s, 5.31, 1.38, 4.00, 0.34, "MLP · SEARCH RUNNING", size=11, color=ORANGE, bold=True)
    add_text(s, 5.31, 1.84, 3.95, 0.62, "50 trials planned", size=22, color=NAVY, bold=True)
    add_text(s, 5.31, 2.53, 3.94, 1.14,
             "Same A-only GroupKFold(5), objective and 200 epochs per fold. Final model and five-seed ensemble are not yet fitted.",
             size=11, color=BLACK, valign=MSO_ANCHOR.TOP)
    add_text(s, 0.53, 4.33, 8.95, 0.48,
             "Approved final MLP rule: train five deterministic seeds (42–46); average flood probabilities and conditional volumes in m³, then apply the 0.5 threshold.",
             size=10.4, color=GRAY)

    # Slide 49: current B run status.
    s = new_content_slide(prs, "Network B rerun is producing the frozen external test",
                          "Progress counts are operational only; no prediction or performance claim is made from B yet.", 49)
    n_ok = len(b_progress.get("results", []))
    n_excluded = len(b_progress.get("failures", []))
    total = 39 * 25
    pending = max(total - n_ok - n_excluded, 0)
    add_card(s, 0.50, 1.28, 2.75, 1.40, "Expected scenarios", f"{total:,}", "39 hydrograph forms × 25 flow factors.", fill=PALE_BLUE, accent=BLUE)
    add_card(s, 3.62, 1.28, 2.75, 1.40, "Completed QA results", f"{n_ok:,}", "Hydraulic outputs passed staging to date.", fill=PALE_GREEN, accent=GREEN)
    add_card(s, 6.74, 1.28, 2.75, 1.40, "QA excluded", f"{n_excluded:,}", "Excluded and reported; reasons can overlap.", fill=PALE_RED, accent=RED)
    bar_x, bar_y, bar_w, bar_h = 0.63, 3.15, 8.98, 0.40
    add_box(s, bar_x, bar_y, bar_w, bar_h, fill=PALE_GRAY, radius=False)
    completed_frac = min((n_ok + n_excluded) / total, 1.0)
    if completed_frac:
        sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(bar_x), Inches(bar_y), Inches(bar_w * completed_frac), Inches(bar_h))
        sh.fill.solid(); sh.fill.fore_color.rgb = BLUE; sh.line.color.rgb = BLUE
    add_text(s, 0.62, 3.63, 9.0, 0.34,
             f"{n_ok + n_excluded:,} / {total:,} scenarios adjudicated · {pending:,} still pending",
             size=11, color=NAVY, bold=True)
    reasons = {}
    for item in b_progress.get("failures", []):
        for reason in item.get("reason_codes", []):
            reasons[reason] = reasons.get(reason, 0) + 1
    reason_text = " · ".join(f"{name.replace('_', ' ')}: {count}" for name, count in sorted(reasons.items()))
    add_text(s, 0.64, 4.13, 8.98, 0.43,
             f"Current exclusion reasons (not mutually exclusive): {reason_text}",
             size=9.6, color=GRAY)
    add_text(s, 0.64, 4.65, 8.98, 0.28,
             "The 5% continuity ceiling is provisional; failed runs do not enter eligible training or scoring.",
             size=9.4, color=RED, bold=True)

    # Slide 50: maps and parity status.
    s = new_content_slide(prs, "New parity plots and flood maps come after the sealed B score",
                          "Historical figures exist, but they belong to a prior benchmark and are not reused as current evidence.", 50)
    steps = [
        (0.50, "1", "Finish MLP", "Complete A-only search and fit the approved five-seed ensemble.", PALE_ORANGE, ORANGE),
        (2.82, "2", "Seal predictions", "Freeze candidate IDs, feature keys and prediction hashes before B labels.", PALE_BLUE, BLUE),
        (5.14, "3", "Score B once", "Verify replay, exact RPT links and QA; then compute transfer metrics.", PALE_GREEN, GREEN),
        (7.46, "4", "Generate visuals", "Node/scenario parity plots and spatial flood maps from verified predictions.", PALE_BLUE, BLUE),
    ]
    for x, number, title, body, fill, accent in steps:
        add_box(s, x, 1.53, 2.08, 1.77, fill=fill)
        add_text(s, x + 0.13, 1.64, 0.38, 0.40, number, size=17, color=accent, bold=True)
        add_text(s, x + 0.48, 1.64, 1.45, 0.40, title, size=10.8, color=NAVY, bold=True)
        add_text(s, x + 0.15, 2.12, 1.78, 0.97, body, size=9.1, color=BLACK, valign=MSO_ANCHOR.TOP)
    add_text(s, 0.68, 3.66, 8.95, 0.66,
             "Why wait: the existing maps/parity use earlier tuned models and an earlier evaluation protocol. Reusing them here would mix versions and overstate the new transfer evidence.",
             size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, 0.80, 4.48, 8.72, 0.30,
             "Current valid model-selection visual: the 50-trial XGBoost inner-CV trace on slide 47.",
             size=10, color=GRAY, align=PP_ALIGN.CENTER)

    # Slide 51: next work and limitations.
    s = new_content_slide(prs, "What the current evidence can support—and what remains",
                          "The project has moved from an unaudited training table to a traceable, label-gated transfer experiment.", 51)
    columns = [
        (0.48, "Evidence now available", [
            "A rebuilt and QA-filtered source dataset with manifests and hashes.",
            "XGBoost selection across 50 grouped trials; candidate provenance verified.",
            "An external network B rerun with run-level artifact links and explicit exclusions."], PALE_GREEN, GREEN),
        (5.12, "Evidence still required", [
            "Finish MLP tuning and fit the final five-seed ensemble.",
            "Seal predictions, then compute B metrics and regenerate maps/parity.",
            "Test another independent network before making broader generalization claims."], PALE_ORANGE, ORANGE),
    ]
    for x, heading, bullets, fill, accent in columns:
        add_box(s, x, 1.25, 4.39, 2.63, fill=fill)
        add_text(s, x + 0.20, 1.40, 3.98, 0.36, heading, size=13, color=accent, bold=True)
        yy = 1.94
        for bullet in bullets:
            add_text(s, x + 0.22, yy, 3.95, 0.54, f"• {bullet}", size=10.1, color=BLACK, valign=MSO_ANCHOR.TOP)
            yy += 0.59
    add_text(s, 0.61, 4.12, 9.0, 0.52,
             "Hydraulic caveat: continuity ≤ 5%, drainage extensions and numerical tolerances are applied provisionally and still need engineering justification.",
             size=10.4, color=RED, bold=True)
    add_text(s, 0.61, 4.68, 9.0, 0.27,
             "Transfer scope: synthetic node inflow hydrographs and one held-out network; rainfall–runoff remains outside scope.",
             size=9.5, color=GRAY)

    # Slide 52: reproducibility.
    s = new_content_slide(prs, "Reproducibility anchors and frozen decisions",
                          "Each result is tied to a preparation, protocol and candidate artifact rather than a mutable database snapshot.", 52)
    rows = [
        ("Prepared A ID", PREP_ID, "138,560 rows · 866 runs · tabular_v4_21"),
        ("A source manifest SHA-256", "4bf0c50f61b2287…", "Run/artifact links and QA source manifest"),
        ("Selection preregistration", "6d5b729e33bd1162…", "Sealed before XGBoost/MLP search"),
        ("XGBoost study SHA-256", "5b3ace9ece0223b8…", "50 trials · seed 42 · GroupKFold(5)"),
        ("B protocol", str(b_progress.get("protocol_sha256", ""))[:16] + "…", "39 × 25 scenarios · separate network · label-free staging"),
    ]
    y = 1.25
    for i, (label, value, detail) in enumerate(rows):
        fill = PALE_GRAY if i % 2 == 0 else WHITE
        add_box(s, 0.48, y, 9.05, 0.63, fill=fill, radius=False)
        add_text(s, 0.63, y + 0.08, 2.05, 0.40, label, size=9.5, color=GRAY, bold=True)
        add_text(s, 2.75, y + 0.08, 2.65, 0.40, value, size=10.1, color=NAVY, bold=True)
        add_text(s, 5.50, y + 0.08, 3.82, 0.40, detail, size=9.1, color=BLACK)
        y += 0.69
    add_text(s, 0.60, 4.85, 8.95, 0.20,
             "Source paths and full digests remain in the experiment manifests and execution log; B target rows are not persisted.",
             size=8.8, color=GRAY)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(OUTPUT)
    print(f"slides={len(prs.slides)} | B accepted={n_ok} | B excluded={n_excluded} | B pending={pending}")
    print(f"A eligible runs={quality['n_runs']} | A rows={quality['n_rows']} | XGB best={study['best_value']:.6f} m3")
    print(f"preregistration sha={prereg.get('sha256', 'stored in file; full hash in execution log')}")


if __name__ == "__main__":
    main()
