import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from .trainer import FEATURE_COLS


def _plot_importance(pipeline, title: str, output_path: Path):
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    df = df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(FEATURE_COLS))))
    ax.barh(df["feature"], df["importance"], color="steelblue")
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_feature_importance_plots(clf_pipeline, reg_pipeline, output_dir: Path):
    """Save horizontal bar charts of feature importances for both models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_importance(
        clf_pipeline,
        "Importancia de Variables — Clasificador",
        output_dir / "feature_importance_classifier.png",
    )
    _plot_importance(
        reg_pipeline,
        "Importancia de Variables — Regresor",
        output_dir / "feature_importance_regressor.png",
    )
    print(f"Gráficos de importancia guardados en {output_dir}")
