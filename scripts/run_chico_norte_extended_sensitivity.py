"""Run the eight remaining frozen novel cases in the documented pilot."""
from scripts import review_chico_norte_hydraulics as review


ROOT = review.ROOT / "outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z"
FACTORS = (0.7, 1.3, 2.7, 3.9, 4.3)
SHAPES = (("pulsos", "novel_20261102_cascade_peaks"),
          ("meseta", "novel_20261102_sustained_late_surge"))
ALREADY_RUN = {("pulsos", 0.7), ("meseta", 4.3)}


def selected_sources():
    return tuple(
        (f"{label}_factor_{factor:.1f}".replace(".", "_"), ROOT,
         f"B:{shape}:f{factor:.3f}")
        for label, shape in SHAPES for factor in FACTORS
        if (label, factor) not in ALREADY_RUN
    )


if __name__ == "__main__":
    review.SOURCES = selected_sources()
    review.main()
