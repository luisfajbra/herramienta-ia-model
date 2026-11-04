"""Run the documented hydraulic sensitivity on a low-factor novel event."""
from scripts import review_chico_norte_hydraulics as review


if __name__ == "__main__":
    review.SOURCES = (
        ("pulsos_nuevos_factor_0_7",
         review.ROOT / "outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z",
         "B:novel_20261102_cascade_peaks:f0.700"),
    )
    review.main()
