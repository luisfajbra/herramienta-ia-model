import json

from swmm_resilience.visualization.runner import write_figure_provenance


def test_figure_provenance_records_the_active_model(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "promotion.json").write_text(
        json.dumps({"family": "xgboost", "classifier_sha256": "aa", "regressor_sha256": "bb",
                    "feature_contract_id": "tabular_v4_21", "prep_id": "p"}),
        encoding="utf-8",
    )

    path = write_figure_provenance(tmp_path / "figs", models)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["classifier_sha256"] == "aa"
    assert record["family"] == "xgboost"
    assert "generated_at_utc" in record
