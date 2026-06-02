import pytest

from swmm_resilience.ml import trainer


def test_train_models_rejects_dataset_without_flooded_rows(tmp_path, tiny_config_factory, trainer_training_df):
    df = trainer_training_df.copy()
    df["inunda"] = 0
    df["vol_inundacion_m3"] = 0.0

    with pytest.raises(ValueError, match="No hay filas inundadas"):
        trainer.train_models(df, tiny_config_factory(), tmp_path / "models")


def test_train_models_writes_artifacts(tmp_path, tiny_config_factory, trainer_training_df):
    trainer.train_models(trainer_training_df, tiny_config_factory(), tmp_path / "models")

    assert (tmp_path / "models" / "classifier.joblib").exists()
    assert (tmp_path / "models" / "regressor.joblib").exists()
    assert (tmp_path / "models" / "training_inp_hash.txt").exists()
