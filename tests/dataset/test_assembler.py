def test_assemble_dataset_passes_through_extra_dynamic_columns(tmp_path):
    import pandas as pd
    from swmm_resilience.dataset.assembler import assemble_dataset

    static_topo_df = pd.DataFrame({
        "node_id": ["N1", "N2"],
        "coord_x": [0.0, 1.0], "coord_y": [0.0, 1.0],
        "elev_fondo": [100.0, 101.0], "prof_max": [1.5, 1.5],
        "n_tuberias_in": [1, 1], "n_tuberias_out": [1, 1],
        "diam_max_in": [0.2, 0.2], "diam_max_out": [0.2, 0.2],
        "pendiente_max_in": [0.01, 0.01], "pendiente_out": [0.01, 0.01],
        "base_inflow_lps": [5.0, 5.0], "dist_outfall_m": [100.0, 100.0],
        "n_nodos_aguas_arriba": [0, 1], "q_pico_acum_base": [10.0, 10.0],
        "upstream_capacity_lps": [50.0, 50.0],
    })
    dynamic_df = pd.DataFrame({
        "node_id": ["N1", "N2"],
        "q_pico_nodo": [2.0, 2.0], "q_pico_acum_escalado": [10.0, 10.0],
        "duracion_horas": [3.0, 3.0], "tiempo_al_pico_h": [0.7, 0.7],
        "shape_id": ["storm_a", "storm_a"],
        "continuity_error_pct": [-7.31, -7.31],
    })
    labels_df = pd.DataFrame({
        "node_id": ["N1", "N2"], "vol_inundacion_m3": [0.0, 0.0], "inunda": [0, 0],
    })
    df = assemble_dataset(
        static_topo_df, [(1.0, dynamic_df, labels_df)], tmp_path / "out.csv",
    )
    assert "continuity_error_pct" in df.columns
    assert df["continuity_error_pct"].tolist() == [-7.31, -7.31]


def test_assemble_dataset_leaves_nan_when_only_some_results_carry_continuity(tmp_path):
    """Fase C2 / revision final (I2): un resultado con continuity_error_pct y
    otro sin ella deben coexistir -- la fila sin medicion queda NaN (y por tanto
    NULL en SQL, que por el Ruling "NULL = no excluido" nunca se filtra).
    Comportamiento deliberado, fijado aqui para que no cambie en silencio."""
    import pandas as pd
    from swmm_resilience.dataset.assembler import assemble_dataset

    static_topo_df = pd.DataFrame({
        "node_id": ["N1", "N2"],
        "coord_x": [0.0, 1.0], "coord_y": [0.0, 1.0],
        "elev_fondo": [100.0, 101.0], "prof_max": [1.5, 1.5],
        "n_tuberias_in": [1, 1], "n_tuberias_out": [1, 1],
        "diam_max_in": [0.2, 0.2], "diam_max_out": [0.2, 0.2],
        "pendiente_max_in": [0.01, 0.01], "pendiente_out": [0.01, 0.01],
        "base_inflow_lps": [5.0, 5.0], "dist_outfall_m": [100.0, 100.0],
        "n_nodos_aguas_arriba": [0, 1], "q_pico_acum_base": [10.0, 10.0],
        "upstream_capacity_lps": [50.0, 50.0],
    })

    def _dynamic(shape_id, continuity):
        frame = pd.DataFrame({
            "node_id": ["N1", "N2"],
            "q_pico_nodo": [2.0, 2.0], "q_pico_acum_escalado": [10.0, 10.0],
            "duracion_horas": [3.0, 3.0], "tiempo_al_pico_h": [0.7, 0.7],
            "shape_id": [shape_id, shape_id],
        })
        if continuity is not None:
            frame["continuity_error_pct"] = continuity
        return frame

    labels_df = pd.DataFrame({
        "node_id": ["N1", "N2"], "vol_inundacion_m3": [0.0, 0.0], "inunda": [0, 0],
    })

    df = assemble_dataset(
        static_topo_df,
        [
            (1.0, _dynamic("with_continuity", -7.31), labels_df),
            (1.0, _dynamic("without_continuity", None), labels_df),
        ],
        tmp_path / "out.csv",
    )

    assert "continuity_error_pct" in df.columns
    measured = df[df["shape_id"] == "with_continuity"]["continuity_error_pct"]
    missing = df[df["shape_id"] == "without_continuity"]["continuity_error_pct"]
    assert measured.tolist() == [-7.31, -7.31]
    assert missing.isna().all()
