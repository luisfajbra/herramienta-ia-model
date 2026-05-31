"""
Local Tkinter desktop application for running simulations and ML workflows.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd


@contextlib.contextmanager
def _suppress_appkit_warnings():
    """Redirect OS-level stderr to /dev/null to suppress macOS AppKit warnings.

    Tkinter on macOS prints harmless Objective-C runtime messages (NSOpenPanel
    identifier override, NSButton height) directly to fd 2, bypassing Python's
    sys.stderr.  This context manager redirects the raw fd for the duration of
    the call and restores it afterwards.  Only active on macOS.
    """
    if sys.platform != "darwin":
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


from swmm_resilience.config import (
    DEFAULT_DB_FILE,
    DEFAULT_INFLOW_MULTIPLIERS,
    DEFAULT_INP_FILE,
    DEFAULT_MODEL_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_CSV,
    SCENARIO_MODE_STEADY,
    SCENARIO_MODE_TIMESERIES,
)
from swmm_resilience.main import infer_scenario_mode, network_results_dir, run_experiment
from swmm_resilience.ml.predict_from_inp import predict_steady_flows_from_inp
from swmm_resilience.visualization.runner import generate_ml_map
from swmm_resilience import reset as reset_module
from swmm_resilience.reset import RESET_CATEGORIES
from swmm_resilience.ml.predict_tabular import (
    available_classification_models,
    available_regression_models,
    predict_steady_flows,
)
from swmm_resilience.ml import train as ml_train
from swmm_resilience.utils import normalize_inflow_multipliers

from PIL import Image, ImageTk


def _float_range(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = float(start)
    if step <= 0:
        raise ValueError("El paso debe ser mayor que cero.")
    while current < stop - 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def parse_numeric_values(raw_text: str, label: str) -> list[float]:
    """Parse comma-separated values or range(start, stop, step) with decimal support."""
    text = raw_text.strip()
    if not text:
        raise ValueError(f"Ingresa al menos un valor para {label}.")

    range_match = re.fullmatch(r"range\(([^,]+),([^,]+),([^)]+)\)", text.replace(" ", ""))
    if range_match:
        start, stop, step = (float(value) for value in range_match.groups())
        values = _float_range(start, stop, step)
    else:
        parts = [part for part in re.split(r"[,;\s]+", text) if part]
        values = [float(part) for part in parts]

    if not values:
        raise ValueError(f"No se pudieron leer valores para {label}.")
    return values


def parse_target_nodes(raw_text: str, all_nodes: bool) -> list[str] | None:
    """Parse a node list from comma, semicolon, or whitespace separated text."""
    if all_nodes:
        return None

    nodes = [node for node in re.split(r"[,;\s]+", raw_text.strip()) if node]
    if not nodes:
        raise ValueError("Seleccionaste subgrupo, pero no escribiste nodos.")
    return nodes


class ResilienciaDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.worker_thread = None
        self.db_viewer = None
        self.db_viewer_window = None

        self.root.title("resiliencIA - Herramienta local")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)

        self._build_state()
        self._build_ui()
        self._toggle_mode()
        self.append_log("Listo. Configura el cuestionario y ejecuta la corrida.\n")

    def _build_state(self):
        self.inp_var = tk.StringVar(value=str(DEFAULT_INP_FILE))
        self.db_var = tk.StringVar(value=str(DEFAULT_DB_FILE))
        self.csv_var = tk.StringVar(value=str(DEFAULT_OUTPUT_CSV))
        self.artifacts_dir_var = tk.StringVar(value=str(DEFAULT_MODEL_ARTIFACTS_DIR))
        self.deltas_var = tk.StringVar(
            value=",".join(str(value) for value in DEFAULT_INFLOW_MULTIPLIERS)
        )
        self.scenario_mode_var = tk.StringVar(value=infer_scenario_mode(DEFAULT_INP_FILE))
        self.all_nodes_var = tk.BooleanVar(value=True)
        self.target_nodes_var = tk.StringVar(value="")
        self.reset_db_var = tk.BooleanVar(value=False)
        self.predict_source_var = tk.StringVar(value="csv")
        self.predict_network_var = tk.StringVar(value="")
        self.predict_inp_var = tk.StringVar(value=str(DEFAULT_INP_FILE))
        self.predict_flows_var = tk.StringVar(value="1.0,1.5,2.0")
        self.predict_all_nodes_var = tk.BooleanVar(value=True)
        self.predict_target_nodes_var = tk.StringVar(value="")
        self.predict_regressor_var = tk.StringVar(value=ml_train.default_regression_model_name())
        self.predict_classifier_var = tk.StringVar(
            value=ml_train.default_classification_model_name()
        )
        self.status_var = tk.StringVar(value="Listo")

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.sim_tab = ttk.Frame(self.notebook, padding=12)
        self.predict_tab = ttk.Frame(self.notebook, padding=12)
        self.ml_tab = ttk.Frame(self.notebook, padding=12)
        self.db_tab = ttk.Frame(self.notebook, padding=12)
        self.reset_tab = ttk.Frame(self.notebook, padding=12)
        self.results_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.sim_tab, text="Cuestionario de corrida")
        self.notebook.add(self.predict_tab, text="Predicción ML")
        self.notebook.add(self.ml_tab, text="Entrenamiento ML")
        self.notebook.add(self.db_tab, text="Base de datos")
        self.notebook.add(self.reset_tab, text="Mantenimiento")
        self.notebook.add(self.results_tab, text="Resultados")

        self._build_simulation_tab()
        self._build_prediction_tab()
        self._build_ml_tab()
        self._build_db_tab()
        self._build_reset_tab()
        self._build_results_tab()

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=1, column=0, sticky="ew")

    def _build_simulation_tab(self):
        self.sim_tab.columnconfigure(0, weight=1)
        self.sim_tab.rowconfigure(2, weight=1)

        files = ttk.LabelFrame(self.sim_tab, text="1. Archivos del proyecto", padding=12)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        self._add_path_row(files, 0, "Archivo .inp", self.inp_var, self._browse_inp)
        self._add_path_row(files, 1, "Base SQLite", self.db_var, self._browse_db_save)
        self._add_path_row(files, 2, "Dataset CSV", self.csv_var, self._browse_csv_save)

        scenario = ttk.LabelFrame(self.sim_tab, text="2. Cuestionario de escenario", padding=12)
        scenario.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        scenario.columnconfigure(1, weight=1)

        ttk.Label(scenario, text="Tipo de evaluacion").grid(row=0, column=0, sticky="w")
        mode_frame = ttk.Frame(scenario)
        mode_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Hidrograma interno (TIMESERIES)",
            value=SCENARIO_MODE_TIMESERIES,
            variable=self.scenario_mode_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Steady flow (Baseline en INFLOWS)",
            value=SCENARIO_MODE_STEADY,
            variable=self.scenario_mode_var,
        ).grid(row=0, column=1, sticky="w", padx=(16, 0))
        ttk.Label(
            scenario,
            text=(
                "Usa 'Hidrograma interno' si el .inp toma caudales desde [TIMESERIES]. "
                "Usa 'Steady flow' si el caudal esta en Baseline dentro de [INFLOWS]."
            ),
            foreground="#555555",
            wraplength=760,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))

        ttk.Label(scenario, text="Factores multiplicadores").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.deltas_entry = ttk.Entry(scenario, textvariable=self.deltas_var)
        self.deltas_entry.grid(row=2, column=1, sticky="ew", pady=(10, 0))
        ttk.Label(
            scenario,
            text=(
                "Puedes usar: 1,1.5,2  o  range(1,4,0.5). "
                "3 = triplicar la serie o el Baseline, segun el modo elegido."
            ),
            foreground="#555555",
        ).grid(row=3, column=1, sticky="w")

        ttk.Label(scenario, text="Nodos a evaluar").grid(row=4, column=0, sticky="w", pady=(10, 0))
        nodes_frame = ttk.Frame(scenario)
        nodes_frame.grid(row=4, column=1, sticky="ew", pady=(10, 0))
        nodes_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            nodes_frame,
            text="Aplicar a todos los nodos",
            variable=self.all_nodes_var,
            command=self._toggle_nodes,
        ).grid(row=0, column=0, sticky="w")
        self.target_nodes_entry = ttk.Entry(nodes_frame, textvariable=self.target_nodes_var)
        self.target_nodes_entry.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        ttk.Label(
            scenario,
            text="Si desmarcas todos: escribe IDs separados por coma, por ejemplo J1,J2,J3",
            foreground="#555555",
        ).grid(row=5, column=1, sticky="w")

        ttk.Checkbutton(
            scenario,
            text="Reemplazar base existente antes de correr",
            variable=self.reset_db_var,
        ).grid(row=6, column=1, sticky="w", pady=(12, 0))
        ttk.Label(
            scenario,
            text=(
                "Si esta opcion queda desmarcada, la corrida se agrega a la base actual "
                "(append). Si la marcas, se reinicia la base y el entrenamiento posterior "
                "usara solo las nuevas corridas."
            ),
            foreground="#555555",
            wraplength=760,
        ).grid(row=7, column=1, sticky="w", pady=(4, 0))

        actions = ttk.Frame(scenario)
        actions.grid(row=8, column=1, sticky="e", pady=(14, 0))
        self.run_button = ttk.Button(actions, text="Ejecutar corrida", command=self._run_simulation)
        self.run_button.grid(row=0, column=0)
        ttk.Button(actions, text="Limpiar log", command=self._clear_log).grid(row=0, column=1, padx=(8, 0))

        self.log = tk.Text(self.sim_tab, height=18, wrap="word")
        self.log.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.log.configure(state="disabled")

    def _build_ml_tab(self):
        self.ml_tab.columnconfigure(0, weight=1)
        self.ml_tab.rowconfigure(1, weight=1)

        intro = ttk.LabelFrame(self.ml_tab, text="Entrenamiento", padding=12)
        intro.grid(row=0, column=0, sticky="ew")
        intro.columnconfigure(0, weight=1)
        ttk.Label(
            intro,
            text=(
                "Entrena y compara modelos usando el dataset configurado. "
                "Ejecuta primero las corridas para generar/actualizar el CSV."
            ),
            wraplength=900,
        ).grid(row=0, column=0, sticky="w")
        self.train_button = ttk.Button(intro, text="Entrenar modelos ML", command=self._train_models)
        self.train_button.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.ml_log = tk.Text(self.ml_tab, height=22, wrap="word")
        self.ml_log.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.ml_log.configure(state="disabled")

    def _build_prediction_tab(self):
        self.predict_tab.columnconfigure(0, weight=1)
        self.predict_tab.rowconfigure(1, weight=1)

        form = ttk.LabelFrame(
            self.predict_tab,
            text="Inferencia ML tabular sin correr PySWMM",
            padding=12,
        )
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Modo de inferencia").grid(row=0, column=0, sticky="w")
        source_frame = ttk.Frame(form)
        source_frame.grid(row=0, column=1, sticky="w", padx=(8, 8))
        ttk.Radiobutton(
            source_frame,
            text="Desde dataset CSV",
            value="csv",
            variable=self.predict_source_var,
            command=self._toggle_prediction_source,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            source_frame,
            text="Desde archivo .inp",
            value="inp",
            variable=self.predict_source_var,
            command=self._toggle_prediction_source,
        ).grid(row=0, column=1, sticky="w", padx=(20, 0))

        ttk.Label(form, text="Dataset entrenable").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.predict_csv_entry = ttk.Entry(form, textvariable=self.csv_var)
        self.predict_csv_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        self.predict_csv_button = ttk.Button(form, text="Buscar...", command=self._browse_csv_open)
        self.predict_csv_button.grid(row=1, column=2, pady=(10, 0))

        ttk.Label(form, text="Red dentro del CSV").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.predict_network_entry = ttk.Entry(form, textvariable=self.predict_network_var)
        self.predict_network_entry.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        ttk.Label(
            form,
            text="Opcional si el CSV contiene una sola red. Usa nombre del .inp o prefijo del network_hash.",
            foreground="#555555",
        ).grid(row=3, column=1, sticky="w", padx=(8, 0))

        ttk.Label(form, text="Red .inp a evaluar").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.predict_inp_entry = ttk.Entry(form, textvariable=self.predict_inp_var)
        self.predict_inp_entry.grid(row=4, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        self.predict_inp_button = ttk.Button(form, text="Buscar...", command=self._browse_predict_inp)
        self.predict_inp_button.grid(row=4, column=2, pady=(10, 0))

        ttk.Label(form, text="Artefactos entrenados").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.artifacts_dir_entry = ttk.Entry(form, textvariable=self.artifacts_dir_var)
        self.artifacts_dir_entry.grid(row=5, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        self.artifacts_dir_button = ttk.Button(
            form,
            text="Buscar...",
            command=self._browse_artifacts_dir,
        )
        self.artifacts_dir_button.grid(row=5, column=2, pady=(10, 0))

        ttk.Label(form, text="Factores a evaluar").grid(row=6, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(form, textvariable=self.predict_flows_var).grid(
            row=6, column=1, sticky="ew", padx=(8, 8), pady=(10, 0)
        )
        ttk.Label(
            form,
            text=(
                "Ejemplo: 1.0,1.5,2.0  o  range(1,3,0.25). "
                "Por ahora la inferencia ML directa es solo para steady flow."
            ),
            foreground="#555555",
        ).grid(row=7, column=1, sticky="w", padx=(8, 0))

        ttk.Label(form, text="Nodos").grid(row=8, column=0, sticky="w", pady=(10, 0))
        nodes = ttk.Frame(form)
        nodes.grid(row=8, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        nodes.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            nodes,
            text="Evaluar todos",
            variable=self.predict_all_nodes_var,
            command=self._toggle_prediction_nodes,
        ).grid(row=0, column=0, sticky="w")
        self.predict_target_nodes_entry = ttk.Entry(nodes, textvariable=self.predict_target_nodes_var)
        self.predict_target_nodes_entry.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        ttk.Label(form, text="Modelo clasificación").grid(row=9, column=0, sticky="w", pady=(10, 0))
        self.predict_classifier_combo = ttk.Combobox(
            form,
            textvariable=self.predict_classifier_var,
            values=available_classification_models(),
            state="readonly",
            width=28,
        )
        self.predict_classifier_combo.grid(row=9, column=1, sticky="w", padx=(8, 8), pady=(10, 0))

        ttk.Label(form, text="Modelo regresión").grid(row=10, column=0, sticky="w", pady=(10, 0))
        self.predict_regressor_combo = ttk.Combobox(
            form,
            textvariable=self.predict_regressor_var,
            values=available_regression_models(),
            state="readonly",
            width=28,
        )
        self.predict_regressor_combo.grid(row=10, column=1, sticky="w", padx=(8, 8), pady=(10, 0))

        actions = ttk.Frame(form)
        actions.grid(row=11, column=1, sticky="e", pady=(12, 0))
        self.predict_button = ttk.Button(actions, text="Predecir con ML", command=self._predict_with_ml)
        self.predict_button.grid(row=0, column=0)
        ttk.Button(actions, text="Limpiar resultados", command=self._clear_prediction_output).grid(
            row=0, column=1, padx=(8, 0)
        )

        self.prediction_output = tk.Text(self.predict_tab, height=22, wrap="none")
        self.prediction_output.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.prediction_output.configure(state="disabled")
        self._toggle_prediction_nodes()
        self._toggle_prediction_source()

    def _build_reset_tab(self):
        self.reset_tab.columnconfigure(0, weight=1)
        self.reset_tab.rowconfigure(1, weight=1)

        options_frame = ttk.LabelFrame(
            self.reset_tab, text="¿Qué deseas limpiar?", padding=12
        )
        options_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self._reset_vars: dict[str, tk.BooleanVar] = {
            key: tk.BooleanVar(value=False) for key in RESET_CATEGORIES
        }

        labels = {
            "db":        "Base de datos (corridas y resultados)",
            "plots":     "Imágenes generadas (mapas PNG)",
            "dataset":   "Archivos dataset_ml.csv",
            "artifacts": "Artefactos ML (.joblib, manifest.json, métricas)",
            "temporal":  "Series temporales (archivos Parquet)",
        }
        for i, (key, label) in enumerate(labels.items()):
            ttk.Checkbutton(
                options_frame, text=label, variable=self._reset_vars[key]
            ).grid(row=i, column=0, sticky="w", pady=2)

        def _toggle_all():
            all_on = all(v.get() for v in self._reset_vars.values())
            for v in self._reset_vars.values():
                v.set(not all_on)

        btn_row = ttk.Frame(options_frame)
        btn_row.grid(row=len(labels), column=0, sticky="w", pady=(10, 0))
        ttk.Button(btn_row, text="Seleccionar todo / Ninguno", command=_toggle_all).pack(side="left")

        warning = ttk.Label(
            options_frame,
            text="⚠ Esta operación es irreversible. Afecta TODAS las redes.",
            foreground="red",
        )
        warning.grid(row=len(labels) + 1, column=0, sticky="w", pady=(8, 0))

        ttk.Button(
            options_frame, text="Limpiar seleccionado", command=self._run_reset
        ).grid(row=len(labels) + 2, column=0, sticky="w", pady=(10, 0))

        self.reset_log = tk.Text(self.reset_tab, height=20, wrap="word")
        self.reset_log.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        self.reset_log.configure(state="disabled")

    def _build_results_tab(self):
        self.results_tab.columnconfigure(0, weight=0)
        self.results_tab.columnconfigure(1, weight=1)
        self.results_tab.rowconfigure(0, weight=1)
        self.results_tab.rowconfigure(1, weight=0)

        left = ttk.LabelFrame(self.results_tab, text="Carpetas", width=200)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 6))
        left.grid_propagate(False)

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=4)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical")
        self.results_tree = ttk.Treeview(
            tree_frame, yscrollcommand=scroll.set, height=20
        )
        scroll.config(command=self.results_tree.yview)
        scroll.pack(side="right", fill="y")
        self.results_tree.pack(fill="both", expand=True)

        self._swmm_root = self.results_tree.insert(
            "", "end", text="SWMM", open=True
        )
        self._ml_root = self.results_tree.insert(
            "", "end", text="ML Tabular", open=True
        )
        self._surr_root = self.results_tree.insert(
            "", "end", text="Surrogado (CNN)", open=True
        )
        self._lstm_root = self.results_tree.insert(
            "", "end", text="Surrogado (LSTM)", open=True
        )

        ttk.Button(left, text="\u21ba Actualizar", command=self._refresh_results_tree).pack(
            pady=(0, 6)
        )

        right = ttk.LabelFrame(self.results_tab, text="Vista previa")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(
            right, text="Selecciona una imagen", anchor="center", background="#f0f0f0"
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.surr_model_var = tk.StringVar(value="cnn")

        pred = ttk.LabelFrame(self.results_tab, text="Predictor Surrogado", padding=12)
        pred.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        pred.columnconfigure(1, weight=1)
        pred.columnconfigure(3, weight=1)

        ttk.Label(pred, text="Modelo:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        model_combo = ttk.Combobox(
            pred, textvariable=self.surr_model_var,
            values=["cnn", "lstm"], state="readonly", width=8,
        )
        model_combo.grid(row=0, column=1, sticky="w")
        model_combo.bind("<<ComboboxSelected>>", lambda e: self._update_predict_button_state())

        ttk.Label(pred, text="Multiplicador:").grid(row=0, column=2, sticky="w", padx=(16, 8))
        self.surr_mult_var = tk.StringVar(value="1.0")
        ttk.Entry(pred, textvariable=self.surr_mult_var, width=12).grid(
            row=0, column=3, sticky="w"
        )

        ttk.Label(pred, text="Red .inp:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.surr_inp_entry = ttk.Entry(pred, textvariable=self.predict_inp_var)
        self.surr_inp_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(pred, text="Buscar...", command=self._browse_predict_inp).grid(
            row=1, column=3, padx=(6, 0), pady=(6, 0)
        )

        self.surr_predict_all_btn = ttk.Button(
            pred, text="Regenerar todos", command=self._predict_surrogate_all
        )
        self.surr_predict_all_btn.grid(row=2, column=1, sticky="w", pady=(10, 0))

        self.surr_predict_btn = ttk.Button(
            pred, text="Predecir y mostrar", command=self._predict_surrogate
        )
        self.surr_predict_btn.grid(row=2, column=3, sticky="e", pady=(10, 0))

        self.results_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self._refresh_results_tree()
        self._update_predict_button_state()

        # ── Temporal Windows Summary ────────────────────────────────────────────
        win_frame = ttk.LabelFrame(self.results_tab, text="Resumen de Ventanas Temporales", padding=12)
        win_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        win_frame.columnconfigure(0, weight=1)

        self.win_summary_btn = ttk.Button(
            win_frame, text="Generar Resumen", command=self._generate_window_summary
        )
        self.win_summary_btn.pack(anchor="w", pady=(0, 6))

        win_tree_frame = ttk.Frame(win_frame)
        win_tree_frame.pack(fill="both", expand=True)

        win_columns = ("multiplier", "duration_min", "time_skip_days",
                       "mean_capacity", "n_peaks", "n_drains", "n_swales")
        self.win_tree = ttk.Treeview(
            win_tree_frame, columns=win_columns, show="headings", height=8,
        )
        self.win_tree.heading("multiplier", text="Qx")
        self.win_tree.heading("duration_min", text="Duración (min)")
        self.win_tree.heading("time_skip_days", text="Salto (días)")
        self.win_tree.heading("mean_capacity", text="Cap. Media (LPS)")
        self.win_tree.heading("n_peaks", text="Picos")
        self.win_tree.heading("n_drains", text="Drenajes")
        self.win_tree.heading("n_swales", text="Cunetas")

        self.win_tree.column("multiplier", width=70)
        self.win_tree.column("duration_min", width=110)
        self.win_tree.column("time_skip_days", width=100)
        self.win_tree.column("mean_capacity", width=130)
        self.win_tree.column("n_peaks", width=60)
        self.win_tree.column("n_drains", width=80)
        self.win_tree.column("n_swales", width=80)

        win_scroll = ttk.Scrollbar(win_tree_frame, orient="vertical", command=self.win_tree.yview)
        self.win_tree.configure(yscrollcommand=win_scroll.set)

        self.win_tree.pack(side="left", fill="both", expand=True)
        win_scroll.pack(side="right", fill="y")

    def _refresh_results_tree(self):
        """Rescan all three map directories and populate the tree."""
        inp_path = Path(self.predict_inp_var.get()).expanduser()
        network_dir = inp_path.parent

        for root_key in (self._swmm_root, self._ml_root, self._surr_root, self._lstm_root):
            for child in self.results_tree.get_children(root_key):
                self.results_tree.delete(child)

        swmm_dir = network_dir / "results" / "plots"
        if swmm_dir.exists():
            for f in sorted(swmm_dir.glob("flood_map_qx*_swmm.png")):
                display = f.stem.replace("_swmm", "")
                self.results_tree.insert(self._swmm_root, "end", iid=str(f), text=display)

        ml_dir = network_dir / "ml" / "results"
        if ml_dir.exists():
            for f in sorted(ml_dir.glob("flood_map_qx*_ml.png")):
                display = f.stem.replace("_ml", "")
                self.results_tree.insert(self._ml_root, "end", iid=str(f), text=display)

        maps_dir = network_dir / "results" / "temporal" / "maps"
        if maps_dir.exists():
            for f in sorted(maps_dir.glob("surrogate_map_cnn_qx*.png")):
                display = f.stem
                self.results_tree.insert(self._surr_root, "end", iid=str(f), text=display)
            for f in sorted(maps_dir.glob("surrogate_map_lstm_qx*.png")):
                display = f.stem
                self.results_tree.insert(self._lstm_root, "end", iid=str(f), text=display)

    def _on_tree_select(self, event):
        """Display the selected image in the right panel."""
        selection = self.results_tree.selection()
        if not selection:
            return
        item = selection[0]
        if item in (self._swmm_root, self._ml_root, self._surr_root, self._lstm_root):
            return
        image_path = Path(item)
        if not image_path.exists():
            return
        self._display_image(image_path)

    def _display_image(self, image_path: Path):
        """Load and display an image scaled to fit the preview panel."""
        try:
            image = Image.open(image_path)
            panel_width = self.image_label.winfo_width()
            panel_height = self.image_label.winfo_height()
            max_w = max(panel_width - 20, 200)
            max_h = max(panel_height - 20, 150)
            image.thumbnail((max_w, max_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self._current_photo = photo
        except Exception as exc:
            self.image_label.configure(image="")
            self.image_label.configure(text=f"Error al cargar imagen:\n{exc}")

    def _update_predict_button_state(self):
        """Enable/disable the predict button based on artifact presence."""
        inp_path = Path(self.predict_inp_var.get()).expanduser()
        artifacts_dir = inp_path.parent / "results" / "temporal" / "model_artifacts"
        model_type = self.surr_model_var.get()
        prefix = "surrogate_cnn" if model_type == "cnn" else "surrogate_lstm"
        artifacts_exist = (artifacts_dir / f"{prefix}_weights.pt").exists()
        state = "normal" if artifacts_exist else "disabled"
        self.surr_predict_btn.configure(state=state)
        self.surr_predict_all_btn.configure(state=state)

    def _predict_surrogate(self):
        """Run surrogate prediction and display the resulting map."""
        try:
            multiplier = float(self.surr_mult_var.get())
            if multiplier <= 0:
                raise ValueError("El multiplicador debe ser mayor que cero.")
            inp_path = Path(self.predict_inp_var.get()).expanduser()
            if not inp_path.exists():
                raise ValueError(f"No existe el archivo .inp: {inp_path}")
            db_path = Path(self.db_var.get()).expanduser()
            model_type = self.surr_model_var.get()
        except Exception as exc:
            messagebox.showerror("Valor inválido", str(exc))
            return

        def worker():
            import matplotlib
            matplotlib.use("Agg")

            from swmm_resilience.ml.temporal.predict import (
                plot_surrogate_map,
                predict_surrogate_from_multiplier,
            )
            preds = predict_surrogate_from_multiplier(
                multiplier=multiplier,
                db_path=db_path,
                model_type=model_type,
            )
            map_path = plot_surrogate_map(
                predictions=preds,
                inp_path=inp_path,
                multiplier=multiplier,
                model_type=model_type,
            )
            print(f"Mapa guardado: {map_path}")
            self.root.after(0, self._on_surrogate_done, str(map_path))

        self._run_in_thread("Predicción surrogada", worker, self.append_log)

    def _predict_surrogate_all(self):
        """Regenerate surrogate maps for all default multipliers."""
        try:
            inp_path = Path(self.predict_inp_var.get()).expanduser()
            if not inp_path.exists():
                raise ValueError(f"No existe el archivo .inp: {inp_path}")
            db_path = Path(self.db_var.get()).expanduser()
            model_type = self.surr_model_var.get()
        except Exception as exc:
            messagebox.showerror("Valor inválido", str(exc))
            return

        def worker():
            import matplotlib
            matplotlib.use("Agg")

            from swmm_resilience.ml.temporal.predict import (
                plot_surrogate_map,
                predict_surrogate_from_multiplier,
            )
            for multiplier in DEFAULT_INFLOW_MULTIPLIERS:
                print(f"Generando mapa Qx{multiplier:.2f}...")
                preds = predict_surrogate_from_multiplier(
                    multiplier=multiplier,
                    db_path=db_path,
                    model_type=model_type,
                )
                plot_surrogate_map(
                    predictions=preds,
                    inp_path=inp_path,
                    multiplier=multiplier,
                    model_type=model_type,
                )
            self.root.after(0, self._on_regenerate_all_done)

        self._run_in_thread("Regenerar todos los mapas", worker, self.append_log)

    def _on_regenerate_all_done(self):
        """After regenerating all maps: refresh tree."""
        self._refresh_results_tree()
        self.notebook.select(self.results_tab)
        messagebox.showinfo("Listo", "Todos los mapas surrogados fueron regenerados.")

    def _on_surrogate_done(self, map_path_str: str):
        """After surrogate prediction completes: refresh tree, select new map."""
        self._refresh_results_tree()
        self.notebook.select(self.results_tab)
        map_path = Path(map_path_str)
        if map_path.exists():
            try:
                self.results_tree.selection_set(map_path_str)
                self.results_tree.focus(map_path_str)
                self.results_tree.see(map_path_str)
            except tk.TclError:
                pass
            self._display_image(map_path)

    def _generate_window_summary(self):
        """Run build_temporal_window_summary and populate the tree."""
        db_path = Path(self.db_var.get()).expanduser()
        inp_path = Path(self.predict_inp_var.get()).expanduser()

        def worker():
            import sqlite3
            conn = sqlite3.connect(db_path)
            try:
                # derive a network_hash from the DB using the inp filename
                inp_name = inp_path.name
                row = conn.execute(
                    "SELECT network_hash FROM runs WHERE network_file = ? LIMIT 1",
                    (inp_name,),
                ).fetchone()
                network_hash = row[0] if row else ""
            finally:
                conn.close()

            from swmm_resilience.ml.temporal.predict import build_temporal_window_summary
            df = build_temporal_window_summary(network_hash=network_hash, db_path=db_path)
            self.root.after(0, self._display_window_summary, df)

        self._run_in_thread("Resumen de ventanas", worker, self.append_log)

    def _display_window_summary(self, df: pd.DataFrame):
        """Populate the temporal window tree with summary data."""
        for row in self.win_tree.get_children():
            self.win_tree.delete(row)
        for _, r in df.iterrows():
            self.win_tree.insert("", "end", values=(
                f"Qx{r['inflow_multiplier']:.2f}" if "inflow_multiplier" in r.index and pd.notna(r["inflow_multiplier"]) else "N/A",
                f"{r['duration_min']:.0f}" if "duration_min" in r.index and pd.notna(r["duration_min"]) else "—",
                f"{r['time_skip_days']:.0f}" if "time_skip_days" in r.index and pd.notna(r["time_skip_days"]) else "—",
                f"{r['mean_capacity_lps']:.1f}" if "mean_capacity_lps" in r.index and pd.notna(r["mean_capacity_lps"]) else "—",
                int(r["n_peaks"]) if "n_peaks" in r.index and pd.notna(r["n_peaks"]) else 0,
                int(r["n_drains"]) if "n_drains" in r.index and pd.notna(r["n_drains"]) else 0,
                int(r["n_swales"]) if "n_swales" in r.index and pd.notna(r["n_swales"]) else 0,
            ))

    def append_reset_log(self, text: str):
        self.reset_log.configure(state="normal")
        self.reset_log.insert("end", text)
        self.reset_log.see("end")
        self.reset_log.configure(state="disabled")

    def _run_reset(self):
        selected = {k for k, v in self._reset_vars.items() if v.get()}
        if not selected:
            messagebox.showwarning("Sin selección", "Selecciona al menos una categoría para limpiar.")
            return

        labels = {
            "db":        "Base de datos",
            "plots":     "Imágenes PNG",
            "dataset":   "dataset_ml.csv",
            "artifacts": "Artefactos ML",
            "temporal":  "Parquet temporales",
        }
        items = "\n  • ".join(labels[k] for k in selected)
        confirmed = messagebox.askyesno(
            "Confirmar limpieza",
            f"Se eliminarán permanentemente:\n\n  • {items}\n\n¿Continuar?",
            icon="warning",
        )
        if not confirmed:
            return

        self.reset_log.configure(state="normal")
        self.reset_log.delete("1.0", "end")
        self.reset_log.configure(state="disabled")

        db_path = Path(self.db_var.get()).expanduser()

        def worker():
            reset_module.reset(
                db="db" in selected,
                plots="plots" in selected,
                dataset="dataset" in selected,
                artifacts="artifacts" in selected,
                temporal="temporal" in selected,
                db_path=db_path,
                callback=None,
            )

        self._run_in_thread("Limpieza", worker, self.append_reset_log)

    def _build_db_tab(self):
        self.db_tab.columnconfigure(0, weight=1)

        panel = ttk.LabelFrame(self.db_tab, text="Visor SQLite", padding=12)
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="Base SQLite").grid(row=0, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.db_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(panel, text="Buscar...", command=self._browse_db_save).grid(row=0, column=2)
        ttk.Button(panel, text="Abrir visor de BD", command=self._open_db_viewer).grid(
            row=1, column=1, sticky="e", pady=(12, 0)
        )

    def _add_path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(0, 6))
        ttk.Button(parent, text="Buscar...", command=command).grid(row=row, column=2, pady=(0, 6))

    def _browse_inp(self):
        with _suppress_appkit_warnings():
            path = filedialog.askopenfilename(
                title="Selecciona archivo SWMM .inp",
                filetypes=[("SWMM input", "*.inp"), ("Todos", "*.*")],
            )
        if path:
            self.inp_var.set(path)
            self.predict_inp_var.set(path)
            self.scenario_mode_var.set(infer_scenario_mode(Path(path).expanduser()))
            self.csv_var.set(str(network_results_dir(Path(path).expanduser()) / DEFAULT_OUTPUT_CSV.name))
            self._sync_artifacts_dir()

    def _browse_predict_inp(self):
        with _suppress_appkit_warnings():
            path = filedialog.askopenfilename(
                title="Selecciona archivo SWMM .inp para inferencia",
                filetypes=[("SWMM input", "*.inp"), ("Todos", "*.*")],
            )
        if path:
            self.predict_inp_var.set(path)
            self._update_predict_button_state()

    def _browse_db_save(self):
        with _suppress_appkit_warnings():
            path = filedialog.asksaveasfilename(
                title="Selecciona base SQLite",
                defaultextension=".db",
                filetypes=[("SQLite", "*.db"), ("Todos", "*.*")],
            )
        if path:
            self.db_var.set(path)

    def _browse_csv_save(self):
        with _suppress_appkit_warnings():
            path = filedialog.asksaveasfilename(
                title="Selecciona dataset CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
            )
        if path:
            self.csv_var.set(path)
            self._sync_artifacts_dir()

    def _browse_csv_open(self):
        with _suppress_appkit_warnings():
            path = filedialog.askopenfilename(
                title="Selecciona dataset CSV",
                filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
            )
        if path:
            self.csv_var.set(path)
            self._sync_artifacts_dir()

    def _browse_artifacts_dir(self):
        with _suppress_appkit_warnings():
            path = filedialog.askdirectory(title="Selecciona carpeta de artefactos ML")
        if path:
            self.artifacts_dir_var.set(path)

    def _toggle_mode(self):
        self.deltas_entry.configure(state="normal")
        self._toggle_nodes()

    def _toggle_nodes(self):
        self.target_nodes_entry.configure(state="disabled" if self.all_nodes_var.get() else "normal")

    def _toggle_prediction_nodes(self):
        self.predict_target_nodes_entry.configure(
            state="disabled" if self.predict_all_nodes_var.get() else "normal"
        )

    def _toggle_prediction_source(self):
        use_csv = self.predict_source_var.get() == "csv"
        csv_state = "normal" if use_csv else "disabled"
        inp_state = "disabled" if use_csv else "normal"
        self.predict_csv_entry.configure(state=csv_state)
        self.predict_csv_button.configure(state=csv_state)
        self.predict_network_entry.configure(state=csv_state)
        self.predict_inp_entry.configure(state=inp_state)
        self.predict_inp_button.configure(state=inp_state)
        self.artifacts_dir_entry.configure(state=inp_state)
        self.artifacts_dir_button.configure(state=inp_state)

    def _sync_artifacts_dir(self):
        csv_path = Path(self.csv_var.get()).expanduser()
        self.artifacts_dir_var.set(str(ml_train.default_artifact_dir(csv_path)))

    def _set_running(self, running: bool):
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.train_button.configure(state=state)
        self.predict_button.configure(state=state)
        self.status_var.set("Ejecutando..." if running else "Listo")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def append_ml_log(self, text: str):
        self.ml_log.configure(state="normal")
        self.ml_log.insert("end", text)
        self.ml_log.see("end")
        self.ml_log.configure(state="disabled")

    def append_prediction_output(self, text: str):
        self.prediction_output.configure(state="normal")
        self.prediction_output.insert("end", text)
        self.prediction_output.see("end")
        self.prediction_output.configure(state="disabled")

    def _clear_prediction_output(self):
        self.prediction_output.configure(state="normal")
        self.prediction_output.delete("1.0", "end")
        self.prediction_output.configure(state="disabled")

    def _run_in_thread(self, title: str, worker, log_callback, on_done=None):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Proceso en curso", "Ya hay una tarea ejecutandose.")
            return

        def wrapped():
            writer = _CallbackLogWriter(self.root, log_callback)
            self.root.after(0, self._set_running, True)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    print(f"\n=== {title} ===")
                    worker()
                    print(f"\n=== {title} finalizado ===")
                self.root.after(0, lambda: messagebox.showinfo("Listo", f"{title} finalizado."))
                if on_done:
                    self.root.after(0, on_done)
            except Exception:
                error = traceback.format_exc()
                self.root.after(0, log_callback, f"\n{error}\n")
                self.root.after(0, lambda: messagebox.showerror("Error", f"{title} fallo. Revisa el log."))
            finally:
                self.root.after(0, self._set_running, False)

        self.worker_thread = threading.Thread(target=wrapped, daemon=True)
        self.worker_thread.start()

    def _run_simulation(self):
        try:
            inp_file = Path(self.inp_var.get()).expanduser()
            db_file = Path(self.db_var.get()).expanduser()
            output_csv = Path(self.csv_var.get()).expanduser()
            if not inp_file.exists():
                raise ValueError(f"No existe el archivo .inp: {inp_file}")
            if output_csv == DEFAULT_OUTPUT_CSV and inp_file != DEFAULT_INP_FILE:
                output_csv = network_results_dir(inp_file) / DEFAULT_OUTPUT_CSV.name
                self.csv_var.set(str(output_csv))

            target_nodes = parse_target_nodes(
                self.target_nodes_var.get(),
                all_nodes=self.all_nodes_var.get(),
            )

            deltas = normalize_inflow_multipliers(
                parse_numeric_values(self.deltas_var.get(), "factores multiplicadores"),
                minimum=1.0,
                label="Los factores multiplicadores",
            )
            scenario_mode = self.scenario_mode_var.get()
        except Exception as exc:
            messagebox.showerror("Configuracion invalida", str(exc))
            return

        if self.reset_db_var.get():
            self._close_db_viewer()

        def worker():
            run_experiment(
                inp_file=inp_file,
                db_file=db_file,
                output_csv=output_csv,
                inflow_multipliers=deltas,
                target_nodes=target_nodes,
                scenario_mode=scenario_mode,
                reset_db=self.reset_db_var.get(),
            )

        self._run_in_thread("Corrida SWMM", worker, self.append_log,
                            on_done=self._refresh_results_tree)

    def _train_models(self):
        def worker():
            csv_path = Path(self.csv_var.get()).expanduser()
            artifacts_dir = Path(self.artifacts_dir_var.get()).expanduser()
            if not csv_path.exists():
                raise FileNotFoundError(f"No se encontro el dataset: {csv_path}")

            print(f"Dataset: {csv_path}")
            print(f"Directorio artefactos: {artifacts_dir}")
            print(f"Test size: {ml_train.ML_TEST_SIZE}")
            print(f"Random state: {ml_train.ML_RANDOM_STATE}")
            print(f"CV folds: {ml_train.ML_CV_FOLDS}")
            print(f"Split strategy: {ml_train.ML_SPLIT_STRATEGY}")
            print(f"Group column: {ml_train.ML_GROUP_COLUMN}")
            feature_space, pca_components = ml_train.describe_feature_space()
            print(f"Espacio de features: {feature_space}")
            print(f"Componentes PCA: {pca_components}")
            print(f"Modelos de regresion: {', '.join(ml_train.build_models().keys())}")
            print(f"Modelos de clasificacion: {', '.join(ml_train.build_classification_models().keys())}")
            print(f"Regresor para persistencia: {self.predict_regressor_var.get()}")
            print(f"Clasificador para persistencia: {self.predict_classifier_var.get()}")

            regression_df = ml_train.evaluate_models(
                csv_path=csv_path,
                target=ml_train.ML_TARGET_REGRESSION,
                test_size=ml_train.ML_TEST_SIZE,
                random_state=ml_train.ML_RANDOM_STATE,
                cv_folds=ml_train.ML_CV_FOLDS,
            )
            ml_train.print_results_table(regression_df)
            ml_train.print_regression_scenario_breakdown(regression_df)
            ml_train.save_results(
                regression_df,
                ml_train.ML_TARGET_REGRESSION,
                prefix="regression_comparison",
                output_dir=csv_path.parent,
            )

            classification_df = ml_train.evaluate_classification_models(
                csv_path=csv_path,
                target=ml_train.ML_TARGET_CLASSIFICATION,
                test_size=ml_train.ML_TEST_SIZE,
                random_state=ml_train.ML_RANDOM_STATE,
                cv_folds=ml_train.ML_CV_FOLDS,
            )
            ml_train.print_classification_results_table(classification_df)
            ml_train.print_classification_scenario_breakdown(classification_df)
            ml_train.save_results(
                classification_df,
                ml_train.ML_TARGET_CLASSIFICATION,
                prefix="classification_comparison",
                output_dir=csv_path.parent,
            )
            artifacts = ml_train.fit_and_save_inference_models(
                csv_path=csv_path,
                regressor_name=self.predict_regressor_var.get(),
                classifier_name=self.predict_classifier_var.get(),
                output_dir=artifacts_dir,
            )
            print("\nArtefactos de inferencia actualizados:")
            print(f"  Regresion    : {artifacts['regression'].artifact_path}")
            print(f"  Clasificacion: {artifacts['classification'].artifact_path}")

        self._run_in_thread("Entrenamiento ML", worker, self.append_ml_log)

    def _predict_with_ml(self):
        try:
            predict_source = self.predict_source_var.get()
            flow_values = normalize_inflow_multipliers(
                parse_numeric_values(self.predict_flows_var.get(), "factores de prediccion"),
                minimum=1.0,
                label="Los factores de prediccion",
            )
            target_nodes = parse_target_nodes(
                self.predict_target_nodes_var.get(),
                all_nodes=self.predict_all_nodes_var.get(),
            )
            classifier_name = self.predict_classifier_var.get()
            regressor_name = self.predict_regressor_var.get()
            network_selector = self.predict_network_var.get().strip() or None
            csv_path = Path(self.csv_var.get()).expanduser()
            inp_path = Path(self.predict_inp_var.get()).expanduser()
            artifacts_dir = Path(self.artifacts_dir_var.get()).expanduser()

            if predict_source == "csv":
                if not csv_path.exists():
                    raise ValueError(f"No existe el dataset CSV: {csv_path}")
            else:
                if not inp_path.exists():
                    raise ValueError(f"No existe el archivo .inp: {inp_path}")
                if not artifacts_dir.exists():
                    raise ValueError(
                        f"No existe la carpeta de artefactos: {artifacts_dir}. "
                        "Entrena y guarda modelos primero."
                    )
        except Exception as exc:
            messagebox.showerror("Configuracion invalida", str(exc))
            return

        self._clear_prediction_output()

        def worker():
            print(f"Modo: {'dataset CSV' if predict_source == 'csv' else 'archivo .inp'}")
            print(f"Factores: {flow_values}")
            print(f"Nodos: {'todos' if target_nodes is None else ', '.join(target_nodes)}")
            print(f"Clasificador solicitado: {classifier_name}")
            print(f"Regresor solicitado: {regressor_name}")
            if predict_source == "csv":
                print(f"Dataset: {csv_path}")
                print(f"Red seleccionada: {network_selector or 'auto'}")
                result = predict_steady_flows(
                    inflow_multipliers=flow_values,
                    dataset_csv=csv_path,
                    target_nodes=target_nodes,
                    classifier_name=classifier_name,
                    regressor_name=regressor_name,
                    network_selector=network_selector,
                )
            else:
                print(f"Archivo .inp: {inp_path.name}")
                print(f"Artefactos: {artifacts_dir}")
                result = predict_steady_flows_from_inp(
                    inflow_multipliers=flow_values,
                    inp_file=inp_path,
                    target_nodes=target_nodes,
                    artifacts_dir=artifacts_dir,
                    classifier_name=classifier_name,
                    regressor_name=regressor_name,
                )
            print(f"Clasificador: {result.classifier_name}")
            print(f"Regresor: {result.regressor_name}")
            print()
            display = result.predictions.copy()
            if "inflow_multiplier" in display.columns:
                display = display.rename(columns={"inflow_multiplier": "factor_incremento"})
            elif "delta_inflow_lps" in display.columns:
                display = display.rename(columns={"delta_inflow_lps": "factor_incremento"})
            display["flooded_probability"] = display["flooded_probability"].map(
                lambda value: f"{value:.3f}"
            )
            display["predicted_peak_flooding_lps"] = display[
                "predicted_peak_flooding_lps"
            ].map(lambda value: f"{value:.3f}")
            print(display.to_string(index=False))

            if inp_path.exists():
                print("\nGenerando mapa(s) de inundación ML...")
                net_dir = inp_path.parent
                out_dir = net_dir / "ml" / "results"
                out_dir.mkdir(parents=True, exist_ok=True)

                from swmm_resilience.visualization.flood_map import plot_flood_map
                from swmm_resilience.visualization.runner import _global_vmax

                vmax = None
                if db_path.exists():
                    try:
                        vmax = _global_vmax(db_path)
                    except Exception:
                        pass

                if predict_source == "csv":
                    preds = result.predictions
                    source_col = "inflow_multiplier" if "inflow_multiplier" in preds.columns else "delta_inflow_lps"
                    for mult in flow_values:
                        mult_preds = preds[preds[source_col] == mult].copy()
                        if mult_preds.empty:
                            continue
                        map_df = mult_preds.rename(columns={
                            "predicted_flooded": "flooded",
                            "predicted_peak_flooding_lps": "peak_flooding_lps",
                        })
                        map_df["source"] = "ML Tabular"
                        map_df["inflow_multiplier"] = mult
                        out = out_dir / f"flood_map_qx{mult:.2f}_ml.png"
                        network_name = net_dir.name
                        title = (
                            f"Mapa de inundación — {network_name}\n"
                            f"Factor de caudal: Qx = {mult:.2f} (ML CSV)"
                        )
                        plot_flood_map(
                            node_data=map_df,
                            inp_path=inp_path,
                            output_path=out,
                            title=title,
                            vmax_global=vmax,
                        )
                        print(f"  Guardado: {out}")
                else:
                    for mult in flow_values:
                        map_path = generate_ml_map(
                            inp_path=inp_path,
                            inflow_multiplier=mult,
                            artifacts_dir=artifacts_dir,
                            open_after=True,
                        )
                        print(f"  Guardado: {map_path}")

        self._run_in_thread("Prediccion ML", worker, self.append_prediction_output,
                            on_done=self._refresh_results_tree)

    def _open_db_viewer(self):
        db_path = Path(self.db_var.get()).expanduser()
        if not db_path.exists():
            messagebox.showerror("Base no encontrada", f"No existe la base: {db_path}")
            return

        try:
            from view_db import SQLiteViewerApp

            self._close_db_viewer()
            window = tk.Toplevel(self.root)
            viewer = SQLiteViewerApp(window, db_path)
            self.db_viewer = viewer
            self.db_viewer_window = window
            window.protocol("WM_DELETE_WINDOW", self._close_db_viewer)
        except Exception as exc:
            messagebox.showerror("No se pudo abrir el visor", str(exc))

    def _close_db_viewer(self):
        if self.db_viewer is not None:
            with contextlib.suppress(Exception):
                self.db_viewer.close()
            self.db_viewer = None

        if self.db_viewer_window is not None:
            with contextlib.suppress(Exception):
                if self.db_viewer_window.winfo_exists():
                    self.db_viewer_window.destroy()
            self.db_viewer_window = None


class _CallbackLogWriter:
    def __init__(self, root: tk.Tk, callback):
        self.root = root
        self.callback = callback

    def write(self, text: str):
        if text:
            self.root.after(0, self.callback, text)

    def flush(self):
        return None


def main():
    with _suppress_appkit_warnings():
        root = tk.Tk()
        app = ResilienciaDesktopApp(root)
        root.update()  # force initial render so startup AppKit warnings fire here
    root.mainloop()
