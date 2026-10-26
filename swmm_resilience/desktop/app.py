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
        raise ValueError("The step must be greater than zero.")
    while current < stop - 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def parse_numeric_values(raw_text: str, label: str) -> list[float]:
    """Parse comma-separated values or range(start, stop, step) with decimal support."""
    text = raw_text.strip()
    if not text:
        raise ValueError(f"Enter at least one value for {label}.")

    range_match = re.fullmatch(r"range\(([^,]+),([^,]+),([^)]+)\)", text.replace(" ", ""))
    if range_match:
        start, stop, step = (float(value) for value in range_match.groups())
        values = _float_range(start, stop, step)
    else:
        parts = [part for part in re.split(r"[,;\s]+", text) if part]
        values = [float(part) for part in parts]

    if not values:
        raise ValueError(f"Could not read values for {label}.")
    return values


def parse_target_nodes(raw_text: str, all_nodes: bool) -> list[str] | None:
    """Parse a node list from comma, semicolon, or whitespace separated text."""
    if all_nodes:
        return None

    nodes = [node for node in re.split(r"[,;\s]+", raw_text.strip()) if node]
    if not nodes:
        raise ValueError("You selected a subgroup, but did not enter any nodes.")
    return nodes


class ResilienciaDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.worker_thread = None
        self.db_viewer = None
        self.db_viewer_window = None

        self.root.title("resiliencIA - Local tool")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)

        self._build_state()
        self._build_ui()
        self._toggle_mode()
        self.append_log("Ready. Configure the questionnaire and run the simulation.\n")

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
        self.status_var = tk.StringVar(value="Ready")

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
        self.notebook.add(self.sim_tab, text="Run questionnaire")
        self.notebook.add(self.predict_tab, text="ML prediction")
        self.notebook.add(self.ml_tab, text="ML training")
        self.notebook.add(self.db_tab, text="Database")
        self.notebook.add(self.reset_tab, text="Maintenance")
        self.notebook.add(self.results_tab, text="Results")

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

        files = ttk.LabelFrame(self.sim_tab, text="1. Project files", padding=12)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        self._add_path_row(files, 0, ".inp file", self.inp_var, self._browse_inp)
        self._add_path_row(files, 1, "SQLite database", self.db_var, self._browse_db_save)
        self._add_path_row(files, 2, "CSV dataset", self.csv_var, self._browse_csv_save)

        scenario = ttk.LabelFrame(self.sim_tab, text="2. Scenario questionnaire", padding=12)
        scenario.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        scenario.columnconfigure(1, weight=1)

        ttk.Label(scenario, text="Evaluation type").grid(row=0, column=0, sticky="w")
        mode_frame = ttk.Frame(scenario)
        mode_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Internal hydrograph (TIMESERIES)",
            value=SCENARIO_MODE_TIMESERIES,
            variable=self.scenario_mode_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Steady flow (Baseline in INFLOWS)",
            value=SCENARIO_MODE_STEADY,
            variable=self.scenario_mode_var,
        ).grid(row=0, column=1, sticky="w", padx=(16, 0))
        ttk.Label(
            scenario,
            text=(
                "Use 'Internal hydrograph' if the .inp takes flows from [TIMESERIES]. "
                "Use 'Steady flow' if the flow is in Baseline within [INFLOWS]."
            ),
            foreground="#555555",
            wraplength=760,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))

        ttk.Label(scenario, text="Multiplier factors").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.deltas_entry = ttk.Entry(scenario, textvariable=self.deltas_var)
        self.deltas_entry.grid(row=2, column=1, sticky="ew", pady=(10, 0))
        ttk.Label(
            scenario,
            text=(
                "You can use: 1,1.5,2  or  range(1,4,0.5). "
                "3 = triple the series or the Baseline, depending on the chosen mode."
            ),
            foreground="#555555",
        ).grid(row=3, column=1, sticky="w")

        ttk.Label(scenario, text="Nodes to evaluate").grid(row=4, column=0, sticky="w", pady=(10, 0))
        nodes_frame = ttk.Frame(scenario)
        nodes_frame.grid(row=4, column=1, sticky="ew", pady=(10, 0))
        nodes_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            nodes_frame,
            text="Apply to all nodes",
            variable=self.all_nodes_var,
            command=self._toggle_nodes,
        ).grid(row=0, column=0, sticky="w")
        self.target_nodes_entry = ttk.Entry(nodes_frame, textvariable=self.target_nodes_var)
        self.target_nodes_entry.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        ttk.Label(
            scenario,
            text="If you uncheck all: enter comma-separated IDs, for example J1,J2,J3",
            foreground="#555555",
        ).grid(row=5, column=1, sticky="w")

        ttk.Checkbutton(
            scenario,
            text="Replace existing database before running",
            variable=self.reset_db_var,
        ).grid(row=6, column=1, sticky="w", pady=(12, 0))
        ttk.Label(
            scenario,
            text=(
                "If this option is unchecked, the run is added to the current database "
                "(append). If checked, the database is reset and the subsequent training "
                "will use only the new runs."
            ),
            foreground="#555555",
            wraplength=760,
        ).grid(row=7, column=1, sticky="w", pady=(4, 0))

        actions = ttk.Frame(scenario)
        actions.grid(row=8, column=1, sticky="e", pady=(14, 0))
        self.run_button = ttk.Button(actions, text="Run simulation", command=self._run_simulation)
        self.run_button.grid(row=0, column=0)
        ttk.Button(actions, text="Clear log", command=self._clear_log).grid(row=0, column=1, padx=(8, 0))

        self.log = tk.Text(self.sim_tab, height=18, wrap="word")
        self.log.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.log.configure(state="disabled")

    def _build_ml_tab(self):
        self.ml_tab.columnconfigure(0, weight=1)
        self.ml_tab.rowconfigure(1, weight=1)

        intro = ttk.LabelFrame(self.ml_tab, text="Training", padding=12)
        intro.grid(row=0, column=0, sticky="ew")
        intro.columnconfigure(0, weight=1)
        ttk.Label(
            intro,
            text=(
                "Trains and compares models using the configured dataset. "
                "Run the simulations first to generate/update the CSV."
            ),
            wraplength=900,
        ).grid(row=0, column=0, sticky="w")
        self.train_button = ttk.Button(intro, text="Train ML models", command=self._train_models)
        self.train_button.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.ml_log = tk.Text(self.ml_tab, height=22, wrap="word")
        self.ml_log.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.ml_log.configure(state="disabled")

    def _build_prediction_tab(self):
        self.predict_tab.columnconfigure(0, weight=1)
        self.predict_tab.rowconfigure(1, weight=1)

        form = ttk.LabelFrame(
            self.predict_tab,
            text="Tabular ML inference without running PySWMM",
            padding=12,
        )
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Inference mode").grid(row=0, column=0, sticky="w")
        source_frame = ttk.Frame(form)
        source_frame.grid(row=0, column=1, sticky="w", padx=(8, 8))
        ttk.Radiobutton(
            source_frame,
            text="From CSV dataset",
            value="csv",
            variable=self.predict_source_var,
            command=self._toggle_prediction_source,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            source_frame,
            text="From .inp file",
            value="inp",
            variable=self.predict_source_var,
            command=self._toggle_prediction_source,
        ).grid(row=0, column=1, sticky="w", padx=(20, 0))

        ttk.Label(form, text="Trainable dataset").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.predict_csv_entry = ttk.Entry(form, textvariable=self.csv_var)
        self.predict_csv_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        self.predict_csv_button = ttk.Button(form, text="Browse...", command=self._browse_csv_open)
        self.predict_csv_button.grid(row=1, column=2, pady=(10, 0))

        ttk.Label(form, text="Network within the CSV").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.predict_network_entry = ttk.Entry(form, textvariable=self.predict_network_var)
        self.predict_network_entry.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        ttk.Label(
            form,
            text="Optional if the CSV contains a single network. Use the .inp name or the network_hash prefix.",
            foreground="#555555",
        ).grid(row=3, column=1, sticky="w", padx=(8, 0))

        ttk.Label(form, text=".inp network to evaluate").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.predict_inp_entry = ttk.Entry(form, textvariable=self.predict_inp_var)
        self.predict_inp_entry.grid(row=4, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        self.predict_inp_button = ttk.Button(form, text="Browse...", command=self._browse_predict_inp)
        self.predict_inp_button.grid(row=4, column=2, pady=(10, 0))

        ttk.Label(form, text="Trained artifacts").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.artifacts_dir_entry = ttk.Entry(form, textvariable=self.artifacts_dir_var)
        self.artifacts_dir_entry.grid(row=5, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        self.artifacts_dir_button = ttk.Button(
            form,
            text="Browse...",
            command=self._browse_artifacts_dir,
        )
        self.artifacts_dir_button.grid(row=5, column=2, pady=(10, 0))

        ttk.Label(form, text="Factors to evaluate").grid(row=6, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(form, textvariable=self.predict_flows_var).grid(
            row=6, column=1, sticky="ew", padx=(8, 8), pady=(10, 0)
        )
        ttk.Label(
            form,
            text=(
                "Example: 1.0,1.5,2.0  or  range(1,3,0.25). "
                "For now, direct ML inference is only for steady flow."
            ),
            foreground="#555555",
        ).grid(row=7, column=1, sticky="w", padx=(8, 0))

        ttk.Label(form, text="Nodes").grid(row=8, column=0, sticky="w", pady=(10, 0))
        nodes = ttk.Frame(form)
        nodes.grid(row=8, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        nodes.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            nodes,
            text="Evaluate all",
            variable=self.predict_all_nodes_var,
            command=self._toggle_prediction_nodes,
        ).grid(row=0, column=0, sticky="w")
        self.predict_target_nodes_entry = ttk.Entry(nodes, textvariable=self.predict_target_nodes_var)
        self.predict_target_nodes_entry.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        ttk.Label(form, text="Classification model").grid(row=9, column=0, sticky="w", pady=(10, 0))
        self.predict_classifier_combo = ttk.Combobox(
            form,
            textvariable=self.predict_classifier_var,
            values=available_classification_models(),
            state="readonly",
            width=28,
        )
        self.predict_classifier_combo.grid(row=9, column=1, sticky="w", padx=(8, 8), pady=(10, 0))

        ttk.Label(form, text="Regression model").grid(row=10, column=0, sticky="w", pady=(10, 0))
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
        self.predict_button = ttk.Button(actions, text="Predict with ML", command=self._predict_with_ml)
        self.predict_button.grid(row=0, column=0)
        ttk.Button(actions, text="Clear results", command=self._clear_prediction_output).grid(
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
            self.reset_tab, text="What do you want to clear?", padding=12
        )
        options_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self._reset_vars: dict[str, tk.BooleanVar] = {
            key: tk.BooleanVar(value=False) for key in RESET_CATEGORIES
        }

        labels = {
            "db":        "Database (runs and results)",
            "plots":     "Generated images (PNG maps)",
            "dataset":   "dataset_ml.csv files",
            "artifacts": "ML artifacts (.joblib, manifest.json, metrics)",
            "temporal":  "Time series (Parquet files)",
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
        ttk.Button(btn_row, text="Select all / None", command=_toggle_all).pack(side="left")

        warning = ttk.Label(
            options_frame,
            text="⚠ This operation is irreversible. It affects ALL networks.",
            foreground="red",
        )
        warning.grid(row=len(labels) + 1, column=0, sticky="w", pady=(8, 0))

        ttk.Button(
            options_frame, text="Clear selected", command=self._run_reset
        ).grid(row=len(labels) + 2, column=0, sticky="w", pady=(10, 0))

        self.reset_log = tk.Text(self.reset_tab, height=20, wrap="word")
        self.reset_log.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        self.reset_log.configure(state="disabled")

    def _build_results_tab(self):
        self.results_tab.columnconfigure(0, weight=0)
        self.results_tab.columnconfigure(1, weight=1)
        self.results_tab.rowconfigure(0, weight=1)
        self.results_tab.rowconfigure(1, weight=0)

        left = ttk.LabelFrame(self.results_tab, text="Folders", width=200)
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
            "", "end", text="Surrogate (CNN)", open=True
        )
        self._lstm_root = self.results_tree.insert(
            "", "end", text="Surrogate (LSTM)", open=True
        )

        ttk.Button(left, text="\u21ba Refresh", command=self._refresh_results_tree).pack(
            pady=(0, 6)
        )

        right = ttk.LabelFrame(self.results_tab, text="Preview")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(
            right, text="Select an image", anchor="center", background="#f0f0f0"
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.surr_model_var = tk.StringVar(value="cnn")

        pred = ttk.LabelFrame(self.results_tab, text="Surrogate Predictor", padding=12)
        pred.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        pred.columnconfigure(1, weight=1)
        pred.columnconfigure(3, weight=1)

        ttk.Label(pred, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        model_combo = ttk.Combobox(
            pred, textvariable=self.surr_model_var,
            values=["cnn", "lstm"], state="readonly", width=8,
        )
        model_combo.grid(row=0, column=1, sticky="w")
        model_combo.bind("<<ComboboxSelected>>", lambda e: self._update_predict_button_state())

        ttk.Label(pred, text="Multiplier:").grid(row=0, column=2, sticky="w", padx=(16, 8))
        self.surr_mult_var = tk.StringVar(value="1.0")
        ttk.Entry(pred, textvariable=self.surr_mult_var, width=12).grid(
            row=0, column=3, sticky="w"
        )

        ttk.Label(pred, text=".inp network:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.surr_inp_entry = ttk.Entry(pred, textvariable=self.predict_inp_var)
        self.surr_inp_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(pred, text="Browse...", command=self._browse_predict_inp).grid(
            row=1, column=3, padx=(6, 0), pady=(6, 0)
        )

        self.surr_predict_all_btn = ttk.Button(
            pred, text="Regenerate all", command=self._predict_surrogate_all
        )
        self.surr_predict_all_btn.grid(row=2, column=1, sticky="w", pady=(10, 0))

        self.surr_predict_btn = ttk.Button(
            pred, text="Predict and show", command=self._predict_surrogate
        )
        self.surr_predict_btn.grid(row=2, column=3, sticky="e", pady=(10, 0))

        self.results_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self._refresh_results_tree()
        self._update_predict_button_state()

        # ── Temporal Windows Summary ────────────────────────────────────────────
        win_frame = ttk.LabelFrame(self.results_tab, text="Time Window Summary", padding=12)
        win_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        win_frame.columnconfigure(0, weight=1)

        self.win_summary_btn = ttk.Button(
            win_frame, text="Generate Summary", command=self._generate_window_summary
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
        self.win_tree.heading("duration_min", text="Duration (min)")
        self.win_tree.heading("time_skip_days", text="Skip (days)")
        self.win_tree.heading("mean_capacity", text="Mean Cap. (LPS)")
        self.win_tree.heading("n_peaks", text="Peaks")
        self.win_tree.heading("n_drains", text="Drains")
        self.win_tree.heading("n_swales", text="Swales")

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
            self.image_label.configure(text=f"Error loading image:\n{exc}")

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
                raise ValueError("The multiplier must be greater than zero.")
            inp_path = Path(self.predict_inp_var.get()).expanduser()
            if not inp_path.exists():
                raise ValueError(f"The .inp file does not exist: {inp_path}")
            db_path = Path(self.db_var.get()).expanduser()
            model_type = self.surr_model_var.get()
        except Exception as exc:
            messagebox.showerror("Invalid value", str(exc))
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
            print(f"Map saved: {map_path}")
            self.root.after(0, self._on_surrogate_done, str(map_path))

        self._run_in_thread("Surrogate prediction", worker, self.append_log)

    def _predict_surrogate_all(self):
        """Regenerate surrogate maps for all default multipliers."""
        try:
            inp_path = Path(self.predict_inp_var.get()).expanduser()
            if not inp_path.exists():
                raise ValueError(f"The .inp file does not exist: {inp_path}")
            db_path = Path(self.db_var.get()).expanduser()
            model_type = self.surr_model_var.get()
        except Exception as exc:
            messagebox.showerror("Invalid value", str(exc))
            return

        def worker():
            import matplotlib
            matplotlib.use("Agg")

            from swmm_resilience.ml.temporal.predict import (
                plot_surrogate_map,
                predict_surrogate_from_multiplier,
            )
            for multiplier in DEFAULT_INFLOW_MULTIPLIERS:
                print(f"Generating map Qx{multiplier:.2f}...")
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

        self._run_in_thread("Regenerate all maps", worker, self.append_log)

    def _on_regenerate_all_done(self):
        """After regenerating all maps: refresh tree."""
        self._refresh_results_tree()
        self.notebook.select(self.results_tab)
        messagebox.showinfo("Done", "All surrogate maps were regenerated.")

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

        self._run_in_thread("Window summary", worker, self.append_log)

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
            messagebox.showwarning("No selection", "Select at least one category to clear.")
            return

        labels = {
            "db":        "Database",
            "plots":     "PNG images",
            "dataset":   "dataset_ml.csv",
            "artifacts": "ML artifacts",
            "temporal":  "Temporal Parquet files",
        }
        items = "\n  • ".join(labels[k] for k in selected)
        confirmed = messagebox.askyesno(
            "Confirm cleanup",
            f"The following will be permanently deleted:\n\n  • {items}\n\nContinue?",
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

        self._run_in_thread("Cleanup", worker, self.append_reset_log)

    def _build_db_tab(self):
        self.db_tab.columnconfigure(0, weight=1)

        panel = ttk.LabelFrame(self.db_tab, text="SQLite viewer", padding=12)
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="SQLite database").grid(row=0, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.db_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(panel, text="Browse...", command=self._browse_db_save).grid(row=0, column=2)
        ttk.Button(panel, text="Open DB viewer", command=self._open_db_viewer).grid(
            row=1, column=1, sticky="e", pady=(12, 0)
        )

    def _add_path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(0, 6))
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, pady=(0, 6))

    def _browse_inp(self):
        with _suppress_appkit_warnings():
            path = filedialog.askopenfilename(
                title="Select SWMM .inp file",
                filetypes=[("SWMM input", "*.inp"), ("All files", "*.*")],
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
                title="Select SWMM .inp file for inference",
                filetypes=[("SWMM input", "*.inp"), ("All files", "*.*")],
            )
        if path:
            self.predict_inp_var.set(path)
            self._update_predict_button_state()

    def _browse_db_save(self):
        with _suppress_appkit_warnings():
            path = filedialog.asksaveasfilename(
                title="Select SQLite database",
                defaultextension=".db",
                filetypes=[("SQLite", "*.db"), ("All files", "*.*")],
            )
        if path:
            self.db_var.set(path)

    def _browse_csv_save(self):
        with _suppress_appkit_warnings():
            path = filedialog.asksaveasfilename(
                title="Select CSV dataset",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            )
        if path:
            self.csv_var.set(path)
            self._sync_artifacts_dir()

    def _browse_csv_open(self):
        with _suppress_appkit_warnings():
            path = filedialog.askopenfilename(
                title="Select CSV dataset",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            )
        if path:
            self.csv_var.set(path)
            self._sync_artifacts_dir()

    def _browse_artifacts_dir(self):
        with _suppress_appkit_warnings():
            path = filedialog.askdirectory(title="Select ML artifacts folder")
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
        self.status_var.set("Running..." if running else "Ready")

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
            messagebox.showwarning("Process running", "A task is already running.")
            return

        def wrapped():
            writer = _CallbackLogWriter(self.root, log_callback)
            self.root.after(0, self._set_running, True)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    print(f"\n=== {title} ===")
                    worker()
                    print(f"\n=== {title} finished ===")
                self.root.after(0, lambda: messagebox.showinfo("Done", f"{title} finished."))
                if on_done:
                    self.root.after(0, on_done)
            except Exception:
                error = traceback.format_exc()
                self.root.after(0, log_callback, f"\n{error}\n")
                self.root.after(0, lambda: messagebox.showerror("Error", f"{title} failed. Check the log."))
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
                raise ValueError(f"The .inp file does not exist: {inp_file}")
            if output_csv == DEFAULT_OUTPUT_CSV and inp_file != DEFAULT_INP_FILE:
                output_csv = network_results_dir(inp_file) / DEFAULT_OUTPUT_CSV.name
                self.csv_var.set(str(output_csv))

            target_nodes = parse_target_nodes(
                self.target_nodes_var.get(),
                all_nodes=self.all_nodes_var.get(),
            )

            deltas = normalize_inflow_multipliers(
                parse_numeric_values(self.deltas_var.get(), "multiplier factors"),
                minimum=1.0,
                label="The multiplier factors",
            )
            scenario_mode = self.scenario_mode_var.get()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
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

        self._run_in_thread("SWMM run", worker, self.append_log,
                            on_done=self._refresh_results_tree)

    def _train_models(self):
        def worker():
            csv_path = Path(self.csv_var.get()).expanduser()
            artifacts_dir = Path(self.artifacts_dir_var.get()).expanduser()
            if not csv_path.exists():
                raise FileNotFoundError(f"Dataset not found: {csv_path}")

            print(f"Dataset: {csv_path}")
            print(f"Artifacts directory: {artifacts_dir}")
            print(f"Test size: {ml_train.ML_TEST_SIZE}")
            print(f"Random state: {ml_train.ML_RANDOM_STATE}")
            print(f"CV folds: {ml_train.ML_CV_FOLDS}")
            print(f"Split strategy: {ml_train.ML_SPLIT_STRATEGY}")
            print(f"Group column: {ml_train.ML_GROUP_COLUMN}")
            feature_space, pca_components = ml_train.describe_feature_space()
            print(f"Feature space: {feature_space}")
            print(f"PCA components: {pca_components}")
            print(f"Regression models: {', '.join(ml_train.build_models().keys())}")
            print(f"Classification models: {', '.join(ml_train.build_classification_models().keys())}")
            print(f"Regressor to persist: {self.predict_regressor_var.get()}")
            print(f"Classifier to persist: {self.predict_classifier_var.get()}")

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
            print("\nInference artifacts updated:")
            print(f"  Regression    : {artifacts['regression'].artifact_path}")
            print(f"  Classification: {artifacts['classification'].artifact_path}")

        self._run_in_thread("ML training", worker, self.append_ml_log)

    def _predict_with_ml(self):
        try:
            predict_source = self.predict_source_var.get()
            flow_values = normalize_inflow_multipliers(
                parse_numeric_values(self.predict_flows_var.get(), "prediction factors"),
                minimum=1.0,
                label="The prediction factors",
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
            db_path = Path(self.db_var.get()).expanduser()

            if predict_source == "csv":
                if not csv_path.exists():
                    raise ValueError(f"The CSV dataset does not exist: {csv_path}")
            else:
                if not inp_path.exists():
                    raise ValueError(f"The .inp file does not exist: {inp_path}")
                if not artifacts_dir.exists():
                    raise ValueError(
                        f"The artifacts folder does not exist: {artifacts_dir}. "
                        "Train and save models first."
                    )
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self._clear_prediction_output()

        def worker():
            import matplotlib
            matplotlib.use("Agg")

            print(f"Mode: {'CSV dataset' if predict_source == 'csv' else '.inp file'}")
            print(f"Factors: {flow_values}")
            print(f"Nodes: {'all' if target_nodes is None else ', '.join(target_nodes)}")
            print(f"Requested classifier: {classifier_name}")
            print(f"Requested regressor: {regressor_name}")
            if predict_source == "csv":
                print(f"Dataset: {csv_path}")
                print(f"Selected network: {network_selector or 'auto'}")
                result = predict_steady_flows(
                    inflow_multipliers=flow_values,
                    dataset_csv=csv_path,
                    target_nodes=target_nodes,
                    classifier_name=classifier_name,
                    regressor_name=regressor_name,
                    network_selector=network_selector,
                )
            else:
                print(f".inp file: {inp_path.name}")
                print(f"Artifacts: {artifacts_dir}")
                result = predict_steady_flows_from_inp(
                    inflow_multipliers=flow_values,
                    inp_file=inp_path,
                    target_nodes=target_nodes,
                    artifacts_dir=artifacts_dir,
                    classifier_name=classifier_name,
                    regressor_name=regressor_name,
                )
            print(f"Classifier: {result.classifier_name}")
            print(f"Regressor: {result.regressor_name}")
            print()
            display = result.predictions.copy()
            if "inflow_multiplier" in display.columns:
                display = display.rename(columns={"inflow_multiplier": "factor_incremento"})
            elif "delta_inflow_lps" in display.columns:
                display = display.rename(columns={"delta_inflow_lps": "factor_incremento"})
            display["flooded_probability"] = display["flooded_probability"].map(
                lambda value: f"{value:.3f}"
            )
            display["predicted_total_flood_volume_m3"] = display[
                "predicted_total_flood_volume_m3"
            ].map(lambda value: f"{value:.3f}")
            print(display.to_string(index=False))

            if inp_path.exists():
                print("\nGenerating ML flood map(s)...")
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
                            "predicted_total_flood_volume_m3": "total_flood_volume_m3",
                        })
                        map_df["source"] = "ML Tabular"
                        map_df["inflow_multiplier"] = mult
                        out = out_dir / f"flood_map_qx{mult:.2f}_ml.png"
                        network_name = net_dir.name
                        title = (
                            f"Flood Map - {network_name}\n"
                            f"Flow Multiplier: Qx = {mult:.2f} (ML CSV)"
                        )
                        plot_flood_map(
                            node_data=map_df,
                            inp_path=inp_path,
                            output_path=out,
                            title=title,
                            vmax_global=vmax,
                        )
                        print(f"  Saved: {out}")
                else:
                    for mult in flow_values:
                        map_path = generate_ml_map(
                            inp_path=inp_path,
                            inflow_multiplier=mult,
                            artifacts_dir=artifacts_dir,
                            open_after=True,
                        )
                        print(f"  Saved: {map_path}")

        self._run_in_thread("ML prediction", worker, self.append_prediction_output,
                            on_done=self._refresh_results_tree)

    def _open_db_viewer(self):
        db_path = Path(self.db_var.get()).expanduser()
        if not db_path.exists():
            messagebox.showerror("Database not found", f"The database does not exist: {db_path}")
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
            messagebox.showerror("Could not open the viewer", str(exc))

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
