"""
Local Tkinter desktop application for running simulations and ML workflows.
"""

from __future__ import annotations

import contextlib
import re
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from swmm_resilience.config import (
    DEFAULT_DB_FILE,
    DEFAULT_DELTA_INFLOWS_LPS,
    DEFAULT_HYDROGRAPH_FILE,
    DEFAULT_INP_FILE,
    DEFAULT_OUTPUT_CSV,
)
from swmm_resilience.main import run_experiment
from swmm_resilience.ml.predict_tabular import (
    available_classification_models,
    available_regression_models,
    predict_steady_flows,
)
from swmm_resilience.ml import train as ml_train


def parse_lps_values(raw_text: str) -> list[float]:
    """Parse comma-separated values or range(start, stop, step) into L/s values."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Ingresa al menos un caudal en L/s.")

    range_match = re.fullmatch(r"range\(([^,]+),([^,]+),([^)]+)\)", text.replace(" ", ""))
    if range_match:
        start, stop, step = (int(value) for value in range_match.groups())
        values = list(range(start, stop, step))
    else:
        parts = [part for part in re.split(r"[,;\s]+", text) if part]
        values = [float(part) for part in parts]

    if not values:
        raise ValueError("No se pudieron leer caudales en L/s.")
    return values


def parse_target_nodes(raw_text: str, all_nodes: bool) -> list[str] | None:
    """Parse a node list from comma, semicolon, or whitespace separated text."""
    if all_nodes:
        return None

    nodes = [node for node in re.split(r"[,;\s]+", raw_text.strip()) if node]
    if not nodes:
        raise ValueError("Seleccionaste subgrupo, pero no escribiste nodos.")
    return nodes


def parse_hydrograph_multipliers(count_text: str, step_text: str) -> list[float]:
    """Build hydrograph multipliers as [step, 2*step, ..., count*step]."""
    count = int(count_text)
    step = float(step_text)
    if count <= 0:
        raise ValueError("La cantidad de deltas del hidrograma debe ser mayor que cero.")
    if step <= 0:
        raise ValueError("El paso de deltas del hidrograma debe ser mayor que cero.")
    return [round(step * index, 6) for index in range(1, count + 1)]


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
        self.mode_var = tk.StringVar(value="steady")
        self.deltas_var = tk.StringVar(
            value=",".join(str(value) for value in DEFAULT_DELTA_INFLOWS_LPS)
        )
        self.hydrograph_var = tk.StringVar(
            value="" if DEFAULT_HYDROGRAPH_FILE is None else str(DEFAULT_HYDROGRAPH_FILE)
        )
        self.hydrograph_delta_count_var = tk.StringVar(value="1")
        self.hydrograph_delta_step_var = tk.StringVar(value="1")
        self.all_nodes_var = tk.BooleanVar(value=True)
        self.target_nodes_var = tk.StringVar(value="")
        self.reset_db_var = tk.BooleanVar(value=True)
        self.predict_flows_var = tk.StringVar(value="10,20,30")
        self.predict_all_nodes_var = tk.BooleanVar(value=True)
        self.predict_target_nodes_var = tk.StringVar(value="")
        regression_models = available_regression_models()
        classification_models = available_classification_models()
        self.predict_regressor_var = tk.StringVar(value="xgboost" if "xgboost" in regression_models else regression_models[0])
        self.predict_classifier_var = tk.StringVar(
            value=(
                "xgboost_classifier"
                if "xgboost_classifier" in classification_models
                else classification_models[0]
            )
        )
        self.status_var = tk.StringVar(value="Listo")

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")

        self.sim_tab = ttk.Frame(notebook, padding=12)
        self.predict_tab = ttk.Frame(notebook, padding=12)
        self.ml_tab = ttk.Frame(notebook, padding=12)
        self.db_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.sim_tab, text="Cuestionario de corrida")
        notebook.add(self.predict_tab, text="Predicción ML")
        notebook.add(self.ml_tab, text="Entrenamiento ML")
        notebook.add(self.db_tab, text="Base de datos")

        self._build_simulation_tab()
        self._build_prediction_tab()
        self._build_ml_tab()
        self._build_db_tab()

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
            text="Steady flow / caudales constantes",
            value="steady",
            variable=self.mode_var,
            command=self._toggle_mode,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Hidrograma externo",
            value="hydrograph",
            variable=self.mode_var,
            command=self._toggle_mode,
        ).grid(row=0, column=1, sticky="w", padx=(20, 0))

        ttk.Label(scenario, text="Caudales steady L/s").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.deltas_entry = ttk.Entry(scenario, textvariable=self.deltas_var)
        self.deltas_entry.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Label(
            scenario,
            text="Puedes usar: 2,4,6,8  o  range(2,102,2)",
            foreground="#555555",
        ).grid(row=2, column=1, sticky="w")

        ttk.Label(scenario, text="CSV hidrograma").grid(row=3, column=0, sticky="w", pady=(10, 0))
        hydro_frame = ttk.Frame(scenario)
        hydro_frame.grid(row=3, column=1, sticky="ew", pady=(10, 0))
        hydro_frame.columnconfigure(0, weight=1)
        self.hydrograph_entry = ttk.Entry(hydro_frame, textvariable=self.hydrograph_var)
        self.hydrograph_entry.grid(row=0, column=0, sticky="ew")
        self.hydrograph_button = ttk.Button(
            hydro_frame, text="Buscar...", command=self._browse_hydrograph
        )
        self.hydrograph_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Label(
            scenario,
            text="Formato esperado: minute,inflow_lps",
            foreground="#555555",
        ).grid(row=4, column=1, sticky="w")

        hydro_deltas = ttk.Frame(scenario)
        hydro_deltas.grid(row=5, column=1, sticky="w", pady=(10, 0))
        ttk.Label(scenario, text="Deltas hidrograma").grid(row=5, column=0, sticky="w", pady=(10, 0))
        ttk.Label(hydro_deltas, text="Cantidad").grid(row=0, column=0, sticky="w")
        self.hydrograph_delta_count_entry = ttk.Entry(
            hydro_deltas,
            textvariable=self.hydrograph_delta_count_var,
            width=8,
        )
        self.hydrograph_delta_count_entry.grid(row=0, column=1, sticky="w", padx=(6, 14))
        ttk.Label(hydro_deltas, text="Paso multiplicador").grid(row=0, column=2, sticky="w")
        self.hydrograph_delta_step_entry = ttk.Entry(
            hydro_deltas,
            textvariable=self.hydrograph_delta_step_var,
            width=8,
        )
        self.hydrograph_delta_step_entry.grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(
            scenario,
            text="Ejemplo: cantidad=3 y paso=0.5 genera 0.5x, 1.0x, 1.5x sobre todo el hidrograma.",
            foreground="#555555",
        ).grid(row=6, column=1, sticky="w")

        ttk.Label(scenario, text="Nodos a evaluar").grid(row=7, column=0, sticky="w", pady=(10, 0))
        nodes_frame = ttk.Frame(scenario)
        nodes_frame.grid(row=7, column=1, sticky="ew", pady=(10, 0))
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
        ).grid(row=8, column=1, sticky="w")

        ttk.Checkbutton(
            scenario,
            text="Reiniciar base de datos antes de correr",
            variable=self.reset_db_var,
        ).grid(row=9, column=1, sticky="w", pady=(12, 0))

        actions = ttk.Frame(scenario)
        actions.grid(row=10, column=1, sticky="e", pady=(14, 0))
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
            text="Evaluar caudales con modelo ML, sin correr PySWMM",
            padding=12,
        )
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Dataset entrenable").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.csv_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(form, text="Buscar...", command=self._browse_csv_open).grid(row=0, column=2)

        ttk.Label(form, text="Caudales a evaluar L/s").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(form, textvariable=self.predict_flows_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 8), pady=(10, 0)
        )
        ttk.Label(
            form,
            text="Ejemplo: 10,20,30  o  range(2,102,2). Por ahora esta prediccion ML es para steady flow.",
            foreground="#555555",
        ).grid(row=2, column=1, sticky="w", padx=(8, 0))

        ttk.Label(form, text="Nodos").grid(row=3, column=0, sticky="w", pady=(10, 0))
        nodes = ttk.Frame(form)
        nodes.grid(row=3, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        nodes.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            nodes,
            text="Evaluar todos",
            variable=self.predict_all_nodes_var,
            command=self._toggle_prediction_nodes,
        ).grid(row=0, column=0, sticky="w")
        self.predict_target_nodes_entry = ttk.Entry(nodes, textvariable=self.predict_target_nodes_var)
        self.predict_target_nodes_entry.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        ttk.Label(form, text="Modelo clasificación").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.predict_classifier_combo = ttk.Combobox(
            form,
            textvariable=self.predict_classifier_var,
            values=available_classification_models(),
            state="readonly",
            width=28,
        )
        self.predict_classifier_combo.grid(row=4, column=1, sticky="w", padx=(8, 8), pady=(10, 0))

        ttk.Label(form, text="Modelo regresión").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.predict_regressor_combo = ttk.Combobox(
            form,
            textvariable=self.predict_regressor_var,
            values=available_regression_models(),
            state="readonly",
            width=28,
        )
        self.predict_regressor_combo.grid(row=5, column=1, sticky="w", padx=(8, 8), pady=(10, 0))

        actions = ttk.Frame(form)
        actions.grid(row=6, column=1, sticky="e", pady=(12, 0))
        self.predict_button = ttk.Button(actions, text="Predecir con ML", command=self._predict_with_ml)
        self.predict_button.grid(row=0, column=0)
        ttk.Button(actions, text="Limpiar resultados", command=self._clear_prediction_output).grid(
            row=0, column=1, padx=(8, 0)
        )

        self.prediction_output = tk.Text(self.predict_tab, height=22, wrap="none")
        self.prediction_output.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.prediction_output.configure(state="disabled")
        self._toggle_prediction_nodes()

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
        path = filedialog.askopenfilename(
            title="Selecciona archivo SWMM .inp",
            filetypes=[("SWMM input", "*.inp"), ("Todos", "*.*")],
        )
        if path:
            self.inp_var.set(path)

    def _browse_hydrograph(self):
        path = filedialog.askopenfilename(
            title="Selecciona hidrograma CSV",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if path:
            self.hydrograph_var.set(path)

    def _browse_db_save(self):
        path = filedialog.asksaveasfilename(
            title="Selecciona base SQLite",
            defaultextension=".db",
            filetypes=[("SQLite", "*.db"), ("Todos", "*.*")],
        )
        if path:
            self.db_var.set(path)

    def _browse_csv_save(self):
        path = filedialog.asksaveasfilename(
            title="Selecciona dataset CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if path:
            self.csv_var.set(path)

    def _browse_csv_open(self):
        path = filedialog.askopenfilename(
            title="Selecciona dataset CSV",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if path:
            self.csv_var.set(path)

    def _toggle_mode(self):
        hydrograph_enabled = self.mode_var.get() == "hydrograph"
        self.deltas_entry.configure(state="disabled" if hydrograph_enabled else "normal")
        state = "normal" if hydrograph_enabled else "disabled"
        self.hydrograph_entry.configure(state=state)
        self.hydrograph_button.configure(state=state)
        self.hydrograph_delta_count_entry.configure(state=state)
        self.hydrograph_delta_step_entry.configure(state=state)
        self._toggle_nodes()

    def _toggle_nodes(self):
        self.target_nodes_entry.configure(state="disabled" if self.all_nodes_var.get() else "normal")

    def _toggle_prediction_nodes(self):
        self.predict_target_nodes_entry.configure(
            state="disabled" if self.predict_all_nodes_var.get() else "normal"
        )

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

    def _run_in_thread(self, title: str, worker, log_callback):
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

            target_nodes = parse_target_nodes(
                self.target_nodes_var.get(),
                all_nodes=self.all_nodes_var.get(),
            )

            if self.mode_var.get() == "steady":
                deltas = parse_lps_values(self.deltas_var.get())
                hydrograph_file = None
                hydrograph_multipliers = None
            else:
                hydrograph_file = Path(self.hydrograph_var.get()).expanduser()
                if not hydrograph_file.exists():
                    raise ValueError(f"No existe el CSV de hidrograma: {hydrograph_file}")
                deltas = None
                hydrograph_multipliers = parse_hydrograph_multipliers(
                    self.hydrograph_delta_count_var.get(),
                    self.hydrograph_delta_step_var.get(),
                )
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
                delta_inflows_lps=deltas,
                hydrograph_file=hydrograph_file,
                hydrograph_multipliers=hydrograph_multipliers,
                target_nodes=target_nodes,
                reset_db=self.reset_db_var.get(),
            )

        self._run_in_thread("Corrida SWMM", worker, self.append_log)

    def _train_models(self):
        def worker():
            csv_path = Path(self.csv_var.get()).expanduser()
            if not csv_path.exists():
                raise FileNotFoundError(f"No se encontro el dataset: {csv_path}")

            print(f"Dataset: {csv_path}")
            print(f"Test size: {ml_train.ML_TEST_SIZE}")
            print(f"Random state: {ml_train.ML_RANDOM_STATE}")
            print(f"CV folds: {ml_train.ML_CV_FOLDS}")
            print(f"Modelos de regresion: {', '.join(ml_train.build_models().keys())}")
            print(f"Modelos de clasificacion: {', '.join(ml_train.build_classification_models().keys())}")

            regression_df = ml_train.evaluate_models(
                csv_path=csv_path,
                target=ml_train.ML_TARGET_REGRESSION,
                test_size=ml_train.ML_TEST_SIZE,
                random_state=ml_train.ML_RANDOM_STATE,
                cv_folds=ml_train.ML_CV_FOLDS,
            )
            ml_train.print_results_table(regression_df)
            ml_train.save_results(
                regression_df,
                ml_train.ML_TARGET_REGRESSION,
                prefix="regression_comparison",
            )

            classification_df = ml_train.evaluate_classification_models(
                csv_path=csv_path,
                target=ml_train.ML_TARGET_CLASSIFICATION,
                test_size=ml_train.ML_TEST_SIZE,
                random_state=ml_train.ML_RANDOM_STATE,
                cv_folds=ml_train.ML_CV_FOLDS,
            )
            ml_train.print_classification_results_table(classification_df)
            ml_train.save_results(
                classification_df,
                ml_train.ML_TARGET_CLASSIFICATION,
                prefix="classification_comparison",
            )

        self._run_in_thread("Entrenamiento ML", worker, self.append_ml_log)

    def _predict_with_ml(self):
        try:
            csv_path = Path(self.csv_var.get()).expanduser()
            if not csv_path.exists():
                raise ValueError(f"No existe el dataset CSV: {csv_path}")
            flow_values = parse_lps_values(self.predict_flows_var.get())
            target_nodes = parse_target_nodes(
                self.predict_target_nodes_var.get(),
                all_nodes=self.predict_all_nodes_var.get(),
            )
            classifier_name = self.predict_classifier_var.get()
            regressor_name = self.predict_regressor_var.get()
        except Exception as exc:
            messagebox.showerror("Configuracion invalida", str(exc))
            return

        self._clear_prediction_output()

        def worker():
            print(f"Dataset: {csv_path}")
            print(f"Caudales L/s: {flow_values}")
            print(f"Nodos: {'todos' if target_nodes is None else ', '.join(target_nodes)}")
            print(f"Clasificador solicitado: {classifier_name}")
            print(f"Regresor solicitado: {regressor_name}")
            result = predict_steady_flows(
                flow_values_lps=flow_values,
                dataset_csv=csv_path,
                target_nodes=target_nodes,
                classifier_name=classifier_name,
                regressor_name=regressor_name,
            )
            print(f"Clasificador: {result.classifier_name}")
            print(f"Regresor: {result.regressor_name}")
            print()
            display = result.predictions.copy()
            display["flooded_probability"] = display["flooded_probability"].map(
                lambda value: f"{value:.3f}"
            )
            display["predicted_flooding_volume_m3"] = display[
                "predicted_flooding_volume_m3"
            ].map(lambda value: f"{value:.3f}")
            print(display.to_string(index=False))

        self._run_in_thread("Prediccion ML", worker, self.append_prediction_output)

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
    root = tk.Tk()
    ResilienciaDesktopApp(root)
    root.mainloop()
