"""
Quick SQLite viewer for the SWMM resilience database.
"""

import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from swmm_resilience.config import DEFAULT_DB_FILE
from swmm_resilience.database.schema import create_schema


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_CANDIDATES = [
    DEFAULT_DB_FILE,
    BASE_DIR / "swmm_resilience.db",
]
DEFAULT_LIMIT = 300


def resolve_db_path() -> Path:
    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_DB_CANDIDATES[0]


class SQLiteViewerApp:
    def __init__(self, root: tk.Tk, db_path: Path):
        self.root = root
        self.db_path = db_path
        self.conn = None
        self.current_columns = []
        self.run_label_to_id = {"(todos)": None}
        self.run_id_to_flow = {}

        self.root.title("Visor SQLite - SWMM Resilience")
        self.root.geometry("1380x820")
        self.root.minsize(1100, 650)

        self._build_ui()
        self._connect_db()
        self._load_metadata()
        self._refresh_summary()
        self._load_table()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        top = ttk.Frame(self.root, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Base de datos:").grid(row=0, column=0, sticky="w")
        self.db_label = ttk.Label(top, text=str(self.db_path.resolve()))
        self.db_label.grid(row=0, column=1, sticky="w")

        self.summary_vars = {
            "runs": tk.StringVar(value="runs: -"),
            "node_results": tk.StringVar(value="node_results: -"),
            "link_results": tk.StringVar(value="link_results: -"),
            "run_summary": tk.StringVar(value="run_summary: -"),
        }

        summary = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        summary.grid(row=1, column=0, sticky="ew")
        for idx, key in enumerate(self.summary_vars):
            ttk.Label(
                summary,
                textvariable=self.summary_vars[key],
                relief="ridge",
                padding=(12, 8),
                width=24,
                anchor="center",
            ).grid(row=0, column=idx, padx=(0 if idx == 0 else 8, 0), sticky="ew")
            summary.columnconfigure(idx, weight=1)

        controls = ttk.LabelFrame(self.root, text="Explorador", padding=12)
        controls.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(2, weight=1)

        filter_bar = ttk.Frame(controls)
        filter_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        filter_bar.columnconfigure(7, weight=1)

        ttk.Label(filter_bar, text="Tabla").grid(row=0, column=0, sticky="w")
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(
            filter_bar,
            textvariable=self.table_var,
            state="readonly",
            width=24,
        )
        self.table_combo.grid(row=0, column=1, sticky="w", padx=(6, 14))
        self.table_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_table())

        ttk.Label(filter_bar, text="run_id").grid(row=0, column=2, sticky="w")
        self.run_var = tk.StringVar()
        self.run_combo = ttk.Combobox(
            filter_bar,
            textvariable=self.run_var,
            state="readonly",
            width=38,
        )
        self.run_combo.grid(row=0, column=3, sticky="w", padx=(6, 14))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_table())

        ttk.Label(filter_bar, text="Limite").grid(row=0, column=4, sticky="w")
        self.limit_var = tk.StringVar(value=str(DEFAULT_LIMIT))
        ttk.Entry(filter_bar, textvariable=self.limit_var, width=8).grid(
            row=0, column=5, sticky="w", padx=(6, 14)
        )

        ttk.Button(filter_bar, text="Cargar", command=self._load_table).grid(
            row=0, column=6, sticky="w"
        )
        ttk.Button(filter_bar, text="Refrescar BD", command=self._refresh_all).grid(
            row=0, column=7, sticky="e", padx=(10, 0)
        )

        self.info_var = tk.StringVar(value="Listo.")
        ttk.Label(controls, textvariable=self.info_var).grid(
            row=1, column=0, sticky="w", pady=(0, 8)
        )

        table_frame = ttk.Frame(controls)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

    def _connect_db(self):
        if not self.db_path.exists():
            raise FileNotFoundError(f"No se encontro la base de datos: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        create_schema(self.conn)

    def _fetch_one(self, query: str, params=()):
        cur = self.conn.cursor()
        return cur.execute(query, params).fetchone()

    def _fetch_all(self, query: str, params=()):
        cur = self.conn.cursor()
        return cur.execute(query, params).fetchall()

    def _load_metadata(self):
        tables = [
            row[0]
            for row in self._fetch_all(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        if not tables:
            raise RuntimeError("La base de datos no tiene tablas visibles.")

        self.table_combo["values"] = tables
        self.table_var.set("runs" if "runs" in tables else tables[0])

        if "runs" in tables:
            run_rows = self._fetch_all(
                """
                SELECT run_id, delta_inflow_lps, executed_at, status
                FROM runs
                ORDER BY executed_at DESC
                """
            )
        else:
            run_rows = []

        self.run_label_to_id = {"(todos)": None}
        self.run_id_to_flow = {}
        run_options = ["(todos)"]
        for run_id, delta_inflow_lps, executed_at, status in run_rows:
            flow_text = "-" if delta_inflow_lps is None else f"{float(delta_inflow_lps):.4f} L/s"
            label = f"{run_id} | Q={flow_text} | {status} | {executed_at}"
            self.run_label_to_id[label] = run_id
            self.run_id_to_flow[run_id] = delta_inflow_lps
            run_options.append(label)
        self.run_combo["values"] = run_options
        self.run_var.set("(todos)")

    def _refresh_summary(self):
        for table_name, var in self.summary_vars.items():
            try:
                count = self._fetch_one(f"SELECT COUNT(*) FROM {table_name}")[0]
                var.set(f"{table_name}: {count}")
            except sqlite3.Error:
                var.set(f"{table_name}: -")

    def _refresh_all(self):
        try:
            self._load_metadata()
            self._refresh_summary()
            self._load_table()
        except Exception as exc:
            messagebox.showerror("Error al refrescar", str(exc))

    def _table_has_run_id(self, table_name: str) -> bool:
        cols = self._fetch_all(f"PRAGMA table_info({table_name})")
        return any(col[1] == "run_id" for col in cols)

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree["columns"] = ()

    def _selected_run_id(self) -> str | None:
        label = self.run_var.get().strip()
        if not label or label == "(todos)":
            return None
        return self.run_label_to_id.get(label, label)

    def _load_table(self):
        table_name = self.table_var.get().strip()
        if not table_name:
            return

        try:
            limit = int(self.limit_var.get())
        except ValueError:
            messagebox.showwarning("Limite invalido", "Ingresa un numero entero en el limite.")
            return

        if limit <= 0:
            messagebox.showwarning("Limite invalido", "El limite debe ser mayor que cero.")
            return

        where = ""
        params = []
        selected_run_id = self._selected_run_id()
        if selected_run_id and self._table_has_run_id(table_name):
            where = " WHERE run_id = ?"
            params.append(selected_run_id)

        order_by = ""
        cols = {col[1] for col in self._fetch_all(f"PRAGMA table_info({table_name})")}
        if "delta_inflow_lps" in cols:
            order_by = " ORDER BY delta_inflow_lps, rowid"
        elif "run_id" in cols:
            order_by = " ORDER BY run_id, rowid"

        query = f"SELECT * FROM {table_name}{where}{order_by} LIMIT ?"
        params.append(limit)

        try:
            cur = self.conn.cursor()
            rows = cur.execute(query, params).fetchall()
            columns = [desc[0] for desc in cur.description]
        except sqlite3.Error as exc:
            messagebox.showerror("Error SQLite", str(exc))
            return

        self._render_rows(columns, rows)

        row_count = len(rows)
        selected_flow = self.run_id_to_flow.get(selected_run_id)
        flow_text = "(todos)"
        if selected_run_id:
            flow_text = "-" if selected_flow is None else f"{float(selected_flow):.4f} L/s"
        self.info_var.set(
            f"Tabla: {table_name} | Filas mostradas: {row_count} | "
            f"Filtro run_id: {selected_run_id or '(todos)'} | "
            f"Caudal: {flow_text} | Limite: {limit}"
        )

    def _render_rows(self, columns, rows):
        self._clear_tree()
        self.current_columns = columns
        self.tree["columns"] = columns

        for col in columns:
            heading = "caudal_inyectado_lps" if col == "delta_inflow_lps" else col
            self.tree.heading(col, text=heading)
            width = 130
            if col.endswith("_id"):
                width = 220
            elif "executed_at" in col:
                width = 170
            elif col == "delta_inflow_lps":
                width = 160
            elif col in {"network_file", "scenario_type", "spatial_pattern"}:
                width = 220
            self.tree.column(col, width=width, minwidth=90, stretch=True, anchor="w")

        for row in rows:
            normalized = ["" if value is None else str(value) for value in row]
            self.tree.insert("", "end", values=normalized)

    def close(self):
        if self.conn is not None:
            self.conn.close()


def main():
    root = tk.Tk()
    try:
        app = SQLiteViewerApp(root, resolve_db_path())
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("No se pudo abrir el visor", str(exc))
        root.destroy()
        return

    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
