# English figures for the base deck

Scripts that regenerate, in English, the pictures of
`Presentacion_MLSMFP_05-10_resultados_fase_c2.pptx` whose text was baked into
the PNG in Spanish. They only write to `docs/figures_en/` (see
`docs/figures_en/INDEX.md`); they do not touch any presentation or project code.

Run from the repository root (Windows):

```
venv\Scripts\python.exe scripts\figures_en\hydrograph_panels.py
venv\Scripts\python.exe scripts\figures_en\gantt_schedule.py
venv\Scripts\python.exe scripts\figures_en\flowcharts.py
```

| Script | Output | Source |
|---|---|---|
| `hydrograph_panels.py` | `slide03_hydrograph_shapes_en.png`, `slide06_new_training_shapes_en.png`, `slide11_validation_shapes_en.png` | `data/hydrograph_shapes/*.csv`, `data/hydrograph_validation/*.csv` |
| `gantt_schedule.py` | `slide02_gantt_schedule_en.png` | data block at the top of the script (transcribed from the original picture) |
| `flowcharts.py` | `slide15_flowchart_training_en.png`, `slide15_flowchart_validation_en.png` | node positions measured on the original pictures |

All scripts use the non-interactive `Agg` backend. Output pixel sizes equal
those of the original pictures (`SCALE` in `hydrograph_panels.py` can be raised
for higher resolution with the same aspect ratio). To move a schedule date,
edit `TASKS`, `MILESTONES` or `DEADLINES` in `gantt_schedule.py`.
