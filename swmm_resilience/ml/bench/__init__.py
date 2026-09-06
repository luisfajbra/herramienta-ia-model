"""Banco ML: preprocesamiento, entrenamiento y evaluación multi-modelo.

Las tres etapas son módulos separados con una única dirección de dependencia.
Ver docs/superpowers/specs/2026-09-05-ml-bench-preproc-train-eval-design.md.
"""

from .schemas import PreparedDataset

__all__ = ["PreparedDataset"]
