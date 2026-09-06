"""Etapa 1 del banco: SQL -> PreparedDataset.

NO-OBJETIVOS de esta etapa, y son deliberados: no imputa, no escala, no
aplica PCA, no elimina filas con nulos permitidos por el contrato, y no
transforma el target. Todo eso pertenece al Pipeline de cada familia, que se
ajusta dentro de cada fold. Si algo de eso aparece en este archivo, es un bug
— y tests/ml/bench/test_stage1_imports.py lo detecta.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import PreparedDataset


def prepare_dataset(db_path: Path, **kwargs) -> PreparedDataset:
    """Implementado en la Task 4."""
    raise NotImplementedError
