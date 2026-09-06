"""La etapa 1 no puede ajustar transformadores: garantía anti-fuga estructural.

Un splitter (sklearn.model_selection) no ajusta nada y sí está permitido —
folds.py lo necesita para reproducir exactamente los folds de evaluator.py.
Cualquier otro submódulo de sklearn, xgboost o torch queda prohibido.
"""

import ast
from pathlib import Path

import pytest

STAGE1_MODULES = ("preprocess.py", "folds.py", "quality.py")

ALLOWED_SKLEARN_PREFIXES = ("sklearn.model_selection",)
FORBIDDEN_ROOTS = ("xgboost", "torch")

BENCH_DIR = Path(__file__).resolve().parents[3] / "swmm_resilience" / "ml" / "bench"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("module_name", STAGE1_MODULES)
def test_stage1_module_imports_no_transformer_or_estimator(module_name):
    path = BENCH_DIR / module_name
    assert path.exists(), f"{module_name} no existe todavía"

    offenders = []
    for imported in _imported_modules(path):
        root = imported.split(".")[0]
        if root in FORBIDDEN_ROOTS:
            offenders.append(imported)
        elif root == "sklearn" and not imported.startswith(ALLOWED_SKLEARN_PREFIXES):
            offenders.append(imported)

    assert not offenders, (
        f"{module_name} importa {offenders}. La etapa 1 sólo puede importar de "
        "sklearn.model_selection; imputar o escalar aquí introduciría fuga de "
        "datos porque se ajustaría sobre el dataset completo. Ver spec §5.1."
    )


def test_the_guard_would_catch_a_real_violation(tmp_path):
    """Verifica que el detector no es un test vacío que siempre pasa."""
    offending = tmp_path / "offending.py"
    offending.write_text(
        "from sklearn.preprocessing import StandardScaler\n", encoding="utf-8"
    )
    assert "sklearn.preprocessing" in _imported_modules(offending)
