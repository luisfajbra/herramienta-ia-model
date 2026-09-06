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
    """Extract all imported module names from a file's AST (direct imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def _resolve_intra_package_import(import_spec: str, import_level: int, bench_dir: Path) -> Path | None:
    """Resolve relative/absolute intra-package imports to a file path.

    Returns the resolved .py file path under bench_dir, or None if:
    - it's an absolute import outside the bench package
    - the resolved file doesn't exist
    """
    if import_level == 0:
        # Absolute import like "from swmm_resilience.ml.bench.x import y"
        bench_prefix = "swmm_resilience.ml.bench."
        if import_spec.startswith(bench_prefix):
            module_path = import_spec[len(bench_prefix):].replace(".", "/")
            resolved = bench_dir / (module_path + ".py")
            return resolved if resolved.exists() else None
        return None
    else:
        # Relative import like "from .x import y" or "from ..x import y"
        # Calculate the parent level: level 1 is current, level 2 is parent, etc.
        module_path = import_spec.replace(".", "/") if import_spec else ""
        resolved = bench_dir / (module_path + ".py") if module_path else bench_dir / "__init__.py"
        return resolved if resolved.exists() else None


def _offenders(path: Path, bench_dir: Path = BENCH_DIR) -> list[str]:
    """Find forbidden imports in a file, including transitive intra-package imports.

    Returns a list of (source_file: module_name) strings where source_file is relative
    to bench_dir for transitive hits, or the module name for direct hits.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found_offenders: list[str] = []
    visited: set[Path] = set()

    def scan_file(file_path: Path, from_file: str | None = None) -> None:
        """Recursively scan a file and its intra-package imports."""
        if file_path in visited:
            return
        visited.add(file_path)

        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except Exception:
            return

        for node in ast.walk(tree):
            # Handle direct imports and intra-package imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    root = module_name.split(".")[0]
                    if root in FORBIDDEN_ROOTS:
                        source = f"{from_file}: {module_name}" if from_file else module_name
                        found_offenders.append(source)
                    elif root == "sklearn" and not module_name.startswith(ALLOWED_SKLEARN_PREFIXES):
                        source = f"{from_file}: {module_name}" if from_file else module_name
                        found_offenders.append(source)
                    else:
                        # Check if this is an intra-package import
                        resolved = _resolve_intra_package_import(module_name, 0, bench_dir)
                        if resolved:
                            rel_path = str(resolved.relative_to(bench_dir)).replace("\\", "/")
                            scan_file(resolved, rel_path)

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    # from . import x or from .. import x
                    # node.names contains the submodules to scan
                    if node.level > 0:
                        for alias in node.names:
                            submodule_name = alias.name
                            resolved = _resolve_intra_package_import(submodule_name, node.level, bench_dir)
                            if resolved:
                                rel_path = str(resolved.relative_to(bench_dir)).replace("\\", "/")
                                scan_file(resolved, rel_path)
                    continue

                module_name = node.module

                if node.level == 0:
                    # Absolute import
                    root = module_name.split(".")[0]
                    if root in FORBIDDEN_ROOTS:
                        source = f"{from_file}: {module_name}" if from_file else module_name
                        found_offenders.append(source)
                    elif root == "sklearn" and not module_name.startswith(ALLOWED_SKLEARN_PREFIXES):
                        source = f"{from_file}: {module_name}" if from_file else module_name
                        found_offenders.append(source)
                    else:
                        # Check if intra-package
                        resolved = _resolve_intra_package_import(module_name, 0, bench_dir)
                        if resolved:
                            rel_path = str(resolved.relative_to(bench_dir)).replace("\\", "/")
                            scan_file(resolved, rel_path)
                else:
                    # Relative import: from .sub import x
                    # Resolve relative to bench_dir (stage 1 modules are at bench_dir root)
                    resolved = _resolve_intra_package_import(module_name, node.level, bench_dir)
                    if resolved:
                        rel_path = str(resolved.relative_to(bench_dir)).replace("\\", "/")
                        scan_file(resolved, rel_path)

    scan_file(path)
    return found_offenders


@pytest.mark.parametrize("module_name", STAGE1_MODULES)
def test_stage1_module_imports_no_transformer_or_estimator(module_name):
    path = BENCH_DIR / module_name
    assert path.exists(), f"{module_name} no existe todavía"

    offenders = _offenders(path, BENCH_DIR)

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
    result = _offenders(offending, tmp_path)
    assert result == ["sklearn.preprocessing"], f"Expected ['sklearn.preprocessing'], got {result}"


def test_transitive_in_package_import_violation(tmp_path):
    """Verifica que el detector atrapa violaciones indirectas dentro del paquete."""
    # Create a helper module that imports StandardScaler
    helper = tmp_path / "_scaling.py"
    helper.write_text(
        "from sklearn.preprocessing import StandardScaler\n", encoding="utf-8"
    )

    # Create a main module that imports from the helper
    main = tmp_path / "main.py"
    main.write_text("from ._scaling import StandardScaler\n", encoding="utf-8")

    # Scanning main should find the violation transitively
    result = _offenders(main, tmp_path)
    assert "_scaling.py: sklearn.preprocessing" in result, (
        f"Expected transitive violation report, got {result}"
    )


def test_bare_relative_import_violation(tmp_path):
    """Verifica que 'from . import x' atrapa violaciones en el submodulo."""
    # Create a helper module that imports StandardScaler
    helper = tmp_path / "_helper.py"
    helper.write_text(
        "from sklearn.preprocessing import StandardScaler\n", encoding="utf-8"
    )

    # Create a main module that uses bare relative import
    main = tmp_path / "main.py"
    main.write_text("from . import _helper\n", encoding="utf-8")

    # Scanning main should find the violation transitively
    result = _offenders(main, tmp_path)
    assert "_helper.py: sklearn.preprocessing" in result, (
        f"Expected violation from bare import, got {result}"
    )
