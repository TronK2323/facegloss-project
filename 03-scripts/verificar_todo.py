"""Project bootstrap and validation for Facegloss.

What this script does:
1. Creates missing project folders.
2. Validates key files.
3. Checks required and optional libraries.
4. Verifies Python syntax for main scripts.
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = [
    "01-data/raw",
    "01-data/interim",
    "01-data/processed",
    "01-data/exports",
    "02-notebooks",
    "03-scripts",
    "04-reports/graphs",
    "04-reports/weekly-reports",
    "04-reports/final-deliverables",
    "05-dashboards",
    "06-docs",
    "07-tests/unit",
    "07-tests/integration",
    "08-config",
    "09-logs",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "dashboard_facegloss.py",
    ".gitignore",
    "03-scripts/verificar_todo.py",
]

REQUIRED_LIBRARIES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
    "streamlit": "streamlit",
    "sklearn": "scikit-learn",
}

OPTIONAL_LIBRARIES = {
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "lightgbm": "lightgbm",
    "openpyxl": "openpyxl",
    "tqdm": "tqdm",
    "bs4": "beautifulsoup4",
    "requests": "requests",
}

SYNTAX_CHECK_FILES = [
    "dashboard_facegloss.py",
    "03-scripts/verificar_todo.py",
]


def print_title(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def check_python_version() -> bool:
    print_title("FACEGLOSS PROJECT CHECK")
    print(f"[OK] Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("[ERROR] Python 3.10+ is required.")
        return False
    return True


def ensure_directories() -> bool:
    print_title("DIRECTORY STRUCTURE")
    created = 0
    for relative_path in REQUIRED_DIRECTORIES:
        full_path = PROJECT_ROOT / relative_path
        if full_path.exists():
            print(f"[OK] {relative_path}")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            created += 1
            print(f"[CREATED] {relative_path}")
    print(f"\nSummary: {len(REQUIRED_DIRECTORIES) - created} existing, {created} created.")
    return True


def check_files() -> bool:
    print_title("KEY FILES")
    missing: list[str] = []
    for relative_path in REQUIRED_FILES:
        full_path = PROJECT_ROOT / relative_path
        if full_path.exists():
            print(f"[OK] {relative_path}")
        else:
            print(f"[MISSING] {relative_path}")
            missing.append(relative_path)
    if missing:
        print("\nMissing required files:")
        for item in missing:
            print(f" - {item}")
        return False
    return True


def check_libraries(libraries: dict[str, str], strict: bool) -> bool:
    mode = "REQUIRED LIBRARIES" if strict else "OPTIONAL LIBRARIES"
    print_title(mode)
    missing: list[str] = []
    for import_name, package_name in libraries.items():
        try:
            importlib.import_module(import_name)
            print(f"[OK] {package_name}")
        except Exception:
            print(f"[MISSING] {package_name}")
            missing.append(package_name)

    if not missing:
        return True

    label = "required" if strict else "optional"
    print(f"\nMissing {label} libraries:")
    for package_name in missing:
        print(f" - {package_name}")
    print("\nInstall command:")
    print(f"python -m pip install {' '.join(missing)}")

    return not strict


def check_syntax() -> bool:
    print_title("SYNTAX CHECK")
    failures: list[str] = []
    for relative_path in SYNTAX_CHECK_FILES:
        full_path = PROJECT_ROOT / relative_path
        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"[OK] {relative_path}")
        except Exception as error:
            print(f"[ERROR] {relative_path} -> {error}")
            failures.append(relative_path)
    return not failures


def main() -> int:
    python_ok = check_python_version()
    directories_ok = ensure_directories()
    files_ok = check_files()
    required_libs_ok = check_libraries(REQUIRED_LIBRARIES, strict=True)
    _ = check_libraries(OPTIONAL_LIBRARIES, strict=False)
    syntax_ok = check_syntax()

    print_title("FINAL RESULT")
    if all([python_ok, directories_ok, files_ok, required_libs_ok, syntax_ok]):
        print("[SUCCESS] Project is ready.")
        print("Next step: streamlit run dashboard_facegloss.py")
        return 0

    print("[FAIL] There are pending issues to resolve.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
