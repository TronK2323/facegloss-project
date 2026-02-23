"""Punto de entrada alternativo para lanzar el dashboard.

Permite evolucionar la estructura del proyecto sin romper el script histórico.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard_facegloss.py"
    runpy.run_path(str(dashboard_path), run_name="__main__")
