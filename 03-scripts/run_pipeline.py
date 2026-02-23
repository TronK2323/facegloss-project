"""Pipeline mínimo y reproducible del proyecto.

Este script valida la configuración base y deja trazado un flujo estándar:
preparar datos -> entrenar -> exportar artefactos.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from facegloss.data import project_reference_date
from facegloss.models import CHURN_THRESHOLD_DAYS
from facegloss.recommender import top_k_default


def main() -> None:
    config_path = ROOT_DIR / "configs" / "params.yaml"

    print("[Facegloss] Inicio de pipeline")
    print(f"[1/3] Configuración detectada: {config_path.exists()} ({config_path})")
    print(f"[2/3] Parámetros base: fecha_ref={project_reference_date().date()} | churn={CHURN_THRESHOLD_DAYS} días")
    print(f"[3/3] Recomendador: top_k={top_k_default()}")
    print("[Facegloss] Pipeline base listo para ampliar con entrenamiento/exportación real")


if __name__ == "__main__":
    main()
