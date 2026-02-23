# Facegloss AI Project

Proyecto de optimización mediante Inteligencia Artificial para Facegloss.eu.

## Estructura actual del repositorio

- `app/`
  - `dashboard.py` - Punto de entrada alternativo para lanzar el dashboard.
- `src/facegloss/`
  - `data.py` - Utilidades de datos.
  - `features.py` - Funciones de ingeniería de variables.
  - `models.py` - Constantes y utilidades de modelos.
  - `recommender.py` - Helpers del recomendador.
- `configs/`
  - `params.yaml` - Parámetros centrales del proyecto.
- `dashboard_facegloss.py` - Dashboard principal en Streamlit (script legado activo).
- `requirements.txt` - Dependencias del proyecto.
- `01-data/`
  - `exports/` - Artefactos exportados (`.pkl`) para modelos y vectorizadores.
- `02-notebooks/` - Notebooks de exploración, segmentación, recomendación y churn.
- `03-scripts/`
  - `verificar_todo.py` - Verificación del entorno.
  - `run_pipeline.py` - Esqueleto de pipeline para automatizar flujo.
- `04-reports/`
  - `final-deliverables/` - Entregables finales.
  - `revision_tareas.md` - Propuesta de tareas de mejora detectadas.
- `docs/`
  - `roadmap.md` - Plan de ejecución por hitos.
  - `arquitectura.md` - Mapa rápido de componentes.
- `tests/` - Pruebas automatizadas.

## Comandos rápidos

```bash
make check
make test
make run-dashboard
```

## Autor

- Bernat Casals Riera
- Bernatcasalsriera@gmail.com

## Fecha de inicio

- 16/02/2026
