# Contribuir al proyecto

## Flujo recomendado
1. Crear rama de trabajo.
2. Hacer cambios pequeños y con commits descriptivos.
3. Ejecutar checks antes de commit.

## Checks mínimos
- `python -m py_compile dashboard_facegloss.py 03-scripts/verificar_todo.py tests/test_project_regressions.py`
- `pytest -q`
