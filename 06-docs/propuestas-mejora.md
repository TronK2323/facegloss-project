# Propuestas de mejora profesional

## Cambios recomendados (siguiente iteracion)

1. Modularizar el dashboard
- Mover logica de negocio a `src/facegloss/` y dejar `dashboard_facegloss.py` como capa de UI.

2. Anadir testing real
- Crear tests para funciones de limpieza, segmentacion RFM y scoring de churn en `07-tests/unit/`.

3. Integrar CI
- Configurar GitHub Actions para ejecutar `python 03-scripts/verificar_todo.py` y tests en cada push.

4. Versionado de datos y modelos
- Usar DVC o al menos versionado por fecha para artefactos en `01-data/exports/`.

5. Configuracion centralizada
- Definir variables de entorno en `08-config/.env.example` y cargarlas desde una sola capa.

6. Calidad de codigo
- Incluir `ruff` y `black` para estilo, y `pre-commit` para controles automaticos antes de commit.

