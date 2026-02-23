# Revisión rápida de la base de código: tareas propuestas

## 1) Tarea de corrección tipográfica
- **Problema detectado:** En el encabezado del dashboard aparece la frase "Crema blanco", que es una incongruencia gramatical en español.
- **Ubicación:** `dashboard_facegloss.py` (docstring inicial).
- **Tarea propuesta:** Ajustar el texto a "Crema blanca" (o "blanco crema", según intención de marca) y revisar el resto de textos UI para mantener consistencia de estilo.
- **Criterio de aceptación:** El encabezado queda corregido y no hay errores tipográficos evidentes en títulos principales.

## 2) Tarea para solucionar un fallo
- **Problema detectado:** El script de verificación usa `except:` genérico al importar librerías, lo que puede ocultar errores reales (por ejemplo, errores internos de módulo) clasificándolos como "librería faltante".
- **Ubicación:** `03-scripts/verificar_todo.py`.
- **Tarea propuesta:** Sustituir `except:` por `except ImportError:` y mostrar un mensaje diferente para excepciones inesperadas.
- **Criterio de aceptación:**
  - Los módulos no instalados se reportan como faltantes.
  - Errores no relacionados con importación quedan visibles para diagnóstico.

## 3) Tarea de comentario/documentación
- **Problema detectado:** El `README.md` describe una estructura de carpetas (`raw/`, `processed/`, `graphs/`, `weekly-reports/`, `05-dashboards/`) que no existe en el repositorio actual.
- **Ubicación:** `README.md`.
- **Tarea propuesta:** Actualizar el README para reflejar la estructura real del proyecto (o crear las carpetas faltantes si son parte del diseño esperado).
- **Criterio de aceptación:** La sección "Estructura del Proyecto" coincide con el árbol real del repositorio.

## 4) Tarea para mejorar una prueba
- **Problema detectado:** No hay suite de pruebas automatizadas para funciones clave del dashboard y scripts auxiliares.
- **Ubicación:** Proyecto completo (sin carpeta de tests actualmente).
- **Tarea propuesta:** Crear pruebas con `pytest` para:
  - Validar invariantes de `generar_datos()` (columnas esperadas, tamaños y rangos básicos).
  - Verificar `delta_pct()` ante casos borde (divisor 0, valores negativos).
  - Añadir un smoke test para `03-scripts/verificar_todo.py` (salida sin excepción).
- **Criterio de aceptación:** `pytest` corre en local y valida al menos los casos mínimos de regresión.
