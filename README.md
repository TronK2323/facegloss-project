# Facegloss AI Project

Proyecto de optimizacion con IA para Facegloss.eu. Este repositorio incluye analisis, modelos y dashboard ejecutivo.

## Estructura profesional

01-data/
- raw/
- interim/
- processed/
- exports/

02-notebooks/

03-scripts/

04-reports/
- graphs/
- weekly-reports/
- final-deliverables/

05-dashboards/

06-docs/

07-tests/
- unit/
- integration/

08-config/

09-logs/

## Arranque rapido (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python 03-scripts/verificar_todo.py
streamlit run dashboard_facegloss.py
```

## Dependencias

- `requirements.txt`: minimo para dashboard y flujo principal.
- `requirements-dev.txt`: extras para notebooks y analisis avanzado.

## Verificacion automatica

El script `03-scripts/verificar_todo.py`:
1. crea carpetas faltantes,
2. valida archivos clave,
3. revisa librerias requeridas y opcionales,
4. ejecuta chequeo de sintaxis.

## Autor

Bernat Casals Riera
bernatcasalsriera@gmail.com

## Inicio del proyecto

16/02/2026
