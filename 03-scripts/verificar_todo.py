# verificar_todo.py
print("\n" + "="*60)
print("🔍 VERIFICACIÓN COMPLETA - FACEGLOSS PROJECT")
print("="*60 + "\n")

# Verificar Python
import sys
print(f"✅ Python: {sys.version.split()[0]}")

# Verificar librerías
print("\n📦 LIBRERÍAS:\n")

librerias = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'sklearn': 'scikit-learn',
    'lightgbm': 'lightgbm',
    'plotly': 'plotly',
    'streamlit': 'streamlit',
    'openpyxl': 'openpyxl',
    'tqdm': 'tqdm',
    'bs4': 'beautifulsoup4',
    'requests': 'requests'
}

correctas = 0
errores = []
errores_inesperados = []

for lib_import, lib_nombre in librerias.items():
    try:
        __import__(lib_import)
        print(f"   ✅ {lib_nombre:<25} OK")
        correctas += 1
    except ImportError:
        print(f"   ❌ {lib_nombre:<25} FALTA")
        errores.append(lib_nombre)
    except Exception as exc:
        print(f"   ⚠️  {lib_nombre:<25} ERROR INESPERADO: {type(exc).__name__}")
        errores_inesperados.append((lib_nombre, exc))

print(f"\n{'='*60}")
print(f"📊 RESULTADO: {correctas}/{len(librerias)} instaladas")
print(f"{'='*60}\n")

if correctas == len(librerias):
    print("🎉 ¡PERFECTO! TODO INSTALADO CORRECTAMENTE\n")
    print("✅ Estás 100% listo para empezar el proyecto Facegloss\n")
    print("📋 PRÓXIMOS PASOS:")
    print("   1. Preparar reunión con Facegloss")
    print("   2. Solicitar accesos a Shopify")
    print("   3. Empezar análisis de datos\n")
else:
    print(f"⚠️  Faltan {len(errores)} librería(s):\n")
    for lib in errores:
        print(f"   pip install {lib}")
    print()

if errores_inesperados:
    print("⚠️  Se detectaron errores inesperados al importar:\n")
    for lib, exc in errores_inesperados:
        print(f"   - {lib}: {type(exc).__name__}: {exc}")
    print()

print("="*60 + "\n")
