from pathlib import Path
import subprocess
import sys


def test_dashboard_header_typo_fixed():
    content = Path('dashboard_facegloss.py').read_text(encoding='utf-8')
    assert 'Crema blanca' in content
    assert 'Crema blanco' not in content


def test_verificar_todo_uses_specific_import_error_handling():
    content = Path('03-scripts/verificar_todo.py').read_text(encoding='utf-8')
    assert 'except ImportError:' in content
    assert 'except:' not in content


def test_verificar_todo_smoke_runs_without_crashing():
    result = subprocess.run(
        [sys.executable, '03-scripts/verificar_todo.py'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert 'VERIFICACIÓN COMPLETA - FACEGLOSS PROJECT' in result.stdout


def test_readme_describes_current_structure():
    readme = Path('README.md').read_text(encoding='utf-8')
    assert 'dashboard_facegloss.py' in readme
    assert '01-data/' in readme
    assert '04-reports/' in readme
