from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from facegloss.features import delta_pct
from facegloss.recommender import top_k_default


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
    assert 'src/facegloss/' in readme


def test_delta_pct_edges():
    assert delta_pct(10, 0) == 0.0
    assert delta_pct(120, 100) == 20.0
    assert delta_pct(80, 100) == -20.0


def test_default_top_k_positive():
    assert top_k_default() > 0
