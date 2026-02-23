.PHONY: test check run-dashboard

check:
	python -m py_compile dashboard_facegloss.py 03-scripts/verificar_todo.py tests/test_project_regressions.py

test:
	pytest -q

run-dashboard:
	streamlit run dashboard_facegloss.py
