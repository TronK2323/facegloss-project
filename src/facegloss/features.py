"""Ingeniería de variables (feature engineering) para Facegloss."""

from __future__ import annotations


def delta_pct(current: float, previous: float) -> float:
    """Calcula el delta porcentual con protección frente a divisor cero."""
    return (current - previous) / previous * 100 if previous else 0.0
