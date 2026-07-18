"""PROGRAM (JEMPO) — pacote de backtest.

Exporta o contrato público do motor event-driven (Etapa 1) e das métricas de
performance (Etapa 3). Monte Carlo e plots entram em etapas seguintes e não são
reexportados aqui.
"""

from backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    PosicaoInterna,
    ResultadoBacktest,
    TradeRegistro,
)
from backtest.metrics import (
    MetricasBacktest,
    atribuir_por_motivo,
    atribuir_por_setor,
    calcular_metricas,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "ResultadoBacktest",
    "TradeRegistro",
    "PosicaoInterna",
    "MetricasBacktest",
    "calcular_metricas",
    "atribuir_por_motivo",
    "atribuir_por_setor",
]
