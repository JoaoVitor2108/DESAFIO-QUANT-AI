"""PROGRAM (JEMPO) — pacote de backtest.

Exporta o contrato público do motor event-driven da Etapa 1. Monte Carlo,
métricas e plots entram em etapas seguintes e não são reexportados aqui.
"""

from backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    PosicaoInterna,
    ResultadoBacktest,
    TradeRegistro,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "ResultadoBacktest",
    "TradeRegistro",
    "PosicaoInterna",
]
