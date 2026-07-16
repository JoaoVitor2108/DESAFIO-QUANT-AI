"""Testes do PROGRAM — BacktestEngine (Etapa 1).

Motor de backtest event-driven: calendário B3, loop diário, execução simulada
com custos, detecção intraday de stop/take, mark-to-market e integração com o
contrato público do ORQUESTRADOR.

Convenções travadas destes testes (ver CONTEXTO_FIXO_FINAL.md e o prompt da
Etapa 1):
- Todas as datas nos contratos/dataclasses do PROGRAM são `pd.Timestamp` NAIVE.
- O calendário BMF sai naive (`.tz is None`).
- Fakes determinísticos espelham o padrão de `tests/test_orchestrator.py`.
"""

import pandas as pd

from backtest.engine import BacktestEngine


def _d(s: str) -> pd.Timestamp:
    """Timestamp naive (convenção do PROGRAM)."""
    return pd.Timestamp(s)


# ── Grupo E — Calendário e bordas (parcial: 23, 24 na Subetapa 1.1) ───────────


def test_calendario_bmf_pula_sexta_santa_2024():
    """R5 — 2024-03-29 (Sexta-feira Santa, B3 fechada) NÃO aparece no calendário.

    `pd.bdate_range` incluiria essa data por ignorar feriados nacionais; o
    calendário 'BMF' do pandas_market_calendars a exclui.
    """
    engine = BacktestEngine(journal=None, orquestrador=None)
    cal = engine._calendario_bmf(_d("2024-03-01"), _d("2024-06-30"))
    assert cal.tz is None
    assert _d("2024-03-29") not in cal


def test_calendario_bmf_pula_corpus_christi_2024():
    """R5 — 2024-05-30 (Corpus Christi, B3 fechada) NÃO aparece no calendário."""
    engine = BacktestEngine(journal=None, orquestrador=None)
    cal = engine._calendario_bmf(_d("2024-03-01"), _d("2024-06-30"))
    assert cal.tz is None
    assert _d("2024-05-30") not in cal
