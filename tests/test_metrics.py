"""Testes do PROGRAM — métricas de performance (Etapa 3, Subetapa 3.1).

Módulo puro `backtest/metrics.py`: consome `ResultadoBacktest` e devolve
`MetricasBacktest` (dataclass frozen). Sem I/O exceto a chamada opcional a
`journal.get_macro` para a taxa livre de risco (Selic — não há CDI no JOURNAL,
ver decisão travada da Etapa 3).

Convenções dos testes:
- Fixtures determinísticas; asserts de VALOR EXATO (mordem mutação), não `is
  not None`.
- Equity é `pd.Series` naive indexada por dias úteis (convenção do PROGRAM).
- `std` amostral (ddof=1) — padrão de mercado para Sharpe. Documentado no módulo.
"""

from __future__ import annotations

import dataclasses
import math
import statistics

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, ResultadoBacktest, TradeRegistro
from backtest.metrics import (
    MetricasBacktest,
    _downside_deviation,
    _sharpe_sortino_vol,
    atribuir_por_motivo,
    atribuir_por_setor,
    calcular_metricas,
)

FUSO = "America/Sao_Paulo"
_COLS = [f.name for f in dataclasses.fields(TradeRegistro)]


# ── Builders determinísticos ──────────────────────────────────────────────────


def mk_trade(
    pnl_liquido: float,
    motivo: str = "prazo",
    setor: str = "Energia",
    custo_entrada: float = 10.0,
    custo_saida: float = 12.0,
) -> dict:
    """Linha de trade fechado com todos os campos de `TradeRegistro`. Só
    `pnl_liquido`, `custo_entrada/saida`, `motivo` e `setor` importam à Etapa 3.1;
    o resto tem defaults inertes para espelhar a saída real do engine."""
    return {
        "ticker": "AAAA3",
        "setor": setor,
        "qtd": 100,
        "preco_entrada": 10.0,
        "preco_saida": 11.0,
        "data_entrada": pd.Timestamp("2024-01-02"),
        "data_saida": pd.Timestamp("2024-01-10"),
        "motivo": motivo,
        "custo_entrada": custo_entrada,
        "custo_saida": custo_saida,
        "pnl_bruto": pnl_liquido + custo_entrada + custo_saida,
        "pnl_liquido": pnl_liquido,
        "y_pred_entrada": 0.05,
        "score_econ_entrada": 0.5,
        "rank_entrada": 1,
        "dias_uteis_ate_saida": 6,
    }


def _trades_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows)[_COLS]


def mk_resultado(
    equity: list[float],
    trades: list[dict] | None = None,
    capital_inicial: float = 100_000.0,
    capital_final: float | None = None,
    n_dias_uteis: int | None = None,
) -> ResultadoBacktest:
    """Monta um `ResultadoBacktest` mínimo para exercitar as métricas."""
    trades = trades or []
    if equity:
        idx = pd.bdate_range("2024-01-02", periods=len(equity))
        equity_series = pd.Series(equity, index=idx, dtype=float, name="equity")
        data_inicio, data_fim = idx[0], idx[-1]
    else:
        equity_series = pd.Series(dtype=float, name="equity")
        data_inicio = data_fim = pd.Timestamp("2024-01-02")
    return ResultadoBacktest(
        trades=_trades_df(trades),
        equity_diario=equity_series,
        avisos=[],
        config=BacktestConfig(capital_inicial=capital_inicial),
        data_inicio=data_inicio,
        data_fim=data_fim,
        n_dias_uteis=n_dias_uteis if n_dias_uteis is not None else len(equity),
        n_trades=len(trades),
        capital_final=capital_final if capital_final is not None
        else (equity[-1] if equity else capital_inicial),
    )


class FakeJournalMacro:
    """Fake ESTRITO do JOURNAL para macro. Replica `get_macro(data_limite) ->
    dict[str, pd.Series]` (exige tz-aware) e devolve `selic_diaria` em % a.a. com
    índice tz-aware SP, como o real. `raise_exc`/`vazio` cobrem R8."""

    def __init__(
        self, selic_anual_pct: float = 10.0, raise_exc: bool = False,
        vazio: bool = False,
    ) -> None:
        self._selic = selic_anual_pct
        self._raise = raise_exc
        self._vazio = vazio

    def get_macro(self, data_limite: pd.Timestamp) -> dict[str, pd.Series]:
        if getattr(data_limite, "tzinfo", None) is None:
            raise ValueError(f"get_macro exige tz-aware; recebeu {data_limite!r}")
        if self._raise:
            raise RuntimeError("BCB SGS indisponível (simulado)")
        if self._vazio:
            return {"selic_diaria": pd.Series(dtype=float, name="selic_diaria")}
        idx = pd.bdate_range("2023-01-01", periods=800, tz=FUSO)
        return {
            "selic_diaria": pd.Series(
                [self._selic] * len(idx), index=idx, name="selic_diaria"
            )
        }


# ── Grupo A — Retorno e volatilidade ──────────────────────────────────────────


def test_retorno_total_capital_dobrou():
    r = mk_resultado([100.0, 200.0], capital_inicial=100.0)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.retorno_total == 1.0


def test_retorno_anualizado_252_dias_iguala_retorno_total():
    equity = [100.0] + [110.0] * 251  # 252 pontos = 252 dias úteis
    r = mk_resultado(equity, capital_inicial=100.0, capital_final=110.0,
                     n_dias_uteis=252)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.retorno_total == pytest.approx(0.10)
    assert m.retorno_anualizado == pytest.approx(0.10)


def test_volatilidade_zero_com_equity_constante():
    r = mk_resultado([100.0, 100.0, 100.0])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.volatilidade_anualizada == 0.0


def test_volatilidade_anualizada_bate_com_valor_conhecido():
    # retornos exatos [0.02, 0.04]; σ amostral = sqrt(0.0002).
    r = mk_resultado([100.0, 102.0, 106.08])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    esperado = math.sqrt(0.0002) * math.sqrt(252)
    assert m.volatilidade_anualizada == pytest.approx(esperado)


def test_retorno_negativo_capital_final_menor():
    r = mk_resultado([100_000.0, 90_000.0], capital_inicial=100_000.0)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.retorno_total == pytest.approx(-0.1)


# ── Grupo B — Sharpe e Sortino ────────────────────────────────────────────────


def test_sharpe_zero_quando_equity_constante():
    r = mk_resultado([100.0, 100.0, 100.0])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.sharpe_anualizado == 0.0


def test_sharpe_positivo_com_trend_up():
    # retornos [0.01, 0.02, 0.01, 0.02]; rf = 0.
    equity = [100.0, 101.0, 103.02, 104.0502, 106.131204]
    r = mk_resultado(equity)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    rets = [0.01, 0.02, 0.01, 0.02]
    esperado = math.sqrt(252) * statistics.mean(rets) / statistics.stdev(rets)
    assert m.sharpe_anualizado > 0
    assert m.sharpe_anualizado == pytest.approx(esperado)


def test_sortino_maior_que_sharpe_quando_volatilidade_upside():
    # retornos [0.05, 0.05, 0.05, -0.01]: downside << vol total.
    equity = [100.0, 105.0, 110.25, 115.7625, 114.604875]
    r = mk_resultado(equity)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.sortino_anualizado > m.sharpe_anualizado > 0
    assert math.isfinite(m.sortino_anualizado)


def test_taxa_livre_risco_override_vs_journal_cdi():
    r = mk_resultado([100.0, 101.0, 102.0])
    journal = FakeJournalMacro(selic_anual_pct=10.0)
    m = calcular_metricas(r, journal=journal, taxa_livre_risco_override=0.05)
    assert m.taxa_livre_risco_anualizada == 0.05
    assert m.fonte_taxa_livre_risco == "override"


def test_sortino_convencao_divide_por_n_total():
    """Sortino usa N TOTAL no denominador do downside deviation, não só
    `n_negativos` (convenção Sortino 1994, Nawrocki 1999).

    Contra-fato: se dividisse por `n_negativos`, uma série com 2 dias muito
    negativos em 500 daria downside_dev enorme (0.05) e Sortino ínfimo.

    Setup determinístico: 498 retornos = 0.001 e 2 retornos = -0.05; MAR = 0.

        downside_dev correto = sqrt((2 × 0.05²) / 500) = sqrt(1e-5) ≈ 0.00316228
        (NÃO sqrt((2 × 0.05²) / 2) = 0.05)
    """
    retornos = np.array([0.001] * 498 + [-0.05] * 2, dtype=float)

    # Downside deviation intermediário — valor EXATO (morde a mutação N→n_neg).
    dd_esperado = math.sqrt((2 * 0.05**2) / 500)  # ≈ 0.0031622776601
    dd = _downside_deviation(retornos, 0.0)
    assert dd == pytest.approx(dd_esperado, rel=1e-12)
    assert dd == pytest.approx(0.0031622776601, abs=1e-12)
    assert dd != pytest.approx(0.05)  # o valor que sairia dividindo por n_neg

    # Sortino final — valor exato derivado da definição.
    mu = retornos.mean()  # 0.398 / 500 = 0.000796
    sortino_esperado = math.sqrt(252) * mu / dd_esperado
    _, _, sortino = _sharpe_sortino_vol(retornos, 0.0)
    assert sortino == pytest.approx(sortino_esperado, rel=1e-12)


def test_payoff_zero_quando_zero_trades_vencedores():
    """Convenção degenerada simétrica ao test 20/31: todos os trades perdedores
    → payoff_medio = 0.0 e profit_factor = 0.0 (não None, não -inf)."""
    trades = [mk_trade(-100.0) for _ in range(5)]
    r = mk_resultado([100.0, 99.0], trades=trades)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.hit_rate == 0.0
    assert m.payoff_medio == 0.0
    assert m.profit_factor == 0.0
    assert m.n_trades_vencedores == 0
    assert m.n_trades_perdedores == 5


def test_sharpe_com_taxa_livre_zero_por_falha(caplog):
    r = mk_resultado([100.0, 101.0, 102.0])
    journal = FakeJournalMacro(raise_exc=True)
    with caplog.at_level("WARNING"):
        m = calcular_metricas(r, journal=journal)
    assert m.taxa_livre_risco_anualizada == 0.0
    assert m.fonte_taxa_livre_risco == "zero_por_falha"
    assert any("taxa livre" in rec.message.lower() for rec in caplog.records)


# ── Grupo C — Drawdown ────────────────────────────────────────────────────────


def test_mdd_zero_com_equity_monotonica_crescente():
    r = mk_resultado([100.0, 110.0, 120.0])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.maximum_drawdown == 0.0
    assert m.duracao_mdd_dias == 0
    assert m.dias_ate_recuperacao_mdd is None


def test_mdd_negativo_com_queda():
    r = mk_resultado([100.0, 90.0, 100.0])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.maximum_drawdown == pytest.approx(-0.10)


def test_duracao_mdd_conta_dias_pico_a_vale():
    # pico em pos 1 (110), vale em pos 3 (95) → duração 2 dias úteis.
    r = mk_resultado([100.0, 110.0, 105.0, 95.0, 100.0])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.duracao_mdd_dias == 2


def test_recuperacao_mdd_none_se_nao_recuperou():
    # pico 110 nunca reatingido após o vale.
    r = mk_resultado([100.0, 110.0, 90.0, 95.0, 100.0])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.dias_ate_recuperacao_mdd is None


def test_recuperacao_mdd_com_valor_conhecido():
    # vale em pos 2 (90), pico 110 reatingido em pos 4 → 2 dias úteis.
    r = mk_resultado([100.0, 110.0, 90.0, 95.0, 110.0, 120.0])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.dias_ate_recuperacao_mdd == 2


# ── Grupo D — Trades ──────────────────────────────────────────────────────────


def test_hit_rate_none_com_zero_trades():
    r = mk_resultado([100.0, 101.0], trades=[])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.hit_rate is None
    assert m.n_trades_total == 0


def test_hit_rate_5_de_10_trades_positivos_da_05():
    trades = [mk_trade(100.0) for _ in range(5)] + [mk_trade(-50.0) for _ in range(5)]
    r = mk_resultado([100.0, 101.0], trades=trades)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.hit_rate == pytest.approx(0.5)
    assert m.n_trades_vencedores == 5
    assert m.n_trades_perdedores == 5


def test_payoff_medio_ganho_medio_sobre_perda_media():
    trades = [mk_trade(100.0), mk_trade(200.0), mk_trade(-50.0), mk_trade(-150.0)]
    r = mk_resultado([100.0, 101.0], trades=trades)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    # mean(ganhos)=150, mean(perdas)=-100 → payoff 1.5
    assert m.payoff_medio == pytest.approx(1.5)


def test_profit_factor_soma_ganhos_sobre_soma_perdas():
    trades = [mk_trade(100.0), mk_trade(200.0), mk_trade(-50.0), mk_trade(-150.0)]
    r = mk_resultado([100.0, 101.0], trades=trades)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    # sum(ganhos)=300, sum(perdas)=-200 → 1.5
    assert m.profit_factor == pytest.approx(1.5)


def test_payoff_infinito_quando_zero_trades_perdedores():
    trades = [mk_trade(100.0), mk_trade(200.0)]
    r = mk_resultado([100.0, 101.0], trades=trades)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.payoff_medio == float("inf")


# ── Grupo F — Custos ──────────────────────────────────────────────────────────


def test_custo_total_soma_entrada_e_saida_de_todos_trades():
    trades = [
        mk_trade(100.0, custo_entrada=10.0, custo_saida=12.0),
        mk_trade(-50.0, custo_entrada=20.0, custo_saida=8.0),
    ]
    r = mk_resultado([100.0, 101.0], trades=trades)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.custo_total_pago == pytest.approx(50.0)  # 10+12+20+8


def test_custo_como_pct_bate_com_valor_esperado():
    trades = [mk_trade(100.0, custo_entrada=100.0, custo_saida=150.0)]
    r = mk_resultado([100.0, 101.0], trades=trades, capital_inicial=100_000.0)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.custo_total_pago == pytest.approx(250.0)
    assert m.custo_como_pct_capital_inicial == pytest.approx(250.0 / 100_000.0)


def test_custo_zero_quando_zero_trades():
    r = mk_resultado([100.0, 101.0], trades=[])
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.custo_total_pago == 0.0
    assert m.custo_como_pct_capital_inicial == 0.0


# ── Grupo E — Atribuição (R5/R6) ──────────────────────────────────────────────


def _trades_atribuicao() -> list[dict]:
    """Trades determinísticos por motivo:
    - stop:  -50, -30           → n=2, total=-80,  medio=-40, hit=0.0
    - take:  +100, +200, +60    → n=3, total=360,  medio=120, hit=1.0
    - prazo: +40, -10           → n=2, total=30,   medio=15,  hit=0.5
    total_pnl = 310.
    """
    return [
        mk_trade(-50.0, motivo="stop"),
        mk_trade(-30.0, motivo="stop"),
        mk_trade(100.0, motivo="take"),
        mk_trade(200.0, motivo="take"),
        mk_trade(60.0, motivo="take"),
        mk_trade(40.0, motivo="prazo"),
        mk_trade(-10.0, motivo="prazo"),
    ]


def test_atribuir_por_motivo_soma_pct_total_da_1():
    r = mk_resultado([100.0, 101.0], trades=_trades_atribuicao())
    df = atribuir_por_motivo(r)
    assert df["pct_do_total_pnl"].sum() == pytest.approx(1.0)


def test_atribuir_por_motivo_ordenado_por_pnl_total():
    r = mk_resultado([100.0, 101.0], trades=_trades_atribuicao())
    df = atribuir_por_motivo(r)
    assert list(df.index) == ["take", "prazo", "stop"]
    assert list(df["pnl_liquido_total"]) == [360.0, 30.0, -80.0]


def test_atribuir_por_setor_agrupa_corretamente():
    trades = [
        mk_trade(100.0, setor="Energia"),
        mk_trade(50.0, setor="Energia"),
        mk_trade(-20.0, setor="Bancos"),
    ]
    r = mk_resultado([100.0, 101.0], trades=trades)
    df = atribuir_por_setor(r)
    assert df.index.name == "setor"
    assert list(df.index) == ["Energia", "Bancos"]
    assert df.loc["Energia", "n_trades"] == 2
    assert df.loc["Energia", "pnl_liquido_total"] == pytest.approx(150.0)
    assert df.loc["Bancos", "pnl_liquido_total"] == pytest.approx(-20.0)


def test_atribuir_com_zero_trades_retorna_dataframe_vazio():
    r = mk_resultado([100.0, 101.0], trades=[])
    df_m = atribuir_por_motivo(r)
    df_s = atribuir_por_setor(r)
    assert df_m.empty
    assert df_s.empty
    assert "pnl_liquido_total" in df_m.columns
    assert df_m.index.name == "motivo"


def test_atribuir_por_motivo_valores_batem_com_soma_manual():
    r = mk_resultado([100.0, 101.0], trades=_trades_atribuicao())
    df = atribuir_por_motivo(r)
    # motivo "take": 3 trades, +100/+200/+60.
    assert df.loc["take", "n_trades"] == 3
    assert df.loc["take", "pnl_liquido_total"] == pytest.approx(360.0)
    assert df.loc["take", "pnl_liquido_medio"] == pytest.approx(120.0)
    assert df.loc["take", "hit_rate"] == pytest.approx(1.0)
    assert df.loc["take", "pct_do_total_pnl"] == pytest.approx(360.0 / 310.0)
    # motivo "stop": 2 perdedores.
    assert df.loc["stop", "hit_rate"] == pytest.approx(0.0)
    assert df.loc["stop", "pnl_liquido_medio"] == pytest.approx(-40.0)


# ── Grupo G — Robustez ────────────────────────────────────────────────────────


def test_n_dias_uteis_zero_levanta_valueerror():
    r = mk_resultado([], n_dias_uteis=0)
    with pytest.raises(ValueError):
        calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)


def test_dois_calculos_identicos_produzem_metricas_identicas():
    trades = [mk_trade(100.0), mk_trade(-50.0)]
    r = mk_resultado([100.0, 105.0, 98.0, 103.0], trades=trades)
    journal = FakeJournalMacro(selic_anual_pct=11.0)
    m1 = calcular_metricas(r, journal=journal)
    m2 = calcular_metricas(r, journal=journal)
    assert m1 == m2


def test_todos_trades_vencedores_gera_payoff_infinito():
    trades = [mk_trade(100.0), mk_trade(50.0), mk_trade(25.0)]
    r = mk_resultado([100.0, 101.0], trades=trades)
    m = calcular_metricas(r, journal=None, taxa_livre_risco_override=0.0)
    assert m.payoff_medio == float("inf")
    assert m.profit_factor == float("inf")
    assert isinstance(m, MetricasBacktest)
