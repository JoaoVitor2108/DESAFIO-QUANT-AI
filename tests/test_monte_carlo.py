"""Testes do PROGRAM — Monte Carlo (Etapa 2).

Módulo puro `backtest/monte_carlo.py`: consome `ResultadoBacktest` (dados) e um
retorno realizado escalar do Ibov, e devolve `ResultadoMonteCarlo` (dataclass
frozen). Sem I/O, sem API.

Semântica travada (ver prompt da Etapa 2):
- Bootstrap com reposição testa ROBUSTEZ DO RETORNO ("dado que essa é a
  distribuição de retorno por trade, o agregado é robusto?").
- Permutação sem reposição testa RISCO/SEQUÊNCIA (MDD) — a soma dos P&L é
  comutativa, logo o retorno total NÃO muda; só a equity curve intermediária e o
  MDD mudam.
- MDD reamostrado é APROXIMAÇÃO (trades encadeados sem gaps de tempo).

Convenções dos testes: fixtures determinísticas; asserts de VALOR EXATO onde faz
sentido; tolerância explícita com racional onde a variância de bootstrap é
intrínseca (ex.: mediana de 10k reamostragens).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, ResultadoBacktest, TradeRegistro
from backtest.monte_carlo import (
    ResultadoMonteCarlo,
    rodar_bootstrap_retornos_trades,
    rodar_permutacao_ordem_trades,
)

_COLS = [f.name for f in dataclasses.fields(TradeRegistro)]
CAPITAL = 100_000.0


# ── Builders determinísticos ──────────────────────────────────────────────────


def _trade_row(
    pnl_liquido: float,
    custo_entrada: float = 10.0,
    custo_saida: float = 12.0,
    setor: str = "Energia",
    motivo: str = "prazo",
) -> dict:
    """Linha de `TradeRegistro` fechado. Só `pnl_liquido` (e, no teste R8, os
    custos e o `pnl_bruto`) importam ao Monte Carlo; o resto tem defaults inertes
    que espelham a saída real do engine."""
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


def mk_res(
    pnls: list[float],
    capital_inicial: float = CAPITAL,
    rows: list[dict] | None = None,
) -> ResultadoBacktest:
    """Monta um `ResultadoBacktest` mínimo para exercitar o Monte Carlo. `pnls` é
    a lista de `pnl_liquido` dos trades; `rows` permite controlar todos os campos
    (usado no teste R8 de custos embutidos)."""
    linhas = rows if rows is not None else [_trade_row(p) for p in pnls]
    if linhas:
        trades = pd.DataFrame(linhas)[_COLS]
    else:
        trades = pd.DataFrame(columns=_COLS)
    equity = pd.Series(
        [capital_inicial, capital_inicial],
        index=pd.bdate_range("2024-01-02", periods=2),
        dtype=float,
        name="equity",
    )
    return ResultadoBacktest(
        trades=trades,
        equity_diario=equity,
        avisos=[],
        config=BacktestConfig(capital_inicial=capital_inicial),
        data_inicio=pd.Timestamp("2024-01-02"),
        data_fim=pd.Timestamp("2024-01-03"),
        n_dias_uteis=2,
        n_trades=len(linhas),
        capital_final=capital_inicial + float(sum(r["pnl_liquido"] for r in linhas)),
    )


# Conjunto padrão com dispersão (retorno realizado = 10000/100000 = 0.10).
_PNLS_ALTERNADO = [500.0 if i % 2 == 0 else -300.0 for i in range(100)]


# ── Grupo A — Determinismo e reprodutibilidade ────────────────────────────────


def test_bootstrap_dois_runs_mesma_seed_produzem_resultados_identicos():
    res = mk_res(_PNLS_ALTERNADO)
    a = rodar_bootstrap_retornos_trades(res, n_simulacoes=2_000, seed=12345)
    b = rodar_bootstrap_retornos_trades(res, n_simulacoes=2_000, seed=12345)
    assert np.array_equal(a.retornos_simulados, b.retornos_simulados)
    assert np.array_equal(a.mdds_simulados, b.mdds_simulados)


def test_seeds_diferentes_produzem_resultados_diferentes():
    res = mk_res(_PNLS_ALTERNADO)
    a = rodar_bootstrap_retornos_trades(res, n_simulacoes=2_000, seed=12345)
    b = rodar_bootstrap_retornos_trades(res, n_simulacoes=2_000, seed=999)
    assert not np.array_equal(a.retornos_simulados, b.retornos_simulados)


def test_bootstrap_nao_usa_np_random_global():
    res = mk_res(_PNLS_ALTERNADO)
    np.random.seed(0)
    a = rodar_bootstrap_retornos_trades(res, n_simulacoes=2_000, seed=12345)
    np.random.seed(999)
    b = rodar_bootstrap_retornos_trades(res, n_simulacoes=2_000, seed=12345)
    assert np.array_equal(a.retornos_simulados, b.retornos_simulados)


# ── Grupo B — Bootstrap com reposição ─────────────────────────────────────────


def test_bootstrap_amostra_com_reposicao():
    # 3 trades de P&L distintos. SEM reposição, toda simulação usaria os 3 trades
    # e a soma (logo o retorno) seria invariante. COM reposição, o retorno varia.
    res = mk_res([1000.0, -400.0, 200.0])
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=5_000, seed=12345)
    assert np.unique(r.retornos_simulados).size > 1
    assert r.retorno_desvio > 0.0


def test_bootstrap_retorno_p50_proximo_do_retorno_realizado():
    res = mk_res(_PNLS_ALTERNADO)
    retorno_realizado = sum(_PNLS_ALTERNADO) / CAPITAL  # 0.10
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=10_000, seed=12345)
    # Mediana de 10k reamostragens de 100 trades converge ao retorno realizado.
    # Tolerância explícita: variância de bootstrap é intrínseca (racional no prompt).
    assert abs(r.retorno_p50 - retorno_realizado) < 0.005


def test_bootstrap_um_trade_apenas_funciona():
    res = mk_res([750.0])
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=1_000, seed=12345)
    esperado = 750.0 / CAPITAL
    assert r.retorno_desvio == 0.0
    assert np.allclose(r.retornos_simulados, esperado)
    assert r.retorno_p05 == pytest.approx(esperado)
    assert r.retorno_p95 == pytest.approx(esperado)


def test_bootstrap_todos_trades_zero_retorna_distribuicao_colapsada():
    res = mk_res([0.0, 0.0, 0.0, 0.0])
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=1_000, seed=12345)
    assert np.all(r.retornos_simulados == 0.0)
    assert np.all(r.mdds_simulados == 0.0)
    assert r.retorno_medio == 0.0
    assert r.prob_retorno_positivo == 0.0
    assert r.prob_mdd_melhor_que_20pct == 1.0


def test_bootstrap_reamostra_pnl_liquido_nao_bruto():
    # pnl_bruto = 1000, custos = 100+100, pnl_liquido = 800. O bootstrap deve
    # reamostrar o LÍQUIDO (custos embutidos), nunca o bruto.
    row = _trade_row(800.0, custo_entrada=100.0, custo_saida=100.0)
    assert row["pnl_bruto"] == 1000.0
    res = mk_res([], rows=[row])
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=500, seed=12345)
    esperado_liquido = 800.0 / CAPITAL
    esperado_bruto = 1000.0 / CAPITAL
    assert np.allclose(r.retornos_simulados, esperado_liquido)
    assert not np.allclose(r.retornos_simulados, esperado_bruto)


# ── Grupo D — MDD reamostrado (bootstrap) ─────────────────────────────────────


def test_mdd_bootstrap_todos_negativos_ou_zero():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=5_000, seed=12345)
    assert np.all(r.mdds_simulados <= 0.0)


def test_mdd_com_todos_trades_vencedores_igual_zero():
    # Equity só sobe → sem drawdown → MDD 0 em toda simulação.
    res = mk_res([100.0, 250.0, 80.0, 300.0])
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=2_000, seed=12345)
    assert np.all(r.mdds_simulados == 0.0)
    assert r.mdd_p05 == 0.0
    assert r.mdd_p95 == 0.0


def test_mdd_p05_menor_ou_igual_mdd_p95():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=5_000, seed=12345)
    assert r.mdd_p05 <= r.mdd_p95


# ── Grupo E — Comparação com Ibov ─────────────────────────────────────────────


def test_prob_supera_ibov_none_quando_ibov_none():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=1_000, seed=12345, retorno_ibov=None)
    assert r.prob_supera_ibov is None
    assert r.retorno_ibov_referencia is None


def test_prob_supera_ibov_igual_zero_quando_ibov_maior_que_todos():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(
        res, n_simulacoes=2_000, seed=12345, retorno_ibov=10.0
    )
    assert r.prob_supera_ibov == 0.0
    assert r.retorno_ibov_referencia == 10.0


def test_prob_supera_ibov_igual_um_quando_ibov_menor_que_todos():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(
        res, n_simulacoes=2_000, seed=12345, retorno_ibov=-10.0
    )
    assert r.prob_supera_ibov == 1.0


# ── Grupo F — Robustez ────────────────────────────────────────────────────────


def test_zero_trades_levanta_valueerror():
    res = mk_res([])
    with pytest.raises(ValueError):
        rodar_bootstrap_retornos_trades(res, n_simulacoes=1_000, seed=12345)


def test_percentis_p05_menor_que_p50_menor_que_p95():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=10_000, seed=12345)
    assert r.retorno_p05 < r.retorno_p50 < r.retorno_p95


def test_retorno_medio_bate_com_media_da_distribuicao():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=5_000, seed=12345)
    assert r.retorno_medio == pytest.approx(float(np.mean(r.retornos_simulados)))


# ── Grupo G — Metadados e contrato ────────────────────────────────────────────


def test_bootstrap_retorna_tecnica_string_correta():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_bootstrap_retornos_trades(res, n_simulacoes=500, seed=12345)
    assert isinstance(r, ResultadoMonteCarlo)
    assert r.tecnica == "bootstrap_retornos"
    assert r.n_simulacoes == 500
    assert r.seed == 12345


# ══════════════════════════════════════════════════════════════════════════════
# Subetapa 2.2 — Permutação sem reposição (Técnica B)
#
# Semântica travada: permutar a ORDEM dos mesmos N trades NÃO muda o retorno
# total (soma comutativa — teste 13), mas muda a equity curve intermediária e
# portanto o MDD (teste 15). Bootstrap testa robustez do RETORNO; permutação
# testa robustez do RISCO DE SEQUÊNCIA. Duas perguntas independentes.
# ══════════════════════════════════════════════════════════════════════════════


# ── Grupo A (deferido da 2.1) — Determinismo da permutação ────────────────────


def test_permutacao_dois_runs_mesma_seed_produzem_resultados_identicos():
    res = mk_res(_PNLS_ALTERNADO)
    a = rodar_permutacao_ordem_trades(res, n_simulacoes=2_000, seed=12345)
    b = rodar_permutacao_ordem_trades(res, n_simulacoes=2_000, seed=12345)
    assert np.array_equal(a.retornos_simulados, b.retornos_simulados)
    assert np.array_equal(a.mdds_simulados, b.mdds_simulados)


# ── Grupo C — Permutação sem reposição ────────────────────────────────────────


def test_permutacao_preserva_conjunto_de_trades():
    # 5 P&L com somas de subconjunto DISTINTAS (100,200,400,800,1600): só o
    # conjunto COMPLETO soma 3100. Se alguma simulação largasse/duplicasse um
    # trade (bootstrap), a soma — logo o retorno — mudaria. Retorno constante e
    # igual a 3100/capital em TODAS as 10k prova que cada simulação usa
    # exatamente os N trades originais, sem reposição. Distingue de bootstrap.
    pnls = [100.0, 200.0, 400.0, 800.0, 1600.0]
    res = mk_res(pnls)
    r = rodar_permutacao_ordem_trades(res, n_simulacoes=10_000, seed=12345)
    esperado = sum(pnls) / CAPITAL  # 3100/100000
    assert np.unique(r.retornos_simulados).size == 1
    assert np.allclose(r.retornos_simulados, esperado)
    # Contraste explícito: o bootstrap no MESMO fixture varia (reposição).
    b = rodar_bootstrap_retornos_trades(res, n_simulacoes=10_000, seed=12345)
    assert b.retorno_desvio > 0.0


def test_permutacao_n_igual_2_funciona():
    # n=2 é o mínimo permitido (R7). As duas ordens possíveis de [+1000, -500]:
    #   [+1000, -500] → equity 100000,101000,100500 → MDD = -500/101000
    #   [-500, +1000] → equity 100000,99500,100500  → MDD = -500/100000
    # Ambas devem aparecer nas 10k; o retorno é invariante (500/100000).
    res = mk_res([1000.0, -500.0])
    r = rodar_permutacao_ordem_trades(res, n_simulacoes=10_000, seed=12345)
    assert np.unique(r.retornos_simulados).size == 1
    assert r.retorno_medio == pytest.approx(500.0 / CAPITAL)
    assert np.unique(np.round(r.mdds_simulados, 12)).size == 2
    assert r.mdds_simulados.min() == pytest.approx(-500.0 / CAPITAL)
    assert r.mdds_simulados.max() == pytest.approx(-500.0 / 101_000.0)


def test_permutacao_n_igual_1_levanta_valueerror():
    res = mk_res([500.0])
    with pytest.raises(ValueError, match="2 trades"):
        rodar_permutacao_ordem_trades(res, n_simulacoes=1_000, seed=12345)


def test_permutacao_retorno_final_igual_em_todas_simulacoes():
    """Verdade matemática: soma é comutativa. Permutar a ordem dos trades NÃO
    muda o retorno total, apenas a equity curve intermediária. P&L inteiros →
    somas parciais exatas em float64 → igualdade EXATA (não aproximação).

    Red aqui = bug de implementação (provável reposição em vez de permutação).
    """
    pnls = [100.0, -50.0, 200.0, -75.0, 30.0]  # soma = 205, inteiros exatos
    res = mk_res(pnls)
    r = rodar_permutacao_ordem_trades(res, n_simulacoes=10_000, seed=12345)
    assert np.all(r.retornos_simulados == r.retornos_simulados[0])
    assert r.retorno_desvio == 0.0
    assert r.retornos_simulados[0] == pytest.approx(205.0 / CAPITAL)


# ── Grupo D — MDD reamostrado (permutação) ────────────────────────────────────


def test_mdd_permutacao_mesmo_conjunto_trades_produz_mdds_diferentes():
    """A permutação existe para testar RISCO DE SEQUÊNCIA: o retorno total é
    insensível à ordem (teste 13), mas o MDD não é.

    Fixture [+1000, +1000, -500, -500] (capital 100k). Enumerando as 24 ordens,
    o MDD reamostrado varia entre dois extremos EXATOS (todos negativos — com
    perdas sempre há drawdown, nenhuma ordem dá MDD 0; MDD 0 exigiria só
    vencedores, ver teste 16):
      - PIOR (perdedores primeiro, [-500,-500,...]): pico 100000, vale 99000
        → MDD = -1000/100000 = -0.01
      - MELHOR (perdas separadas, [1000,-500,1000,-500]): quedas de 500 em picos
        101000 e 101500 → MDD = -500/101000 ≈ -0.004950
    Como 4! = 24 ordens cabem nas 10k simulações, ambos os extremos aparecem.
    """
    res = mk_res([1000.0, 1000.0, -500.0, -500.0])
    r = rodar_permutacao_ordem_trades(res, n_simulacoes=10_000, seed=12345)
    # Há variação REAL de MDD (não ruído numérico) e percentis bem ordenados.
    assert r.mdd_p05 != r.mdd_p95
    assert r.mdd_p05 < r.mdd_p95
    # Extremos teóricos exatos presentes na distribuição.
    assert r.mdds_simulados.min() == pytest.approx(-1000.0 / CAPITAL)   # -0.01
    assert r.mdds_simulados.max() == pytest.approx(-500.0 / 101_000.0)  # melhor
    assert np.all(r.mdds_simulados < 0.0)  # nenhuma ordem zera o MDD


# ── Grupo G (deferido da 2.1) — Metadado da permutação ────────────────────────


def test_permutacao_retorna_tecnica_string_correta():
    res = mk_res(_PNLS_ALTERNADO)
    r = rodar_permutacao_ordem_trades(res, n_simulacoes=500, seed=12345)
    assert isinstance(r, ResultadoMonteCarlo)
    assert r.tecnica == "permutacao_ordem"
    assert r.n_simulacoes == 500
    assert r.seed == 12345
