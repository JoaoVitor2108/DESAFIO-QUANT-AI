"""Testes do PROGRAM — BacktestEngine (Etapa 1).

Motor de backtest event-driven: calendário B3, loop diário, execução simulada
com custos, detecção intraday de stop/take, mark-to-market e integração com o
contrato público do ORQUESTRADOR.

Convenções travadas (ver CONTEXTO_FIXO_FINAL.md e o prompt da Etapa 1):
- Todas as datas nos contratos/dataclasses do PROGRAM são `pd.Timestamp` NAIVE.
- O calendário BMF sai naive (`.tz is None`).
- Fakes ESTRITOS (tests/fakes.py) replicam a exigência tz-aware das fronteiras:
  se o engine esquecer o adaptador `_para_sp`, o teste FALHA (não passa em
  "naive-land").

Cobertura R1–R9 e Literal de motivo: ver o mapa no fim do arquivo.
"""

import pandas as pd
import pytest

from backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    _para_naive,
    _para_sp,
)
from fakes import (
    FUSO,
    FakeJournalBacktest,
    FakeOrquestrador,
    mk_decisao,
    mk_fechamento,
    mk_ordem,
    preco_df,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _d(s: str) -> pd.Timestamp:
    """Timestamp naive (convenção do PROGRAM)."""
    return pd.Timestamp(s)


def _sp(s: str) -> pd.Timestamp:
    """Timestamp tz-aware America/Sao_Paulo (chaves de cronograma / asserts)."""
    return pd.Timestamp(s, tz=FUSO)


# Pregões B3 confirmados via pandas_market_calendars ('BMF').
D11 = [
    "2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07",
    "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14",
    "2024-03-15",
]
D5 = D11[:5]
# Janela ao redor da Sexta-feira Santa (2024-03-29 NÃO é pregão).
DSANTA = ["2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01",
          "2024-04-02"]


def _serie_df(datas, o, h, l, c, overrides=None):
    """DataFrame OHLCV com valores-base por coluna e overrides por data:
    `overrides = {"2024-03-06": {"l": 90, "c": 91}, ...}`."""
    overrides = overrides or {}
    op, hi, lo, cl = [], [], [], []
    for d in datas:
        ov = overrides.get(d, {})
        op.append(ov.get("o", o))
        hi.append(ov.get("h", h))
        lo.append(ov.get("l", l))
        cl.append(ov.get("c", c))
    return preco_df(datas, op, hi, lo, cl)


def _engine(dados, cronograma, setores=None, config=None):
    journal = FakeJournalBacktest(dados=dados, setores=setores)
    orq = FakeOrquestrador(cronograma=cronograma)
    engine = BacktestEngine(journal, orq, config=config)
    return engine, journal, orq


# ══════════════════════════════════════════════════════════════════════════════
# Grupo A — Fluxo básico
# ══════════════════════════════════════════════════════════════════════════════


def test_capital_inicial_sem_ordens():
    """A1/R7 — 5 dias sem ordens: caixa 100k, equity_diario constante em 100k."""
    engine, _, _ = _engine(dados={}, cronograma={})
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-07"))
    assert res.capital_final == pytest.approx(100_000.0)
    assert res.n_trades == 0
    assert len(res.equity_diario) == 5
    assert all(v == pytest.approx(100_000.0) for v in res.equity_diario)


def test_abre_e_fecha_por_prazo():
    """A2/R1/R3/R4 — abre PETR4, ORQUESTRADOR fecha por prazo; preço de saída =
    Open[D+1], custos e P&L exatos."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-11": {"o": 110, "h": 112, "l": 108, "c": 110}})
    cron = {
        _sp("2024-03-01"): mk_decisao(
            _sp("2024-03-01"),
            novas_ordens=[mk_ordem("PETR4.SA", _sp("2024-03-04"), setor="Energia")],
        ),
        _sp("2024-03-08"): mk_decisao(
            _sp("2024-03-08"),
            fechamentos=[mk_fechamento("PETR4.SA", "prazo", _sp("2024-03-08"))],
        ),
    }
    engine, _, orq = _engine({"PETR4.SA": df}, cron)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))

    assert res.n_trades == 1
    t = res.trades.iloc[0]
    assert t["motivo"] == "prazo"
    assert t["qtd"] == 150  # int(0.15*100000/100)
    assert t["preco_entrada"] == pytest.approx(100.0)
    assert t["preco_saida"] == pytest.approx(110.0)  # Open[2024-03-11]
    assert t["data_entrada"] == _d("2024-03-04")
    assert t["data_saida"] == _d("2024-03-11")
    assert t["custo_entrada"] == pytest.approx(0.004 * 100 * 150)   # 60
    assert t["custo_saida"] == pytest.approx(0.004 * 110 * 150)     # 66
    assert t["pnl_bruto"] == pytest.approx((110 - 100) * 150)       # 1500
    assert t["pnl_liquido"] == pytest.approx(1500 - 60 - 66)        # 1374
    assert ("PETR4.SA", "Energia", 100.0, _sp("2024-03-04")) in orq.execucoes
    assert ("PETR4.SA", _sp("2024-03-11")) in orq.fechamentos_notificados


def test_abre_e_fecha_por_reversao():
    """A3 — idêntico ao prazo, mas motivo='reversao'."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-11": {"o": 110, "h": 112, "l": 108, "c": 110}})
    cron = {
        _sp("2024-03-01"): mk_decisao(
            _sp("2024-03-01"),
            novas_ordens=[mk_ordem("PETR4.SA", _sp("2024-03-04"))],
        ),
        _sp("2024-03-08"): mk_decisao(
            _sp("2024-03-08"),
            fechamentos=[mk_fechamento("PETR4.SA", "reversao", _sp("2024-03-08"))],
        ),
    }
    engine, _, _ = _engine({"PETR4.SA": df}, cron)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert res.n_trades == 1
    assert res.trades.iloc[0]["motivo"] == "reversao"
    assert res.trades.iloc[0]["preco_saida"] == pytest.approx(110.0)


def test_multiplas_posicoes_simultaneas():
    """A4 — abre 3 tickers em sequência, todos ativos ao mesmo tempo, fecham em
    datas distintas."""
    dfa = _serie_df(D11, 100, 101, 99, 100)
    dfb = _serie_df(D11, 100, 101, 99, 100)
    dfc = _serie_df(D11, 100, 101, 99, 100)
    cron = {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
            novas_ordens=[mk_ordem("AAAA3.SA", _sp("2024-03-04"))]),
        _sp("2024-03-05"): mk_decisao(_sp("2024-03-05"),
            novas_ordens=[mk_ordem("BBBB3.SA", _sp("2024-03-06"))]),
        _sp("2024-03-07"): mk_decisao(_sp("2024-03-07"),
            novas_ordens=[mk_ordem("CCCC3.SA", _sp("2024-03-08"))]),
        _sp("2024-03-11"): mk_decisao(_sp("2024-03-11"),
            fechamentos=[mk_fechamento("AAAA3.SA", "prazo", _sp("2024-03-11"))]),
        _sp("2024-03-12"): mk_decisao(_sp("2024-03-12"),
            fechamentos=[mk_fechamento("BBBB3.SA", "prazo", _sp("2024-03-12"))]),
        _sp("2024-03-13"): mk_decisao(_sp("2024-03-13"),
            fechamentos=[mk_fechamento("CCCC3.SA", "prazo", _sp("2024-03-13"))]),
    }
    engine, _, _ = _engine(
        {"AAAA3.SA": dfa, "BBBB3.SA": dfb, "CCCC3.SA": dfc}, cron)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))

    assert res.n_trades == 3
    assert set(res.trades["motivo"]) == {"prazo"}
    saidas = sorted(res.trades["data_saida"])
    assert saidas == [_d("2024-03-12"), _d("2024-03-13"), _d("2024-03-14")]
    # Simultaneidade: todas entram (≤ 03-08) antes da 1ª saída (03-12).
    assert res.trades["data_entrada"].max() < res.trades["data_saida"].min()


def test_ordem_pulada_por_caixa_insuficiente():
    """A5/R1 — 2ª ordem com notional > caixa restante: aviso 'caixa_insuficiente',
    sem crash, sem furar o caixa."""
    cfg = BacktestConfig(sizing_pct=0.6)  # drena o caixa rápido
    dfa = _serie_df(D11, 100, 101, 99, 100)
    dfb = _serie_df(D11, 100, 101, 99, 100)
    cron = {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
            novas_ordens=[mk_ordem("AAAA3.SA", _sp("2024-03-04"))]),
        _sp("2024-03-05"): mk_decisao(_sp("2024-03-05"),
            novas_ordens=[mk_ordem("BBBB3.SA", _sp("2024-03-06"))]),
    }
    engine, _, orq = _engine({"AAAA3.SA": dfa, "BBBB3.SA": dfb}, cron, config=cfg)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))

    tipos = [a["tipo"] for a in res.avisos]
    assert "caixa_insuficiente" in tipos
    assert not any(e[0] == "BBBB3.SA" for e in orq.execucoes)  # B não executou
    assert res.capital_final >= 0  # não furou o caixa


# ══════════════════════════════════════════════════════════════════════════════
# Grupo B — Detecção intraday
# ══════════════════════════════════════════════════════════════════════════════


def _cron_abre(ticker="PETR4.SA", exec_dia="2024-03-04"):
    return {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
            novas_ordens=[mk_ordem(ticker, _sp(exec_dia))]),
    }


def test_stop_dispara_por_low():
    """B6/R4 — Low[D] ≤ stop_price → fecha por stop; preço = stop exato (92)."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-06": {"o": 95, "h": 96, "l": 90, "c": 91}})
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    stops = res.trades[res.trades["motivo"] == "stop"]
    assert len(stops) == 1
    t = stops.iloc[0]
    assert t["preco_saida"] == pytest.approx(0.92 * 100)  # 92.0 exato
    assert t["data_saida"] == _d("2024-03-06")


def test_take_dispara_por_high():
    """B7/R4 — High[D] ≥ take_price → fecha por take; preço = take exato (115)."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-06": {"o": 105, "h": 120, "l": 104, "c": 118}})
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    takes = res.trades[res.trades["motivo"] == "take"]
    assert len(takes) == 1
    assert takes.iloc[0]["preco_saida"] == pytest.approx(1.15 * 100)  # 115.0 exato


def test_stop_e_take_mesmo_dia_prioriza_stop():
    """B8/R1 — Low<stop E High>take no mesmo dia: prioridade STOP (conservador)."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-06": {"o": 100, "h": 120, "l": 90, "c": 100}})
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert res.n_trades == 1
    assert res.trades.iloc[0]["motivo"] == "stop"
    assert res.trades.iloc[0]["preco_saida"] == pytest.approx(92.0)


def test_stop_intraday_impede_fechamento_orquestrador():
    """B9/R1 Passo 3 — stop no Passo 1 e ORQUESTRADOR devolve fechamento do mesmo
    ticker no Passo 3 → RuntimeError (invariante violada)."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-06": {"o": 95, "h": 96, "l": 90, "c": 91}})
    cron = _cron_abre()
    cron[_sp("2024-03-06")] = mk_decisao(_sp("2024-03-06"),
        fechamentos=[mk_fechamento("PETR4.SA", "prazo", _sp("2024-03-06"))])
    engine, _, _ = _engine({"PETR4.SA": df}, cron)
    with pytest.raises(RuntimeError):
        engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))


def test_stop_no_dia_da_entrada_pode_disparar():
    """B10/R4 — Low[D_entrada] ≤ stop_price no MESMO dia da entrada dispara stop.
    Entrada é no Open; Low do mesmo dia pode furar o stop."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-04": {"o": 100, "h": 101, "l": 90, "c": 95}})
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre(exec_dia="2024-03-04"))
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert res.n_trades == 1
    t = res.trades.iloc[0]
    assert t["motivo"] == "stop"
    assert t["data_entrada"] == _d("2024-03-04")
    assert t["data_saida"] == _d("2024-03-04")
    assert t["dias_uteis_ate_saida"] == 0


def test_sem_dados_intraday_pula_verificacao():
    """B11/R8 — ticker sem barra em D: não crasha, mantém posição, loga aviso."""
    # Sem a barra de 2024-03-06 (gap intraday).
    datas = [d for d in D11 if d != "2024-03-06"]
    df = _serie_df(datas, 100, 101, 99, 100)
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    tipos_data = [(a["tipo"], a["data"], a["ticker"]) for a in res.avisos]
    assert ("sem_dados_intraday", _d("2024-03-06"), "PETR4.SA") in tipos_data
    # Não fechou por stop no dia sem dado; só fecha por fim_backtest no fim.
    assert set(res.trades["motivo"]) == {"fim_backtest"}


def test_notificacao_fechamento_stop_sem_motivo():
    """B12/contrato — notificar_fechamento é chamado com 2 args (ticker, data),
    sem motivo."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-06": {"o": 95, "h": 96, "l": 90, "c": 91}})
    engine, _, orq = _engine({"PETR4.SA": df}, _cron_abre())
    engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert ("PETR4.SA", _sp("2024-03-06")) in orq.fechamentos_notificados
    assert all(len(ch) == 2 for ch in orq.fechamentos_notificados)


# ══════════════════════════════════════════════════════════════════════════════
# Grupo C — Custos e P&L
# ══════════════════════════════════════════════════════════════════════════════


def _abre_e_fecha_prazo(entry_open=100, exit_open=110):
    """Cenário base: abre PETR4 @entry_open (03-04), fecha por prazo @exit_open
    (03-11)."""
    df = _serie_df(D11, entry_open, entry_open + 1, entry_open - 1, entry_open,
                   overrides={"2024-03-11": {
                       "o": exit_open, "h": exit_open + 2,
                       "l": exit_open - 2, "c": exit_open}})
    cron = {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
            novas_ordens=[mk_ordem("PETR4.SA", _sp("2024-03-04"))]),
        _sp("2024-03-08"): mk_decisao(_sp("2024-03-08"),
            fechamentos=[mk_fechamento("PETR4.SA", "prazo", _sp("2024-03-08"))]),
    }
    return _engine({"PETR4.SA": df}, cron)


def test_custo_entrada_exato_04pct():
    """C13/R3 — custo_entrada = 0.004 × preco_entrada × qtd."""
    engine, _, _ = _abre_e_fecha_prazo()
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    t = res.trades.iloc[0]
    assert t["custo_entrada"] == pytest.approx(0.004 * 100 * t["qtd"])


def test_custo_saida_exato_04pct():
    """C14/R3 — custo_saida = 0.004 × preco_saida × qtd."""
    engine, _, _ = _abre_e_fecha_prazo()
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    t = res.trades.iloc[0]
    assert t["custo_saida"] == pytest.approx(0.004 * 110 * t["qtd"])


def test_pnl_bruto_vs_liquido():
    """C15/R3 — preço +10%: pnl_bruto = (P_saida-P_entrada)×qtd; pnl_liquido =
    pnl_bruto - custo_entrada - custo_saida."""
    engine, _, _ = _abre_e_fecha_prazo(entry_open=100, exit_open=110)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    t = res.trades.iloc[0]
    assert t["pnl_bruto"] == pytest.approx((110 - 100) * t["qtd"])
    assert t["pnl_liquido"] == pytest.approx(
        t["pnl_bruto"] - t["custo_entrada"] - t["custo_saida"])


def test_pnl_liquido_negativo_quando_custos_comem_ganho():
    """C16/R3 — ganho bruto de 0.5% (< ~0.8% round-trip): pnl_liquido < 0."""
    engine, _, _ = _abre_e_fecha_prazo(entry_open=100, exit_open=100.5)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    t = res.trades.iloc[0]
    assert t["pnl_bruto"] > 0
    assert t["pnl_liquido"] < 0


def test_sizing_15pct_do_equity_corrente():
    """C17/R6 — a 2ª posição dimensiona por 15% do equity_hoje (fim de D-1), que
    já reflete a 1ª posição marcada a mercado (não capital inicial, não caixa)."""
    dfa = _serie_df(D5, 100, 101, 99, 100)
    dfb = _serie_df(D5, 100, 101, 99, 100)
    cron = {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
            novas_ordens=[mk_ordem("AAAA3.SA", _sp("2024-03-04"))]),
        _sp("2024-03-05"): mk_decisao(_sp("2024-03-05"),
            novas_ordens=[mk_ordem("BBBB3.SA", _sp("2024-03-06"))]),
    }
    engine, _, orq = _engine({"AAAA3.SA": dfa, "BBBB3.SA": dfb}, cron)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-07"))

    # equity_hoje visto ao decidir 03-05 = caixa(84940) + 150×Close[03-04](100)
    eq_0305 = dict((d, e) for d, e in orq.decidir_calls)[_sp("2024-03-05")]
    assert eq_0305 == pytest.approx(99_940.0)
    # qtd_B = int(0.15 × 99940 / 100) = 149 (≠ 150 do capital, ≠ 127 do caixa)
    tb = res.trades[res.trades["ticker"] == "BBBB3.SA"].iloc[0]
    assert tb["qtd"] == 149


# ══════════════════════════════════════════════════════════════════════════════
# Grupo D — Mark-to-market e equity
# ══════════════════════════════════════════════════════════════════════════════


def test_equity_hoje_reflete_fim_D_menos_1():
    """D18/R2 — equity_hoje recebido ao decidir dia 5 = caixa + Σ qtd×Close[dia 4]."""
    df = _serie_df(D11, 100, 101, 99, 100)
    engine, _, orq = _engine({"PETR4.SA": df}, _cron_abre())
    engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    # dia 5 = 2024-03-07 (5º pregão). Posição PETR4 aberta (entrada 03-04).
    eq = dict((d, e) for d, e in orq.decidir_calls)[_sp("2024-03-07")]
    # caixa após abrir = 100000 - (150×100 + 0.004×150×100) = 84940;
    # + 150 × Close[03-06]=100 → 99940
    assert eq == pytest.approx(99_940.0)


def test_equity_diario_registrado_todo_dia_util():
    """D19/R1 Passo 5 — um ponto de equity por pregão: n_dias_uteis == len."""
    df = _serie_df(D11, 100, 101, 99, 100)
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert res.n_dias_uteis == len(res.equity_diario) == len(D11)


def test_equity_sobe_com_marcacao_favoravel():
    """D20 — Close sobe dia a dia: equity_diario sobe (monotônico no período de
    marcação favorável)."""
    ov = {"2024-03-04": {"c": 102}, "2024-03-05": {"c": 104},
          "2024-03-06": {"c": 106}, "2024-03-07": {"c": 108}}
    df = _serie_df(D5, 100, 109, 99, 100, overrides=ov)  # High<115 evita take
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-07"))
    # dias com a posição marcada em Close crescente: 03-04..03-07
    tail = res.equity_diario.loc[_d("2024-03-04"):_d("2024-03-07")]
    assert tail.is_monotonic_increasing
    assert tail.iloc[-1] > tail.iloc[0]


def test_equity_desce_com_marcacao_desfavoravel():
    """D21 — Close cai dia a dia: equity_diario cai (sem furar o stop=92)."""
    ov = {"2024-03-04": {"c": 99, "l": 98}, "2024-03-05": {"c": 98, "l": 97},
          "2024-03-06": {"c": 97, "l": 96}, "2024-03-07": {"c": 96, "l": 95}}
    df = _serie_df(D5, 100, 101, 99, 100, overrides=ov)
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-07"))
    tail = res.equity_diario.loc[_d("2024-03-04"):_d("2024-03-07")]
    assert tail.is_monotonic_decreasing
    assert tail.iloc[-1] < tail.iloc[0]


def test_equity_pos_fechamento_intraday_atualiza_caixa():
    """D22/R1 — stop dispara em D: caixa liberado; equity_fim_D reflete o caixa."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-06": {"o": 95, "h": 96, "l": 90, "c": 91}})
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    # caixa após abrir=84940; stop: +150×92 - 0.004×92×150 = +13800-55.2
    esperado = 84_940.0 + 150 * 92 - 0.004 * 92 * 150  # 98684.8
    assert res.equity_diario.loc[_d("2024-03-06")] == pytest.approx(esperado)


def test_close_marcacao_fallback_para_close_anterior_quando_barra_ausente():
    """D/R8 — política de fallback do MTM: quando falta a barra do dia D para uma
    posição aberta (pregão pulado por flakiness do yfinance), `_close_marcacao`
    carrega o último Close disponível ≤ D (carry-forward), NÃO crasha nem usa o
    preço de entrada. Aqui a barra de 2024-03-06 está ausente e o Close[03-05]=107
    (distintivo) deve ser usado para marcar a posição."""
    datas = [d for d in D11 if d != "2024-03-06"]
    df = _serie_df(datas, 100, 101, 99, 100,
                   overrides={"2024-03-05": {"c": 107, "h": 108}})
    engine, _, _ = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    # 03-06 existe no equity_diario (o loop marca o dia mesmo sem barra do ticker)
    assert _d("2024-03-06") in res.equity_diario.index
    # caixa após abrir=84940; MTM carrega Close[03-05]=107 → 84940 + 150×107
    assert res.equity_diario.loc[_d("2024-03-06")] == pytest.approx(84_940.0 + 150 * 107)
    # fallback é carry-forward, NÃO o preço de entrada (100) nem "mtm_sem_preco"
    assert not any(a["tipo"] == "mtm_sem_preco" and a["ticker"] == "PETR4.SA"
                   for a in res.avisos)


# ══════════════════════════════════════════════════════════════════════════════
# Grupo E — Calendário e bordas
# ══════════════════════════════════════════════════════════════════════════════


def test_calendario_bmf_pula_sexta_santa_2024():
    """E23/R5 — 2024-03-29 (Sexta-feira Santa) NÃO aparece no calendário."""
    engine = BacktestEngine(journal=None, orquestrador=None)
    cal = engine._calendario_bmf(_d("2024-03-01"), _d("2024-06-30"))
    assert cal.tz is None
    assert _d("2024-03-29") not in cal


def test_calendario_bmf_pula_corpus_christi_2024():
    """E24/R5 — 2024-05-30 (Corpus Christi) NÃO aparece no calendário."""
    engine = BacktestEngine(journal=None, orquestrador=None)
    cal = engine._calendario_bmf(_d("2024-03-01"), _d("2024-06-30"))
    assert cal.tz is None
    assert _d("2024-05-30") not in cal


def test_fechamento_forcado_no_fim_backtest():
    """E25/R8 — posição aberta no último dia: fechada por 'fim_backtest' pelo
    Close[data_fim]."""
    df = _serie_df(D5, 100, 101, 99, 100, overrides={"2024-03-07": {"c": 105}})
    engine, _, orq = _engine({"PETR4.SA": df}, _cron_abre())
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-07"))
    assert res.n_trades == 1
    t = res.trades.iloc[0]
    assert t["motivo"] == "fim_backtest"
    assert t["preco_saida"] == pytest.approx(105.0)  # Close[2024-03-07]
    assert t["data_saida"] == _d("2024-03-07")
    assert ("PETR4.SA", _sp("2024-03-07")) in orq.fechamentos_notificados


def test_fechamento_prazo_pula_para_proximo_pregao_apos_feriado():
    """E26/R1 Passo 3 — D+1 é feriado (Sexta Santa): fechamento executa no próximo
    pregão (04-01), não em 03-29."""
    df = _serie_df(DSANTA, 100, 101, 99, 100,
                   overrides={"2024-04-01": {"o": 110, "h": 112, "l": 108, "c": 110}})
    cron = {
        _sp("2024-03-25"): mk_decisao(_sp("2024-03-25"),
            novas_ordens=[mk_ordem("PETR4.SA", _sp("2024-03-26"))]),
        _sp("2024-03-28"): mk_decisao(_sp("2024-03-28"),
            fechamentos=[mk_fechamento("PETR4.SA", "prazo", _sp("2024-03-28"))]),
    }
    engine, _, _ = _engine({"PETR4.SA": df}, cron)
    res = engine.rodar_backtest(_d("2024-03-25"), _d("2024-04-02"))
    t = res.trades[res.trades["motivo"] == "prazo"].iloc[0]
    assert t["data_saida"] == _d("2024-04-01")  # pulou 2024-03-29
    assert t["preco_saida"] == pytest.approx(110.0)  # Open[2024-04-01]


def test_ticker_sem_dados_gera_aviso_e_nao_crasha():
    """E27/R8 — ticker sem QUALQUER dado (JBSS3/ELET3 flaky): aviso e sem crash;
    ordem não executada."""
    # Journal só tem PETR4; ordem para JBSS3 (ausente) → DadoIndisponivel.
    cron = {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
            novas_ordens=[mk_ordem("JBSS3.SA", _sp("2024-03-04"))]),
    }
    engine, _, orq = _engine({"PETR4.SA": _serie_df(D5, 100, 101, 99, 100)}, cron)
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-07"))  # não levanta
    assert any(a["ticker"] == "JBSS3.SA" and a["tipo"] == "sem_dados_execucao"
               for a in res.avisos)
    assert not any(e[0] == "JBSS3.SA" for e in orq.execucoes)
    assert res.n_trades == 0


# ══════════════════════════════════════════════════════════════════════════════
# Grupo F — Integração com o contrato do ORQUESTRADOR
# ══════════════════════════════════════════════════════════════════════════════


def test_notificacao_execucao_com_todos_campos():
    """F28/contrato — notificar_execucao recebe (ticker, setor, preco, data)."""
    df = _serie_df(D11, 100, 101, 99, 100)
    cron = {_sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
        novas_ordens=[mk_ordem("PETR4.SA", _sp("2024-03-04"), setor="Energia")])}
    engine, _, orq = _engine({"PETR4.SA": df}, cron)
    engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert orq.execucoes[0] == ("PETR4.SA", "Energia", 100.0, _sp("2024-03-04"))


def test_notificacao_fechamento_apenas_ticker_e_data():
    """F29/contrato — notificar_fechamento recebe apenas (ticker, data)."""
    engine, _, orq = _abre_e_fecha_prazo()
    engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert len(orq.fechamentos_notificados) >= 1
    assert all(len(ch) == 2 for ch in orq.fechamentos_notificados)


def test_conflito_stop_intraday_vs_fechamento_orquestrador_levanta():
    """F30/R1 Passo 3 — RuntimeError com mensagem clara mencionando ticker e data."""
    df = _serie_df(D11, 100, 101, 99, 100,
                   overrides={"2024-03-06": {"o": 95, "h": 96, "l": 90, "c": 91}})
    cron = _cron_abre()
    cron[_sp("2024-03-06")] = mk_decisao(_sp("2024-03-06"),
        fechamentos=[mk_fechamento("PETR4.SA", "prazo", _sp("2024-03-06"))])
    engine, _, _ = _engine({"PETR4.SA": df}, cron)
    with pytest.raises(RuntimeError, match=r"PETR4\.SA.*2024-03-06"):
        engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))


def test_equity_hoje_e_argumento_de_decidir():
    """F31/R2 — FakeOrquestrador registra equity_hoje; bate com o MTM esperado
    (1º dia = capital)."""
    df = _serie_df(D11, 100, 101, 99, 100)
    engine, _, orq = _engine({"PETR4.SA": df}, _cron_abre())
    engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    calls = dict((d, e) for d, e in orq.decidir_calls)
    assert calls[_sp("2024-03-01")] == pytest.approx(100_000.0)  # 1º dia


def test_ordem_das_notificacoes_no_dia():
    """F32/R1 — no dia D a ordem é: Passo 1 (notificar_fechamento por stop) ANTES
    do Passo 2 (decidir) ANTES do Passo 4 (notificar_execucao)."""
    # AAAA3 aberta (entra 03-04), STOP em 03-06. No mesmo 03-06, o cronograma
    # emite uma nova ordem BBBB3 com execução no próprio 03-06.
    dfa = _serie_df(D11, 100, 101, 99, 100,
                    overrides={"2024-03-06": {"o": 95, "h": 96, "l": 90, "c": 91}})
    dfb = _serie_df(D11, 100, 101, 99, 100)
    cron = {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
            novas_ordens=[mk_ordem("AAAA3.SA", _sp("2024-03-04"))]),
        _sp("2024-03-06"): mk_decisao(_sp("2024-03-06"),
            novas_ordens=[mk_ordem("BBBB3.SA", _sp("2024-03-06"))]),
    }
    engine, _, orq = _engine({"AAAA3.SA": dfa, "BBBB3.SA": dfb}, cron)
    engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))

    log = orq.log
    i_fech = next(i for i, e in enumerate(log)
                  if e[0] == "fechamento" and e[1] == "AAAA3.SA")
    i_dec = next(i for i, e in enumerate(log)
                 if e[0] == "decidir" and e[1] == _sp("2024-03-06"))
    i_exec = next(i for i, e in enumerate(log)
                  if e[0] == "execucao" and e[1] == "BBBB3.SA")
    assert i_fech < i_dec < i_exec


# ══════════════════════════════════════════════════════════════════════════════
# Grupo G — Determinismo e reprodutibilidade
# ══════════════════════════════════════════════════════════════════════════════


def test_dois_runs_identicos_produzem_resultados_identicos():
    """G33/R9 — dois runs com Fakes idênticos: trades e equity byte-a-byte iguais."""
    def _run():
        df = _serie_df(D11, 100, 101, 99, 100,
                       overrides={"2024-03-11": {"o": 110, "h": 112, "l": 108, "c": 110}})
        cron = {
            _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"),
                novas_ordens=[mk_ordem("PETR4.SA", _sp("2024-03-04"))]),
            _sp("2024-03-08"): mk_decisao(_sp("2024-03-08"),
                fechamentos=[mk_fechamento("PETR4.SA", "prazo", _sp("2024-03-08"))]),
        }
        eng, _, _ = _engine({"PETR4.SA": df}, cron)
        return eng.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))

    r1, r2 = _run(), _run()
    assert r1.trades.equals(r2.trades)
    assert r1.equity_diario.equals(r2.equity_diario)
    assert r1.capital_final == r2.capital_final


def test_ordem_alfabetica_de_tickers_na_iteracao_stop_take():
    """G34/R9 — dois tickers com stop no mesmo dia: iteração em ordem alfabética
    (AAAA3 antes de ZZZZ3)."""
    ov = {"2024-03-06": {"o": 95, "h": 96, "l": 90, "c": 91}}
    dfa = _serie_df(D11, 100, 101, 99, 100, overrides=ov)
    dfz = _serie_df(D11, 100, 101, 99, 100, overrides=ov)
    cron = {
        _sp("2024-03-01"): mk_decisao(_sp("2024-03-01"), novas_ordens=[
            mk_ordem("ZZZZ3.SA", _sp("2024-03-04")),
            mk_ordem("AAAA3.SA", _sp("2024-03-04")),
        ]),
    }
    engine, _, orq = _engine({"AAAA3.SA": dfa, "ZZZZ3.SA": dfz}, cron)
    engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    stops_0306 = [t for t, d in orq.fechamentos_notificados if d == _sp("2024-03-06")]
    assert stops_0306 == ["AAAA3.SA", "ZZZZ3.SA"]


def test_config_default_bate_com_documentacao():
    """G35 — BacktestConfig default travado."""
    c = BacktestConfig()
    assert c.capital_inicial == 100_000.0
    assert c.corretagem == 0.003
    assert c.slippage == 0.001
    assert c.stop_pct == 0.08
    assert c.take_pct == 0.15
    assert c.sizing_pct == 0.15
    assert c.custo_perna == pytest.approx(0.004)


# ══════════════════════════════════════════════════════════════════════════════
# Adaptador de fronteira tz
# ══════════════════════════════════════════════════════════════════════════════


def test_adaptador_tz_para_sp_e_idempotente():
    """Adaptador — _para_sp devolve tz-aware SP; idempotente; mesma data de parede."""
    out = _para_sp(_d("2024-03-15"))
    assert out.tzinfo is not None
    assert str(out.tz) == "America/Sao_Paulo"
    assert out.date() == pd.Timestamp("2024-03-15").date()
    assert _para_sp(_para_sp(_d("2024-03-15"))) == _para_sp(_d("2024-03-15"))


def test_adaptador_tz_preserva_hora_de_parede_sem_deslocar():
    """Adaptador — 10:00 naive → 10:00 SP (tz_localize), NÃO 07:00 (tz_convert de
    UTC). Pega o bug clássico tz_localize vs tz_convert."""
    out = _para_sp(pd.Timestamp("2024-03-15 10:00"))
    assert out == pd.Timestamp("2024-03-15 10:00", tz=FUSO)
    assert out.hour == 10
    assert out != pd.Timestamp("2024-03-15 10:00", tz="UTC").tz_convert(FUSO)


def test_engine_nunca_passa_naive_para_agentes():
    """Adaptador — run mínimo (abre + fecha) com Fakes ESTRITOS não levanta
    ValueError: prova que todo call-site do engine usa o adaptador tz."""
    engine, _, _ = _abre_e_fecha_prazo()
    # Fakes levantam ValueError em qualquer data naive; se passar, nenhum vazou.
    res = engine.rodar_backtest(_d("2024-03-01"), _d("2024-03-15"))
    assert res.n_trades == 1


# ── Mapa de cobertura (auditoria do checkpoint) ───────────────────────────────
# R1 fluxo diário: test_abre_e_fecha_por_prazo, test_ordem_das_notificacoes_no_dia
# R2 equity_hoje = MTM fim D-1: test_equity_hoje_reflete_fim_D_menos_1,
#    test_equity_hoje_e_argumento_de_decidir, test_sizing_15pct_do_equity_corrente
# R3 custos 0.4%/perna e P&L: test_custo_entrada/_saida, test_pnl_*
# R4 preços de execução (Open D+1 / stop / take / fim): test_stop/_take/_fechamento_*
# R5 calendário BMF: test_calendario_bmf_pula_*
# R6 sizing: test_sizing_15pct_do_equity_corrente
# R7 capital/estado inicial: test_capital_inicial_sem_ordens
# R8 robustez: test_sem_dados_intraday, test_ticker_sem_dados, test_fechamento_forcado
# R9 determinismo/ordem: test_dois_runs_identicos, test_ordem_alfabetica
# Motivo Literal: stop(B6), take(B7), prazo(A2), reversao(A3), fim_backtest(E25)
