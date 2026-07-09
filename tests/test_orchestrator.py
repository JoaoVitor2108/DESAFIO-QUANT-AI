"""Testes do ORQUESTRADOR (JEMPO).

Etapa 1: smoke dos dataclasses do contrato público + record interno.
Etapa 2: fixtures dos Fakes (FakeJournal/FakeMathML/FakeEcon) + construtor,
status() e bookkeeping (notificar_execucao/notificar_fechamento). As etapas
seguintes adicionam os testes de comportamento (drawdown, seleção, timing,
fechamentos, decidir).
"""

from datetime import time

import pandas as pd
import pytest

from agents.econ import ScoreEcon
from agents.orchestrator import (
    DecisaoDia,
    FechamentoOrdem,
    Ordem,
    OrchestratorAgent,
    OrchestratorConfig,
    PosicaoAberta,
)

FUSO = "America/Sao_Paulo"

# Contrato de 8 colunas de MathMLAgent.prever_universo (§3.1 do spec).
_COLS_MATHML = [
    "ticker", "y_pred", "score_econ", "tem_evento", "rank",
    "volume_relativo", "data_noticia_mais_recente", "setor",
]


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


# ── Fixtures dos Fakes (locais ao teste, determinísticas) ─────────────────────


class FakeJournal:
    """Trivial: argumento obrigatório do __init__, nunca chamado pelo
    ORQUESTRADOR. Existe só para satisfazer a assinatura do construtor."""


def _mk_score(score_total: float, data_noticia=None,
              tem_evento: bool = True) -> ScoreEcon:
    """Monta um ScoreEcon válido (12 campos obrigatórios). Só `score_total` e
    `data_noticia_mais_recente` importam para as regras do ORQUESTRADOR."""
    return ScoreEcon(
        ticker="X", data_referencia=_ts("2024-01-01"), score_total=score_total,
        comp_noticia=score_total, comp_saude_financeira=0.0, comp_setorial=0.0,
        comp_macro=0.0, confianca=0.8, tem_evento=tem_evento, n_noticias=1,
        justificativa="fake", modelo="fake", data_noticia_mais_recente=data_noticia,
    )


class FakeEcon:
    """avaliar(ticker, data_limite) → ScoreEcon controlado por ticker."""

    def __init__(self, por_ticker: dict | None = None, default_score: float = 0.0):
        self._por_ticker = por_ticker or {}
        self._default = default_score
        self.chamadas: list[tuple] = []  # rastreio p/ asserts de anti-lookahead

    def avaliar(self, ticker, data_limite):
        self.chamadas.append((ticker, data_limite))
        if ticker in self._por_ticker:
            return self._por_ticker[ticker]
        return _mk_score(self._default)


class FakeMathML:
    """prever_universo(tickers, data_limite) → DataFrame pré-fabricado por data.
    Ignora `tickers` (determinismo) e devolve DataFrame vazio de 8 colunas se a
    data não tiver cenário registrado."""

    def __init__(self, df_por_data: dict | None = None):
        self._df_por_data = df_por_data or {}
        self.chamadas: list[tuple] = []

    def prever_universo(self, tickers, data_limite):
        self.chamadas.append((tuple(tickers), data_limite))
        if data_limite in self._df_por_data:
            return self._df_por_data[data_limite].copy()
        return pd.DataFrame(columns=_COLS_MATHML)


_UNIVERSO_STUB = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "MGLU3.SA"]


def _make_agent(config: OrchestratorConfig | None = None,
                econ: FakeEcon | None = None,
                math_ml: FakeMathML | None = None,
                tickers_ativos=None) -> OrchestratorAgent:
    return OrchestratorAgent(
        journal=FakeJournal(),
        econ=econ or FakeEcon(),
        math_ml=math_ml or FakeMathML(),
        config=config or OrchestratorConfig(),
        tickers_ativos=tickers_ativos or (lambda data: list(_UNIVERSO_STUB)),
    )


# ── Etapa 1: dataclasses ──────────────────────────────────────────────────────


def test_config_defaults_travados():
    c = OrchestratorConfig()
    assert c.score_econ_min == 0.30
    assert c.volume_relativo_min == 1.5
    assert c.max_posicoes == 3
    assert c.max_por_setor == 2
    assert c.sizing_pct == 0.15
    assert c.stop_loss_pct == 0.08
    assert c.take_profit_pct == 0.15
    assert c.prazo_max_dias_uteis == 5
    assert c.score_reversao == -0.30
    assert c.dd_janela_dias_uteis == 21
    assert c.dd_limite == 0.10
    assert c.dd_pausa_dias_uteis == 5
    assert c.hora_corte_b3 == time(17, 5)


def test_config_e_imutavel():
    c = OrchestratorConfig()
    try:
        c.sizing_pct = 0.20  # type: ignore[misc]
    except Exception as e:
        assert e.__class__.__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("OrchestratorConfig deveria ser frozen")


def test_dataclasses_output_instanciam():
    ordem = Ordem(
        ticker="PETR4.SA", setor="Energia", data_decisao=_ts("2024-03-01"),
        data_execucao=_ts("2024-03-04"), rank=1, y_pred=0.02, score_econ=0.5,
        volume_relativo=1.8, sizing_pct=0.15, motivo_execucao_atrasada=None,
    )
    fech = FechamentoOrdem(ticker="PETR4.SA", motivo="prazo",
                           data_gatilho=_ts("2024-03-11"))
    dec = DecisaoDia(
        data=_ts("2024-03-01"), novas_ordens=[ordem], fechamentos=[fech],
        pausado=False, motivo_pausa=None, dd_corrente=0.0,
        posicoes_abertas_snapshot=["PETR4.SA"],
    )
    assert dec.novas_ordens[0].ticker == "PETR4.SA"
    assert dec.fechamentos[0].motivo == "prazo"
    assert dec.pausado is False


def test_posicao_aberta_interna_instancia():
    pos = PosicaoAberta(
        ticker="VALE3.SA", setor="Mineração", preco_entrada=60.0,
        data_execucao=_ts("2024-03-04"), stop_price=55.2, take_price=69.0,
        prazo_max=_ts("2024-03-11"),
    )
    assert pos.stop_price == 55.2
    assert pos.take_price == 69.0


# ── Etapa 2: construtor, status, notificações ─────────────────────────────────


def test_agente_instancia_com_fakes():
    agent = _make_agent()
    assert isinstance(agent, OrchestratorAgent)


def test_status_estado_inicial():
    agent = _make_agent()
    st = agent.status()
    assert st == {
        "n_posicoes_abertas": 0,
        "tickers": [],
        "dd_corrente": 0.0,
        "pausado_ate": None,
        "ultima_data_decidida": None,
        "n_equity_pontos": 0,
    }


def test_notificar_execucao_happy_path_calcula_stop_take_prazo():
    agent = _make_agent()
    agent.notificar_execucao("PETR4.SA", "Energia", 100.0, _ts("2024-03-04"))

    st = agent.status()
    assert st["n_posicoes_abertas"] == 1
    assert st["tickers"] == ["PETR4.SA"]

    pos = agent._posicoes["PETR4.SA"]
    assert pos.preco_entrada == 100.0
    assert pos.setor == "Energia"
    assert pos.stop_price == pytest.approx(92.0)   # 100 × (1 - 0.08)
    assert pos.take_price == pytest.approx(115.0)  # 100 × (1 + 0.15)
    # prazo = data_execucao + 5 dias úteis (04/03 seg → 11/03 seg)
    assert pos.prazo_max == _ts("2024-03-04") + pd.tseries.offsets.BusinessDay(5)
    assert pos.prazo_max == _ts("2024-03-11")


def test_notificar_execucao_dupla_levanta_valueerror():
    agent = _make_agent()
    agent.notificar_execucao("VALE3.SA", "Mineração", 60.0, _ts("2024-03-04"))
    with pytest.raises(ValueError, match="dupla notificação"):
        agent.notificar_execucao("VALE3.SA", "Mineração", 61.0, _ts("2024-03-05"))


def test_notificar_execucao_preco_nao_positivo_levanta_valueerror():
    agent = _make_agent()
    with pytest.raises(ValueError, match="preco_execucao"):
        agent.notificar_execucao("VALE3.SA", "Mineração", 0.0, _ts("2024-03-04"))


def test_notificar_execucao_data_naive_levanta_valueerror():
    agent = _make_agent()
    with pytest.raises(ValueError, match="timezone-aware"):
        agent.notificar_execucao("VALE3.SA", "Mineração", 60.0,
                                 pd.Timestamp("2024-03-04"))


def test_notificar_fechamento_happy_path_remove_posicao():
    agent = _make_agent()
    agent.notificar_execucao("ITUB4.SA", "Financeiro", 30.0, _ts("2024-03-04"))
    agent.notificar_fechamento("ITUB4.SA", _ts("2024-03-11"))
    assert agent.status()["n_posicoes_abertas"] == 0
    assert "ITUB4.SA" not in agent._posicoes


def test_notificar_fechamento_ticker_inexistente_levanta_valueerror():
    agent = _make_agent()
    with pytest.raises(ValueError, match="sem posição aberta"):
        agent.notificar_fechamento("ABEV3.SA", _ts("2024-03-11"))


def test_notificar_fechamento_data_naive_levanta_valueerror():
    agent = _make_agent()
    agent.notificar_execucao("ITUB4.SA", "Financeiro", 30.0, _ts("2024-03-04"))
    with pytest.raises(ValueError, match="timezone-aware"):
        agent.notificar_fechamento("ITUB4.SA", pd.Timestamp("2024-03-11"))


def test_status_reflete_multiplas_posicoes():
    agent = _make_agent()
    agent.notificar_execucao("PETR4.SA", "Energia", 100.0, _ts("2024-03-04"))
    agent.notificar_execucao("VALE3.SA", "Mineração", 60.0, _ts("2024-03-04"))
    st = agent.status()
    assert st["n_posicoes_abertas"] == 2
    assert set(st["tickers"]) == {"PETR4.SA", "VALE3.SA"}
    # status devolve cópia — mutar o retorno não afeta o estado interno
    st["tickers"].append("XXXX")
    assert "XXXX" not in agent.status()["tickers"]


# ── Etapa 3: _atualizar_pausa (drawdown + circuit-breaker) ────────────────────

_BD = pd.tseries.offsets.BusinessDay


def _seed_equity(agent: OrchestratorAgent, valores, data_inicio="2024-01-01"):
    """Popula _equity_series com `valores` em dias úteis consecutivos.
    Retorna a próxima data útil (para a chamada de _atualizar_pausa sob teste)."""
    datas = pd.bdate_range(data_inicio, periods=len(valores), tz=FUSO)
    agent._equity_series = [(d, float(v)) for d, v in zip(datas, valores)]
    return datas[-1] + _BD(1)


def test_drawdown_janela_incompleta_circuit_breaker_inativo():  # caso 15
    agent = _make_agent()
    prox = _seed_equity(agent, [100.0] * 19)  # 19 pontos; após append → 20 (<21)
    dd = agent._atualizar_pausa(prox, 50.0)   # queda enorme, mas janela incompleta
    assert dd == 0.0
    assert agent._dd_corrente == 0.0
    assert agent._pausado_ate is None
    assert agent.status()["n_equity_pontos"] == 20


def test_drawdown_exatamente_10pct_nao_pausa():  # caso 16
    agent = _make_agent()
    prox = _seed_equity(agent, [100.0] * 20)  # após append → 21 (janela completa)
    dd = agent._atualizar_pausa(prox, 90.0)   # dd = (100-90)/100 = 0.10 (não > 0.10)
    assert dd == pytest.approx(0.10)
    assert agent._pausado_ate is None


def test_drawdown_10_01pct_pausa_5du():  # caso 17
    agent = _make_agent()
    prox = _seed_equity(agent, [100.0] * 20)
    dd = agent._atualizar_pausa(prox, 89.99)  # dd = 0.1001 > 0.10 → pausa
    assert dd == pytest.approx(0.1001)
    assert agent._pausado_ate == prox + _BD(5)


def test_pausa_ativa_impede_novo_trigger():  # caso 18
    agent = _make_agent()
    prox = _seed_equity(agent, [100.0] * 20)
    agent._atualizar_pausa(prox, 85.0)        # dd = 0.15 → pausa em prox+5du
    pausa_original = agent._pausado_ate
    # dia seguinte, ainda dentro da pausa, com dd ainda maior → não redispara
    d2 = prox + _BD(1)
    agent._atualizar_pausa(d2, 80.0)
    assert agent._pausado_ate == pausa_original  # inalterada


def test_fim_da_pausa_permite_novo_trigger():  # caso 19
    agent = _make_agent()
    prox = _seed_equity(agent, [100.0] * 20)
    # pausa anterior JÁ expirada (data >= _pausado_ate) + novo dd → redispara
    agent._pausado_ate = prox - _BD(1)
    agent._atualizar_pausa(prox, 80.0)        # prox >= _pausado_ate, dd = 0.20
    assert agent._pausado_ate == prox + _BD(5)  # pausa fresca


def test_novo_drawdown_durante_pausa_nao_estende():  # caso 20
    agent = _make_agent()
    prox = _seed_equity(agent, [100.0] * 20)
    agent._atualizar_pausa(prox, 85.0)        # dd = 0.15 → pausa P = prox+5du
    P = agent._pausado_ate
    assert P == prox + _BD(5)
    d2 = prox + _BD(1)                        # d2 < P (dentro da pausa)
    agent._atualizar_pausa(d2, 70.0)          # dd = 0.30, maior ainda
    assert agent._pausado_ate == P            # não empurra a data de fim


def test_dd_corrente_no_pico_e_zero():
    agent = _make_agent()
    prox = _seed_equity(agent, [100.0] * 20)
    dd = agent._atualizar_pausa(prox, 100.0)  # equity no pico → dd = 0.0 (sem pausa)
    assert dd == 0.0
    assert agent._pausado_ate is None


# ── Etapa 4: _selecionar_ordens (pool + top-N + limite setorial) ──────────────


def _row(ticker, setor, rank, score_econ=0.9, volume_relativo=2.0,
         data_noticia=pd.NaT):
    """Uma linha do contrato de 8 colunas do prever_universo. Defaults passam
    nos filtros (score 0.9 > 0.30; volume 2.0 > 1.5)."""
    return {
        "ticker": ticker, "y_pred": -0.01 * rank, "score_econ": score_econ,
        "tem_evento": True, "rank": rank, "volume_relativo": volume_relativo,
        "data_noticia_mais_recente": data_noticia, "setor": setor,
    }


def _df(rows):
    return pd.DataFrame(rows, columns=_COLS_MATHML)


def test_selecao_pool_vazio_zero_ordens():  # caso 1
    agent = _make_agent()
    # df vazio
    assert agent._selecionar_ordens(_df([]), _ts("2024-03-01")) == []
    # df com linhas que reprovam nos filtros (score e volume baixos)
    df = _df([_row("AAAA.SA", "Energia", 1, score_econ=0.1, volume_relativo=1.0)])
    assert agent._selecionar_ordens(df, _ts("2024-03-01")) == []


def test_selecao_uma_candidata_uma_ordem_sizing_15pct():  # caso 2
    agent = _make_agent()
    df = _df([_row("PETR4.SA", "Energia", 1)])
    ordens = agent._selecionar_ordens(df, _ts("2024-03-01"))
    assert len(ordens) == 1
    o = ordens[0]
    assert o.ticker == "PETR4.SA"
    assert o.setor == "Energia"
    assert o.rank == 1
    assert o.sizing_pct == 0.15
    assert o.data_decisao == _ts("2024-03-01")


def test_selecao_tres_setores_distintos_tres_ordens():  # caso 3
    agent = _make_agent()
    df = _df([
        _row("PETR4.SA", "Energia", 1),
        _row("VALE3.SA", "Mineração", 2),
        _row("ITUB4.SA", "Financeiro", 3),
    ])
    ordens = agent._selecionar_ordens(df, _ts("2024-03-01"))
    assert [o.ticker for o in ordens] == ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]


def test_selecao_top3_mesmo_setor_pega_apenas_dois():  # caso 4
    agent = _make_agent()
    df = _df([
        _row("BBAS3.SA", "Financeiro", 1),
        _row("ITUB4.SA", "Financeiro", 2),
        _row("BBDC4.SA", "Financeiro", 3),
        _row("SANB11.SA", "Financeiro", 4),
    ])
    ordens = agent._selecionar_ordens(df, _ts("2024-03-01"))
    assert [o.rank for o in ordens] == [1, 2]  # rank 3 e 4 pulados (setor saturado)


def test_selecao_mesmo_setor_mais_rank4_outro_setor():  # caso 5
    agent = _make_agent()
    df = _df([
        _row("BBAS3.SA", "Financeiro", 1),
        _row("ITUB4.SA", "Financeiro", 2),
        _row("BBDC4.SA", "Financeiro", 3),   # pulado: Financeiro saturado
        _row("VALE3.SA", "Mineração", 4),    # entra: outro setor
    ])
    ordens = agent._selecionar_ordens(df, _ts("2024-03-01"))
    assert [o.rank for o in ordens] == [1, 2, 4]
    assert ordens[2].ticker == "VALE3.SA"


def test_selecao_duas_posicoes_abertas_um_slot():  # caso 6
    agent = _make_agent()
    agent.notificar_execucao("PETR4.SA", "Energia", 100.0, _ts("2024-02-28"))
    agent.notificar_execucao("ITUB4.SA", "Financeiro", 30.0, _ts("2024-02-28"))
    df = _df([
        _row("VALE3.SA", "Mineração", 1),   # setor livre → deve entrar
        _row("MGLU3.SA", "Varejo", 2),      # não entra: só 1 slot livre
    ])
    ordens = agent._selecionar_ordens(df, _ts("2024-03-01"))
    assert len(ordens) == 1
    assert ordens[0].ticker == "VALE3.SA"


def test_selecao_tres_posicoes_abertas_zero_ordens():  # caso 7
    agent = _make_agent()
    agent.notificar_execucao("PETR4.SA", "Energia", 100.0, _ts("2024-02-28"))
    agent.notificar_execucao("VALE3.SA", "Mineração", 60.0, _ts("2024-02-28"))
    agent.notificar_execucao("ITUB4.SA", "Financeiro", 30.0, _ts("2024-02-28"))
    df = _df([_row("MGLU3.SA", "Varejo", 1)])
    assert agent._selecionar_ordens(df, _ts("2024-03-01")) == []


def test_selecao_volume_relativo_nan_reprova():  # caso 13
    agent = _make_agent()
    df = _df([
        _row("AAAA.SA", "Energia", 1, volume_relativo=float("nan")),  # reprovado
        _row("BBBB.SA", "Mineração", 2),                              # aprovado
    ])
    ordens = agent._selecionar_ordens(df, _ts("2024-03-01"))
    assert [o.ticker for o in ordens] == ["BBBB.SA"]


def test_selecao_score_econ_nan_reprova():  # caso 14
    agent = _make_agent()
    df = _df([
        _row("AAAA.SA", "Energia", 1, score_econ=float("nan")),  # reprovado
        _row("BBBB.SA", "Mineração", 2),                         # aprovado
    ])
    ordens = agent._selecionar_ordens(df, _ts("2024-03-01"))
    assert [o.ticker for o in ordens] == ["BBBB.SA"]


# ── Etapa 5: _resolver_data_execucao (D+1 vs D+2, corte 17h05 B3) ─────────────


def _resolver(agent, data, data_noticia):
    row = pd.Series(_row("X.SA", "Energia", 1, data_noticia=data_noticia))
    return agent._resolver_data_execucao(row, data)


def test_execucao_noticia_pos_17h05_vira_d2():  # caso 8
    agent = _make_agent()
    d = _ts("2024-03-05")                       # terça; D-1 = 04/03 (segunda)
    data_exec, motivo = _resolver(agent, d, _ts("2024-03-04 18:00"))
    assert data_exec == _ts("2024-03-07")       # D+2 = quinta
    assert motivo == "noticia_pos_17h"


def test_execucao_noticia_pre_17h05_vira_d1():  # caso 9
    agent = _make_agent()
    d = _ts("2024-03-05")
    data_exec, motivo = _resolver(agent, d, _ts("2024-03-04 15:00"))
    assert data_exec == _ts("2024-03-06")       # D+1 = quarta
    assert motivo is None


def test_execucao_noticia_nat_vira_d1():  # caso 10
    agent = _make_agent()
    d = _ts("2024-03-05")
    data_exec, motivo = _resolver(agent, d, pd.NaT)
    assert data_exec == _ts("2024-03-06")
    assert motivo is None


def test_execucao_sexta_pre_17h05_vira_segunda():  # caso 11
    agent = _make_agent()
    d = _ts("2024-03-01")                       # sexta; D-1 = 29/02 (quinta)
    data_exec, motivo = _resolver(agent, d, _ts("2024-02-29 15:00"))
    assert data_exec == _ts("2024-03-04")       # próximo dia útil = segunda
    assert motivo is None


def test_execucao_sexta_pos_17h05_vira_terca():  # caso 12
    agent = _make_agent()
    d = _ts("2024-03-01")                       # sexta
    data_exec, motivo = _resolver(agent, d, _ts("2024-02-29 18:00"))
    assert data_exec == _ts("2024-03-05")       # D+2 = terça
    assert motivo == "noticia_pos_17h"


def test_execucao_noticia_exatamente_17h05_vira_d1():  # fronteira: <= corte
    agent = _make_agent()
    d = _ts("2024-03-05")
    data_exec, motivo = _resolver(agent, d, _ts("2024-03-04 17:05"))
    assert data_exec == _ts("2024-03-06")       # exatamente no corte → D+1
    assert motivo is None


def test_execucao_normaliza_hora_da_decisao_e_preserva_tz():
    agent = _make_agent()
    d = _ts("2024-03-05 10:00")                 # decisão às 10h
    data_exec, motivo = _resolver(agent, d, pd.NaT)
    assert data_exec == _ts("2024-03-06")       # normalizado p/ midnight
    assert data_exec.hour == 0
    assert str(data_exec.tz) == "America/Sao_Paulo"  # sem round-trip p/ UTC


# ── Etapa 6: _verificar_fechamentos (prazo + reversão via ECON) ───────────────


def test_fechamento_reversao_prazo_nao_vencido():  # caso 21
    econ = FakeEcon(por_ticker={"PETR4.SA": _mk_score(-0.5)})
    agent = _make_agent(econ=econ)
    agent.notificar_execucao("PETR4.SA", "Energia", 100.0, _ts("2024-03-04"))
    # prazo_max = 04/03 + 5du = 11/03; decidimos em 06/03 (prazo NÃO vencido)
    fechs = agent._verificar_fechamentos(_ts("2024-03-06"))
    assert len(fechs) == 1
    assert fechs[0].ticker == "PETR4.SA"
    assert fechs[0].motivo == "reversao"
    assert fechs[0].data_gatilho == _ts("2024-03-06")


def test_fechamento_prazo_prevalece_sobre_reversao():  # caso 22
    # ECON também reverteria, mas o prazo venceu → ECON não deve nem ser chamado
    econ = FakeEcon(por_ticker={"PETR4.SA": _mk_score(-0.5)})
    agent = _make_agent(econ=econ)
    agent.notificar_execucao("PETR4.SA", "Energia", 100.0, _ts("2024-03-04"))
    fechs = agent._verificar_fechamentos(_ts("2024-03-11"))  # data == prazo_max
    assert len(fechs) == 1
    assert fechs[0].motivo == "prazo"
    assert econ.chamadas == []  # ECON não chamado (prazo tem prioridade)


def test_fechamento_score_exato_menos_030_nao_fecha():  # fronteira estrita
    econ = FakeEcon(por_ticker={"VALE3.SA": _mk_score(-0.30)})
    agent = _make_agent(econ=econ)
    agent.notificar_execucao("VALE3.SA", "Mineração", 60.0, _ts("2024-03-04"))
    fechs = agent._verificar_fechamentos(_ts("2024-03-06"))
    assert fechs == []  # -0.30 não é < -0.30


def test_fechamento_score_menos_029_nao_fecha():
    econ = FakeEcon(por_ticker={"VALE3.SA": _mk_score(-0.29)})
    agent = _make_agent(econ=econ)
    agent.notificar_execucao("VALE3.SA", "Mineração", 60.0, _ts("2024-03-04"))
    assert agent._verificar_fechamentos(_ts("2024-03-06")) == []


def test_fechamento_sem_posicoes_retorna_vazio_sem_chamar_econ():
    econ = FakeEcon(default_score=-0.9)
    agent = _make_agent(econ=econ)
    assert agent._verificar_fechamentos(_ts("2024-03-06")) == []
    assert econ.chamadas == []


def test_fechamento_uma_por_prazo_outra_por_reversao():
    econ = FakeEcon(por_ticker={"B.SA": _mk_score(-0.5)}, default_score=0.0)
    agent = _make_agent(econ=econ)
    # A: prazo_max 11/03 (vence em 11/03); B: prazo_max 15/03 (não vence)
    agent.notificar_execucao("A.SA", "Energia", 100.0, _ts("2024-03-04"))
    agent.notificar_execucao("B.SA", "Mineração", 50.0, _ts("2024-03-08"))
    fechs = agent._verificar_fechamentos(_ts("2024-03-11"))
    motivos = {f.ticker: f.motivo for f in fechs}
    assert motivos == {"A.SA": "prazo", "B.SA": "reversao"}
    # A saiu por prazo → ECON só foi chamado para B
    assert [t for t, _ in econ.chamadas] == ["B.SA"]


def test_fechamento_anti_lookahead_data_limite_igual_data():
    econ = FakeEcon(default_score=0.0)
    agent = _make_agent(econ=econ)
    agent.notificar_execucao("PETR4.SA", "Energia", 100.0, _ts("2024-03-04"))
    agent.notificar_execucao("VALE3.SA", "Mineração", 60.0, _ts("2024-03-04"))
    data = _ts("2024-03-06")
    agent._verificar_fechamentos(data)
    # toda chamada ao ECON usa data_limite == data (kwarg explícito, §10)
    assert len(econ.chamadas) == 2
    assert all(dl == data for _, dl in econ.chamadas)


# ── Etapa 7: decidir() + validações + integração ──────────────────────────────


def test_tickers_ativos_default_aponta_para_config():  # ponto 3
    import config
    agent = OrchestratorAgent(FakeJournal(), FakeEcon(), FakeMathML(),
                              OrchestratorConfig())
    assert agent._tickers_ativos is config.tickers_ativos


def test_decidir_happy_path_gera_ordens():
    data = _ts("2024-03-05")
    df = _df([_row("PETR4.SA", "Energia", 1), _row("VALE3.SA", "Mineração", 2)])
    agent = _make_agent(math_ml=FakeMathML({data: df}))
    dec = agent.decidir(data, 100_000.0)
    assert dec.pausado is False
    assert dec.motivo_pausa is None
    assert [o.ticker for o in dec.novas_ordens] == ["PETR4.SA", "VALE3.SA"]
    assert all(o.sizing_pct == 0.15 for o in dec.novas_ordens)
    assert dec.fechamentos == []
    assert dec.dd_corrente == 0.0
    assert dec.posicoes_abertas_snapshot == []  # nada aberto ainda


def test_decidir_universo_vazio_sem_ordens_sem_erro():  # ponto 4
    data = _ts("2024-03-05")
    agent = _make_agent(math_ml=FakeMathML({}),           # df sempre vazio
                        tickers_ativos=lambda d: [])       # universo vazio
    dec = agent.decidir(data, 100_000.0)
    assert dec.novas_ordens == []
    assert dec.pausado is False


def test_decidir_duas_vezes_mesma_data_valueerror():  # caso 25
    data = _ts("2024-03-05")
    agent = _make_agent(math_ml=FakeMathML({data: _df([])}))
    agent.decidir(data, 100_000.0)
    with pytest.raises(ValueError, match="posterior"):
        agent.decidir(data, 100_000.0)


def test_decidir_data_retroativa_valueerror():
    agent = _make_agent()
    agent.decidir(_ts("2024-03-05"), 100_000.0)
    with pytest.raises(ValueError, match="posterior"):
        agent.decidir(_ts("2024-03-04"), 100_000.0)


def test_decidir_equity_nao_positivo_valueerror():  # caso 26
    agent = _make_agent()
    with pytest.raises(ValueError, match="equity_hoje"):
        agent.decidir(_ts("2024-03-05"), 0.0)
    with pytest.raises(ValueError, match="equity_hoje"):
        agent.decidir(_ts("2024-03-05"), -10.0)


def test_decidir_data_naive_valueerror():  # caso 27
    agent = _make_agent()
    with pytest.raises(ValueError, match="timezone-aware"):
        agent.decidir(pd.Timestamp("2024-03-05"), 100_000.0)


def test_decidir_marca_data_mesmo_se_passo_posterior_levanta():  # ponto 2
    class MathMLQueLevanta:
        def prever_universo(self, tickers, data_limite):
            raise RuntimeError("falha simulada no MATH&ML")

    agent = _make_agent(math_ml=MathMLQueLevanta())
    data = _ts("2024-03-05")
    with pytest.raises(RuntimeError):
        agent.decidir(data, 100_000.0)
    # a chamada aconteceu → _ultima_data_decidida atualizou antes do erro
    assert agent._ultima_data_decidida == data
    with pytest.raises(ValueError, match="posterior"):
        agent.decidir(data, 100_000.0)


def test_decidir_determinismo_tres_execucoes():  # caso 28
    data = _ts("2024-03-05")
    df = _df([
        _row("PETR4.SA", "Energia", 1),
        _row("VALE3.SA", "Mineração", 2),
        _row("ITUB4.SA", "Financeiro", 3),
    ])

    def run():
        agent = _make_agent(math_ml=FakeMathML({data: df.copy()}),
                            econ=FakeEcon(default_score=0.0))
        return agent.decidir(data, 100_000.0)

    d1, d2, d3 = run(), run(), run()
    assert d1 == d2 == d3


def test_decidir_anti_lookahead_data_limite_igual_data():  # integração anti-LA
    data = _ts("2024-03-05")
    df = _df([_row("PETR4.SA", "Energia", 1)])
    econ = FakeEcon(default_score=0.0)
    mm = FakeMathML({data: df})
    agent = _make_agent(econ=econ, math_ml=mm)
    # pré-abre posição p/ o ECON ser exercitado no _verificar_fechamentos
    agent.notificar_execucao("XPTO.SA", "Energia", 50.0, _ts("2024-03-04"))
    agent.decidir(data, 100_000.0)
    assert mm.chamadas and econ.chamadas             # ambos exercitados
    assert all(dl == data for _, dl in mm.chamadas)  # MATH&ML: data_limite == data
    assert all(dl == data for _, dl in econ.chamadas)  # ECON: idem


def _executar(agent, dec, precos):
    """Simula o loop do PROGRAM: fecha o que decidiu fechar, abre as novas."""
    for f in dec.fechamentos:
        agent.notificar_fechamento(f.ticker, dec.data)
    for o in dec.novas_ordens:
        agent.notificar_execucao(o.ticker, o.setor, precos[o.ticker],
                                 o.data_execucao)


def test_integracao_fluxo_completo_prazo_e_reversao():
    # PETR4 entra 04/03 (exec 05/03, prazo 12/03); VALE3 entra 05/03
    # (exec 06/03) e reverte em 06/03. PETR4 fecha por prazo em 12/03.
    dias = pd.bdate_range("2024-03-04", periods=7, tz=FUSO)  # 04..12/03 (úteis)
    precos = {"PETR4.SA": 100.0, "VALE3.SA": 60.0}
    econ = FakeEcon(por_ticker={"VALE3.SA": _mk_score(-0.5)}, default_score=0.0)
    cenarios = {
        dias[0]: _df([_row("PETR4.SA", "Energia", 1)]),      # 04/03
        dias[1]: _df([_row("VALE3.SA", "Mineração", 1)]),    # 05/03
    }
    agent = _make_agent(econ=econ, math_ml=FakeMathML(cenarios))

    resultados = {}
    for d in dias:
        dec = agent.decidir(d, 100_000.0)
        resultados[d] = dec
        _executar(agent, dec, precos)

    # 04/03: entra PETR4
    assert [o.ticker for o in resultados[dias[0]].novas_ordens] == ["PETR4.SA"]
    assert agent.status()["n_equity_pontos"] == 7

    # 05/03: entra VALE3 (PETR4 continua aberta)
    assert [o.ticker for o in resultados[dias[1]].novas_ordens] == ["VALE3.SA"]

    # 06/03: VALE3 reverte (prazo não vencido) → fechamento reversao
    fech_06 = resultados[dias[2]].fechamentos
    assert [(f.ticker, f.motivo) for f in fech_06] == [("VALE3.SA", "reversao")]
    # snapshot no retorno reflete estado ANTES da notificação do PROGRAM
    assert set(resultados[dias[2]].posicoes_abertas_snapshot) == {"PETR4.SA",
                                                                  "VALE3.SA"}

    # 12/03: PETR4 fecha por prazo
    fech_12 = resultados[dias[6]].fechamentos
    assert [(f.ticker, f.motivo) for f in fech_12] == [("PETR4.SA", "prazo")]

    # fim: sem posições abertas
    assert agent.status()["n_posicoes_abertas"] == 0


def test_integracao_circuit_breaker_ponta_a_ponta():
    dias = pd.bdate_range("2024-01-01", periods=28, tz=FUSO)
    # 21 dias flat, queda no 22º (idx 21), recuperação a partir do idx 26
    equities = ([100_000.0] * 21) + ([85_000.0] * 5) + ([100_000.0] * 2)
    candidato = _df([_row("PETR4.SA", "Energia", 1)])
    # só o dia de "volta a operar" (idx 26) tem candidato
    agent = _make_agent(math_ml=FakeMathML({dias[26]: candidato}))

    resultados = {}
    for d, eq in zip(dias, equities):
        resultados[d] = agent.decidir(d, eq)

    # dia da queda (idx 21): já pausado, dd ~15%
    assert resultados[dias[21]].pausado is True
    assert resultados[dias[21]].motivo_pausa == "drawdown"
    assert resultados[dias[21]].dd_corrente == pytest.approx(0.15)
    # durante a pausa (idx 22..25): pausado, sem novas ordens
    for i in range(22, 26):
        assert resultados[dias[i]].pausado is True
        assert resultados[dias[i]].novas_ordens == []
    # fim da pausa (idx 26, equity recuperada): volta a operar e gera ordem
    assert resultados[dias[26]].pausado is False
    assert [o.ticker for o in resultados[dias[26]].novas_ordens] == ["PETR4.SA"]
    # durante a pausa, MATH&ML NÃO foi consultado (retorno antecipado do §7 p5)
    datas_consultadas = {dl for _, dl in agent._math_ml.chamadas}
    assert all(dias[i] not in datas_consultadas for i in range(21, 26))
