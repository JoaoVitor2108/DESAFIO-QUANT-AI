"""
Testes dos helpers PUROS da calibração do ECON — determinísticos, sem rede.

A orquestração ao vivo (`calibrar`, `teste_placebo`) depende de chave + amostra de
notícias e não é exercida aqui; cobrimos a lógica isolada que sustenta o relatório
de mitigação de viés. Execute: pytest tests/test_econ_calibration.py -v
"""
import pandas as pd
import pytest

from unittest.mock import MagicMock

from config import FUSO
from agents.econ import ScoreEcon
from agents.sources.noticia import Noticia
from calibration.econ_calibration import (
    segmentar_por_exposicao,
    anonimizar_noticias,
    auditar_justificativa,
    calcular_ic,
    calcular_ic_com_ic,
    classificar_degradacao,
    contar_eventos_por_segmento,
    baseline_sentimento_simples,
    diagnosticar_colinearidade,
    _beta_setorial,
    RELIABLE_CUTOFF,
    TRAINING_CUTOFF,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


def _noticia(titulo, conteudo="") -> Noticia:
    return Noticia(titulo=titulo, conteudo=conteudo, url="http://x",
                   publicado_em=ts("2021-05-10"), fonte="valor.globo.com",
                   peso_fonte=0.9, ticker="PETR4.SA")


# ── segmentar_por_exposicao (DEFESA 1) ─────────────────────────────────────────


class TestSegmentar:
    # Só o TRAINING_CUTOFF importa para lookahead: tudo <= jul/2025 estava no treino
    # (mesmo risco de cola), independentemente do reliable cutoff.
    def test_calibracao_2020_2021_e_dentro_treino(self):
        assert segmentar_por_exposicao(ts("2021-06-15")) == "dentro_treino"

    def test_entre_reliable_e_training_ainda_e_dentro_treino(self):
        # fev–jul/2025 estava no treino: NÃO é zona intermediária de risco
        assert segmentar_por_exposicao(ts("2025-04-01")) == "dentro_treino"

    def test_apos_training_cutoff_e_limpo(self):
        assert segmentar_por_exposicao(ts("2025-09-01")) == "limpo"

    def test_fronteira_training_ainda_dentro_treino(self):
        # o próprio instante do training cutoff ainda está no treino
        assert segmentar_por_exposicao(TRAINING_CUTOFF) == "dentro_treino"

    def test_apos_training_cutoff_um_segundo_e_limpo(self):
        assert segmentar_por_exposicao(TRAINING_CUTOFF + pd.Timedelta(seconds=1)) == "limpo"


# ── anonimizar_noticias (DEFESA 2 — placebo) ───────────────────────────────────


class TestAnonimizar:
    def test_remove_nome_do_titulo_e_conteudo(self):
        n = _noticia("Petrobras eleva dividendos", "A Petrobras anunciou hoje...")
        out = anonimizar_noticias([n], "Petrobras")
        assert "Petrobras" not in out[0].titulo
        assert "Petrobras" not in out[0].conteudo

    def test_case_insensitive(self):
        n = _noticia("PETROBRAS e petrobras")
        out = anonimizar_noticias([n], "Petrobras")
        assert "petrobras" not in out[0].titulo.lower()

    def test_nao_muta_original(self):
        n = _noticia("Petrobras sobe")
        anonimizar_noticias([n], "Petrobras")
        assert n.titulo == "Petrobras sobe"  # original intacto

    def test_modo_swap_usa_substituto(self):
        n = _noticia("Petrobras eleva produção")
        out = anonimizar_noticias([n], "Petrobras", modo="swap", substituto="PRIO")
        assert "PRIO" in out[0].titulo
        assert "Petrobras" not in out[0].titulo

    def test_swap_exige_substituto(self):
        n = _noticia("Petrobras")
        with pytest.raises(ValueError):
            anonimizar_noticias([n], "Petrobras", modo="swap")


# ── auditar_justificativa (P8) ─────────────────────────────────────────────────


class TestAuditar:
    def test_justificativa_limpa_sem_flags(self):
        texto = ("O plano de capex eleva a produção esperada e melhora a geração "
                 "de caixa; fundamentos sólidos sustentam o múltiplo.")
        assert auditar_justificativa(texto) == []

    def test_detecta_desfecho_expost(self):
        texto = "A ação caiu 8% nos dias seguintes, confirmando o risco."
        flags = auditar_justificativa(texto)
        assert flags  # não vazio

    def test_detecta_posteriormente(self):
        assert auditar_justificativa("Posteriormente o mercado reagiu.")


# ── calcular_ic ────────────────────────────────────────────────────────────────


class TestIC:
    def test_correlacao_monotonica_positiva(self):
        scores = [-1.0, -0.5, 0.0, 0.5, 1.0]
        retornos = [-0.02, -0.01, 0.0, 0.03, 0.05]  # monotônico crescente
        ic = calcular_ic(scores, retornos)
        assert ic == pytest.approx(1.0, abs=1e-9)

    def test_amostra_insuficiente_retorna_nan(self):
        assert pd.isna(calcular_ic([0.1], [0.2]))


# ── v4-P4: beta setorial sem lookahead ─────────────────────────────────────────


def _journal_beta():
    """Journal falso que registra as janelas pedidas em get_precos/get_retorno_ibovespa."""
    j = MagicMock()
    idx = pd.date_range("2024-01-01", periods=40, freq="B", tz=FUSO)
    j.get_precos.return_value = pd.DataFrame({"Close": range(1, 41)}, index=idx)
    j.get_retorno_ibovespa.return_value = pd.Series(
        [0.01] * 39, index=idx[1:], name="retorno_ibovespa")
    return j


class TestBetaSetorial:
    def test_beta_nao_usa_precos_pos_data_limite(self):
        j = _journal_beta()
        data_limite = ts("2024-03-01 17:00")

        _beta_setorial(j, "PETR4.SA", data_limite, janela_dias=60)

        # toda chamada de preço/ibov para estimar beta termina ATÉ data_limite
        for chamada in j.get_precos.call_args_list:
            fim = chamada.args[2] if len(chamada.args) > 2 else chamada.kwargs.get("data_limite")
            assert fim <= data_limite
        for chamada in j.get_retorno_ibovespa.call_args_list:
            fim = chamada.args[1] if len(chamada.args) > 1 else chamada.kwargs.get("data_limite")
            assert fim <= data_limite

    def test_beta_retorna_float_ou_none(self):
        j = _journal_beta()
        b = _beta_setorial(j, "PETR4.SA", ts("2024-03-01 17:00"))
        assert b is None or isinstance(b, float)


# ── v4-P5: classificação de degradação ─────────────────────────────────────────


def _score(avisos, confianca=0.0, tem_evento=True):
    return ScoreEcon(ticker="PETR4.SA", data_referencia=ts("2024-03-15"),
                     score_total=0.0, comp_noticia=0.0, comp_saude_financeira=0.0,
                     comp_setorial=0.0, comp_macro=0.0, confianca=confianca,
                     tem_evento=tem_evento, n_noticias=1, justificativa="x",
                     modelo="m", avisos=avisos)


class TestClassificarDegradacao:
    def test_sem_chave(self):
        s = _score(["ANTHROPIC_API_KEY ausente ou cliente indisponível; score neutro."])
        assert classificar_degradacao(s) == "sem_chave"

    def test_erro_api(self):
        s = _score(["erro na chamada ao Claude: timeout"])
        assert classificar_degradacao(s) == "erro_api"

    def test_malformada(self):
        s = _score(["resposta do Claude não contém bloco tool_use válido."])
        assert classificar_degradacao(s) == "malformada"

    def test_avaliacao_ok_nao_e_degradacao(self):
        s = _score([], confianca=0.8)
        assert classificar_degradacao(s) is None

    def test_divergencia_nao_e_degradacao(self):
        # divergência score×notícia é aviso, mas a avaliação ocorreu (confiança > 0)
        s = _score(["divergência score_total (+0.90) vs componente_noticia (-0.80) > 0.25"],
                   confianca=0.7)
        assert classificar_degradacao(s) is None


# ── v4-C1: contagem de eventos por segmento ────────────────────────────────────


class TestContarEventos:
    def test_conta_por_segmento(self):
        eventos = [("PETR4.SA", ts("2021-06-15")),   # dentro_treino
                   ("VALE3.SA", ts("2021-07-01")),   # dentro_treino
                   ("PETR4.SA", ts("2025-09-01"))]   # limpo
        out = contar_eventos_por_segmento(eventos)
        assert out["dentro_treino"] == 2
        assert out["limpo"] == 1
        assert out["total"] == 3

    def test_avisa_quando_limpo_pequeno(self):
        eventos = [("PETR4.SA", ts("2025-09-01"))]  # 1 evento limpo < 30
        out = contar_eventos_por_segmento(eventos)
        assert out["aviso_n_limpo"] is True


# ── v4-C2: IC com intervalo de confiança (bootstrap) ───────────────────────────


class TestICBootstrap:
    def test_bootstrap_converge_para_spearman(self):
        scores = [i / 100 for i in range(-50, 50)]
        retornos = [s + 0.001 for s in scores]  # quase perfeito monotônico
        out = calcular_ic_com_ic(scores, retornos, n_bootstrap=200, seed=42)
        assert out["ic"] == pytest.approx(1.0, abs=1e-6)
        assert out["p_valor"] < 0.05
        assert out["ic95_low"] <= out["ic"] <= out["ic95_high"]

    def test_intervalo_maior_em_amostra_pequena(self):
        import random
        random.seed(0)
        scores = [random.gauss(0, 1) for _ in range(8)]
        retornos = [random.gauss(0, 1) for _ in range(8)]
        out = calcular_ic_com_ic(scores, retornos, n_bootstrap=200, seed=1)
        largura = out["ic95_high"] - out["ic95_low"]
        assert largura > 0.3  # amostra pequena → intervalo largo


# ── v4-C3: baseline de sentimento lexical ──────────────────────────────────────


class TestBaseline:
    def test_noticia_positiva_score_positivo(self):
        n = _noticia("Empresa registra lucro recorde e alta de receita")
        assert baseline_sentimento_simples([n]) > 0

    def test_noticia_negativa_score_negativo(self):
        n = _noticia("Empresa tem prejuízo e queda nas vendas; investigação")
        assert baseline_sentimento_simples([n]) < 0

    def test_noticia_neutra_score_zero(self):
        n = _noticia("Empresa realiza assembleia ordinária na data prevista")
        assert baseline_sentimento_simples([n]) == 0.0


# ── v4-C4: diagnóstico de colinearidade implícita ──────────────────────────────


class TestColinearidade:
    def test_correlacao_score_x_fundamento(self):
        # score_total cresce junto com o ROE nos contextos → correlação alta
        scores = [_score([], confianca=0.8) for _ in range(5)]
        for s, v in zip(scores, [-1.0, -0.5, 0.0, 0.5, 1.0]):
            s.score_total = v
        contextos = [{"fundamentos": {"roe": v, "pl": 10.0}}
                     for v in [-1.0, -0.5, 0.0, 0.5, 1.0]]
        out = diagnosticar_colinearidade(scores, contextos)
        assert out["roe"] == pytest.approx(1.0, abs=1e-9)
        assert abs(out["pl"]) < 1e-9 or pd.isna(out["pl"])  # pl constante → ~0/NaN
