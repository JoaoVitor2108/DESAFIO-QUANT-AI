"""
Testes dos helpers PUROS da calibração do ECON — determinísticos, sem rede.

A orquestração ao vivo (`calibrar`, `teste_placebo`) depende de chave + amostra de
notícias e não é exercida aqui; cobrimos a lógica isolada que sustenta o relatório
de mitigação de viés. Execute: pytest tests/test_econ_calibration.py -v
"""
import pandas as pd
import pytest

from config import FUSO
from agents.sources.noticia import Noticia
from calibration.econ_calibration import (
    segmentar_por_exposicao,
    anonimizar_noticias,
    auditar_justificativa,
    calcular_ic,
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
    def test_calibracao_2020_2021_e_teto_otimista(self):
        assert segmentar_por_exposicao(ts("2021-06-15")) == "teto_otimista"

    def test_entre_reliable_e_training_e_intermediario(self):
        assert segmentar_por_exposicao(ts("2025-04-01")) == "intermediario"

    def test_apos_training_cutoff_e_limpo(self):
        assert segmentar_por_exposicao(ts("2025-09-01")) == "limpo"

    def test_fronteira_reliable_inclusiva_no_teto(self):
        # exatamente no reliable cutoff ainda é conhecido pelo modelo
        assert segmentar_por_exposicao(RELIABLE_CUTOFF) == "teto_otimista"

    def test_fronteira_training_ainda_intermediario(self):
        # o próprio instante do training cutoff ainda está no treino
        assert segmentar_por_exposicao(TRAINING_CUTOFF) == "intermediario"

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
