"""Testes da comparação entre versões de prompt (Etapa 2) — sem rede, sem API.

O que importa aqui é a lógica de DECISÃO: resumir uma versão, comparar contra o
baseline e a anterior, e aplicar os critérios de parada. A rodada paga em si é do
`baseline_econ`. Execute: pytest tests/test_etapa2_prompt.py -v
"""
import pandas as pd
import pytest

from config import FUSO
from calibration.etapa2_prompt import (
    DELTA_CONVERGENCIA,
    decidir_parada,
    resumo_versao,
    tabela_comparativa,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


def _df(n: int = 40, ic_sinal: int = 1) -> pd.DataFrame:
    linhas = []
    for i in range(n):
        y = (i % 10) / 100.0 - 0.05
        linhas.append({
            "data": ts("2025-09-01") + pd.Timedelta(days=i),
            "ticker": "PETR4.SA", "y": y, "score": y * ic_sinal,
            "score_lexical": y / 2, "limpo": True, "chamou_api": True,
            "tem_evento": True, "degradacao": None, "justificativa": "x",
            "custo_usd": 0.005, "latencia_llm_s": 4.0,
            "tokens_in": 3000, "tokens_out": 300,
        })
    return pd.DataFrame(linhas)


class TestResumoVersao:
    def test_traz_as_metricas_da_tabela_comparativa(self):
        r = resumo_versao("v2", _df())
        for chave in ("versao", "ic_completo", "ic_limpo", "gap_limpo",
                      "custo", "n", "taxa_degradacao"):
            assert chave in r

    def test_gap_limpo_e_econ_menos_lexical(self):
        r = resumo_versao("v2", _df())
        assert r["gap_limpo"] == pytest.approx(r["ic_limpo"] - r["lexical_limpo"])

    def test_custo_soma_a_coluna(self):
        r = resumo_versao("v2", _df(n=40))
        assert r["custo"] == pytest.approx(40 * 0.005)


class TestTabelaComparativa:
    def test_uma_linha_por_versao_em_ordem(self):
        linhas = tabela_comparativa([resumo_versao("v1", _df()),
                                     resumo_versao("v2", _df(ic_sinal=-1))]).splitlines()
        corpo = [x for x in linhas if x.startswith("| v")]
        assert len(corpo) == 2
        assert corpo[0].startswith("| v1") and corpo[1].startswith("| v2")

    def test_acumula_o_custo(self):
        txt = tabela_comparativa([resumo_versao("v1", _df()), resumo_versao("v2", _df())])
        assert "0.4000" in txt  # 2 x 40 x 0.005 acumulado


class TestDecidirParada:
    """R7: alguns critérios param sozinhos, outros exigem perguntar ao humano."""

    @staticmethod
    def _r(ic_limpo, gap=0.01, custo_acum=5.0):
        return {"ic_limpo": ic_limpo, "gap_limpo": gap, "custo_acumulado": custo_acum}

    def test_sucesso_quando_ic_supera_meta_e_gap_positivo(self):
        d = decidir_parada(self._r(0.18), anteriores=[], iteracao=1)
        assert d["parar"] is True and d["motivo"] == "sucesso"

    def test_nao_para_por_ic_alto_se_gap_negativo(self):
        """IC alto perdendo do léxico não é sucesso — é o léxico carregando."""
        d = decidir_parada(self._r(0.18, gap=-0.01), anteriores=[], iteracao=1)
        assert d["motivo"] != "sucesso"

    def test_piora_vs_baseline_na_iteracao_1_exige_humano(self):
        d = decidir_parada(self._r(-0.02), anteriores=[{"ic_limpo": 0.0137}], iteracao=1)
        assert d["parar"] is True and d["perguntar"] is True

    def test_convergencia_apos_duas_iteracoes_sem_ganho(self):
        hist = [{"ic_limpo": 0.050}, {"ic_limpo": 0.0505}]
        d = decidir_parada(self._r(0.051), anteriores=hist, iteracao=3)
        assert d["parar"] is True and d["motivo"] == "convergencia"

    def test_ganho_acima_do_limiar_nao_converge(self):
        hist = [{"ic_limpo": 0.050}, {"ic_limpo": 0.080}]
        d = decidir_parada(self._r(0.11), anteriores=hist, iteracao=3)
        assert d["parar"] is False

    def test_para_por_orcamento(self):
        d = decidir_parada(self._r(0.05, custo_acum=14.5), anteriores=[], iteracao=2)
        assert d["parar"] is True and d["motivo"] == "orcamento"

    def test_para_no_teto_de_iteracoes(self):
        d = decidir_parada(self._r(0.05), anteriores=[{"ic_limpo": 0.01}] * 4, iteracao=5)
        assert d["parar"] is True and d["motivo"] == "max_iteracoes"

    def test_limiar_de_convergencia_e_o_documentado(self):
        assert DELTA_CONVERGENCIA == 0.005
