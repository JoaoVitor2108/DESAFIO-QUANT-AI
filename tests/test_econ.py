"""
Testes do EconAgent — determinísticos, sem rede.

O cliente Anthropic é sempre injetado (mock) ou ausente; nenhuma chamada real
é feita aqui. Execute: pytest tests/test_econ.py -v
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config import FUSO
from agents.journal import Fundamentals
from agents.sources.noticia import Noticia
from agents.sources.bloomberg_earnings import EarningsBloomberg
from agents import econ as econ_mod
from agents.econ import EconAgent, ScoreEcon


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _noticia(titulo="Petrobras anuncia novo plano de investimentos") -> Noticia:
    return Noticia(
        titulo=titulo,
        conteudo="Conteúdo da notícia para teste.",
        url="https://valor.globo.com/empresa",
        publicado_em=ts("2024-03-10 09:00"),
        fonte="valor.globo.com",
        peso_fonte=0.9,
        ticker="PETR4.SA",
    )


def _journal_mock(noticias):
    """JournalAgent falso: devolve dados fixos para os coletores do ECON."""
    j = MagicMock()
    j.get_noticias.return_value = noticias
    j.get_setor.return_value = "Petróleo, Gás e Biocombustíveis"
    j.get_fundamentals.return_value = Fundamentals(
        ticker="PETR4.SA",
        data_referencia=ts("2024-03-15"),
        pl=4.5, pvp=1.1, roe=0.30, margem_liquida=0.18,
        divida_liquida_ebitda=0.8, receita=600e9, lucro_liquido=100e9,
        setor="Petróleo, Gás e Biocombustíveis",
    )
    j.get_macro.return_value = {
        "selic_meta": pd.Series([11.25], index=[ts("2024-03-01")]),
        "ipca_12m": pd.Series([4.5], index=[ts("2024-02-01")]),
        "ptax_usdbrl": pd.Series([4.95], index=[ts("2024-03-14")]),
    }
    j.get_retornos_setor.return_value = {
        "retorno_medio": 0.03, "retorno_mediano": 0.025,
        "n_tickers": 4, "tickers": ["PETR4.SA"], "setor": "Petróleo, Gás e Biocombustíveis",
    }
    return j


def _client_mock(input_dict):
    """Cliente Anthropic falso: messages.create devolve um bloco tool_use fixo."""
    bloco = SimpleNamespace(type="tool_use", name="registrar_avaliacao", input=input_dict)
    resposta = SimpleNamespace(content=[bloco], stop_reason="tool_use")
    client = MagicMock()
    client.messages.create.return_value = resposta
    return client


_TOOL_OK = {
    "score_total": 0.6,
    "componente_noticia": 0.7,
    "componente_saude_financeira": 0.5,
    "componente_setorial": 0.3,
    "componente_macro": -0.1,
    "confianca": 0.8,
    "justificativa": "Plano de capex eleva produção futura; fundamentos sólidos.",
}


# ── Testes ────────────────────────────────────────────────────────────────────


def test_sem_noticia_nao_chama_claude(tmp_path):
    client = _client_mock(_TOOL_OK)
    agent = EconAgent(journal=_journal_mock([]), client=client, cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert isinstance(score, ScoreEcon)
    assert score.tem_evento is False
    assert score.n_noticias == 0
    assert score.score_total == 0.0
    assert score.comp_noticia == 0.0
    assert score.comp_saude_financeira == 0.0
    assert score.comp_setorial == 0.0
    assert score.comp_macro == 0.0
    assert score.confianca == 0.0
    client.messages.create.assert_not_called()


def test_parse_tool_use(tmp_path):
    client = _client_mock(_TOOL_OK)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=client, cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.tem_evento is True
    assert score.n_noticias == 1
    assert score.score_total == 0.6
    assert score.comp_noticia == 0.7
    assert score.comp_saude_financeira == 0.5
    assert score.comp_setorial == 0.3
    assert score.comp_macro == -0.1
    assert score.confianca == 0.8
    assert "capex" in score.justificativa.lower() or score.justificativa
    assert score.modelo == agent.model
    client.messages.create.assert_called_once()


def test_score_clamp(tmp_path):
    fora = dict(_TOOL_OK, score_total=1.5, componente_noticia=-2.0,
                componente_macro=9.9, confianca=1.3)
    client = _client_mock(fora)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=client, cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.score_total == 1.0
    assert score.comp_noticia == -1.0
    assert score.comp_macro == 1.0
    assert 0.0 <= score.confianca <= 1.0
    assert score.confianca == 1.0


def test_degrada_sem_chave(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # client=None força a criação do cliente real, que falha por falta de chave
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=None, cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.score_total == 0.0
    assert score.confianca == 0.0
    assert any("chave" in a.lower() or "key" in a.lower() for a in score.avisos)


def test_erro_api_degrada(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("API fora do ar")
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=client, cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.score_total == 0.0
    assert score.confianca == 0.0
    assert any(a for a in score.avisos)


def test_cache_evita_segunda_chamada(tmp_path):
    client = _client_mock(_TOOL_OK)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=client, cache_dir=tmp_path)

    s1 = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))
    s2 = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert s1.score_total == s2.score_total
    client.messages.create.assert_called_once()


def test_data_limite_naive_levanta(tmp_path):
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)
    with pytest.raises(ValueError):
        agent.avaliar("PETR4.SA", pd.Timestamp("2024-03-15 17:00"))  # naive


def test_resposta_malformada_nao_cacheia(tmp_path):
    # Resposta sem bloco tool_use → neutro degradado; NÃO pode ser cacheado,
    # senão uma falha transitória do modelo contaminaria 24h de chamadas.
    sem_tool = SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")],
                               stop_reason="end_turn")
    client = MagicMock()
    client.messages.create.return_value = sem_tool
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=client, cache_dir=tmp_path)

    s1 = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))
    s2 = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert s1.score_total == 0.0
    assert s1.confianca == 0.0
    assert any(a for a in s1.avisos)
    assert client.messages.create.call_count == 2  # 2ª tentativa, não cache hit


def test_clamp_retorna_float(tmp_path):
    fora = dict(_TOOL_OK, score_total=1.5, componente_noticia=-2.0)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(fora),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert isinstance(score.score_total, float)
    assert isinstance(score.comp_noticia, float)
    assert isinstance(score.confianca, float)


def test_contexto_sem_nan(tmp_path):
    journal = _journal_mock([_noticia()])
    journal.get_macro.return_value = {
        "selic_meta": pd.Series([11.25], index=[ts("2024-03-01")]),
        "ptax_usdbrl": pd.Series([float("nan")], index=[ts("2024-03-14")]),  # valor faltante
    }
    agent = EconAgent(journal=journal, client=_client_mock(_TOOL_OK), cache_dir=tmp_path)

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "NaN" not in contexto


# ── P4: versão do prompt na chave de cache ─────────────────────────────────────


def test_cache_invalida_ao_mudar_prompt_version(tmp_path, monkeypatch):
    client = _client_mock(_TOOL_OK)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=client, cache_dir=tmp_path)

    monkeypatch.setattr(econ_mod, "_PROMPT_VERSION", "v1")
    agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))
    monkeypatch.setattr(econ_mod, "_PROMPT_VERSION", "v2")  # prompt mudou
    agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    # Versões diferentes do prompt não podem compartilhar cache.
    assert client.messages.create.call_count == 2


# ── P7: coerência score_total × comp_noticia (Opção A) ─────────────────────────


def test_divergencia_score_noticia_gera_aviso(tmp_path):
    fora = dict(_TOOL_OK, score_total=0.9, componente_noticia=-0.8)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(fora),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert any("diverg" in a.lower() for a in score.avisos)


def test_score_coerente_nao_gera_aviso_divergencia(tmp_path):
    coerente = dict(_TOOL_OK, score_total=0.6, componente_noticia=0.7)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(coerente),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert not any("diverg" in a.lower() for a in score.avisos)


# ── P2: hooks de calibração (override) ─────────────────────────────────────────


def test_noticias_override_pula_journal(tmp_path):
    journal = _journal_mock([])  # journal não tem notícia
    client = _client_mock(_TOOL_OK)
    agent = EconAgent(journal=journal, client=client, cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"),
                          noticias_override=[_noticia()])

    journal.get_noticias.assert_not_called()
    assert score.tem_evento is True
    assert score.n_noticias == 1
    client.messages.create.assert_called_once()


def test_nome_override_no_contexto(tmp_path):
    # _montar_contexto só troca o campo da empresa; anonimizar o TEXTO da notícia
    # é responsabilidade de anonimizar_noticias (camada de calibração). Por isso
    # usamos uma notícia de título neutro (sem o nome real embutido).
    neutra = _noticia(titulo="Empresa anuncia novo plano de investimentos")
    agent = EconAgent(journal=_journal_mock([neutra]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [neutra], [],
                                      nome_override="EMPRESA_ANON")

    assert "EMPRESA_ANON" in contexto
    assert "Petrobras" not in contexto


# ── P2 (v3): identidade_pura — omitir contexto fundamental ─────────────────────


def test_sem_contexto_fundamental_omite_fundamentos(tmp_path):
    journal = _journal_mock([_noticia()])
    agent = EconAgent(journal=journal, client=_client_mock(_TOOL_OK), cache_dir=tmp_path)

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [],
                                      incluir_contexto_fundamental=False)

    assert "fundamentos" not in contexto
    assert "macro" not in contexto
    assert "retornos_setor" not in contexto
    journal.get_fundamentals.assert_not_called()
    journal.get_macro.assert_not_called()
    journal.get_retornos_setor.assert_not_called()


def test_identidade_pura_omite_ticker(tmp_path):
    # identidade_pura deve esconder a IDENTIDADE: o símbolo do ticker (PETR4.SA)
    # entregaria a empresa que o placebo quer ocultar.
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)
    contexto = agent._montar_contexto(
        "PETR4.SA", ts("2024-03-15 17:00"),
        [_noticia(titulo="Empresa anuncia plano")], [],
        nome_override="a companhia", incluir_contexto_fundamental=False)
    assert "PETR4" not in contexto
    assert '"ticker"' not in contexto


def test_com_contexto_fundamental_inclui_fundamentos(tmp_path):
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "fundamentos" in contexto and "macro" in contexto


# ── P3 (v3): isolamento físico — avaliar nunca busca preços ────────────────────


def test_avaliar_nunca_busca_precos(tmp_path):
    journal = _journal_mock([_noticia()])
    agent = EconAgent(journal=journal, client=_client_mock(_TOOL_OK), cache_dir=tmp_path)

    agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    # get_precos só é usado na MEDIÇÃO DE ALVO EX-POST da calibração, nunca no
    # caminho de decisão — senão preços pós-data_limite vazariam para o payload.
    journal.get_precos.assert_not_called()


# ── P5 (v3): limiar de divergência apertado para 0.25 ──────────────────────────


def test_divergencia_no_novo_limiar_gera_aviso(tmp_path):
    # diff = 0.3: aviso com _DIVERGENCIA_MAX=0.25 (não geraria com o antigo 0.5)
    fora = dict(_TOOL_OK, score_total=0.4, componente_noticia=0.1)
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(fora),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert any("diverg" in a.lower() for a in score.avisos)


# ── P6 (v3): timestamp da notícia mais recente no ScoreEcon ────────────────────


def test_data_noticia_preenchida_com_evento(tmp_path):
    n1 = _noticia()  # publicado_em = 2024-03-10 09:00
    n2 = _noticia()
    object.__setattr__(n2, "publicado_em", ts("2024-03-12 14:00"))  # mais recente
    agent = EconAgent(journal=_journal_mock([n1, n2]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.data_noticia_mais_recente == ts("2024-03-12 14:00")


def test_data_noticia_none_sem_evento(tmp_path):
    agent = EconAgent(journal=_journal_mock([]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.data_noticia_mais_recente is None


# ── v4-P6: hashes individuais das notícias usadas (auditoria) ──────────────────


def test_noticias_hashes_preenchido_com_evento(tmp_path):
    n1, n2 = _noticia(), _noticia(titulo="Outra notícia diferente")
    agent = EconAgent(journal=_journal_mock([n1, n2]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert len(score.noticias_hashes) == 2
    assert all(isinstance(h, str) and h for h in score.noticias_hashes)
    assert len(set(score.noticias_hashes)) == 2  # notícias distintas → hashes distintos


def test_noticias_hashes_vazio_sem_evento(tmp_path):
    agent = EconAgent(journal=_journal_mock([]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.noticias_hashes == []


# ── Earnings Bloomberg no dossiê (Subetapa B) ─────────────────────────────────
#
# O bloco de CONTEXTO DE RESULTADO TRIMESTRAL entra só quando a notícia cai a
# ≤5 dias úteis DEPOIS de uma divulgação de resultado. Sem consenso no dossiê o
# Haiku infere a expectativa de memória — contaminação medida na Etapa 2.


def _earnings(
    ticker="PETR4.SA",
    data_divulgacao="2024-03-14",
    periodo="Q4 23",
    lpa_estimado=1.871,
    lpa_realizado=2.154,
    lpa_comparavel=2.070,
    surpresa_pct=10.64,
    var_preco_pct=-6.15,
) -> EarningsBloomberg:
    return EarningsBloomberg(
        ticker=ticker,
        data_divulgacao=ts(data_divulgacao),
        periodo=periodo,
        periodo_ref=ts("2023-12-31"),
        lpa_estimado=lpa_estimado,
        lpa_realizado=lpa_realizado,
        lpa_comparavel=lpa_comparavel,
        surpresa_pct=surpresa_pct,
        var_preco_pct=var_preco_pct,
        pl=3.8,
    )


class _EarningsFake:
    """Fonte de earnings em memória, com o mesmo corte anti-lookahead da real."""

    def __init__(self, earnings: list[EarningsBloomberg]):
        self._earnings = sorted(earnings, key=lambda e: e.data_divulgacao)

    def buscar_earnings_proximos(self, ticker, data_limite, janela_dias=5):
        import numpy as np
        elegiveis = [
            e for e in self._earnings
            if e.ticker == ticker
            and e.data_divulgacao <= data_limite
            and np.busday_count(e.data_divulgacao.date(), data_limite.date()) <= janela_dias
        ]
        return elegiveis[-1] if elegiveis else None


def _agent_com_earnings(tmp_path, earnings: list[EarningsBloomberg], noticias=None):
    return EconAgent(
        journal=_journal_mock(noticias if noticias is not None else [_noticia()]),
        client=_client_mock(_TOOL_OK),
        cache_dir=tmp_path,
        earnings_source=_EarningsFake(earnings),
    )


def test_dossie_inclui_earnings_quando_proximo_de_resultado(tmp_path):
    # Resultado em 14/03 (qui); avaliação em 15/03 → 1 dia útil depois.
    agent = _agent_com_earnings(tmp_path, [_earnings(data_divulgacao="2024-03-14")])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "CONTEXTO DE RESULTADO TRIMESTRAL:" in contexto
    assert "Q4 23" in contexto


def test_dossie_sem_earnings_quando_longe_de_resultado(tmp_path):
    # Resultado em 01/02; avaliação em 15/03 → ~30 dias úteis, fora da janela.
    agent = _agent_com_earnings(tmp_path, [_earnings(data_divulgacao="2024-02-01")])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "CONTEXTO DE RESULTADO TRIMESTRAL" not in contexto


def test_dossie_sem_earnings_quando_source_none(tmp_path):
    # Backwards-compatible: sem fonte injetada o dossiê é o de antes.
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path)

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "CONTEXTO DE RESULTADO TRIMESTRAL" not in contexto


def test_earnings_no_dossie_respeita_anti_lookahead(tmp_path):
    # Resultado sai em 20/03, avaliação em 15/03: não pode vazar para o dossiê.
    agent = _agent_com_earnings(tmp_path, [_earnings(data_divulgacao="2024-03-20")])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "CONTEXTO DE RESULTADO TRIMESTRAL" not in contexto
    assert "Q4 23" not in contexto


def test_formato_do_bloco_earnings_no_dossie(tmp_path):
    agent = _agent_com_earnings(tmp_path, [_earnings(data_divulgacao="2024-03-14")])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert contexto.endswith(
        "CONTEXTO DE RESULTADO TRIMESTRAL:\n"
        "Período: Q4 23 | Divulgado em: 2024-03-14\n"
        "- LPA realizado (GAAP): R$ 2.154\n"
        "- LPA comparável (ajustado): R$ 2.070\n"
        "- Estimativa consenso: R$ 1.871\n"
        "- Surpresa vs consenso: +10.64% (calculada sobre comparável)\n"
        "- Reação do preço na sessão de divulgação: -6.15%"
    )


def test_bloco_omite_reacao_do_preco_no_proprio_dia_da_divulgacao(tmp_path):
    # `Var prç%` do Terminal é a SESSÃO DE REAÇÃO, que em 18 de 21 amostras é
    # D+1 (resultado sai after-market). Avaliando em D, essa sessão ainda não
    # aconteceu — e pior, ela é a primeira perna do alvo y (D→D+5). Mostrá-la
    # seria lookahead estrutural.
    agent = _agent_com_earnings(tmp_path, [_earnings(data_divulgacao="2024-03-15")])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 23:59"), [_noticia()], [])

    assert "CONTEXTO DE RESULTADO TRIMESTRAL" in contexto  # consenso/LPA seguem
    assert "Reação do preço" not in contexto
    assert "-6.15" not in contexto


def test_bloco_inclui_reacao_do_preco_a_partir_do_dia_util_seguinte(tmp_path):
    agent = _agent_com_earnings(tmp_path, [_earnings(data_divulgacao="2024-03-14")])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 23:59"), [_noticia()], [])

    assert "- Reação do preço na sessão de divulgação: -6.15%" in contexto


def test_bloco_omite_linha_de_lpa_ausente(tmp_path):
    # Sem comparável: a linha some, não vira "R$ None".
    agent = _agent_com_earnings(tmp_path, [_earnings(lpa_comparavel=None)])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "None" not in contexto
    assert "LPA comparável" not in contexto
    assert "- LPA realizado (GAAP): R$ 2.154" in contexto


def test_sem_bloco_quando_realizado_e_comparavel_ausentes(tmp_path):
    # Só consenso (linha de calendário futuro) não é resultado divulgado.
    agent = _agent_com_earnings(
        tmp_path, [_earnings(lpa_realizado=None, lpa_comparavel=None, surpresa_pct=None)])

    contexto = agent._montar_contexto("PETR4.SA", ts("2024-03-15 17:00"), [_noticia()], [])

    assert "CONTEXTO DE RESULTADO TRIMESTRAL" not in contexto


def test_identidade_pura_nao_recebe_earnings(tmp_path):
    # O placebo esconde a identidade da empresa; período e LPAs a entregariam.
    agent = _agent_com_earnings(tmp_path, [_earnings(data_divulgacao="2024-03-14")])

    contexto = agent._montar_contexto(
        "PETR4.SA", ts("2024-03-15 17:00"), [_noticia(titulo="Empresa anuncia plano")], [],
        nome_override="a companhia", incluir_contexto_fundamental=False)

    assert "CONTEXTO DE RESULTADO TRIMESTRAL" not in contexto
    assert "Q4 23" not in contexto


def test_cache_separa_avaliacao_com_e_sem_earnings(tmp_path):
    # Mesmo ticker/data/notícia, dossiês diferentes: o cache NÃO pode servir a
    # avaliação sem earnings para a chamada com earnings (degradação silenciosa).
    noticias = [_noticia()]
    sem = EconAgent(journal=_journal_mock(noticias), client=_client_mock(_TOOL_OK),
                    cache_dir=tmp_path)
    sem.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    client_com = _client_mock(_TOOL_OK)
    com = EconAgent(journal=_journal_mock(noticias), client=client_com, cache_dir=tmp_path,
                    earnings_source=_EarningsFake([_earnings(data_divulgacao="2024-03-14")]))
    com.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    client_com.messages.create.assert_called_once()


def test_falha_da_fonte_de_earnings_nao_derruba_avaliacao(tmp_path):
    # Degradação graciosa: o backtest não pode quebrar por uma fonte de dados.
    fonte_quebrada = MagicMock()
    fonte_quebrada.buscar_earnings_proximos.side_effect = RuntimeError("excel corrompido")
    agent = EconAgent(journal=_journal_mock([_noticia()]), client=_client_mock(_TOOL_OK),
                      cache_dir=tmp_path, earnings_source=fonte_quebrada)

    score = agent.avaliar("PETR4.SA", ts("2024-03-15 17:00"))

    assert score.score_total == 0.6
    assert any("earnings" in a.lower() for a in score.avisos)
