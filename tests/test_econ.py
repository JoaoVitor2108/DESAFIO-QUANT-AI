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
