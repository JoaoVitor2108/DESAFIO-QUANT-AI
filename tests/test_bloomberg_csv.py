"""
Testes do BloombergCSVSource (Etapa 2, subetapa 2.1) — fonte isolada.

Fixture: CSV pequeno sintético (8 notícias), no schema do parser da Etapa 1
(`data,ticker,titulo,fonte,url,peso,corpo,resumo_ia`). NÃO depende do CSV real
de 164 notícias.

Convenção de janela (decisão do João): borda INCLUSIVA nos dois lados
(`data_inicio <= publicado_em <= data_limite`), para casar com a cascata do
JournalAgent e com GDELT/NewsAPI.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

import agents.sources.bloomberg_csv as mod
from agents.sources.bloomberg_csv import BloombergCSVSource
from agents.sources.noticia import Noticia


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/Sao_Paulo")


# 8 linhas: cobre ticker distinto, mesmo instante (tiebreak), corpo/resumo
# vazios, peso != 1.0, e notícias fora da janela de outubro.
_LINHAS = [
    # data                         ticker      titulo                              fonte                   url peso corpo                        resumo_ia
    ("2025-10-05T09:00:00-03:00", "PETR4.SA", "Petrobras anuncia dividendo extra", "Bloomberg News",       "", "1.0", "Corpo A sobre dividendos.", "Resumo A curado."),
    ("2025-10-10T14:30:00-03:00", "PETR4.SA", "Petrobras fecha contrato",          "Bloomberg First Word", "", "1.0", "Corpo B.",                  ""),
    ("2025-10-10T14:30:00-03:00", "PETR4.SA", "Aaa manchete no mesmo horario",     "Bloomberg",            "", "1.0", "Corpo C.",                  ""),
    ("2025-10-07T11:00:00-03:00", "VALE3.SA", "Vale producao recorde",             "Bloomberg News",       "", "1.0", "",                          ""),
    ("2025-11-01T08:00:00-03:00", "ITUB4.SA", "Itau lucro trimestral",             "Bloomberg Portuguese", "", "0.9", "Corpo D.",                  "Resumo D."),
    ("2025-09-01T09:00:00-03:00", "PETR4.SA", "Petrobras fora janela antes",       "Bloomberg News",       "", "1.0", "Corpo antes.",              ""),
    ("2025-12-01T09:00:00-03:00", "PETR4.SA", "Petrobras fora janela depois",      "Bloomberg News",       "", "1.0", "Corpo depois.",             ""),
    ("2025-10-05T09:00:00-03:00", "PETR3.SA", "Petro ON ticker diferente",         "Bloomberg News",       "", "1.0", "Corpo PETR3.",              ""),
]
_COLUNAS = ("data", "ticker", "titulo", "fonte", "url", "peso", "corpo", "resumo_ia")


def _escrever_csv(tmp_path: Path) -> Path:
    caminho = tmp_path / "noticias.csv"
    df = pd.DataFrame(_LINHAS, columns=list(_COLUNAS))
    df.to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def _src(tmp_path: Path) -> BloombergCSVSource:
    return BloombergCSVSource(_escrever_csv(tmp_path))


# janela ampla que pega tudo de 2025
_ANO = (ts("2025-01-01T00:00:00-03:00"), ts("2025-12-31T23:59:59-03:00"))


# ── Grupo A — Leitura básica ─────────────────────────────────────────────────


def test_carrega_csv_e_retorna_dataframe(tmp_path):
    df = _src(tmp_path)._carregar_csv()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert set(_COLUNAS).issubset(set(df.columns))


def test_csv_inexistente_retorna_lista_vazia_com_aviso(tmp_path, caplog):
    src = BloombergCSVSource(tmp_path / "nao_existe.csv")
    with caplog.at_level(logging.WARNING):
        resultado = src.buscar("PETR4.SA", *_ANO)
    assert resultado == []
    assert any("não encontrado" in r.message.lower() or "nao encontrado" in r.message.lower()
               for r in caplog.records)


def test_lazy_load_carrega_uma_unica_vez(tmp_path, monkeypatch):
    src = _src(tmp_path)
    contador = {"n": 0}
    real = mod.pd.read_csv

    def spy(*a, **k):
        contador["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod.pd, "read_csv", spy)
    src.buscar("PETR4.SA", *_ANO)
    src.buscar("PETR4.SA", *_ANO)
    assert contador["n"] == 1


def test_ordenacao_saida_por_data_ascendente_tiebreak_titulo(tmp_path):
    noticias = _src(tmp_path).buscar("PETR4.SA", *_ANO)
    titulos = [n.titulo for n in noticias]
    assert titulos == [
        "Petrobras fora janela antes",       # 2025-09-01
        "Petrobras anuncia dividendo extra", # 2025-10-05
        "Aaa manchete no mesmo horario",     # 2025-10-10  (tiebreak: 'Aaa' < 'Petrobras')
        "Petrobras fecha contrato",          # 2025-10-10
        "Petrobras fora janela depois",      # 2025-12-01
    ]


# ── Grupo B — Filtragem ──────────────────────────────────────────────────────


def test_filtra_por_ticker_exato(tmp_path):
    noticias = _src(tmp_path).buscar("PETR4.SA", *_ANO)
    assert len(noticias) == 5
    assert all(n.ticker == "PETR4.SA" for n in noticias)
    # PETR3.SA (mesmo instante de uma PETR4) não pode vazar.
    assert "Petro ON ticker diferente" not in [n.titulo for n in noticias]


def test_filtra_por_data_inicio_e_fim_ambos_inclusivos(tmp_path):
    # Janela [10-05 09:00, 10-10 14:30] com bordas exatas das notícias.
    inicio = ts("2025-10-05T09:00:00-03:00")
    limite = ts("2025-10-10T14:30:00-03:00")
    noticias = _src(tmp_path).buscar("PETR4.SA", inicio, limite)
    titulos = {n.titulo for n in noticias}
    # borda inicial inclusiva:
    assert "Petrobras anuncia dividendo extra" in titulos
    # borda final inclusiva (duas notícias exatamente em 10-10 14:30):
    assert "Petrobras fecha contrato" in titulos
    assert "Aaa manchete no mesmo horario" in titulos
    # fora da janela não entra:
    assert "Petrobras fora janela antes" not in titulos
    assert "Petrobras fora janela depois" not in titulos


def test_ticker_sem_noticias_retorna_lista_vazia(tmp_path):
    assert _src(tmp_path).buscar("WEGE3.SA", *_ANO) == []


def test_range_data_vazio_retorna_lista_vazia(tmp_path):
    # fim < início → janela vazia.
    assert _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-10T00:00:00-03:00"), ts("2025-10-01T00:00:00-03:00")
    ) == []


# ── Grupo C — Reconciliação de schema ────────────────────────────────────────


def test_concatena_resumo_ia_e_corpo_no_conteudo(tmp_path):
    noticias = _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-05T00:00:00-03:00"), ts("2025-10-05T23:59:59-03:00")
    )
    assert len(noticias) == 1
    # resumo_ia primeiro (mais denso), corpo depois.
    assert noticias[0].conteudo == "Resumo A curado.\n\nCorpo A sobre dividendos."


def test_resumo_ia_vazio_usa_apenas_corpo(tmp_path):
    noticias = _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-10T14:30:00-03:00"), ts("2025-10-10T14:30:00-03:00")
    )
    fecha = next(n for n in noticias if n.titulo == "Petrobras fecha contrato")
    assert fecha.conteudo == "Corpo B."


def test_ambos_vazios_mantem_noticia_com_conteudo_vazio_e_aviso(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        noticias = _src(tmp_path).buscar("VALE3.SA", *_ANO)
    assert len(noticias) == 1
    assert noticias[0].conteudo == ""
    assert any("sem corpo nem resumo" in r.message.lower() for r in caplog.records)


def test_peso_bloomberg_e_1_ponto_0(tmp_path):
    noticias = _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-05T00:00:00-03:00"), ts("2025-10-05T23:59:59-03:00")
    )
    assert noticias[0].peso_fonte == 1.0


def test_peso_diferente_de_1_loga_aviso_mas_mantem(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        noticias = _src(tmp_path).buscar("ITUB4.SA", *_ANO)
    assert len(noticias) == 1
    assert noticias[0].peso_fonte == 0.9
    assert any("peso" in r.message.lower() for r in caplog.records)


# ── Grupo D — Integração com Noticia dataclass ───────────────────────────────


def test_todos_campos_obrigatorios_do_noticia_preenchidos(tmp_path):
    noticias = _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-05T00:00:00-03:00"), ts("2025-10-05T23:59:59-03:00")
    )
    n = noticias[0]
    assert isinstance(n, Noticia)
    assert n.titulo == "Petrobras anuncia dividendo extra"
    assert n.conteudo == "Resumo A curado.\n\nCorpo A sobre dividendos."
    assert n.url == ""
    assert n.fonte == "Bloomberg News"
    assert n.peso_fonte == 1.0
    assert n.ticker == "PETR4.SA"
    assert n.publicado_em == ts("2025-10-05T09:00:00-03:00")


def test_timezone_preservado_como_sao_paulo(tmp_path):
    noticias = _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-05T00:00:00-03:00"), ts("2025-10-05T23:59:59-03:00")
    )
    pub = noticias[0].publicado_em
    assert pub.tzinfo is not None
    assert pub.utcoffset() == timedelta(hours=-3)
    assert pub.isoformat().endswith("-03:00")


def test_url_vazia_para_bloomberg_e_ok(tmp_path):
    noticias = _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-05T00:00:00-03:00"), ts("2025-10-05T23:59:59-03:00")
    )
    assert noticias[0].url == ""


# ── Grupo E — Determinismo ───────────────────────────────────────────────────


def test_dois_runs_mesmo_csv_produzem_resultados_identicos(tmp_path):
    src = _src(tmp_path)
    r1 = src.buscar("PETR4.SA", *_ANO)
    r2 = src.buscar("PETR4.SA", *_ANO)
    assert r1 == r2


def test_ordenacao_estavel_com_datas_repetidas(tmp_path):
    noticias = _src(tmp_path).buscar(
        "PETR4.SA", ts("2025-10-10T14:30:00-03:00"), ts("2025-10-10T14:30:00-03:00")
    )
    assert [n.titulo for n in noticias] == [
        "Aaa manchete no mesmo horario",
        "Petrobras fecha contrato",
    ]
