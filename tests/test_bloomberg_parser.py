"""
Testes do parser Bloomberg raw → CSV (Etapa 1 do gate de custo).

Fixtures são strings que simulam o formato bruto da coluna A do Excel
exportado do Terminal Bloomberg — NÃO dependem do arquivo real (exceto o
Grupo E, opcional, que valida a extração de ponta a ponta).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

import openpyxl

from agents.sources.bloomberg_parser import (
    NoticiaBloomberg,
    RelatorioParsing,
    _detectar_coluna_conteudo,
    _limpar_chrome_residual,
    _parsear_worksheet,
    escrever_csv,
    mapear_codigo_fonte,
    parsear_excel_bloomberg,
    parsear_texto_aba,
    parsear_timestamp_bloomberg,
    ticker_da_aba,
)

# ── Fixtures de texto bruto ──────────────────────────────────────────────────

# Bloco completo com Resumo da Bloomberg AI, By autor e corpo (Bloomberg) --.
BLOCO_COMPLETO = (
    "<Back> Voltar\n"
    "Anterior  Próxima  Enviar    Opções    Traduzir                    Notícia\n"
    " 01/15/2025 14:30:00[BN]\n"
    "Esta notícia foi atualizada. Clique aqui para a última versão.\n"
    "      Subscribe Latin America Credit Trends\n"
    "Petrobras Joins Debt Sale Deluge With $2 Billion Bond Offering\n"
    "   Resumo da Bloomberg AI\n"
    "     •  Petrobras is offering $2 billion in dollar bonds.\n"
    "     •  The oil company is joining several other corporates.\n"
    "                                              PETR4 BZ Equity\n"
    " By Giovanna Bellotti Azevedo\n"
    "\n"
    " (Bloomberg) -- Brazil state oil producer is tapping global    Painel gráfico »\n"
    " markets on Wednesday, joining a flurry of companies.\n"
)

# Bloco simples, sem resumo, corpo direto após By.
BLOCO_SEM_RESUMO = (
    "<Back> Voltar\n"
    "Anterior  Próxima  Enviar    Opções    Traduzir                    Notícia\n"
    " 03/10/2025 15:31:48[BFW]\n"
    "Braskem Pulls From $1 Billion Credit Facility Amid Debt Review\n"
    " By Rachel Gamarski\n"
    " (Bloomberg) -- Braskem SA tapped the entirety of a credit line.\n"
)


def _texto(*blocos: str) -> str:
    return "".join(blocos)


# ── Grupo A — Extração básica ────────────────────────────────────────────────


def test_parseia_data_bloomberg_correta():
    ts = parsear_timestamp_bloomberg("01/15/2025 14:30:00")
    assert ts is not None
    assert ts.isoformat() == "2025-01-15T14:30:00-03:00"


def test_axia3_remapeado_para_elet3_por_alias():
    # Rebrand pós-privatização: as abas da 2ª coleta vêm como AXIA3, mas o
    # UNIVERSO_HISTORICO indexa a empresa pelo nome histórico ELET3.
    assert ticker_da_aba("AXIA3") == "ELET3.SA"
    assert ticker_da_aba("axia3") == "ELET3.SA"


def test_ticker_sem_alias_mantem_original():
    for aba in ("PETR4", "vale3", "BBSE3", "ASAI3", "ELET3"):
        assert ticker_da_aba(aba) == f"{aba.upper()}.SA"


def test_ticker_da_aba_ganha_sufixo_sa():
    assert ticker_da_aba("petr4") == "PETR4.SA"
    assert ticker_da_aba("VALE3") == "VALE3.SA"


def test_codigo_bn_mapeia_bloomberg_news():
    fonte, conhecido = mapear_codigo_fonte("BN")
    assert fonte == "Bloomberg News"
    assert conhecido is True


def test_codigo_bfw_mapeia_bloomberg_first_word():
    fonte, conhecido = mapear_codigo_fonte("BFW")
    assert fonte == "Bloomberg First Word"
    assert conhecido is True


def test_codigo_desconhecido_registra_aviso():
    texto = (
        " 01/15/2025 14:30:00[XYZ]\n"
        "Um título suficientemente longo para passar no filtro\n"
        " (Bloomberg) -- corpo qualquer.\n"
    )
    noticias, rel = parsear_texto_aba(texto, "petr4")
    assert len(noticias) == 1
    assert noticias[0].fonte == "Bloomberg"
    assert "XYZ" in rel.codigos_fonte_desconhecidos


# ── Grupo B — Estrutura da notícia ───────────────────────────────────────────


def test_detecta_multiplas_noticias_por_delimitador_back_voltar():
    texto = _texto(BLOCO_COMPLETO, BLOCO_SEM_RESUMO)
    noticias, _ = parsear_texto_aba(texto, "petr4")
    assert len(noticias) == 2


def test_extrai_titulo_apos_data_ignorando_subscribe():
    noticias, _ = parsear_texto_aba(BLOCO_COMPLETO, "petr4")
    assert noticias[0].titulo == (
        "Petrobras Joins Debt Sale Deluge With $2 Billion Bond Offering"
    )


def test_extrai_resumo_ia_quando_presente():
    noticias, _ = parsear_texto_aba(BLOCO_COMPLETO, "petr4")
    resumo = noticias[0].resumo_ia
    assert "Petrobras is offering $2 billion in dollar bonds." in resumo
    assert "joining several other corporates" in resumo
    # O rótulo da seção não deve vazar para o resumo.
    assert "Resumo da Bloomberg AI" not in resumo


def test_extrai_corpo_apos_bloomberg_ou_by_autor():
    noticias, _ = parsear_texto_aba(BLOCO_COMPLETO, "petr4")
    corpo = noticias[0].corpo
    assert corpo.startswith("(Bloomberg) -- Brazil state oil producer")
    # Chrome inline deve ter sido removido.
    assert "Painel gráfico" not in corpo
    assert "BZ Equity" not in corpo


# ── Grupo C — Filtros de qualidade ───────────────────────────────────────────


def test_pula_noticia_com_titulo_vazio_e_loga(caplog):
    texto = (
        " 01/15/2025 14:30:00[BN]\n"
        "curto\n"  # < 10 chars → título inválido
        " (Bloomberg) -- corpo.\n"
    )
    with caplog.at_level(logging.WARNING):
        noticias, rel = parsear_texto_aba(texto, "petr4")
    assert noticias == []
    assert rel.n_puladas_titulo_vazio == 1
    assert any("título" in r.message.lower() for r in caplog.records)


def test_pula_noticia_com_data_invalida_e_loga(caplog):
    texto = (
        " 13/45/2025 99:99:99[BN]\n"  # mês/dia/hora impossíveis
        "Um título suficientemente longo para o filtro\n"
        " (Bloomberg) -- corpo.\n"
    )
    with caplog.at_level(logging.WARNING):
        noticias, rel = parsear_texto_aba(texto, "petr4")
    assert noticias == []
    assert rel.n_puladas_data_invalida == 1
    assert any("data" in r.message.lower() for r in caplog.records)


def test_corpo_sem_marcadores_usa_linhas_apos_titulo():
    # Itens traduzidos (código VAL) chegam sem "(Bloomberg) --" nem "By":
    # o corpo real vem logo após o título e não pode ser descartado.
    texto = (
        " 08/01/2025 10:54:15[VAL]\n"
        "Zig contratou o Bradesco BBI para estruturar uma rodada\n"
        "de captação, disse o presidente da empresa ao Valor.\n"
        "A companhia foi abordada por diferentes players.\n"
    )
    noticias, _ = parsear_texto_aba(texto, "bbdc4")
    assert len(noticias) == 1
    assert noticias[0].titulo == "Zig contratou o Bradesco BBI para estruturar uma rodada"
    assert "captação" in noticias[0].corpo
    assert "diferentes players" in noticias[0].corpo


def test_deduplica_data_titulo_iguais():
    texto = _texto(BLOCO_SEM_RESUMO, BLOCO_SEM_RESUMO)  # idênticos
    noticias, rel = parsear_texto_aba(texto, "petr4")
    assert len(noticias) == 1
    assert rel.n_duplicatas_removidas == 1


# ── Grupo D — Robustez ───────────────────────────────────────────────────────


def test_aba_vazia_registra_e_nao_crasha():
    noticias, rel = parsear_texto_aba("", "wege3")
    assert noticias == []
    assert rel.n_puladas_titulo_vazio == 0
    assert rel.n_puladas_data_invalida == 0


# ── Grupo E — Coluna de conteúdo (A ou B) ────────────────────────────────────
# Parte das abas exportadas do Terminal traz os blocos deslocados para a coluna
# B. `_detectar_coluna_conteudo` reporta a coluna dominante pela contagem de
# date-lines; -1 sinaliza aba sem conteúdo nenhum.


def _aba_sintetica(linhas_por_coluna: dict[int, str]):
    """Worksheet em memória com texto distribuído por coluna (1=A, 2=B)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    for col, texto in linhas_por_coluna.items():
        for i, linha in enumerate(texto.splitlines(), start=1):
            ws.cell(row=i, column=col, value=linha)
    return ws


def test_detecta_coluna_a_quando_conteudo_em_a():
    ws = _aba_sintetica({1: _texto(BLOCO_COMPLETO, BLOCO_SEM_RESUMO)})
    assert _detectar_coluna_conteudo(ws) == 0


def test_detecta_coluna_b_quando_conteudo_em_b():
    ws = _aba_sintetica({2: _texto(BLOCO_COMPLETO, BLOCO_SEM_RESUMO)})
    assert _detectar_coluna_conteudo(ws) == 1


def test_detecta_coluna_retorna_menos_um_quando_aba_vazia():
    ws = _aba_sintetica({1: "linha solta sem date-line\noutra linha\n"})
    assert _detectar_coluna_conteudo(ws) == -1


def test_parseia_aba_com_conteudo_dividido_entre_a_e_b():
    # Caso real (RDOR3/BBSE3): parte dos blocos em A, parte em B. Nenhum
    # dos dois lados pode ser descartado.
    ws = _aba_sintetica({1: BLOCO_COMPLETO, 2: BLOCO_SEM_RESUMO})
    noticias, _ = _parsear_worksheet(ws, "rdor3")
    titulos = {n.titulo for n in noticias}
    assert len(noticias) == 2, titulos
    assert any("Petrobras Joins Debt Sale" in t for t in titulos)
    assert any("Braskem Pulls From" in t for t in titulos)


def test_encoding_utf8_com_acentos_portugueses():
    texto = (
        " 03/09/2025 15:58:30[PBN]\n"
        "Petrobras quebra hiato e se junta à enxurrada global de dívida\n"
        " (Bloomberg) -- A produção de petróleo cresceu com atenção à demanda.\n"
    )
    noticias, _ = parsear_texto_aba(texto, "petr4")
    assert noticias[0].titulo == (
        "Petrobras quebra hiato e se junta à enxurrada global de dívida"
    )
    assert "produção de petróleo" in noticias[0].corpo
    assert noticias[0].fonte == "Bloomberg Portuguese"


def test_dois_runs_identicos_produzem_mesmo_csv(tmp_path):
    noticias, _ = parsear_texto_aba(_texto(BLOCO_COMPLETO, BLOCO_SEM_RESUMO), "petr4")
    csv1 = tmp_path / "run1.csv"
    csv2 = tmp_path / "run2.csv"
    escrever_csv(noticias, csv1)
    escrever_csv(noticias, csv2)
    assert csv1.read_bytes() == csv2.read_bytes()
    # Cabeçalho na ordem exata do schema JEMPO.
    primeira_linha = csv1.read_text(encoding="utf-8").splitlines()[0]
    assert primeira_linha == "data,ticker,titulo,fonte,url,peso,corpo,resumo_ia"


# ── Grupo F — Limpeza de chrome residual do corpo ────────────────────────────


def test_limpar_chrome_noticias_recomendadas():
    corpo = (
        "Conteúdo jornalístico relevante sobre o balanço da empresa. "
        "Notícias recomendadas Outra manchete qualquer que não é conteúdo"
    )
    assert _limpar_chrome_residual(corpo) == (
        "Conteúdo jornalístico relevante sobre o balanço da empresa."
    )


def test_limpar_chrome_para_entrar_em_contato():
    corpo_pt = (
        "Texto real da matéria sobre a companhia. "
        "Para entrar em contato com o repórter: Fulano de Tal, fulano@bloomberg.net"
    )
    assert _limpar_chrome_residual(corpo_pt) == "Texto real da matéria sobre a companhia."
    # A variante em inglês (Bloomberg News) também deve ser cortada.
    corpo_en = (
        "Real story body about the company. "
        "To contact the reporter on this story: John Doe in Sao Paulo"
    )
    assert _limpar_chrome_residual(corpo_en) == "Real story body about the company."


def test_limpar_chrome_tag_ticker_sem_equity():
    corpo = "Ambipar teve forte queda no pregão de hoje. AMBP3 BZ"
    assert _limpar_chrome_residual(corpo) == "Ambipar teve forte queda no pregão de hoje."


def test_limpar_chrome_numeracao_orfa():
    corpo = "O resultado do trimestre veio acima do esperado. 101) 102) 103)"
    assert _limpar_chrome_residual(corpo) == (
        "O resultado do trimestre veio acima do esperado."
    )


def test_limpar_chrome_preserva_conteudo_real():
    # Datas em parênteses, preço-alvo de analista e NOTE: NÃO são chrome.
    corpo = (
        "Citi (Compra, preço-alvo R$ 42) elevou a recomendação em 26/11 (11/26/25). "
        "NOTE: veja a cobertura completa do setor bancário"
    )
    limpo = _limpar_chrome_residual(corpo)
    assert "preço-alvo R$ 42)" in limpo
    assert "(11/26/25)" in limpo
    assert "NOTE:" in limpo


# ── Grupo E — Integração com Excel real (opcional) ───────────────────────────

_EXCEL_REAL = Path("data/raw/Pasta2(Recuperado Automaticamente).xlsx")


@pytest.mark.skipif(not _EXCEL_REAL.exists(), reason="Excel real ausente")
def test_parseia_excel_real_extrai_pelo_menos_40_noticias():
    noticias, relatorio = parsear_excel_bloomberg(_EXCEL_REAL)
    assert relatorio.n_noticias_extraidas >= 40
    assert len(noticias) == relatorio.n_noticias_extraidas
    # As 3 abas vazias devem ser detectadas, não crashar.
    assert set(relatorio.abas_vazias) == {"wege3", "abev3", "lren3"}
    # Todas tz-aware em São Paulo e com peso primário.
    for n in noticias:
        assert n.data.tzinfo is not None
        assert n.peso == 1.0
        assert n.ticker.endswith(".SA")
