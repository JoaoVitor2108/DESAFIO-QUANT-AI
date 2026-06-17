"""
Testes do NewsAPISource.
Execute: pytest tests/test_newsapi.py -v -s

Testes de integração pulam se NEWS_API_KEY não estiver no ambiente. O
comportamento sem chave e o ajuste de janela de 30 dias são testados
deterministicamente (sem rede).
"""
import os

import pandas as pd
import pytest

from agents.sources.newsapi import NewsAPISource
from config import WHITELIST_FONTES

_SP = "America/Sao_Paulo"
_KEY = os.getenv("NEWS_API_KEY", "")
_TEM_CHAVE = bool(_KEY)
_SKIP_SEM_CHAVE = pytest.mark.skipif(not _TEM_CHAVE, reason="NEWS_API_KEY ausente")


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=_SP)


@pytest.fixture
def src(tmp_path):
    return NewsAPISource(cache_dir=tmp_path, whitelist=WHITELIST_FONTES, api_key=_KEY)


# ── Determinísticos (sem rede) ────────────────────────────────────────────────


def test_newsapi_sem_chave_retorna_vazio(tmp_path):
    src = NewsAPISource(cache_dir=tmp_path, whitelist=WHITELIST_FONTES, api_key="")
    agora = pd.Timestamp.now(tz=_SP)
    assert src.buscar("Petrobras", agora - pd.Timedelta(days=5), agora) == []


def test_newsapi_ajusta_data_inicio_alem_30_dias(tmp_path, monkeypatch):
    """data_inicio muito antiga deve ser clampada à janela de 30 dias antes do fetch."""
    capturado = {}
    class _Resp:
        status_code = 200
        def json(self): return {"articles": []}
    def _get(url, params=None, **k):
        capturado["from"] = params["from"]
        return _Resp()
    monkeypatch.setattr("agents.sources.newsapi.requests.get", _get)

    src = NewsAPISource(cache_dir=tmp_path, whitelist=WHITELIST_FONTES, api_key="fake")
    limite = ts("2024-05-31 12:00")
    src.buscar("Petrobras", ts("2024-01-01"), limite)  # 5 meses antes
    # from deve ser ~30 dias antes do limite, não jan/2024
    assert capturado["from"] == "2024-05-01", f"from não foi clampado: {capturado['from']}"


def test_newsapi_inicio_apos_limite_retorna_vazio(tmp_path):
    src = NewsAPISource(cache_dir=tmp_path, whitelist=WHITELIST_FONTES, api_key="fake")
    d = ts("2024-05-10")
    assert src.buscar("Vale", d, d) == []


def test_newsapi_filtra_whitelist_mock(tmp_path, monkeypatch):
    class _Resp:
        status_code = 200
        def json(self):
            return {"articles": [
                {"title": "Bom", "url": "https://reuters.com/a", "publishedAt": "2024-05-10T12:00:00Z",
                 "description": "x", "content": "y"},
                {"title": "Ruim", "url": "https://siteduvidoso.xyz/b", "publishedAt": "2024-05-10T12:00:00Z",
                 "description": "x", "content": "y"},
            ]}
    monkeypatch.setattr("agents.sources.newsapi.requests.get", lambda *a, **k: _Resp())
    src = NewsAPISource(cache_dir=tmp_path, whitelist=WHITELIST_FONTES, api_key="fake")
    limite = ts("2024-05-31 12:00")
    noticias = src.buscar("Petrobras", ts("2024-05-05"), limite)
    assert len(noticias) == 1
    assert "reuters.com" in noticias[0].fonte


# ── Integração (chave real) ───────────────────────────────────────────────────


@_SKIP_SEM_CHAVE
def test_newsapi_retorna_lista_se_chave_presente(src):
    agora = pd.Timestamp.now(tz=_SP)
    noticias = src.buscar("Petrobras", agora - pd.Timedelta(days=7), agora)
    assert isinstance(noticias, list)
    print(f"\nNewsAPI 'Petrobras' (7d): {len(noticias)} notícias")


@_SKIP_SEM_CHAVE
def test_newsapi_timezone_aware(src):
    agora = pd.Timestamp.now(tz=_SP)
    for n in src.buscar("Vale", agora - pd.Timedelta(days=7), agora):
        assert n.publicado_em.tzinfo is not None
        assert "Sao_Paulo" in str(n.publicado_em.tz)


@_SKIP_SEM_CHAVE
def test_newsapi_corte_data_limite(src):
    limite = pd.Timestamp.now(tz=_SP) - pd.Timedelta(days=2)
    for n in src.buscar("Petrobras", limite - pd.Timedelta(days=5), limite):
        assert n.publicado_em <= limite
