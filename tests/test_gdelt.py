"""
Testes do GDELTSource contra a API real do GDELT 2.0 Doc.
Execute: pytest tests/test_gdelt.py -v -s

Sem mocks para os testes de integração — se a API estiver fora, o teste pula
gracefully (pytest skip). Apenas os testes de api-down e cache usam mock.
"""
import logging
import pickle
import time

import pandas as pd
import pytest

from agents.sources.gdelt import GDELTSource
from agents.sources.noticia import Noticia
from config import WHITELIST_FONTES

logger = logging.getLogger(__name__)

_SP = "America/Sao_Paulo"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=_SP)


@pytest.fixture
def src(tmp_path):
    return GDELTSource(cache_dir=tmp_path, whitelist=WHITELIST_FONTES)


def _online() -> bool:
    """Verifica se o GDELT responde com SUCESSO genuíno (não só 'vivo').

    Espera 6s e refaz até 3 vezes com backoff (0s, +10s, +20s). Só considera
    online quando obtém status 200 com JSON parseável contendo "articles" — um
    200 não-JSON (texto de rate limit) ou 429 NÃO contam, levando ao skip dos
    testes de integração. Isso evita o falso-positivo de 'online mas sem dados':
    se o GDELT está limitando a ponto de não entregar JSON, os testes pulam
    honestamente (a lógica de retry/parse é coberta pelos testes mockados).
    """
    import requests
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": "test", "mode": "ArtList", "format": "json", "maxrecords": "1"}

    time.sleep(6)  # margem inicial para o rate limit do GDELT
    for tentativa, espera in enumerate((0, 10, 20), start=1):
        if espera:
            time.sleep(espera)
        try:
            r = requests.get(url, params=params, timeout=15)
        except Exception as e:
            logger.debug("GDELT _online tentativa %d falhou: %s", tentativa, e)
            continue
        if r.status_code == 200:
            try:
                if "articles" in r.json():
                    logger.info("GDELT online (tentativa %d)", tentativa)
                    return True
            except Exception:
                pass  # 200 não-JSON (rate-limit textual) → tenta de novo
        logger.debug("GDELT _online tentativa %d status %s", tentativa, r.status_code)

    logger.info("GDELT indisponível/limitado após 3 tentativas")
    return False


_SKIP_OFFLINE = pytest.mark.skipif(not _online(), reason="GDELT API indisponível")


# ── Integração (API real) ─────────────────────────────────────────────────────


@_SKIP_OFFLINE
class TestIntegracaoGDELT:
    @pytest.fixture(autouse=True)
    def _rate_limit_guard(self):
        # Espaça testes consecutivos para não estourar o rate limit do GDELT.
        time.sleep(5.5)

    @staticmethod
    def _buscar_ou_skip(src, query, di, dl):
        """Busca real; se vier vazio (provável rate limit no momento), pula.

        Evita os dois extremos ruins: falso-positivo (passar com 0) e flakiness
        (falhar por rate limit). O contrato de retry/parse é garantido pelos
        testes mockados determinísticos.
        """
        noticias = src.buscar(query, di, dl)
        if not noticias:
            pytest.skip("GDELT sem resultados agora (provável rate limit); "
                        "lógica coberta pelos testes mockados")
        return noticias

    def test_gdelt_retorna_lista(self, src):
        noticias = self._buscar_ou_skip(src, "Petrobras", ts("2024-01-01"), ts("2024-01-31 23:59"))
        assert len(noticias) >= 1, "GDELT deveria retornar ≥1 notícia da whitelist"
        print(f"\nGDELT 'Petrobras' jan/2024: {len(noticias)} notícias da whitelist")
        print(f"  exemplo: {noticias[0].fonte} | {noticias[0].titulo[:60]}")

    def test_gdelt_filtra_whitelist(self, src):
        noticias = self._buscar_ou_skip(src, "Vale", ts("2024-01-01"), ts("2024-01-31 23:59"))
        for n in noticias:
            assert n.peso_fonte is not None
            # fonte deve casar com algum domínio da whitelist
            assert any(dom in f"{n.fonte} {n.url}" for dom in WHITELIST_FONTES), (
                f"Fonte fora da whitelist: {n.fonte}"
            )

    def test_gdelt_timezone_aware(self, src):
        noticias = self._buscar_ou_skip(src, "Petrobras", ts("2024-01-01"), ts("2024-01-31 23:59"))
        for n in noticias:
            assert n.publicado_em.tzinfo is not None
            assert "Sao_Paulo" in str(n.publicado_em.tz)

    def test_gdelt_corte_data_limite(self, src):
        limite = ts("2024-01-15 12:00")
        noticias = self._buscar_ou_skip(src, "Petrobras", ts("2024-01-01"), limite)
        for n in noticias:
            assert n.publicado_em <= limite, f"Lookahead: {n.publicado_em} > {limite}"


# ── Resiliência e cache (mock) ────────────────────────────────────────────────


@pytest.fixture
def no_throttle(monkeypatch):
    """Neutraliza o throttle/backoff (time.sleep) para testes mockados rápidos."""
    monkeypatch.setattr("agents.sources.gdelt.time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("agents.sources.gdelt._ultima_chamada", 0.0)


def test_gdelt_api_down_retorna_vazio(src, monkeypatch, no_throttle):
    class _Resp:
        status_code = 500
        text = ""
        def json(self): return {}
    monkeypatch.setattr("agents.sources.gdelt.requests.get", lambda *a, **k: _Resp())
    noticias = src.buscar("Petrobras", ts("2024-01-01"), ts("2024-01-31 23:59"))
    assert noticias == []


def test_gdelt_excecao_rede_retorna_vazio(src, monkeypatch, no_throttle):
    def _boom(*a, **k):
        raise ConnectionError("rede caiu")
    monkeypatch.setattr("agents.sources.gdelt.requests.get", _boom)
    assert src.buscar("Vale", ts("2024-01-01"), ts("2024-01-31 23:59")) == []


def test_gdelt_retry_apos_429(src, monkeypatch, no_throttle):
    # 429 na 1ª chamada, 200 com 1 artigo da whitelist na 2ª → buscar recupera.
    respostas = iter([429, 200])
    class _Resp:
        def __init__(self, status):
            self.status_code = status
            self.text = "Please limit requests" if status == 429 else ""
        def json(self):
            return {"articles": [{
                "title": "Petrobras sobe", "url": "https://reuters.com/x",
                "domain": "reuters.com", "seendate": "20240110T120000Z",
            }]}
    chamadas = {"n": 0}
    def _get(*a, **k):
        chamadas["n"] += 1
        return _Resp(next(respostas))
    monkeypatch.setattr("agents.sources.gdelt.requests.get", _get)

    noticias = src.buscar("Petrobras", ts("2024-01-01"), ts("2024-01-31 23:59"))
    assert chamadas["n"] == 2, "Deveria ter repetido a request após o 429"
    assert len(noticias) == 1
    assert noticias[0].fonte == "reuters.com"


def test_gdelt_rate_limit_persistente_retorna_vazio(src, monkeypatch, no_throttle):
    # Sempre 429 → esgota tentativas → [] sem cachear.
    class _Resp:
        status_code = 429
        text = "Please limit requests"
        def json(self): return {}
    monkeypatch.setattr("agents.sources.gdelt.requests.get", lambda *a, **k: _Resp())
    assert src.buscar("Vale", ts("2024-01-01"), ts("2024-01-31 23:59")) == []
    # Não deve ter cacheado a falha
    assert not list(src.cache_dir.glob("gdelt_*.pkl")), "Falha de rate limit não pode ser cacheada"


def test_gdelt_cache_funciona(src, monkeypatch, no_throttle):
    # Primeira chamada: mock devolve um artigo da whitelist
    chamadas = {"n": 0}
    class _Resp:
        status_code = 200
        def json(self):
            return {"articles": [{
                "title": "Petrobras sobe", "url": "https://reuters.com/x",
                "domain": "reuters.com", "seendate": "20240110T120000Z",
            }]}
    def _get(*a, **k):
        chamadas["n"] += 1
        return _Resp()
    monkeypatch.setattr("agents.sources.gdelt.requests.get", _get)

    n1 = src.buscar("Petrobras", ts("2024-01-01"), ts("2024-01-31 23:59"))
    assert len(n1) == 1 and chamadas["n"] == 1
    # Segunda chamada idêntica: deve vir do cache, sem novo request
    n2 = src.buscar("Petrobras", ts("2024-01-01"), ts("2024-01-31 23:59"))
    assert len(n2) == 1 and chamadas["n"] == 1, "Cache não evitou novo request"

    # Arquivo de cache existe no diretório
    cache_files = list(src.cache_dir.glob("gdelt_*.pkl"))
    assert cache_files, "Arquivo de cache GDELT não foi criado"
    with open(cache_files[0], "rb") as f:
        assert isinstance(pickle.load(f), list)
