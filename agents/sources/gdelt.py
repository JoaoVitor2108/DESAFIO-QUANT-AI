"""
GDELTSource — leitor do GDELT 2.0 Doc API para notícias do JOURNAL.

GDELT indexa imprensa global desde 2015, sem chave de API. Não entrega o
corpo do artigo (apenas título, URL, domínio e timestamp), por isso é a
camada de volume histórico — complementada por Bloomberg (curadoria manual)
e NewsAPI (período recente com corpo).

Anti-lookahead: o fetch usa enddatetime convertido para UTC, e há um filtro
final descartando qualquer artigo com publicado_em > data_limite (a API às
vezes devolve um pouco além da janela pedida).
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from agents.sources.noticia import Noticia, peso_para_url

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_FUSO = "America/Sao_Paulo"
_FMT_API = "%Y%m%d%H%M%S"  # formato UTC esperado pelo endpoint

# Rate limit do GDELT: ~1 requisição a cada 5s. Throttle proativo (compartilhado
# entre instâncias, pois a API é um recurso global) + retry com backoff no 429.
_GDELT_MIN_INTERVALO_S = 5.5   # espaçamento mínimo entre requests
_GDELT_MAX_TENTATIVAS = 4      # tentativas antes de desistir sob rate limit
_GDELT_BACKOFF_S = 5.5         # base do backoff (multiplicado pela tentativa)
_ultima_chamada = 0.0          # time.monotonic() da última request (estado de módulo)


def _throttle() -> None:
    """Garante o espaçamento mínimo entre requisições ao GDELT."""
    global _ultima_chamada
    delta = time.monotonic() - _ultima_chamada
    if 0 <= delta < _GDELT_MIN_INTERVALO_S:
        time.sleep(_GDELT_MIN_INTERVALO_S - delta)
    _ultima_chamada = time.monotonic()


def _eh_rate_limit(resp) -> bool:
    """True se a resposta indica rate limit (429 ou 200 com aviso textual)."""
    if resp.status_code == 429:
        return True
    if resp.status_code == 200 and "Please limit requests" in getattr(resp, "text", ""):
        return True
    return False


class GDELTSource:
    """Consulta o GDELT 2.0 Doc API e devolve Noticia filtradas pela whitelist."""

    def __init__(
        self,
        cache_dir: str | Path,
        whitelist: dict[str, float],
        ttl_horas: int = 24,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.whitelist = whitelist
        self.ttl = timedelta(hours=ttl_horas)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_path(self, query: str, data_limite: pd.Timestamp) -> Path:
        raw = f"{query}|{data_limite.isoformat()}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.cache_dir / f"gdelt_{h}.pkl"

    def _ler_cache(self, path: Path):
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > self.ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _gravar_cache(self, path: Path, valor) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump(valor, f)
        except Exception as e:
            logger.warning("Falha ao gravar cache GDELT: %s", e)

    # ── Busca ─────────────────────────────────────────────────────────────────

    def buscar(
        self,
        query: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
        max_results: int = 100,
    ) -> list[Noticia]:
        """Retorna Noticia da whitelist publicadas em [data_inicio, data_limite].

        Timestamps de entrada e saída são timezone-aware em America/Sao_Paulo.
        Em qualquer falha de rede/parse, loga e retorna lista vazia — uma fonte
        fora do ar não pode derrubar o pipeline.
        """
        cache_path = self._cache_path(query, data_limite)
        cached = self._ler_cache(cache_path)
        if cached is not None:
            logger.debug("GDELT cache hit para query '%s' (%d notícias)", query[:60], len(cached))
            return cached

        inicio_utc = data_inicio.tz_convert("UTC")
        limite_utc = data_limite.tz_convert("UTC")
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(min(max_results, 250)),
            "sort": "DateDesc",
            "startdatetime": inicio_utc.strftime(_FMT_API),
            "enddatetime": limite_utc.strftime(_FMT_API),
        }
        logger.debug("GDELT request query='%s' janela %s..%s (UTC)",
                     query[:60], inicio_utc, limite_utc)

        # Laço com retry/backoff: o GDELT responde 429 sob rate limit (~1 req/5s).
        # Falha definitiva retorna [] SEM cachear, para a próxima chamada tentar.
        data = None
        for tentativa in range(1, _GDELT_MAX_TENTATIVAS + 1):
            _throttle()
            try:
                resp = requests.get(_ENDPOINT, params=params, timeout=15)
            except Exception as e:
                logger.warning("GDELT falhou (rede) para query '%s': %s", query[:60], e)
                return []

            if _eh_rate_limit(resp):
                if tentativa == _GDELT_MAX_TENTATIVAS:
                    logger.warning("GDELT rate limit persistente após %d tentativas (query '%s')",
                                   _GDELT_MAX_TENTATIVAS, query[:60])
                    return []
                espera = _GDELT_BACKOFF_S * tentativa
                logger.info("GDELT rate-limited (429) query '%s', tentativa %d/%d; aguardando %.1fs",
                            query[:60], tentativa, _GDELT_MAX_TENTATIVAS, espera)
                time.sleep(espera)
                continue

            if resp.status_code != 200:
                logger.warning("GDELT status %s para query '%s'", resp.status_code, query[:60])
                return []

            try:
                data = resp.json()
            except Exception as e:
                logger.warning("GDELT JSON inválido para query '%s': %s", query[:60], e)
                return []
            break

        if data is None:
            return []

        result: list[Noticia] = []
        for art in data.get("articles", []):
            url = art.get("url", "") or ""
            dominio = art.get("domain", "") or ""
            peso = peso_para_url(f"{dominio} {url}", self.whitelist)
            if peso is None:
                continue  # fora da whitelist

            pub = self._parse_seendate(art.get("seendate", ""))
            if pub is None:
                continue
            if pub > data_limite:
                continue  # filtro de segurança anti-lookahead

            result.append(Noticia(
                titulo=art.get("title", "") or "",
                conteudo="",  # GDELT não entrega corpo do artigo
                url=url,
                publicado_em=pub,
                fonte=dominio or "gdelt",
                peso_fonte=peso,
                ticker=query,
            ))

        result.sort(key=lambda n: n.publicado_em, reverse=True)
        logger.debug("GDELT '%s': %d artigos brutos -> %d na whitelist",
                     query[:60], len(data.get("articles", [])), len(result))
        self._gravar_cache(cache_path, result)
        return result

    @staticmethod
    def _parse_seendate(raw: str):
        """Converte seendate (UTC) → America/Sao_Paulo. None se inparseável."""
        if not raw:
            return None
        # Formato canônico: "20240115T103000Z"
        try:
            return pd.Timestamp(
                year=int(raw[0:4]), month=int(raw[4:6]), day=int(raw[6:8]),
                hour=int(raw[9:11]), minute=int(raw[11:13]), second=int(raw[13:15]),
                tz="UTC",
            ).tz_convert(_FUSO)
        except Exception:
            pass
        # Fallback: deixar o pandas inferir
        try:
            return pd.Timestamp(raw, tz="UTC").tz_convert(_FUSO)
        except Exception:
            logger.warning("GDELT seendate inparseável: %r", raw)
            return None
