"""
NewsAPISource — leitor do NewsAPI (newsapi.org) para notícias do JOURNAL.

Camada de período recente: o plano gratuito cobre apenas os últimos ~30 dias,
mas entrega corpo do artigo (description + content), o que GDELT não faz.
Requer NEWS_API_KEY no ambiente; sem chave, retorna lista vazia (não erro) —
o pipeline roda mesmo sem essa fonte.

Anti-lookahead: publishedAt (UTC) convertido para America/Sao_Paulo e filtrado
contra data_limite.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from agents.sources.noticia import Noticia, peso_para_url

logger = logging.getLogger(__name__)

_ENDPOINT = "https://newsapi.org/v2/everything"
_FUSO = "America/Sao_Paulo"
_JANELA_FREE_DIAS = 30  # plano gratuito só devolve artigos dos últimos 30 dias


class NewsAPISource:
    """Consulta o NewsAPI e devolve Noticia filtradas pela whitelist."""

    def __init__(
        self,
        cache_dir: str | Path,
        whitelist: dict[str, float],
        api_key: str,
        ttl_horas: int = 24,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.whitelist = whitelist
        self.api_key = api_key or ""
        self.ttl = timedelta(hours=ttl_horas)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_path(self, query: str, data_limite: pd.Timestamp) -> Path:
        raw = f"{query}|{data_limite.isoformat()}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.cache_dir / f"newsapi_{h}.pkl"

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
            logger.warning("Falha ao gravar cache NewsAPI: %s", e)

    # ── Busca ─────────────────────────────────────────────────────────────────

    def buscar(
        self,
        query: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
        max_results: int = 50,
    ) -> list[Noticia]:
        """Retorna Noticia da whitelist publicadas em [data_inicio, data_limite].

        Se a chave não estiver presente, retorna []. Se data_inicio for anterior
        à janela de 30 dias do plano free, é ajustada para esse limite (com
        aviso). Em qualquer falha, loga e retorna [].
        """
        if not self.api_key:
            logger.info("NewsAPI sem chave; pulando fonte.")
            return []

        # Plano gratuito: clampar data_inicio à janela de 30 dias.
        cutoff = data_limite - pd.Timedelta(days=_JANELA_FREE_DIAS)
        if data_inicio < cutoff:
            logger.info(
                "NewsAPI: data_inicio %s anterior à janela de %d dias; ajustando para %s",
                data_inicio.date(), _JANELA_FREE_DIAS, cutoff.date(),
            )
            data_inicio = cutoff
        if data_inicio >= data_limite:
            return []

        cache_path = self._cache_path(query, data_limite)
        cached = self._ler_cache(cache_path)
        if cached is not None:
            logger.debug("NewsAPI cache hit para query '%s' (%d notícias)", query[:60], len(cached))
            return cached

        logger.debug("NewsAPI request query='%s' janela %s..%s",
                     query[:60], data_inicio.date(), data_limite.date())
        try:
            resp = requests.get(
                _ENDPOINT,
                params={
                    "q": query,
                    "from": data_inicio.date().isoformat(),
                    "to": data_limite.date().isoformat(),
                    "sortBy": "publishedAt",
                    "language": "pt",
                    "pageSize": str(min(max_results, 100)),
                    "apiKey": self.api_key,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("NewsAPI status %s para query '%s'", resp.status_code, query[:60])
                return []
            data = resp.json()
        except Exception as e:
            logger.warning("NewsAPI falhou para query '%s': %s", query[:60], e)
            return []

        result: list[Noticia] = []
        for art in data.get("articles", []):
            url = art.get("url", "") or ""
            dominio = urlparse(url).netloc.lower()
            peso = peso_para_url(f"{dominio} {url}", self.whitelist)
            if peso is None:
                continue

            try:
                pub = pd.Timestamp(art["publishedAt"]).tz_convert(_FUSO)
            except Exception:
                continue
            if pub > data_limite:
                continue

            conteudo = " ".join(
                p for p in (art.get("description"), art.get("content")) if p
            )
            result.append(Noticia(
                titulo=art.get("title", "") or "",
                conteudo=conteudo,
                url=url,
                publicado_em=pub,
                fonte=dominio or "newsapi",
                peso_fonte=peso,
                ticker=query,
            ))

        result.sort(key=lambda n: n.publicado_em, reverse=True)
        logger.debug("NewsAPI '%s': %d artigos brutos -> %d na whitelist",
                     query[:60], len(data.get("articles", [])), len(result))
        self._gravar_cache(cache_path, result)
        return result
