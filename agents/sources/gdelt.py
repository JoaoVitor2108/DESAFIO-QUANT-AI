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
import os
import pickle
import random
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

# Throttle base (espaçamento mínimo entre requisições distintas ao GDELT).
# Configurável por ambiente: em backtest histórico (alto volume) convém 12-15s.
_GDELT_THROTTLE_SECONDS = float(os.environ.get("GDELT_THROTTLE_SECONDS", "5"))

# Backoff exponencial sob rate limit (429/503): 60s → 120 → 240 → 480 → ~600,
# 5 tentativas, ~15min de espera total no pior caso antes de levantar.
_GDELT_BACKOFF_BASE_SECONDS = 60       # 1ª espera; dobra a cada retry
_GDELT_BACKOFF_MAX_SECONDS = 600       # teto por espera individual
_GDELT_BACKOFF_MAX_TENTATIVAS = 5      # após isso, levanta
_GDELT_JITTER_FRACAO = 0.15            # ±15% de ruído uniforme em cada espera

_ultima_chamada = 0.0          # time.monotonic() da última request (estado de módulo)


class GDELTRateLimitedError(RuntimeError):
    """GDELT respondeu 429/503 mesmo após backoff exponencial.

    Levantada quando se esgotam as tentativas. O JournalAgent deve capturar esta
    exceção e registrar warning explícito (degradação REAL), em vez de tratá-la
    como 'sem notícia'.
    """


class GDELTUnavailableError(RuntimeError):
    """GDELT fora do ar (5xx que não 503, timeout persistente, DNS, JSON inválido).

    Diferente de rate limit: indica indisponibilidade do serviço, não bloqueio por
    IP. Tratamento similar no JournalAgent (warning + degradação consciente).
    """


def _eh_rate_limit(resp) -> bool:
    """True se a resposta indica rate limit (429/503 ou 200 com aviso textual)."""
    if resp.status_code in (429, 503):
        return True
    if resp.status_code == 200 and "Please limit requests" in (getattr(resp, "text", "") or ""):
        return True
    return False


class GDELTSource:
    """Consulta o GDELT 2.0 Doc API e devolve Noticia filtradas pela whitelist."""

    def __init__(
        self,
        cache_dir: str | Path,
        whitelist: dict[str, float],
        ttl_horas: int = 24,
        sleep_fn=None,
        rng: random.Random | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.whitelist = whitelist
        self.ttl = timedelta(hours=ttl_horas)
        # sleep injetável (testes passam fake que registra esperas sem dormir).
        # Default delega a time.sleep via name-lookup, p/ respeitar monkeypatch.
        self._sleep = sleep_fn or (lambda s: time.sleep(s))
        self._rng = rng or random.Random()
        logger.info("GDELTSource: throttle=%.1fs entre chamadas", _GDELT_THROTTLE_SECONDS)

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

    # ── Throttle + backoff ──────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Garante o espaçamento mínimo (`_GDELT_THROTTLE_SECONDS`) entre requests
        distintas. Estado de módulo: o rate limit do GDELT é por IP/global."""
        global _ultima_chamada
        delta = time.monotonic() - _ultima_chamada
        if 0 <= delta < _GDELT_THROTTLE_SECONDS:
            self._sleep(_GDELT_THROTTLE_SECONDS - delta)
        _ultima_chamada = time.monotonic()

    def _esperar_backoff(self, tentativa: int) -> float:
        """Dorme o backoff exponencial da tentativa (1-indexed) com jitter ±15%.
        Retorna o tempo dormido (para log/teste). Não invalida cache."""
        base = _GDELT_BACKOFF_BASE_SECONDS * (2 ** (tentativa - 1))
        base = min(base, _GDELT_BACKOFF_MAX_SECONDS)
        jitter = base * _GDELT_JITTER_FRACAO * (2 * self._rng.random() - 1)  # ±15%
        espera = max(1.0, base + jitter)
        self._sleep(espera)
        return espera

    def _chamar_gdelt_com_backoff(self, url: str, params: dict) -> dict:
        """Chama o GDELT com backoff em 429/503. Throttle uma vez antes do laço
        (os retries já são espaçados pelo backoff).

        Returns: dict do JSON (status 200).
        Raises:
            GDELTRateLimitedError: 429/503 persistente após N tentativas.
            GDELTUnavailableError: timeout/5xx/DNS/JSON inválido.
        """
        self._throttle()
        for tentativa in range(1, _GDELT_BACKOFF_MAX_TENTATIVAS + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning("GDELT timeout/conn (tentativa %d/%d): %s",
                               tentativa, _GDELT_BACKOFF_MAX_TENTATIVAS, e)
                if tentativa < _GDELT_BACKOFF_MAX_TENTATIVAS:
                    self._esperar_backoff(tentativa)
                    continue
                raise GDELTUnavailableError(
                    f"GDELT inacessível após {_GDELT_BACKOFF_MAX_TENTATIVAS} "
                    f"tentativas: {e}") from e

            if _eh_rate_limit(resp):
                logger.warning("GDELT %s (tentativa %d/%d)", resp.status_code,
                               tentativa, _GDELT_BACKOFF_MAX_TENTATIVAS)
                if tentativa < _GDELT_BACKOFF_MAX_TENTATIVAS:
                    espera = self._esperar_backoff(tentativa)
                    logger.warning("  aguardando %.0fs antes de retry", espera)
                    continue
                raise GDELTRateLimitedError(
                    f"GDELT 429/503 persistente após "
                    f"{_GDELT_BACKOFF_MAX_TENTATIVAS} tentativas")

            if resp.status_code != 200:
                raise GDELTUnavailableError(f"GDELT respondeu status {resp.status_code}")

            try:
                return resp.json()
            except ValueError as e:
                raise GDELTUnavailableError(f"GDELT JSON inválido: {e}") from e

        # inalcançável (o laço sempre retorna ou levanta), mas por garantia:
        raise GDELTRateLimitedError("GDELT 429/503 (fallthrough)")

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

        Degradação NÃO é silenciada: rate limit persistente (429/503) levanta
        `GDELTRateLimitedError` e indisponibilidade (timeout/5xx/JSON inválido)
        levanta `GDELTUnavailableError` — o JournalAgent captura e registra. Cache
        existente nunca é invalidado por erro; falha não é cacheada.
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

        # Pode levantar GDELTRateLimitedError / GDELTUnavailableError — propaga
        # de propósito (degradação consciente no JournalAgent), sem cachear falha.
        data = self._chamar_gdelt_com_backoff(_ENDPOINT, params)

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
