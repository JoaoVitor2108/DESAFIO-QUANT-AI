"""
JournalAgent — provedor de dados puro para o sistema JEMPO.

Responsabilidade única: coletar, limpar e entregar dados brutos.
NÃO calcula scores nem faz julgamentos qualitativos — isso é do ECON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

from config import (
    FUSO,
    INICIO_WARMUP,
    LAG_FUNDAMENTALS_DIAS,
    UNIVERSO_HISTORICO,
    tickers_ativos,
)
from agents.sources.cvm import CVMSource

logger = logging.getLogger(__name__)

_CORTE_HORA = 17
_CORTE_MIN = 5

_WHITELIST_PESOS: dict[str, float] = {
    "bloomberg.com": 1.00,
    "reuters.com": 0.95,
    "valor.globo.com": 0.95,
    "valor.com.br": 0.90,
    "broadcast.com.br": 0.90,
    "estadao.com.br": 0.85,
    "infomoney.com.br": 0.75,
}

# ── Exceções ─────────────────────────────────────────────────────────────────


class LookaheadError(Exception):
    pass


class DadoIndisponivel(Exception):
    pass


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class Noticia:
    titulo: str
    conteudo: str
    url: str
    publicado_em: pd.Timestamp  # timezone-aware, America/Sao_Paulo
    fonte: str
    peso_fonte: float
    ticker: Optional[str] = None


@dataclass
class Fundamentals:
    ticker: str
    data_referencia: pd.Timestamp
    trimestre_fim: Optional[pd.Timestamp] = None
    data_recebimento_cvm: Optional[pd.Timestamp] = None  # DT_RECEB da CVM (anti-lookahead preciso)
    tipo_doc: Optional[str] = None             # "ITR" ou "DFP"
    periodicidade: Optional[dict] = None       # {"receita": "TTM", "caixa": "point_in_time", ...}
    pl: Optional[float] = None
    pvp: Optional[float] = None
    roe: Optional[float] = None
    margem_liquida: Optional[float] = None
    divida_liquida_ebitda: Optional[float] = None
    ebitda: Optional[float] = None
    lucro_liquido: Optional[float] = None
    receita: Optional[float] = None
    patrimonio: Optional[float] = None
    divida: Optional[float] = None
    caixa: Optional[float] = None
    setor: Optional[str] = None
    avisos: list[str] = field(default_factory=list)


# ── Cache em disco ────────────────────────────────────────────────────────────


class _DiskCache:
    TTL = timedelta(hours=24)

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, method: str, params: dict) -> Path:
        raw = json.dumps({"m": method, "p": params}, sort_keys=True, default=str)
        key = hashlib.sha256(raw.encode()).hexdigest()[:20]
        return self.cache_dir / f"{key}.pkl"

    def get(self, method: str, params: dict):
        path = self._path(method, params)
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > self.TTL:
            path.unlink(missing_ok=True)
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, method: str, params: dict, value) -> None:
        path = self._path(method, params)
        with open(path, "wb") as f:
            pickle.dump(value, f)


# ── Helpers de timezone ───────────────────────────────────────────────────────


def _validate_aware(ts: pd.Timestamp, name: str = "timestamp") -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{name} deve ser timezone-aware (America/Sao_Paulo); recebeu naive.")


def _ultimo_fechamento_disponivel(agora: pd.Timestamp) -> pd.Timestamp:
    """Último fechamento de candle disponível dado o corte das 17h05 da B3.

    Não considera feriados — limitação conhecida, aceitável para o protótipo.
    """
    _validate_aware(agora, "agora")
    corte = agora.normalize().replace(hour=_CORTE_HORA, minute=_CORTE_MIN, second=0, microsecond=0)
    if agora >= corte:
        return agora.normalize()
    d = agora.normalize() - pd.Timedelta(days=1)
    while d.dayofweek >= 5:  # sáb=5, dom=6
        d -= pd.Timedelta(days=1)
    return d


def _assert_no_lookahead(
    dados,
    data_limite: pd.Timestamp,
    context: str = "",
) -> None:
    """Levanta LookaheadError se qualquer timestamp nos dados ultrapassar data_limite."""
    _validate_aware(data_limite, "data_limite")

    if isinstance(dados, (pd.DataFrame, pd.Series)):
        if dados.empty:
            return
        idx = dados.index
        if hasattr(idx, "tz") and idx.tz is None:
            raise ValueError(f"Index deve ser timezone-aware em '{context}'")
        max_ts = idx.max()
        if max_ts > data_limite:
            raise LookaheadError(
                f"Lookahead em '{context}': max_index={max_ts} > data_limite={data_limite}"
            )
    elif isinstance(dados, list):
        timestamps = [n.publicado_em for n in dados if hasattr(n, "publicado_em")]
        if not timestamps:
            return
        max_ts = max(timestamps)
        if max_ts > data_limite:
            raise LookaheadError(
                f"Lookahead em '{context}': max_publicado_em={max_ts} > data_limite={data_limite}"
            )


def _normalizar_titulo(titulo: str) -> str:
    t = titulo.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:80]


def _peso_url(url: str) -> Optional[float]:
    for domain, peso in _WHITELIST_PESOS.items():
        if domain in url:
            return peso
    return None


# ── JournalAgent ──────────────────────────────────────────────────────────────


class JournalAgent:
    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        bloomberg_dir: Path = Path("data/bloomberg"),
    ) -> None:
        self._cache = _DiskCache(cache_dir)
        self._bloomberg_dir = bloomberg_dir
        self._news_api_key = os.getenv("NEWS_API_KEY", "")
        self._fred_api_key = os.getenv("FRED_API_KEY", "")

    # ── 1. Notícias ───────────────────────────────────────────────────────────

    def get_noticias(
        self,
        query: str,
        data_limite: pd.Timestamp,
        lookback_days: int = 7,
    ) -> list[Noticia]:
        """Coleta notícias de múltiplas fontes com dedup e peso por fonte.

        Ordem de prioridade: Bloomberg CSV > Reuters/Valor via GDELT >
        GDELT geral > NewsAPI. Apenas fontes da whitelist são aceitas.
        """
        _validate_aware(data_limite, "data_limite")
        cache_key = {"q": query, "dl": str(data_limite), "lb": lookback_days}
        cached = self._cache.get("get_noticias", cache_key)
        if cached is not None:
            return cached

        data_inicio = data_limite - pd.Timedelta(days=lookback_days)
        noticias: list[Noticia] = []
        vistos: set[str] = set()

        def _add(n: Noticia) -> None:
            chave = _normalizar_titulo(n.titulo)
            if not chave or chave in vistos:
                return
            if n.publicado_em > data_limite:
                return
            vistos.add(chave)
            noticias.append(n)

        # Camada 1 — Bloomberg CSV (peso máximo)
        for n in self._bloomberg_csv(query, data_inicio, data_limite):
            _add(n)

        # Camada 2 — Reuters e Valor via GDELT com filtro de domínio
        for domain in ("reuters.com", "valor.globo.com", "valor.com.br"):
            for n in self._gdelt(f"{query} domain:{domain}", data_inicio, data_limite):
                _add(n)

        # Camada 3 — GDELT geral (apenas whitelist)
        for n in self._gdelt(query, data_inicio, data_limite):
            _add(n)

        # Camada 4 — NewsAPI (plano free: últimos 30 dias)
        if self._news_api_key:
            cutoff_news = data_limite - pd.Timedelta(days=30)
            inicio_news = max(data_inicio, cutoff_news)
            for n in self._newsapi(query, inicio_news, data_limite):
                _add(n)

        noticias.sort(key=lambda n: (n.peso_fonte, n.publicado_em), reverse=True)
        _assert_no_lookahead(noticias, data_limite, "get_noticias")
        self._cache.set("get_noticias", cache_key, noticias)
        return noticias

    def _bloomberg_csv(
        self,
        query: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> list[Noticia]:
        result: list[Noticia] = []
        if not self._bloomberg_dir.exists():
            return result
        termos = query.lower().split()
        for csv_path in self._bloomberg_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path, dtype=str).fillna("")
                for _, row in df.iterrows():
                    try:
                        pub = pd.Timestamp(row.get("data_publicacao", ""))
                        if pub.tzinfo is None:
                            pub = pub.tz_localize(FUSO)
                        else:
                            pub = pub.tz_convert(FUSO)
                    except Exception:
                        continue
                    if not (data_inicio <= pub <= data_limite):
                        continue
                    titulo = row.get("titulo", "")
                    conteudo = row.get("conteudo", "")
                    if termos and not any(t in f"{titulo} {conteudo}".lower() for t in termos):
                        continue
                    result.append(Noticia(
                        titulo=titulo,
                        conteudo=conteudo,
                        url=row.get("url", ""),
                        publicado_em=pub,
                        fonte="bloomberg_csv",
                        peso_fonte=1.0,
                        ticker=row.get("ticker") or None,
                    ))
            except Exception as e:
                logger.warning("Erro lendo Bloomberg CSV %s: %s", csv_path.name, e)
        return result

    def _gdelt(
        self,
        query: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> list[Noticia]:
        """Consulta GDELT DocSearch v2. Retorna apenas artigos da whitelist."""
        fmt = "%Y%m%d%H%M%S"
        try:
            resp = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": query,
                    "mode": "artlist",
                    "maxrecords": "250",
                    "format": "json",
                    "startdatetime": data_inicio.strftime(fmt),
                    "enddatetime": data_limite.strftime(fmt),
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("GDELT falhou para query '%s': %s", query[:60], e)
            return []

        result: list[Noticia] = []
        for art in data.get("articles", []):
            url = art.get("url", "")
            peso = _peso_url(url)
            if peso is None:
                continue  # descarte: fora da whitelist
            try:
                # GDELT seendate: "20230115T103000Z"
                raw = art.get("seendate", "")
                pub = pd.Timestamp(
                    year=int(raw[0:4]), month=int(raw[4:6]), day=int(raw[6:8]),
                    hour=int(raw[9:11]), minute=int(raw[11:13]), second=int(raw[13:15]),
                    tz="UTC",
                ).tz_convert(FUSO)
            except Exception:
                try:
                    pub = pd.Timestamp(art.get("seendate", "")).tz_convert(FUSO)
                except Exception:
                    continue
            result.append(Noticia(
                titulo=art.get("title", ""),
                conteudo="",  # GDELT não entrega corpo do artigo
                url=url,
                publicado_em=pub,
                fonte="gdelt",
                peso_fonte=peso,
            ))
        return result

    def _newsapi(
        self,
        query: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> list[Noticia]:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": data_inicio.date().isoformat(),
                    "to": data_limite.date().isoformat(),
                    "language": "pt",
                    "sortBy": "publishedAt",
                    "pageSize": "100",
                    "apiKey": self._news_api_key,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("NewsAPI falhou para '%s': %s", query[:60], e)
            return []

        result: list[Noticia] = []
        for art in data.get("articles", []):
            url = art.get("url", "")
            peso = _peso_url(url)
            if peso is None:
                continue
            try:
                pub = pd.Timestamp(art["publishedAt"]).tz_convert(FUSO)
            except Exception:
                continue
            result.append(Noticia(
                titulo=art.get("title", "") or "",
                conteudo=art.get("description", "") or "",
                url=url,
                publicado_em=pub,
                fonte="newsapi",
                peso_fonte=peso,
            ))
        return result

    # ── 2. Preços ─────────────────────────────────────────────────────────────

    def get_precos(
        self,
        ticker: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
        preencher_gaps: bool = False,
    ) -> pd.DataFrame:
        """Retorna OHLCV diário com colunas Close (ajustado) e Close_raw (bruto).

        Colunas entregues: Open, High, Low, Close, Volume, Close_raw, flag_qualidade.
        flag_qualidade marca (sem apagar): volume_zero | nan | outlier_retorno.
        ffill do Close só se preencher_gaps=True, limite de 3 dias.
        """
        _validate_aware(data_inicio, "data_inicio")
        _validate_aware(data_limite, "data_limite")

        if data_inicio < INICIO_WARMUP:
            raise ValueError(
                f"data_inicio {data_inicio.date()} é anterior ao warmup mínimo "
                f"{INICIO_WARMUP.date()}. Dados de treino começam em 2019-01-01."
            )

        cache_key = {
            "t": ticker,
            "di": str(data_inicio.date()),
            "dl": str(data_limite.date()),
            "pg": preencher_gaps,
        }
        cached = self._cache.get("get_precos", cache_key)
        if cached is not None:
            return cached

        end = (data_limite + pd.Timedelta(days=1)).date()
        start = data_inicio.date()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df_adj = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False, multi_level_index=False,
                )
                df_raw = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=False, progress=False, multi_level_index=False,
                )
            except TypeError:
                # Versão antiga do yfinance sem multi_level_index
                df_adj = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                df_raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

        # Achatar MultiIndex se vier (compatibilidade entre versões do yfinance)
        for df in (df_adj, df_raw):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if df_adj.empty:
            raise DadoIndisponivel(f"yfinance não retornou dados para {ticker!r}")

        df = df_adj[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Close_raw: preço sem ajuste de dividendo/split
        if "Close" in df_raw.columns:
            df["Close_raw"] = df_raw["Close"].reindex(df.index)
        else:
            df["Close_raw"] = np.nan
            logger.warning("%s: Close_raw indisponível no download bruto", ticker)

        # Converter index para timezone-aware America/Sao_Paulo
        if df.index.tz is None:
            df.index = pd.DatetimeIndex(df.index).tz_localize(FUSO)
        else:
            df.index = df.index.tz_convert(FUSO)

        # Cortar por data_limite (end do yfinance é exclusivo, mas garantir aqui)
        df = df[df.index <= data_limite]

        # Flags de qualidade — marcar, nunca apagar
        retornos = df["Close"].pct_change()
        flags = pd.Series("", index=df.index, dtype=str)

        def _add_flag(mask: pd.Series, flag: str) -> None:
            flags[mask] = flags[mask].where(flags[mask] == "", flags[mask] + "|") + flag

        _add_flag(df["Volume"] == 0, "volume_zero")
        _add_flag(df["Close"].isna(), "nan")
        _add_flag(retornos.abs() > 0.30, "outlier_retorno")
        df["flag_qualidade"] = flags

        # ffill Close (não Volume), gaps <= 3 dias úteis
        if preencher_gaps:
            df["Close"] = df["Close"].ffill(limit=3)

        _assert_no_lookahead(df, data_limite, f"get_precos({ticker})")
        self._cache.set("get_precos", cache_key, df)
        return df

    # ── 3. Fundamentals ───────────────────────────────────────────────────────

    def get_fundamentals(
        self,
        ticker: str,
        data_limite: pd.Timestamp,
    ) -> Fundamentals:
        """Retorna métricas fundamentalistas brutas.

        Fonte primária: CVM (dados abertos, DT_RECEB como marco de disponibilidade).
        Fallback yfinance: apenas para campo 'setor' (sector classification).
        P/L e P/VP: best-effort com ações em circulação da CVM + preço do yfinance.
        Qualquer falha gera aviso explícito — nunca número silenciosamente errado.
        """
        _validate_aware(data_limite, "data_limite")
        cache_key = {"t": ticker, "dl": str(data_limite.date())}
        cached = self._cache.get("get_fundamentals", cache_key)
        if cached is not None:
            return cached

        result = Fundamentals(ticker=ticker, data_referencia=data_limite)

        # Setor do universo histórico (fonte mais confiável)
        info_universo = UNIVERSO_HISTORICO.get(ticker, {})
        result.setor = info_universo.get("setor")

        # Se não estiver no universo, tentar yfinance (fallback, apenas setor)
        if result.setor is None:
            try:
                result.setor = yf.Ticker(ticker).info.get("sector")
            except Exception as e:
                result.avisos.append(f"setor via yfinance falhou: {e}")

        # ── CVM como fonte primária de fundamentals ───────────────────────────
        cd_cvm = info_universo.get("cd_cvm")
        cnpj = info_universo.get("cnpj")

        if cd_cvm is None:
            result.avisos.append(
                f"cd_cvm não encontrado em UNIVERSO_HISTORICO para {ticker}; "
                "adicione o campo em config.py para habilitar dados da CVM"
            )
            self._cache.set("get_fundamentals", cache_key, result)
            return result

        cvm = CVMSource()
        cvm_data = None
        try:
            cvm_data = cvm.get_fundamentals(cd_cvm, data_limite)
        except Exception as e:
            result.avisos.append(f"CVMSource.get_fundamentals falhou: {e}")

        if cvm_data is None:
            result.avisos.append(
                f"CVM sem dados disponíveis para cd_cvm={cd_cvm} até {data_limite.date()}"
            )
            self._cache.set("get_fundamentals", cache_key, result)
            return result

        # Mapear campos CVM → Fundamentals
        result.trimestre_fim = cvm_data.get("dt_refer")
        result.data_recebimento_cvm = cvm_data.get("dt_receb")
        result.tipo_doc = cvm_data.get("tipo_doc")
        result.periodicidade = cvm_data.get("periodicidade")
        result.lucro_liquido = cvm_data.get("lucro_liquido")
        result.receita = cvm_data.get("receita")
        result.ebitda = cvm_data.get("ebitda_aproximado")
        result.patrimonio = cvm_data.get("patrimonio_liquido")
        result.divida = cvm_data.get("divida_bruta")
        result.caixa = cvm_data.get("caixa")

        for aviso in cvm_data.get("avisos", []):
            result.avisos.append(f"[CVM] {aviso}")

        # Métricas derivadas
        if result.lucro_liquido is not None and result.receita:
            result.margem_liquida = result.lucro_liquido / result.receita

        if result.lucro_liquido is not None and result.patrimonio:
            result.roe = result.lucro_liquido / result.patrimonio

        div_liq = cvm_data.get("divida_liquida")
        if div_liq is not None and result.ebitda:
            result.divida_liquida_ebitda = div_liq / result.ebitda

        # ── P/L e P/VP — best-effort: ações CVM + preço yfinance ─────────────
        try:
            shares = cvm.get_acoes_em_circulacao(cd_cvm, data_limite, cnpj=cnpj)
            if shares is None or shares == 0:
                result.avisos.append("ações em circulação indisponíveis; P/L e P/VP não calculados")
            else:
                preco_df = self.get_precos(ticker, data_limite - pd.Timedelta(days=10), data_limite)
                close_vals = preco_df["Close"].dropna()
                if close_vals.empty:
                    result.avisos.append("preço histórico vazio; P/L e P/VP não calculados")
                else:
                    preco = float(close_vals.iloc[-1])
                    mkt_cap = shares * preco

                    if result.lucro_liquido and result.lucro_liquido > 0:
                        result.pl = mkt_cap / result.lucro_liquido
                    else:
                        result.avisos.append("lucro ≤ 0; P/L não calculado")

                    if result.patrimonio and result.patrimonio > 0:
                        result.pvp = mkt_cap / result.patrimonio
                    else:
                        result.avisos.append("patrimônio ≤ 0; P/VP não calculado")
        except Exception as e:
            result.avisos.append(f"falha best-effort P/L e P/VP: {e}")

        self._cache.set("get_fundamentals", cache_key, result)
        return result

    # ── 4. Macro ──────────────────────────────────────────────────────────────

    def get_macro(self, data_limite: pd.Timestamp) -> dict[str, pd.Series]:
        """Retorna séries macroeconômicas brutas cortadas em data_limite.

        Primário: BCB SGS (sem chave). Fallback: FRED (requer FRED_API_KEY).
        Séries: selic_diaria, selic_meta, ptax_usdbrl, ipca_mensal, ipca_12m.
        """
        _validate_aware(data_limite, "data_limite")
        cache_key = {"dl": str(data_limite.date())}
        cached = self._cache.get("get_macro", cache_key)
        if cached is not None:
            return cached

        _BCB = {
            "selic_diaria": 11,
            "selic_meta": 432,
            "ptax_usdbrl": 1,
            "ipca_mensal": 433,
        }
        _FRED_FALLBACK = {
            "selic_meta": "IRSTCI01BRM156N",
            "ptax_usdbrl": "DEXBZUS",
        }

        result: dict[str, pd.Series] = {}
        for nome, codigo in _BCB.items():
            try:
                result[nome] = self._bcb_serie(codigo, INICIO_WARMUP, data_limite)
            except Exception as e:
                logger.warning("BCB SGS %d (%s) falhou: %s", codigo, nome, e)
                result[nome] = pd.Series(dtype=float, name=nome)
                if nome in _FRED_FALLBACK and self._fred_api_key:
                    try:
                        result[nome] = self._fred_serie(
                            _FRED_FALLBACK[nome], INICIO_WARMUP, data_limite
                        )
                        logger.info("%s: usando FRED fallback", nome)
                    except Exception as ef:
                        logger.warning("FRED fallback %s falhou: %s", nome, ef)

        # Compor IPCA 12m (produto rolante de 12 meses)
        ipca_m = result.get("ipca_mensal", pd.Series(dtype=float))
        if not ipca_m.empty:
            ipca_pct = ipca_m / 100
            ipca_12m = (
                (1 + ipca_pct)
                .rolling(window=12)
                .apply(lambda x: x.prod() - 1, raw=True)
                .dropna()
            ) * 100
            result["ipca_12m"] = ipca_12m.rename("ipca_12m")
        else:
            result["ipca_12m"] = pd.Series(dtype=float, name="ipca_12m")

        # Garantir timezone-aware e cortar por data_limite
        for nome, serie in result.items():
            if serie.empty:
                continue
            if serie.index.tz is None:
                serie.index = serie.index.tz_localize(FUSO)
            else:
                serie.index = serie.index.tz_convert(FUSO)
            result[nome] = serie[serie.index <= data_limite]

        self._cache.set("get_macro", cache_key, result)
        return result

    def _bcb_serie(
        self,
        codigo: int,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> pd.Series:
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
            f"?formato=json"
            f"&dataInicial={data_inicio.strftime('%d/%m/%Y')}"
            f"&dataFinal={data_limite.strftime('%d/%m/%Y')}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y").dt.tz_localize(FUSO)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df.set_index("data")["valor"].rename(str(codigo))

    def _fred_serie(
        self,
        series_id: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> pd.Series:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "observation_start": data_inicio.strftime("%Y-%m-%d"),
                "observation_end": data_limite.strftime("%Y-%m-%d"),
                "api_key": self._fred_api_key,
                "file_type": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        records = []
        for o in resp.json().get("observations", []):
            try:
                records.append((
                    pd.Timestamp(o["date"]).tz_localize(FUSO),
                    float(o["value"]),
                ))
            except (ValueError, KeyError):
                continue
        if not records:
            return pd.Series(dtype=float, name=series_id)
        idx, vals = zip(*records)
        return pd.Series(list(vals), index=pd.DatetimeIndex(idx), name=series_id)

    # ── 5. Retorno Ibovespa ───────────────────────────────────────────────────

    def get_retorno_ibovespa(
        self,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> pd.Series:
        """Retornos diários do ^BVSP na janela solicitada.

        Mesma frequência de get_precos; usado pelo MATH&ML para retorno em excesso.
        """
        _validate_aware(data_inicio, "data_inicio")
        _validate_aware(data_limite, "data_limite")
        cache_key = {"di": str(data_inicio.date()), "dl": str(data_limite.date())}
        cached = self._cache.get("get_retorno_ibovespa", cache_key)
        if cached is not None:
            return cached

        df = self.get_precos("^BVSP", data_inicio, data_limite)
        retornos = df["Close"].pct_change().dropna().rename("retorno_ibovespa")

        _assert_no_lookahead(retornos, data_limite, "get_retorno_ibovespa")
        self._cache.set("get_retorno_ibovespa", cache_key, retornos)
        return retornos

    # ── 6. Setor ──────────────────────────────────────────────────────────────

    def get_setor(self, ticker: str) -> str:
        """Setor padronizado do ticker conforme vocabulário fixo em config.py."""
        info = UNIVERSO_HISTORICO.get(ticker)
        if info is None:
            raise DadoIndisponivel(
                f"Ticker {ticker!r} não está em UNIVERSO_HISTORICO. "
                "Adicione-o em config.py antes de usar."
            )
        return info["setor"]

    # ── 7. Retornos do Setor ──────────────────────────────────────────────────

    def get_retornos_setor(
        self,
        setor: str,
        data_limite: pd.Timestamp,
        janela_dias: int = 60,
    ) -> dict:
        """Retorna agregação de retornos dos pares do setor na janela.

        Lista de tickers é dinâmica via tickers_ativos(data_limite) — sem lista estática.
        Entrega dado bruto (retorno_medio, retorno_mediano, n_tickers, tickers);
        cabe ao ECON ponderar 'momento setorial' a partir desses números.
        """
        _validate_aware(data_limite, "data_limite")
        cache_key = {"s": setor, "dl": str(data_limite.date()), "j": janela_dias}
        cached = self._cache.get("get_retornos_setor", cache_key)
        if cached is not None:
            return cached

        ativos = tickers_ativos(data_limite)
        tickers_setor = [
            t for t in ativos
            if UNIVERSO_HISTORICO.get(t, {}).get("setor") == setor
        ]

        if not tickers_setor:
            return {
                "retorno_medio": None,
                "retorno_mediano": None,
                "n_tickers": 0,
                "tickers": [],
                "setor": setor,
            }

        data_inicio = data_limite - pd.Timedelta(days=janela_dias)
        retornos_acum: list[float] = []
        for tk in tickers_setor:
            try:
                df = self.get_precos(tk, data_inicio, data_limite)
                close = df["Close"].dropna()
                if len(close) < 2:
                    continue
                retornos_acum.append(float(close.iloc[-1] / close.iloc[0] - 1))
            except Exception as e:
                logger.warning("get_retornos_setor: erro em %s: %s", tk, e)

        result = {
            "retorno_medio": float(np.mean(retornos_acum)) if retornos_acum else None,
            "retorno_mediano": float(np.median(retornos_acum)) if retornos_acum else None,
            "n_tickers": len(retornos_acum),
            "tickers": tickers_setor,
            "setor": setor,
        }

        self._cache.set("get_retornos_setor", cache_key, result)
        return result
