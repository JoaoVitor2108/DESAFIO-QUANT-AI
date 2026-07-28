"""
BloombergCSVSource — fonte de notícias Bloomberg via CSV parseado (Etapa 2).

Lê o CSV limpo produzido pelo `bloomberg_parser` (Etapa 1,
`data/bloomberg/parsed/noticias.csv`) e serve o `JournalAgent` no schema
`Noticia`. Bloomberg é a fonte primária do JEMPO (peso 1.0): curada à mão no
Terminal da FGV, sem rate limit, sem dependência de rede em runtime e
determinística.

Reconciliação de schema (parser → Noticia), decidida na Etapa 2:
  data       → publicado_em
  peso       → peso_fonte
  corpo+resumo_ia → conteudo   (resumo_ia primeiro, mais denso; separador \\n\\n)
  titulo/url/ticker/fonte → 1:1
O dataclass `Noticia` NÃO tem 'categoria'; o campo do parser não existe aqui.

Janela de datas: bordas INCLUSIVAS (`data_inicio <= publicado_em <= data_limite`),
para casar com a cascata do JournalAgent e com GDELT/NewsAPI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from agents.sources.noticia import Noticia

logger = logging.getLogger(__name__)

# Colunas exatas que o CSV do parser (Etapa 1) precisa ter.
_COLUNAS_ESPERADAS = ("data", "ticker", "titulo", "fonte", "url", "peso", "corpo", "resumo_ia")


class BloombergCSVSource:
    """Fonte de notícias Bloomberg via CSV parseado, no schema `Noticia`."""

    NOME_FONTE = "Bloomberg"
    PESO_PADRAO = 1.0
    SEPARADOR_CONTEUDO = "\n\n"

    def __init__(
        self,
        caminho_csv: Path = Path("data/bloomberg/parsed/noticias.csv"),
    ) -> None:
        """`caminho_csv`: CSV parseado. Ausente → `buscar` retorna [] com aviso."""
        self._caminho_csv = Path(caminho_csv)
        self._df_cache: Optional[pd.DataFrame] = None  # lazy load

    # ── API pública ───────────────────────────────────────────────────────────

    def buscar(
        self,
        ticker: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> list[Noticia]:
        """Notícias Bloomberg do `ticker` em [data_inicio, data_limite] (inclusivo).

        `ticker`: formato `PETR4.SA` (com sufixo). `data_inicio`/`data_limite`:
        tz-aware America/Sao_Paulo. Retorna lista ordenada por (publicado_em,
        titulo); vazia se sem match, janela inválida ou CSV inexistente.
        """
        self._validar_aware(data_inicio, "data_inicio")
        self._validar_aware(data_limite, "data_limite")
        if data_limite < data_inicio:
            return []

        df = self._carregar_csv()
        if df.empty:
            return []

        mask = (
            (df["ticker"] == ticker)
            & (df["_data"] >= data_inicio)
            & (df["_data"] <= data_limite)
        )
        selecionadas = df[mask]
        noticias = [self._converter_para_noticia(row) for _, row in selecionadas.iterrows()]
        noticias.sort(key=lambda n: (n.publicado_em, n.titulo))
        return noticias

    # ── Carga (lazy) ──────────────────────────────────────────────────────────

    def _carregar_csv(self) -> pd.DataFrame:
        """Lê o CSV inteiro uma única vez e mantém em memória (lazy load).

        Sem cache TTL: fonte local e determinística. Se o arquivo mudar, recrie
        a instância. Arquivo ausente → DataFrame vazio (buscar retornará []).
        """
        if self._df_cache is not None:
            return self._df_cache

        if not self._caminho_csv.exists():
            logger.warning(
                "Bloomberg CSV não encontrado em %s; fonte retornará vazio",
                self._caminho_csv,
            )
            self._df_cache = pd.DataFrame(columns=[*_COLUNAS_ESPERADAS, "_data"])
            return self._df_cache

        df = pd.read_csv(self._caminho_csv, dtype=str).fillna("")

        faltando = set(_COLUNAS_ESPERADAS) - set(df.columns)
        if faltando:
            raise ValueError(
                f"Bloomberg CSV {self._caminho_csv} sem colunas {sorted(faltando)}; "
                f"esperado {list(_COLUNAS_ESPERADAS)}"
            )

        df["_data"] = pd.to_datetime(df["data"], format="ISO8601")
        if df["_data"].dt.tz is None:
            # R3: CSV sem timezone é bug do parser (Etapa 1), não conserto aqui.
            raise ValueError(
                f"Bloomberg CSV {self._caminho_csv}: coluna 'data' tz-naive "
                "(bug do parser da Etapa 1); esperado ISO 8601 tz-aware"
            )

        self._df_cache = df
        return self._df_cache

    # ── Conversão de schema ───────────────────────────────────────────────────

    def _converter_para_noticia(self, row: pd.Series) -> Noticia:
        """Mapeia uma linha do CSV do parser para o dataclass `Noticia`.

        conteudo = resumo_ia + corpo (resumo_ia primeiro). Ambos vazios: mantém
        a notícia com conteudo="" e loga aviso. peso != 1.0: mantém e loga.
        """
        resumo = str(row["resumo_ia"]).strip()
        corpo = str(row["corpo"]).strip()
        partes = [p for p in (resumo, corpo) if p]
        conteudo = self.SEPARADOR_CONTEUDO.join(partes)
        if not conteudo:
            logger.warning(
                "Bloomberg CSV: notícia sem corpo nem resumo (%r); conteudo vazio",
                row["titulo"],
            )

        peso = pd.to_numeric(row["peso"], errors="coerce")
        peso = float(peso) if pd.notna(peso) else self.PESO_PADRAO
        if peso != self.PESO_PADRAO:
            logger.warning(
                "Bloomberg CSV: peso %.3f != %.1f para %r; mantendo (parser sabe o que faz)",
                peso, self.PESO_PADRAO, row["titulo"],
            )

        return Noticia(
            titulo=str(row["titulo"]),
            conteudo=conteudo,
            url=str(row["url"]),
            publicado_em=row["_data"],
            fonte=str(row["fonte"]) or self.NOME_FONTE,
            peso_fonte=peso,
            ticker=str(row["ticker"]) or None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _validar_aware(ts: pd.Timestamp, nome: str) -> None:
        if ts.tzinfo is None:
            raise ValueError(f"{nome} deve ser timezone-aware (America/Sao_Paulo); recebeu naive.")
