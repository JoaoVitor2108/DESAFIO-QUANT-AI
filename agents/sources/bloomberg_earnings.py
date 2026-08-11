"""
Fonte de earnings (consenso vs realizado) do Bloomberg Terminal.

O João coletou na FGV um Excel com uma aba por ticker, cada aba trazendo o
histórico de divulgações de resultado no layout da tela ERN do Terminal.
Esta fonte lê esse arquivo e serve o dossiê do `EconAgent`: sem consenso
explícito no dossiê, o LLM infere a expectativa de mercado de memória —
contaminação medida na Etapa 2 da calibração.

Peculiaridades do arquivo real, todas tratadas aqui:
  - Header na 1ª linha, seguido da linha-lixo "Média dos valores absolutos".
  - `Dt divulg` é TEXTO em MM/DD/YYYY (formato americano do Terminal), não data.
  - `Per ref` é MM/YY; vira o último dia do mês do trimestre de referência.
  - `%Surp` e `Var prç%` são texto com "%"; "N.M." (not meaningful) marca
    trimestre em que o sinal do LPA virou — surpresa percentual sem sentido.
  - Linhas futuras (2026/2027) trazem só `Estimado`, sem realizado.
  - A aba `LREN3 ` tem espaço sobrando; `AXIA3` é o nome comercial de `ELET3`.
    Ambos normalizados por `ticker_da_aba` (mesmo alias do parser de notícias).
  - A aba `AMER3` vem sem a coluna `P/L` — opcional, vira None com aviso.

Surpresa: o Terminal calcula `%Surp` contra `Comp` (LPA comparável, ajustado),
não contra `Divulgado`. Confirmado linha a linha no arquivo real
(PETR4 Q4 25: (2.857 − 1.506)/1.506 = 89.7% = `%Surp`). Por isso `%Surp` é lido
como está e `lpa_comparavel` é exposto junto — quem formatar o dossiê precisa
dos dois para não afirmar uma surpresa que não fecha com os números ao lado.

Anti-lookahead: `buscar_*` só devolvem earnings com `data_divulgacao <=
data_limite`. O Terminal fornece a DATA da divulgação, não a hora; para uma
notícia do próprio dia da divulgação isso admite até um dia de vazamento
intradiário (resultados brasileiros costumam sair após o fechamento).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agents.sources.bloomberg_parser import ticker_da_aba

logger = logging.getLogger(__name__)

# ── Constantes de domínio ─────────────────────────────────────────────────────

_FUSO = "America/Sao_Paulo"
_FMT_DATA_DIVULGACAO = "%m/%d/%Y"  # Terminal exporta no formato americano
_FMT_PERIODO_REF = "%m/%y"         # "06/25" → junho/2025

# Colunas do Terminal → nome interno. `P/L` fica de fora: é opcional (a aba
# AMER3 não a traz) e é resolvida à parte.
_MAPA_COLUNAS = {
    "Dt divulg": "data_divulgacao",
    "Per": "periodo",
    "Per ref": "periodo_ref",
    "Divulgado": "lpa_realizado",
    "Comp": "lpa_comparavel",
    "Estimado": "lpa_estimado",
    "%Surp": "surpresa_pct",
    "Var prç%": "var_preco_pct",
}
_COLUNA_PL = "P/L"

# Sentinela do Terminal para surpresa sem significado (sinal do LPA virou).
_SENTINELA_NAO_MENSURAVEL = "N.M."

_JANELA_PADRAO_DIAS = 5  # dias úteis; mesma régua de 5du do horizonte do JEMPO

_COLUNAS_NORMALIZADAS = (
    "ticker", "data_divulgacao", "periodo", "periodo_ref", "lpa_estimado",
    "lpa_realizado", "lpa_comparavel", "surpresa_pct", "var_preco_pct", "pl",
)


@dataclass(frozen=True)
class EarningsBloomberg:
    """Uma divulgação de resultado trimestral, como o Terminal a reporta."""

    ticker: str                     # PETR4.SA
    data_divulgacao: pd.Timestamp   # tz-aware SP — dia em que o resultado saiu
    periodo: str                    # "Q2 25"
    periodo_ref: pd.Timestamp       # último dia do trimestre de referência
    lpa_estimado: Optional[float]   # consenso antes do anúncio
    lpa_realizado: Optional[float]  # LPA divulgado
    lpa_comparavel: Optional[float] # LPA ajustado; base do %Surp do Terminal
    surpresa_pct: Optional[float]   # (comparável − estimado) / estimado, em %
    var_preco_pct: Optional[float]  # reação do preço no dia da divulgação
    pl: Optional[float]             # P/L na época


# ── Helpers puros ─────────────────────────────────────────────────────────────


def parsear_percentual(valor) -> Optional[float]:
    """"45.64%" → 45.64. "N.M." e vazios → None."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)

    texto = str(valor).strip().rstrip("%").strip()
    if not texto or texto.upper().startswith(_SENTINELA_NAO_MENSURAVEL):
        return None
    try:
        return float(texto.replace(",", ""))
    except ValueError:
        logger.warning("Earnings: percentual não reconhecido %r; tratado como ausente", valor)
        return None


def parsear_periodo_ref(valor) -> pd.Timestamp:
    """"06/25" → 2025-06-30 tz-aware SP (último dia do trimestre de referência)."""
    inicio_mes = pd.to_datetime(str(valor).strip(), format=_FMT_PERIODO_REF, errors="coerce")
    if pd.isna(inicio_mes):
        return pd.NaT
    fim_mes = inicio_mes + pd.offsets.MonthEnd(0)
    return fim_mes.tz_localize(_FUSO)


def _opcional(valor) -> Optional[float]:
    return None if pd.isna(valor) else float(valor)


class BloombergEarningsSource:
    """
    Fonte de dados de earnings (consenso vs realizado) do Bloomberg.
    Lê o Excel coletado na FGV e serve para o dossiê do ECON.
    """

    def __init__(
        self,
        caminho_excel: Path = Path("data/bloomberg/raw/earnings_bloomberg.xlsx"),
    ) -> None:
        """`caminho_excel`: Excel bruto da tela ERN. Ausente → `buscar_*` retornam None."""
        self._caminho = Path(caminho_excel)
        self._df_cache: Optional[pd.DataFrame] = None

    # ── API pública ───────────────────────────────────────────────────────────

    def buscar_earnings_proximos(
        self,
        ticker: str,
        data_limite: pd.Timestamp,
        janela_dias: int = _JANELA_PADRAO_DIAS,
    ) -> Optional[EarningsBloomberg]:
        """
        Busca o earnings mais recente dentro de `janela_dias` úteis
        ANTES de data_limite.

        Lógica: se a notícia do ECON está sendo avaliada em data D,
        e houve anúncio de resultado entre D-janela e D, retorna
        os dados desse earnings. Senão retorna None.

        Anti-lookahead: só retorna earnings com data_divulgacao <= data_limite.
        """
        candidato = self.buscar_ultimo_earnings(ticker, data_limite)
        if candidato is None:
            return None

        dias_uteis = np.busday_count(
            candidato.data_divulgacao.date(), data_limite.date()
        )
        return candidato if dias_uteis <= janela_dias else None

    def buscar_ultimo_earnings(
        self,
        ticker: str,
        data_limite: pd.Timestamp,
    ) -> Optional[EarningsBloomberg]:
        """
        Retorna o earnings mais recente ANTES de data_limite,
        independente da janela. Útil pra contexto geral.
        """
        self._validar_aware(data_limite, "data_limite")

        df = self._carregar_excel()
        if df.empty:
            return None

        # Anti-lookahead: nada com divulgação após o limite; e sem realizado
        # não é uma divulgação, é uma estimativa de calendário futuro.
        elegiveis = df[
            (df["ticker"] == ticker)
            & (df["data_divulgacao"] <= data_limite)
            & (df["lpa_realizado"].notna())
        ]
        if elegiveis.empty:
            return None

        return self._para_dataclass(elegiveis.iloc[-1])

    # ── Carga (lazy) ──────────────────────────────────────────────────────────

    def _carregar_excel(self) -> pd.DataFrame:
        """Lazy load do Excel inteiro.

        Lê todas as abas uma única vez e devolve um DataFrame normalizado,
        ordenado por (ticker, data_divulgacao). Arquivo ausente → vazio.
        """
        if self._df_cache is not None:
            return self._df_cache

        if not self._caminho.exists():
            logger.warning(
                "Excel de earnings não encontrado em %s; fonte retornará vazio",
                self._caminho,
            )
            self._df_cache = pd.DataFrame(columns=list(_COLUNAS_NORMALIZADAS))
            return self._df_cache

        abas = pd.read_excel(self._caminho, sheet_name=None, header=0)
        partes = [
            parseada
            for nome_aba, bruto in abas.items()
            if not (parseada := self._parsear_aba(nome_aba, bruto)).empty
        ]

        if partes:
            df = pd.concat(partes, ignore_index=True)
            df = df.sort_values(
                ["ticker", "data_divulgacao"], kind="mergesort"
            ).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=list(_COLUNAS_NORMALIZADAS))

        self._df_cache = df
        return self._df_cache

    def _parsear_aba(self, nome_aba: str, bruto: pd.DataFrame) -> pd.DataFrame:
        """Uma aba do Terminal → linhas normalizadas. Aba imprestável → vazio."""
        vazio = pd.DataFrame(columns=list(_COLUNAS_NORMALIZADAS))
        ticker = ticker_da_aba(nome_aba)

        faltando = sorted(set(_MAPA_COLUNAS) - set(bruto.columns))
        if faltando:
            raise ValueError(
                f"Excel de earnings {self._caminho}, aba {nome_aba!r}: colunas "
                f"{faltando} ausentes; esperado {sorted(_MAPA_COLUNAS)}"
            )

        df = bruto.rename(columns=_MAPA_COLUNAS)

        if _COLUNA_PL in bruto.columns:
            df["pl"] = pd.to_numeric(bruto[_COLUNA_PL], errors="coerce")
        else:
            logger.warning(
                "Excel de earnings, aba %r: sem coluna %r; P/L ficará ausente",
                nome_aba, _COLUNA_PL,
            )
            df["pl"] = np.nan

        # Descarta a linha-lixo "Média dos valores absolutos" e qualquer outra
        # sem data válida: sem data não há como aplicar o corte anti-lookahead.
        datas = pd.to_datetime(
            df["data_divulgacao"], format=_FMT_DATA_DIVULGACAO, errors="coerce"
        )
        df = df[datas.notna()].copy()
        if df.empty:
            logger.warning("Excel de earnings, aba %r: sem linhas de resultado", nome_aba)
            return vazio

        df["ticker"] = ticker
        df["data_divulgacao"] = datas[datas.notna()].dt.tz_localize(_FUSO)
        df["periodo"] = df["periodo"].astype(str).str.strip()
        # dtypes explícitos: uma aba toda-nula (ex.: AMER3 sem P/L) sairia com
        # dtype object e faria o concat entre abas mudar de tipo.
        df["periodo_ref"] = pd.to_datetime(df["periodo_ref"].map(parsear_periodo_ref), utc=True)
        df["periodo_ref"] = df["periodo_ref"].dt.tz_convert(_FUSO)
        for coluna in ("lpa_estimado", "lpa_realizado", "lpa_comparavel", "pl"):
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce").astype("float64")
        for coluna in ("surpresa_pct", "var_preco_pct"):
            df[coluna] = pd.to_numeric(df[coluna].map(parsear_percentual)).astype("float64")

        return df[list(_COLUNAS_NORMALIZADAS)].sort_values(
            "data_divulgacao", kind="mergesort"
        )

    # ── Conversão de schema ───────────────────────────────────────────────────

    @staticmethod
    def _para_dataclass(row: pd.Series) -> EarningsBloomberg:
        return EarningsBloomberg(
            ticker=str(row["ticker"]),
            data_divulgacao=row["data_divulgacao"],
            periodo=str(row["periodo"]),
            periodo_ref=row["periodo_ref"],
            lpa_estimado=_opcional(row["lpa_estimado"]),
            lpa_realizado=_opcional(row["lpa_realizado"]),
            lpa_comparavel=_opcional(row["lpa_comparavel"]),
            surpresa_pct=_opcional(row["surpresa_pct"]),
            var_preco_pct=_opcional(row["var_preco_pct"]),
            pl=_opcional(row["pl"]),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _validar_aware(ts: pd.Timestamp, nome: str) -> None:
        if ts.tzinfo is None:
            raise ValueError(f"{nome} deve ser timezone-aware (America/Sao_Paulo); recebeu naive.")
