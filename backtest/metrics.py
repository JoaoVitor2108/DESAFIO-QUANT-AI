"""PROGRAM (JEMPO) — métricas de performance e atribuição de retorno (Etapa 3).

Posição no pipeline (pós-processamento do backtest):

    ... → PROGRAM (engine) → ResultadoBacktest → PROGRAM (metrics, este módulo)
                              dados puros          Sharpe/MDD/atribuição

Biblioteca de FUNÇÕES PURAS (R1): sem classe com estado, sem I/O — exceto a
chamada opcional e determinística (dado o cache) a `journal.get_macro`, usada só
para a taxa livre de risco. Cada função recebe um `ResultadoBacktest` e devolve
uma dataclass frozen (`MetricasBacktest`) ou um `pd.DataFrame`. Dois runs com o
mesmo `ResultadoBacktest` produzem saída idêntica byte-a-byte (R7).

Taxa livre de risco (R2): o JOURNAL não expõe CDI — `get_macro(data_limite)`
devolve `selic_diaria` (SGS 11, Selic overnight realizada, % a.a. base 252).
Usamos essa série como risk-free (`fonte="selic"`). Precedência: override do
chamador (`fonte="override"`) > Selic do journal (`fonte="selic"`) > zero com
aviso (`fonte="zero_por_falha"`). O valor `"cdi"` do enum é intencionalmente
inalcançável neste JOURNAL.

Convenções de bordas (R8), documentadas onde ocorrem:
- `n_dias_uteis == 0` → `ValueError` (backtest vazio não faz sentido).
- Equity constante → volatilidade, Sharpe e Sortino = `0.0` (nunca NaN).
- Sem retornos negativos mas com volatilidade → Sortino = `+inf`.
- Todos os trades vencedores → `payoff_medio` e `profit_factor` = `+inf`.
- `n_trades == 0` → hit_rate/payoff/profit_factor = `None`; métricas de equity
  (Sharpe/MDD/vol) permanecem válidas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import ResultadoBacktest

logger = logging.getLogger(__name__)

# Zero magic numbers: fatores de anualização e chaves macro nomeados.
DIAS_UTEIS_ANO = 252
_CHAVE_SELIC = "selic_diaria"  # SGS 11 — Selic overnight realizada, % a.a.
_FUSO = "America/Sao_Paulo"


# ── Contrato público ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricasBacktest:
    """Métricas de performance derivadas de um `ResultadoBacktest`. Imutável.

    Convenções travadas (ver R4/R8): `maximum_drawdown` é negativo (ou 0.0);
    métricas de trade são `None` quando não há trades.

    Convenções de `payoff_medio` e `profit_factor` em casos degenerados:

        n_trades == 0:
            payoff_medio = None
            profit_factor = None
            (indefinido — sem trades, sem métrica)

        n_vencedores == 0 (todos perdedores):
            payoff_medio = 0.0
            profit_factor = 0.0
            (interpretação: zero ganho relativo)

        n_perdedores == 0 (todos vencedores):
            payoff_medio = float('inf')
            profit_factor = float('inf')
            (interpretação: ganho infinito relativo à perda zero)
    """

    # Retorno
    retorno_total: float
    retorno_anualizado: float

    # Risco
    volatilidade_anualizada: float
    sharpe_anualizado: float
    sortino_anualizado: float
    maximum_drawdown: float          # negativo (ou 0.0)
    duracao_mdd_dias: int
    dias_ate_recuperacao_mdd: Optional[int]

    # Trades
    n_trades_total: int
    n_trades_vencedores: int
    n_trades_perdedores: int
    hit_rate: Optional[float]
    payoff_medio: Optional[float]
    profit_factor: Optional[float]

    # Custos
    custo_total_pago: float
    custo_como_pct_capital_inicial: float

    # Metadados
    taxa_livre_risco_anualizada: float
    fonte_taxa_livre_risco: str  # "cdi" | "selic" | "override" | "zero_por_falha"


# ── Retornos diários (R3) ─────────────────────────────────────────────────────


def _retornos_diarios(equity: pd.Series) -> np.ndarray:
    """Retornos diários simples `r_t = equity[t]/equity[t-1] − 1` (R3). A primeira
    observação (r_0) é descartada por não ter t-1; a série resultante tem
    comprimento `len(equity) - 1`."""
    valores = equity.to_numpy(dtype=float)
    if valores.size < 2:
        return np.empty(0, dtype=float)
    return valores[1:] / valores[:-1] - 1.0


# ── Taxa livre de risco (R2/R8) ───────────────────────────────────────────────


def _resolver_taxa_livre_risco(
    resultado: ResultadoBacktest,
    journal,
    override: Optional[float],
) -> tuple[float, str]:
    """Resolve a taxa livre de risco ANUALIZADA e sua fonte (R2/R8).

    Precedência: `override` (fonte 'override') > Selic do journal (fonte 'selic')
    > 0.0 com aviso (fonte 'zero_por_falha'). Selic vem de
    `journal.get_macro(data_fim)['selic_diaria']` (% a.a.), média no intervalo
    `[data_inicio, data_fim]`, convertida para fração.
    """
    if override is not None:
        return float(override), "override"

    if journal is None:
        logger.warning(
            "taxa livre de risco: journal ausente e sem override; usando 0.0"
        )
        return 0.0, "zero_por_falha"

    try:
        macro = journal.get_macro(_para_sp(resultado.data_fim))
        serie = macro.get(_CHAVE_SELIC)
        if serie is None or serie.empty:
            raise ValueError(f"série {_CHAVE_SELIC!r} ausente ou vazia")
        janela = _fatiar_intervalo(serie, resultado.data_inicio, resultado.data_fim)
        if janela.empty:
            raise ValueError("sem observações de Selic no intervalo do backtest")
        return float(janela.mean()) / 100.0, "selic"
    except Exception as e:  # noqa: BLE001
        # Fallback intencional; R8 exige degradação graciosa quando a fonte de
        # taxa livre falha por qualquer motivo (rede, dado ausente, chave errada,
        # etc). Registrado em fonte_taxa_livre_risco="zero_por_falha" para
        # auditabilidade. Difere do _get_precos do engine, que captura só exceções
        # tipadas conhecidas.
        logger.warning(
            "taxa livre de risco: Selic indisponível (%s); usando 0.0", e
        )
        return 0.0, "zero_por_falha"


def _para_sp(data: pd.Timestamp) -> pd.Timestamp:
    """Localiza uma data naive do PROGRAM em `America/Sao_Paulo` (o JOURNAL exige
    tz-aware). Idempotente."""
    ts = pd.Timestamp(data)
    return ts.tz_localize(_FUSO) if ts.tzinfo is None else ts.tz_convert(_FUSO)


def _fatiar_intervalo(
    serie: pd.Series, data_inicio: pd.Timestamp, data_fim: pd.Timestamp
) -> pd.Series:
    """Recorta `serie` (índice tz-aware) ao intervalo `[data_inicio, data_fim]`,
    comparando com as datas do PROGRAM (naive) localizadas em SP."""
    ini, fim = _para_sp(data_inicio), _para_sp(data_fim)
    return serie[(serie.index >= ini) & (serie.index <= fim)]


def _taxa_diaria(taxa_anual: float) -> float:
    """Converte taxa anualizada (base 252) em taxa diária composta."""
    return (1.0 + taxa_anual) ** (1.0 / DIAS_UTEIS_ANO) - 1.0


# ── Drawdown (R4) ─────────────────────────────────────────────────────────────


def _drawdown(equity: pd.Series) -> tuple[float, int, Optional[int]]:
    """Maximum drawdown e suas durações (R4). Retorna
    `(mdd, duracao_pico_a_vale, dias_ate_recuperacao)`:

    - `mdd`: `min((equity − running_max) / running_max)`, número negativo (ou 0.0
      sem drawdown).
    - `duracao_pico_a_vale`: dias úteis do pico ao vale do MDD máximo (0 se não
      há drawdown).
    - `dias_ate_recuperacao`: dias do vale até reatingir o pico anterior; `None`
      se não recuperou até o fim (ou se não há drawdown).
    """
    valores = equity.to_numpy(dtype=float)
    if valores.size == 0:
        return 0.0, 0, None
    running_max = np.maximum.accumulate(valores)
    dd = (valores - running_max) / running_max
    mdd = float(dd.min())
    if mdd == 0.0:
        return 0.0, 0, None

    vale = int(dd.argmin())
    pico_valor = running_max[vale]
    # Pico que originou ESTE drawdown: última vez que o equity atingiu o
    # running_max antes do vale.
    pico = int(np.where(valores[: vale + 1] == pico_valor)[0][-1])
    duracao = vale - pico

    apos = np.where(valores[vale + 1:] >= pico_valor)[0]
    dias_recuperacao = int(apos[0] + 1) if apos.size else None
    return mdd, duracao, dias_recuperacao


# ── Métricas de trades (R4/R8) ────────────────────────────────────────────────


def _metricas_trades(trades: pd.DataFrame) -> dict:
    """Métricas derivadas dos trades fechados (R4). Bordas R8: sem trades →
    hit_rate/payoff/profit_factor = `None`; sem perdedores (mas com vencedores) →
    payoff/profit_factor = `+inf`."""
    n_total = int(len(trades))
    if n_total == 0:
        return {
            "n_trades_total": 0,
            "n_trades_vencedores": 0,
            "n_trades_perdedores": 0,
            "hit_rate": None,
            "payoff_medio": None,
            "profit_factor": None,
            "custo_total_pago": 0.0,
        }

    pnl = trades["pnl_liquido"].to_numpy(dtype=float)
    ganhos = pnl[pnl > 0]
    perdas = pnl[pnl < 0]
    n_venc, n_perd = int(ganhos.size), int(perdas.size)

    hit_rate = n_venc / n_total

    if n_perd == 0:
        payoff = float("inf") if n_venc > 0 else None
        profit_factor = float("inf") if n_venc > 0 else None
    elif n_venc == 0:
        payoff = 0.0
        profit_factor = 0.0
    else:
        payoff = float(ganhos.mean()) / abs(float(perdas.mean()))
        profit_factor = float(ganhos.sum()) / abs(float(perdas.sum()))

    custo_total = float(
        (trades["custo_entrada"] + trades["custo_saida"]).sum()
    )
    return {
        "n_trades_total": n_total,
        "n_trades_vencedores": n_venc,
        "n_trades_perdedores": n_perd,
        "hit_rate": hit_rate,
        "payoff_medio": payoff,
        "profit_factor": profit_factor,
        "custo_total_pago": custo_total,
    }


# ── Métricas de risco a partir dos retornos (R2/R4/R8) ────────────────────────


def _downside_deviation(retornos: np.ndarray, mar: float) -> float:
    """Downside deviation em relação ao MAR (Minimum Acceptable Return).

    Convenção travada (Sortino 1994; Nawrocki 1999):

        downside_dev = sqrt( Σ min(r_t − MAR, 0)² / N )

    onde `N` é o número TOTAL de retornos — não só os negativos. Dividir por N
    total (e não por `n_negativos`) evita que poucos retornos muito negativos
    inflem artificialmente o denominador e colapsem o Sortino: mede "o quanto
    desviou para baixo ao longo de todo o horizonte". Sem nenhum retorno abaixo
    do MAR, retorna `0.0` (o chamador trata como Sortino `+inf`)."""
    desvios = np.minimum(retornos - mar, 0.0)
    return float(np.sqrt(np.sum(desvios**2) / retornos.size))


def _sharpe_sortino_vol(
    retornos: np.ndarray, rf_diario: float
) -> tuple[float, float, float]:
    """Volatilidade anualizada, Sharpe e Sortino (R2/R4).

    Convenções:
    - Volatilidade e Sharpe usam std amostral (ddof=1), padrão de asset pricing
      empírico.
    - Sortino usa downside deviation dividida por N TOTAL de observações
      (Sortino 1994; Nawrocki 1999), não por `n_negativos` — ver
      `_downside_deviation`. O MAR é a taxa livre de risco diária (a mesma do
      Sharpe), de modo que numerador e denominador ficam consistentes.

    Bordas R8: sem variação (equity constante) → tudo 0.0; sem retornos abaixo do
    MAR mas com volatilidade → Sortino = `+inf`.
    """
    if retornos.size < 2:
        return 0.0, 0.0, 0.0

    sigma = float(retornos.std(ddof=1))
    vol_anual = sigma * np.sqrt(DIAS_UTEIS_ANO)
    if sigma == 0.0:
        return vol_anual, 0.0, 0.0

    excesso = float(retornos.mean()) - rf_diario
    sharpe = np.sqrt(DIAS_UTEIS_ANO) * excesso / sigma

    downside = _downside_deviation(retornos, rf_diario)
    sortino = float("inf") if downside == 0.0 else (
        np.sqrt(DIAS_UTEIS_ANO) * excesso / downside
    )
    return float(vol_anual), float(sharpe), float(sortino)


# ── Função principal (contrato) ───────────────────────────────────────────────


def calcular_metricas(
    resultado: ResultadoBacktest,
    journal,
    taxa_livre_risco_override: Optional[float] = None,
) -> MetricasBacktest:
    """Calcula todas as métricas de performance de um `ResultadoBacktest` (R4).

    Função pura (R1/R7): sem estado, sem I/O — exceto `journal.get_macro` para a
    taxa livre de risco (R2). Determinística.

    Args:
        resultado: saída de `BacktestEngine.rodar_backtest`.
        journal: provedor com `get_macro(data_limite) -> dict[str, pd.Series]`
            (usado só para a Selic). Pode ser `None` se `taxa_livre_risco_override`
            for informado.
        taxa_livre_risco_override: taxa livre de risco ANUALIZADA (fração, ex.:
            0.10 = 10% a.a.) que, se informada, tem precedência sobre a Selic do
            journal (R2).

    Returns:
        `MetricasBacktest` frozen.

    Raises:
        ValueError: se `n_dias_uteis == 0` (backtest vazio, R8).
    """
    if resultado.n_dias_uteis == 0:
        raise ValueError(
            "n_dias_uteis == 0: backtest vazio não tem métricas (R8)."
        )

    capital_inicial = resultado.config.capital_inicial
    retorno_total = resultado.capital_final / capital_inicial - 1.0
    retorno_anualizado = (
        (1.0 + retorno_total) ** (DIAS_UTEIS_ANO / resultado.n_dias_uteis) - 1.0
    )

    taxa_anual, fonte = _resolver_taxa_livre_risco(
        resultado, journal, taxa_livre_risco_override
    )
    rf_diario = _taxa_diaria(taxa_anual)

    retornos = _retornos_diarios(resultado.equity_diario)
    vol_anual, sharpe, sortino = _sharpe_sortino_vol(retornos, rf_diario)
    mdd, duracao_mdd, dias_recuperacao = _drawdown(resultado.equity_diario)

    t = _metricas_trades(resultado.trades)
    custo_pct = t["custo_total_pago"] / capital_inicial

    return MetricasBacktest(
        retorno_total=retorno_total,
        retorno_anualizado=retorno_anualizado,
        volatilidade_anualizada=vol_anual,
        sharpe_anualizado=sharpe,
        sortino_anualizado=sortino,
        maximum_drawdown=mdd,
        duracao_mdd_dias=duracao_mdd,
        dias_ate_recuperacao_mdd=dias_recuperacao,
        n_trades_total=t["n_trades_total"],
        n_trades_vencedores=t["n_trades_vencedores"],
        n_trades_perdedores=t["n_trades_perdedores"],
        hit_rate=t["hit_rate"],
        payoff_medio=t["payoff_medio"],
        profit_factor=t["profit_factor"],
        custo_total_pago=t["custo_total_pago"],
        custo_como_pct_capital_inicial=custo_pct,
        taxa_livre_risco_anualizada=taxa_anual,
        fonte_taxa_livre_risco=fonte,
    )


# ── Atribuição de retorno (R5/R6) ─────────────────────────────────────────────

_COLUNAS_ATRIBUICAO = [
    "n_trades",
    "pnl_liquido_total",
    "pnl_liquido_medio",
    "hit_rate",
    "pct_do_total_pnl",
]


def _atribuir(trades: pd.DataFrame, chave: str) -> pd.DataFrame:
    """Agrega os trades fechados por `chave` ('motivo' ou 'setor') e devolve o
    DataFrame de atribuição (R5/R6). Colunas: `n_trades`, `pnl_liquido_total`,
    `pnl_liquido_medio`, `hit_rate`, `pct_do_total_pnl`. Índice = `chave`,
    ordenado por `pnl_liquido_total` decrescente (empates desfeitos pela ordem
    alfabética do índice — determinismo R7).

    `pct_do_total_pnl` é a contribuição relativa ao P&L líquido total; a soma das
    linhas dá 1.0 quando o total é não-nulo (mesmo com grupos negativos). Se o
    total for exatamente 0, a contribuição relativa é indefinida → `NaN`.

    Sem trades → DataFrame vazio com as mesmas colunas e nome de índice.
    """
    if len(trades) == 0:
        vazio = pd.DataFrame(columns=_COLUNAS_ATRIBUICAO)
        vazio.index.name = chave
        return vazio

    total_pnl = float(trades["pnl_liquido"].sum())
    linhas: list[dict] = []
    for nome, grupo in trades.groupby(chave, sort=False):
        pnl = grupo["pnl_liquido"]
        n = int(len(grupo))
        pnl_total = float(pnl.sum())
        linhas.append({
            chave: nome,
            "n_trades": n,
            "pnl_liquido_total": pnl_total,
            "pnl_liquido_medio": float(pnl.mean()),
            "hit_rate": float((pnl > 0).sum()) / n,
            "pct_do_total_pnl": (
                pnl_total / total_pnl if total_pnl != 0.0 else float("nan")
            ),
        })

    df = pd.DataFrame(linhas).set_index(chave)
    # Tiebreak determinístico: alfabético primeiro, depois pnl decrescente
    # (mergesort é estável, então empates ficam em ordem alfabética).
    df = df.sort_index()
    df = df.sort_values("pnl_liquido_total", ascending=False, kind="mergesort")
    return df[_COLUNAS_ATRIBUICAO]


def atribuir_por_motivo(resultado: ResultadoBacktest) -> pd.DataFrame:
    """Atribuição de retorno por motivo de saída (R5): stop / take / prazo /
    reversao / fim_backtest. Ver `_atribuir` para o contrato das colunas."""
    return _atribuir(resultado.trades, "motivo")


def atribuir_por_setor(resultado: ResultadoBacktest) -> pd.DataFrame:
    """Atribuição de retorno por setor (R6). Mesmas colunas de
    `atribuir_por_motivo`, agrupadas pelo `setor` do `TradeRegistro`."""
    return _atribuir(resultado.trades, "setor")
