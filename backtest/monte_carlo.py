"""PROGRAM (JEMPO) — Monte Carlo sobre os trades de um backtest (Etapa 2).

Posição no pipeline (pós-processamento do backtest, ao lado de `metrics.py`):

    ... → PROGRAM (engine) → ResultadoBacktest → PROGRAM (monte_carlo, este módulo)
                              dados puros          distribuições Monte Carlo

Responde "não foi sorte?" por dois ângulos INDEPENDENTES, cada um numa técnica
com distribuição SEPARADA (nunca combinadas):

- **Bootstrap com reposição** (`rodar_bootstrap_retornos_trades`): reamostra N
  trades COM reposição da lista realizada, B vezes. Testa a ROBUSTEZ DO RETORNO —
  "dado que essa é a distribuição de retorno por trade, o agregado é robusto?".

- **Permutação sem reposição** (`rodar_permutacao_ordem_trades`): reembaralha a
  ORDEM cronológica dos mesmos N trades, B vezes. A soma dos P&L é comutativa,
  logo o retorno total NÃO muda; muda a equity curve intermediária e portanto o
  MDD. Testa RISCO/SEQUÊNCIA — "o resultado depende de sorte no ordenamento?".

Biblioteca de FUNÇÕES PURAS: sem I/O, sem chamadas externas, sem dependência de
API. O retorno realizado do Ibov entra como ESCALAR (parâmetro), nunca é buscado
internamente nem reamostrado — é benchmark passivo fixo do mesmo período.

Sharpe NÃO é calculado: o bootstrap sobre trades não corresponde ao Sharpe do
backtest real (que é sobre retornos DIÁRIOS da equity). Só retorno total e MDD
reamostrado por simulação.

Determinismo (R6): seed via `np.random.default_rng(seed)` (nunca `np.random.seed`
global); dois runs com mesmo `ResultadoBacktest`, mesmo `retorno_ibov` e mesma
seed produzem arrays idênticos byte-a-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from backtest.engine import ResultadoBacktest

# Zero magic numbers: parâmetros da simulação e limiares nomeados.
N_SIMULACOES_DEFAULT = 10_000
SEED_DEFAULT = 12345
_PERCENTIS = (5, 25, 50, 75, 95)
_LIMIAR_MDD_TOLERANCIA = -0.20  # proxy de tolerância a risco (R5)

TECNICA_BOOTSTRAP = "bootstrap_retornos"
TECNICA_PERMUTACAO = "permutacao_ordem"


# ── Contrato público ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResultadoMonteCarlo:
    """Distribuições Monte Carlo de UMA técnica (bootstrap OU permutação).
    Imutável. `retornos_simulados`/`mdds_simulados` têm shape `(n_simulacoes,)`.

    `mdds_simulados` são todos ≤ 0 (MDD reamostrado — aproximação, ver
    `_construir_equity_sintetica`). Nos percentis de MDD, `mdd_p05` é o PIOR 5%
    (mais negativo) e `mdd_p95` o MELHOR 5% (mais próximo de zero), consistente com
    `np.percentile` sobre valores negativos.
    """

    tecnica: str  # "bootstrap_retornos" | "permutacao_ordem"
    n_simulacoes: int
    seed: int

    # Distribuição de retorno total
    retornos_simulados: np.ndarray  # shape (n_simulacoes,)
    retorno_p05: float
    retorno_p25: float
    retorno_p50: float
    retorno_p75: float
    retorno_p95: float
    retorno_medio: float
    retorno_desvio: float

    # Distribuição de MDD reamostrado
    mdds_simulados: np.ndarray  # shape (n_simulacoes,), todos ≤ 0
    mdd_p05: float  # pior 5%
    mdd_p25: float
    mdd_p50: float
    mdd_p75: float
    mdd_p95: float  # melhor 5%

    # Probabilidades condicionais
    prob_retorno_positivo: float
    prob_supera_ibov: Optional[float]  # None se retorno_ibov=None
    prob_mdd_melhor_que_20pct: float

    # Metadados
    retorno_ibov_referencia: Optional[float]


# ── Extração e construção (funções puras) ─────────────────────────────────────


def _extrair_pnls(resultado: ResultadoBacktest) -> np.ndarray:
    """Vetor de `pnl_liquido` dos trades fechados (R8: já com custos embutidos —
    o Monte Carlo NUNCA reamostra `pnl_bruto` nem recalcula custos)."""
    if resultado.trades.empty:
        return np.empty(0, dtype=float)
    return resultado.trades["pnl_liquido"].to_numpy(dtype=float)


def _construir_equity_sintetica(
    pnls_por_sim: np.ndarray, capital_inicial: float
) -> np.ndarray:
    """Constrói as equity curves sintéticas de um lote de simulações.

    `pnls_por_sim` tem shape `(B, N)`: cada linha é a sequência de `pnl_liquido`
    reamostrados de uma simulação. Retorna shape `(B, N+1)`: coluna 0 =
    `capital_inicial`; coluna j = capital + soma acumulada dos j primeiros P&L
    (compounding ADITIVO em R$, R3).

    Aproximação (R3): trades reamostrados são aplicados sequencialmente sem gaps
    de tempo. O MDD resultante é 'MDD reamostrado', não o MDD verdadeiro. O MDD
    verdadeiro exige a equity curve DIÁRIA, que se perde no bootstrap. Reportar
    como aproximação nos gráficos.
    """
    cumsum = np.cumsum(pnls_por_sim, axis=1)
    b, n = pnls_por_sim.shape
    equity = np.empty((b, n + 1), dtype=float)
    equity[:, 0] = capital_inicial
    equity[:, 1:] = capital_inicial + cumsum
    return equity


def _mdd_reamostrado(equity: np.ndarray) -> np.ndarray:
    """MDD de cada equity curve sintética, vetorizado sobre o lote.

    `equity` tem shape `(B, N+1)`. Para cada linha: `min((v − running_max) /
    running_max)`. Retorna shape `(B,)`, todos ≤ 0 (0 quando a curva só sobe).

    Helper local (não reusa `metrics._drawdown`) para permitir a vetorização em
    lote e não acoplar o Monte Carlo a um símbolo privado de `metrics.py`.
    """
    running_max = np.maximum.accumulate(equity, axis=1)
    dd = (equity - running_max) / running_max
    return dd.min(axis=1)


# ── Estatísticas e empacotamento ──────────────────────────────────────────────


def _montar_resultado(
    tecnica: str,
    retornos: np.ndarray,
    mdds: np.ndarray,
    n_simulacoes: int,
    seed: int,
    retorno_ibov: Optional[float],
) -> ResultadoMonteCarlo:
    """Calcula percentis, momentos e probabilidades condicionais (R5) e empacota
    num `ResultadoMonteCarlo` frozen. `np.percentile` usa a interpolação 'linear'
    padrão (determinística)."""
    r05, r25, r50, r75, r95 = (float(x) for x in np.percentile(retornos, _PERCENTIS))
    m05, m25, m50, m75, m95 = (float(x) for x in np.percentile(mdds, _PERCENTIS))

    prob_supera_ibov = (
        None if retorno_ibov is None
        else float(np.mean(retornos > retorno_ibov))
    )

    return ResultadoMonteCarlo(
        tecnica=tecnica,
        n_simulacoes=n_simulacoes,
        seed=seed,
        retornos_simulados=retornos,
        retorno_p05=r05,
        retorno_p25=r25,
        retorno_p50=r50,
        retorno_p75=r75,
        retorno_p95=r95,
        retorno_medio=float(np.mean(retornos)),
        retorno_desvio=float(np.std(retornos)),
        mdds_simulados=mdds,
        mdd_p05=m05,
        mdd_p25=m25,
        mdd_p50=m50,
        mdd_p75=m75,
        mdd_p95=m95,
        prob_retorno_positivo=float(np.mean(retornos > 0.0)),
        prob_supera_ibov=prob_supera_ibov,
        prob_mdd_melhor_que_20pct=float(np.mean(mdds > _LIMIAR_MDD_TOLERANCIA)),
        retorno_ibov_referencia=retorno_ibov,
    )


def _rodar(
    tecnica: str,
    pnls_por_sim: np.ndarray,
    capital_inicial: float,
    n_simulacoes: int,
    seed: int,
    retorno_ibov: Optional[float],
) -> ResultadoMonteCarlo:
    """Núcleo compartilhado: dado o lote `(B, N)` de P&L reamostrados, constrói as
    equity curves, mede retorno total e MDD reamostrado por simulação (R2) e
    empacota as estatísticas. Comum a bootstrap e permutação."""
    equity = _construir_equity_sintetica(pnls_por_sim, capital_inicial)
    # Retorno total simulado = equity_final / capital_inicial − 1 (R2).
    retornos = equity[:, -1] / capital_inicial - 1.0
    mdds = _mdd_reamostrado(equity)
    return _montar_resultado(
        tecnica, retornos, mdds, n_simulacoes, seed, retorno_ibov
    )


# ── Técnica A — Bootstrap com reposição ───────────────────────────────────────


def rodar_bootstrap_retornos_trades(
    resultado: ResultadoBacktest,
    n_simulacoes: int = N_SIMULACOES_DEFAULT,
    seed: int = SEED_DEFAULT,
    retorno_ibov: Optional[float] = None,
) -> ResultadoMonteCarlo:
    """Bootstrap com reposição sobre os trades (Técnica A, R1).

    Sorteia N trades COM reposição (N = nº de trades realizados) a partir da lista
    de `pnl_liquido`, repetindo `n_simulacoes` vezes. Cada simulação encadeia os
    trades reamostrados numa equity curve sintética e mede retorno total e MDD
    reamostrado (R2/R3). Testa a robustez do RETORNO agregado.

    Racional (R2): NÃO calcula Sharpe — bootstrap sobre trades não corresponde ao
    Sharpe do backtest real (retornos diários da equity). R8: reamostra o
    `pnl_liquido` (custos já embutidos), nunca o bruto.

    Args:
        resultado: saída de `BacktestEngine.rodar_backtest`.
        n_simulacoes: nº de reamostragens (B). Default 10.000.
        seed: semente do `default_rng` (R6). Default 12345.
        retorno_ibov: retorno realizado do Ibov no MESMO período (escalar). Se
            `None`, `prob_supera_ibov` fica `None` (R4/R7).

    Returns:
        `ResultadoMonteCarlo` com `tecnica="bootstrap_retornos"`.

    Raises:
        ValueError: se não há trades (R7 — sem trades não há o que reamostrar).
    """
    pnls = _extrair_pnls(resultado)
    n = pnls.size
    if n == 0:
        raise ValueError(
            "bootstrap requer ao menos 1 trade; ResultadoBacktest sem trades (R7)."
        )
    rng = np.random.default_rng(seed)
    # Índices COM reposição: shape (B, N). n_trades=1 → sempre o mesmo trade (R7).
    indices = rng.integers(0, n, size=(n_simulacoes, n))
    pnls_por_sim = pnls[indices]
    return _rodar(
        TECNICA_BOOTSTRAP,
        pnls_por_sim,
        resultado.config.capital_inicial,
        n_simulacoes,
        seed,
        retorno_ibov,
    )


# ── Técnica B — Permutação sem reposição ──────────────────────────────────────


def rodar_permutacao_ordem_trades(
    resultado: ResultadoBacktest,
    n_simulacoes: int = N_SIMULACOES_DEFAULT,
    seed: int = SEED_DEFAULT,
    retorno_ibov: Optional[float] = None,
) -> ResultadoMonteCarlo:
    """Permutação sem reposição da ordem cronológica dos trades (Técnica B, R1).

    Reembaralha a sequência dos MESMOS N trades (sem reposição DENTRO de cada
    simulação — cada trade é usado exatamente uma vez), repetindo `n_simulacoes`
    vezes. Como a soma dos P&L é comutativa, o retorno total é INVARIANTE entre
    simulações (a menos de erro de arredondamento de ponto flutuante na soma
    acumulada); o que muda é a equity curve intermediária e portanto o MDD.
    Testa RISCO/SEQUÊNCIA, não retorno.

    Args:
        resultado: saída de `BacktestEngine.rodar_backtest`.
        n_simulacoes: nº de permutações (B). Default 10.000. Permutações podem se
            repetir entre simulações (amostragem independente do espaço de ordens).
        seed: semente do `default_rng` (R6). Default 12345.
        retorno_ibov: retorno realizado do Ibov no MESMO período (escalar). Se
            `None`, `prob_supera_ibov` fica `None` (R4/R7).

    Returns:
        `ResultadoMonteCarlo` com `tecnica="permutacao_ordem"`.

    Raises:
        ValueError: se há menos de 2 trades (R7 — com 0 nada a permutar; com 1 a
            permutação é degenerada, uma sequência só).
    """
    pnls = _extrair_pnls(resultado)
    n = pnls.size
    if n < 2:
        raise ValueError(
            f"permutação requer ao menos 2 trades (recebeu {n}); com <2 a ordem é "
            "degenerada (R7)."
        )
    rng = np.random.default_rng(seed)
    # Permutações independentes por simulação, vetorizadas: argsort de uniformes
    # dá permutações uniformes de range(N), sem reposição dentro da linha.
    permutacoes = np.argsort(rng.random((n_simulacoes, n)), axis=1)
    pnls_por_sim = pnls[permutacoes]
    return _rodar(
        TECNICA_PERMUTACAO,
        pnls_por_sim,
        resultado.config.capital_inicial,
        n_simulacoes,
        seed,
        retorno_ibov,
    )
