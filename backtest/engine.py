"""PROGRAM (JEMPO) — motor de backtest event-driven.

Posição no pipeline:

    JOURNAL → ECON → MATH&ML → ORQUESTRADOR → PROGRAM
      dados   score   ranking   decisão       (este) backtest

O PROGRAM consome as decisões do ORQUESTRADOR (`decidir`) e as executa numa
simulação diária com custos, detecção intraday de stop/take, mark-to-market e
compounding. Devolve um `ResultadoBacktest` — objeto de DADOS puro. Métricas de
performance (Sharpe, MDD, Monte Carlo, plots) NÃO ficam aqui; virão em etapas
seguintes (`backtest/metrics.py`, `backtest/monte_carlo.py`, `backtest/plots.py`).

Fronteiras e convenções travadas (ver CONTEXTO_FIXO_FINAL.md e o prompt da Etapa 1):
- Anti-lookahead é sagrado: a decisão do dia D usa informação estritamente ≤ D-1
  (`equity_hoje` = mark-to-market do fim de D-1, calculado ANTES de qualquer
  leitura intraday de D). Ver `_mtm_fim_dia` e `rodar_backtest`.
- Regras de stop/take (intraday, Low/High) são responsabilidade do PROGRAM; o
  ORQUESTRADOR só devolve fechamentos por prazo/reversão.
- Preços: o backtest opera sobre OHLC AJUSTADO por eventos corporativos
  (splits/bonificações) — Open/High/Low/Close do `journal.get_precos`. `Close_raw`
  é ignorado. Trade-off: perde-se estudo de impacto de split, ganha-se
  consistência de P&L (decisão travada por João, Etapa 1).
- Datas: todos os contratos e dataclasses do PROGRAM usam `pd.Timestamp` NAIVE.
  A conversão para `America/Sao_Paulo` (exigida pelos contratos travados do
  ORQUESTRADOR e do `get_precos`) é feita como ADAPTADOR DE FRONTEIRA, apenas no
  instante de cada chamada — nunca vaza para o estado interno do engine.
- Determinismo: zero aleatoriedade; iterações sobre posições em ordem alfabética
  de ticker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


# ── Configuração (imutável) ───────────────────────────────────────────────────


@dataclass(frozen=True)
class BacktestConfig:
    """Parâmetros do backtest. Todos os defaults vêm das regras travadas do
    desafio (custos por perna, stop/take, sizing). ZERO magic numbers no engine:
    todo parâmetro numérico da simulação sai daqui."""

    capital_inicial: float = 100_000.0
    corretagem: float = 0.003  # 0.3% por perna
    slippage: float = 0.001    # 0.1% por perna
    stop_pct: float = 0.08     # stop = (1 - stop_pct) × preco_entrada
    take_pct: float = 0.15     # take = (1 + take_pct) × preco_entrada
    sizing_pct: float = 0.15   # fração do equity corrente por nova ordem

    @property
    def custo_perna(self) -> float:
        """Custo total por perna (entrada OU saída), sobre o notional. 0.4%."""
        return self.corretagem + self.slippage


# ── Record interno (estado de posição aberta) ─────────────────────────────────


@dataclass
class PosicaoInterna:
    """Estado interno de uma posição aberta no PROGRAM. stop/take são calculados
    sobre o preço REAL de execução (Open do dia de execução). Mutável por ser
    estado de bookkeeping do loop — nunca exposto no `ResultadoBacktest`."""

    ticker: str
    setor: str
    qtd: int
    preco_entrada: float
    data_entrada: pd.Timestamp
    stop_price: float  # (1 - stop_pct) × preco_entrada
    take_price: float  # (1 + take_pct) × preco_entrada
    custo_entrada: float
    y_pred: float
    score_econ: float
    rank: int


# ── Records de output (contrato público) ──────────────────────────────────────


@dataclass(frozen=True)
class TradeRegistro:
    """Um trade fechado (round-trip completo). Imutável — é dado histórico.

    `pnl_bruto = (preco_saida - preco_entrada) × qtd`
    `pnl_liquido = pnl_bruto - custo_entrada - custo_saida`
    """

    ticker: str
    setor: str
    qtd: int
    preco_entrada: float
    preco_saida: float
    data_entrada: pd.Timestamp
    data_saida: pd.Timestamp
    motivo: Literal["stop", "take", "prazo", "reversao", "fim_backtest"]
    custo_entrada: float
    custo_saida: float
    pnl_bruto: float
    pnl_liquido: float
    y_pred_entrada: float
    score_econ_entrada: float
    rank_entrada: int
    dias_uteis_ate_saida: int


@dataclass(frozen=True)
class ResultadoBacktest:
    """Saída de `rodar_backtest`. Objeto de DADOS puro — sem métricas derivadas.

    - `trades`: DataFrame com uma linha por `TradeRegistro` fechado.
    - `equity_diario`: Série indexada por data (naive), valor em R$, um ponto por
      dia útil (mark-to-market do fim do dia).
    - `avisos`: lista de dicts {tipo, data, ticker, detalhe}.
    """

    trades: pd.DataFrame
    equity_diario: pd.Series
    avisos: list[dict]
    config: BacktestConfig
    data_inicio: pd.Timestamp
    data_fim: pd.Timestamp
    n_dias_uteis: int
    n_trades: int
    capital_final: float


# ── Engine ────────────────────────────────────────────────────────────────────


class BacktestEngine:
    """Motor de backtest event-driven do JEMPO.

    Orquestra o loop diário (R1): mark-to-market de D-1, detecção intraday de
    stop/take, `orq.decidir`, execução de fechamentos e novas ordens, e registro
    do equity do fim do dia. Consome apenas o contrato público do ORQUESTRADOR e
    o `journal.get_precos`.
    """

    def __init__(
        self,
        journal,
        orquestrador,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        """
        Args:
            journal: provedor de dados com
                `get_precos(ticker, data_inicio, data_limite)` → DataFrame OHLCV
                (colunas Open/High/Low/Close/Volume). Índice tz-aware SP no real;
                o adaptador de fronteira normaliza datas na chamada.
            orquestrador: instância com o contrato público de `OrchestratorAgent`
                (`decidir`, `notificar_execucao`, `notificar_fechamento`, `status`).
            config: `BacktestConfig`; default = `BacktestConfig()`.
        """
        self._journal = journal
        self._orq = orquestrador
        self._config = config if config is not None else BacktestConfig()

        # Estado do loop — (re)inicializado de fato em `rodar_backtest` (R7).
        self._caixa_livre: float = self._config.capital_inicial
        self._posicoes_internas: dict[str, PosicaoInterna] = {}
        self._trades_fechados: list[TradeRegistro] = []
        self._equity_diario: list[tuple[pd.Timestamp, float]] = []
        self._avisos: list[dict] = []
        self._calendario: pd.DatetimeIndex = pd.DatetimeIndex([])

    # ── Calendário B3 (R5) ────────────────────────────────────────────────────

    def _calendario_bmf(
        self, data_inicio: pd.Timestamp, data_fim: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """Dias de pregão da B3 entre `data_inicio` e `data_fim` (inclusive).

        Usa `pandas_market_calendars` com o calendário 'BMF', que conhece os
        feriados nacionais (Sexta-feira Santa, Corpus Christi, Finados, etc.) —
        ao contrário de `pd.bdate_range`, que só pula sábado/domingo (R5).

        O índice retornado é NAIVE (`.tz is None`), por convenção do PROGRAM: as
        datas do backtest são naive e a conversão para America/Sao_Paulo acontece
        só no adaptador de fronteira das chamadas ao ORQUESTRADOR/JOURNAL.

        Args:
            data_inicio: primeiro dia da janela (naive).
            data_fim: último dia da janela, inclusive (naive).

        Returns:
            pd.DatetimeIndex naive com um ponto por pregão da B3 na janela.
        """
        import pandas_market_calendars as mcal

        bmf = mcal.get_calendar("BMF")
        schedule = bmf.schedule(start_date=data_inicio, end_date=data_fim)
        calendario = schedule.index.tz_localize(None)
        assert calendario.tz is None, "calendário BMF deve ser naive (.tz is None)"
        return calendario

    # ── Loop principal (R1) — Subetapa 1.2 ────────────────────────────────────

    def rodar_backtest(
        self,
        data_inicio: pd.Timestamp,
        data_fim: pd.Timestamp,
    ) -> ResultadoBacktest:
        """Roda o backtest entre `data_inicio` e `data_fim` inclusive.

        Fluxo por dia útil D (R1): (0) `equity_hoje` = mark-to-market do fim de
        D-1 — calculado ANTES de qualquer leitura intraday de D (anti-lookahead,
        R2); (1) detecção intraday de stop/take; (2) `orq.decidir(D, equity_hoje)`;
        (3) execução de fechamentos por prazo/reversão; (4) execução de novas
        ordens; (5) registro do equity do fim de D.

        `equity_hoje` passado ao ORQUESTRADOR é SEMPRE o MTM do fim de D-1 (R2).

        (Implementado na Subetapa 1.2.)
        """
        raise NotImplementedError("rodar_backtest: Subetapa 1.2")

    # ── Métodos privados do loop — Subetapa 1.2 ───────────────────────────────

    def _mtm_fim_dia(self, data: pd.Timestamp) -> float:
        """Mark-to-market do fim do dia `data`: caixa livre + Σ qtd × Close[data].

        Anti-lookahead auditável (R2): nunca lê Close de data futura. (Implementado
        na Subetapa 1.2.)
        """
        raise NotImplementedError("_mtm_fim_dia: Subetapa 1.2")

    def _detectar_stop_take(self, data: pd.Timestamp) -> list[TradeRegistro]:
        """Detecta stop/take intraday no dia `data` para posições abertas (R1
        Passo 1). Prioridade STOP sobre TAKE no mesmo dia. (Subetapa 1.2.)"""
        raise NotImplementedError("_detectar_stop_take: Subetapa 1.2")

    def _executar_fechamento_orquestrador(
        self, fech, data_decisao: pd.Timestamp
    ) -> Optional[TradeRegistro]:
        """Executa um `FechamentoOrdem` por prazo/reversão na abertura do próximo
        pregão (R1 Passo 3 / R4). (Subetapa 1.2.)"""
        raise NotImplementedError("_executar_fechamento_orquestrador: Subetapa 1.2")

    def _executar_nova_ordem(
        self, ordem, equity_hoje: float
    ) -> Optional[PosicaoInterna]:
        """Executa uma `Ordem` nova: sizing, custo, checagem de caixa e abertura
        da posição (R1 Passo 4 / R6). (Subetapa 1.2.)"""
        raise NotImplementedError("_executar_nova_ordem: Subetapa 1.2")

    def _forcar_fechamento_fim_backtest(
        self, data_fim: pd.Timestamp
    ) -> list[TradeRegistro]:
        """Fecha TODAS as posições abertas no fim do backtest pelo Close[data_fim],
        motivo='fim_backtest' (R8). (Subetapa 1.2.)"""
        raise NotImplementedError("_forcar_fechamento_fim_backtest: Subetapa 1.2")

    def _adicionar_aviso(
        self, tipo: str, data: pd.Timestamp, ticker: str, detalhe: str
    ) -> None:
        """Registra um aviso estruturado em `self._avisos` (R8). (Subetapa 1.2.)"""
        raise NotImplementedError("_adicionar_aviso: Subetapa 1.2")

    def _proximo_pregao_na_janela(
        self, data: pd.Timestamp
    ) -> Optional[pd.Timestamp]:
        """Próximo pregão da B3 estritamente após `data`, dentro da janela do
        backtest; None se não houver (R1 Passo 3). (Subetapa 1.2.)"""
        raise NotImplementedError("_proximo_pregao_na_janela: Subetapa 1.2")
