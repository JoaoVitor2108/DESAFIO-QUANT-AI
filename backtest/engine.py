"""PROGRAM (JEMPO) — motor de backtest event-driven.

Posição no pipeline:

    JOURNAL → ECON → MATH&ML → ORQUESTRADOR → PROGRAM
      dados   score   ranking   decisão       (este) backtest

O PROGRAM consome as decisões do ORQUESTRADOR (`decidir`) e as executa numa
simulação diária com custos, detecção intraday de stop/take, mark-to-market e
compounding. Devolve um `ResultadoBacktest` — objeto de DADOS puro. Métricas de
performance (Sharpe, MDD, Monte Carlo, plots) NÃO ficam aqui; virão em etapas
seguintes (`backtest/metrics.py`, `backtest/monte_carlo.py`, `backtest/plots.py`).

Fronteira tz: PROGRAM opera internamente com datas naive (sem timezone). As
chamadas aos contratos do ORQUESTRADOR e JOURNAL — que exigem tz-aware
America/Sao_Paulo — passam pelo adaptador `_para_sp` na saída e são normalizadas
de volta a naive na leitura (via `_get_precos`). Interior do engine não vê tz.

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
- Determinismo: zero aleatoriedade; iterações sobre posições em ordem alfabética
  de ticker.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from agents.journal import DadoIndisponivel, LookaheadError

# Fuso do sistema JEMPO (tz-aware exigido pelos contratos travados).
_FUSO = "America/Sao_Paulo"


# ── Adaptador de fronteira tz (funções puras) ─────────────────────────────────


def _para_sp(data: pd.Timestamp) -> pd.Timestamp:
    """Adaptador de fronteira SAÍDA: normaliza uma data naive do PROGRAM para
    tz-aware `America/Sao_Paulo`, como exigido pelos contratos travados do
    ORQUESTRADOR (`_exigir_aware`) e do JOURNAL (`_validate_aware`).

    Usa `tz_localize` (interpreta a hora como HORA DE PAREDE em SP), nunca
    `tz_convert` — evita o bug clássico de deslocar 10:00 → 07:00. Idempotente:
    se já for tz-aware, apenas converte para SP.
    """
    ts = pd.Timestamp(data)
    return ts.tz_localize(_FUSO) if ts.tzinfo is None else ts.tz_convert(_FUSO)


def _para_naive(data: pd.Timestamp) -> pd.Timestamp:
    """Adaptador de fronteira ENTRADA: normaliza uma data que possa vir tz-aware
    (ex.: `Ordem.data_execucao` do ORQUESTRADOR real) para naive, preservando a
    hora de parede. Idempotente."""
    ts = pd.Timestamp(data)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


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
    o `journal.get_precos` (sempre via `_get_precos`, nunca direto).
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
                `_get_precos` normaliza para naive na leitura.
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
        self._data_inicio: Optional[pd.Timestamp] = None
        self._data_fim: Optional[pd.Timestamp] = None

    # ── Calendário B3 (R5) ────────────────────────────────────────────────────

    def _calendario_bmf(
        self, data_inicio: pd.Timestamp, data_fim: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """Dias de pregão da B3 entre `data_inicio` e `data_fim` (inclusive).

        Usa `pandas_market_calendars` com o calendário 'BMF', que conhece os
        feriados nacionais (Sexta-feira Santa, Corpus Christi, Finados, etc.) —
        ao contrário de `pd.bdate_range`, que só pula sábado/domingo (R5).

        O índice retornado é NAIVE (`.tz is None`), por convenção do PROGRAM.

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

    # ── Acesso a preços — choke-point único do JOURNAL (adaptador tz) ─────────

    def _get_precos(
        self,
        ticker: str,
        data_inicio: pd.Timestamp,
        data_limite: pd.Timestamp,
    ) -> pd.DataFrame:
        """ÚNICO ponto de acesso ao `journal.get_precos`. Adaptador de fronteira
        nos dois sentidos:

        - ENTRADA: localiza `data_inicio`/`data_limite` em SP (o JOURNAL exige
          tz-aware).
        - SAÍDA: normaliza o índice do DataFrame de volta para naive, para o
          interior do engine comparar contra o calendário (naive).

        Robustez (R8): `DadoIndisponivel`/`LookaheadError` (exceções tipadas do
        JOURNAL) viram DataFrame vazio — o chamador loga aviso e segue. NUNCA usa
        `try/except` genérico.

        Nenhum outro método do engine chama `self._journal.get_precos` direto.
        """
        try:
            df = self._journal.get_precos(
                ticker, _para_sp(data_inicio), _para_sp(data_limite)
            )
        except (DadoIndisponivel, LookaheadError):
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def _barra_exata(
        self, df: pd.DataFrame, data: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Linha OHLCV do dia EXATO `data` (naive) no DataFrame, ou None se
        ausente (feriado do ticker / gap / delisting)."""
        if df.empty:
            return None
        alvo = pd.Timestamp(data).normalize()
        sel = df[df.index.normalize() == alvo]
        if sel.empty:
            return None
        return sel.iloc[-1]

    def _close_marcacao(
        self, df: pd.DataFrame, data: pd.Timestamp
    ) -> Optional[float]:
        """Close para mark-to-market em `data`: o Close EXATO do dia se existir;
        senão o último Close disponível ≤ `data` (carry-forward, anti-lookahead);
        None se não houver barra alguma até `data`."""
        if df.empty:
            return None
        exata = self._barra_exata(df, data)
        if exata is not None:
            return float(exata["Close"])
        prior = df[df.index <= pd.Timestamp(data)]
        if prior.empty:
            return None
        return float(prior["Close"].iloc[-1])

    # ── Mark-to-market (R2) ───────────────────────────────────────────────────

    def _mtm_fim_dia(self, data: pd.Timestamp) -> float:
        """Mark-to-market ao fim do dia `data`: caixa livre + Σ qtd_i × Close[data]_i
        para todas as posições internas abertas.

        Convenção travada (R2): `equity_hoje` passado ao ORQUESTRADOR na decisão
        de D é `_mtm_fim_dia(D-1)` — calculado ANTES do Passo 1, para nunca ler
        preço ≥ D na decisão (anti-lookahead). No 1º dia usa `capital_inicial`.

        Anti-lookahead auditável (R2): cada leitura pede `data_limite=data`, e o
        assert abaixo garante que NENHUM Close com índice > `data` entra no
        cálculo. A garantia de nível de decisão vem de o chamador passar D-1.
        """
        equity = self._caixa_livre
        for ticker in sorted(self._posicoes_internas):
            pos = self._posicoes_internas[ticker]
            df = self._get_precos(ticker, self._data_inicio, data)
            assert df.empty or df.index.max() <= pd.Timestamp(data), (
                f"lookahead em _mtm_fim_dia({data}): {ticker} tem índice "
                f"{df.index.max()} > {data}"
            )
            close = self._close_marcacao(df, data)
            if close is None:
                self._adicionar_aviso(
                    "mtm_sem_preco", data, ticker,
                    "sem Close para marcar a mercado; usando preço de entrada",
                )
                close = pos.preco_entrada
            equity += pos.qtd * close
        return equity

    # ── Passo 1 — detecção intraday de stop/take (R1/R4) ──────────────────────

    def _detectar_stop_take(self, data: pd.Timestamp) -> list[TradeRegistro]:
        """Detecta stop/take intraday no dia `data` para as posições abertas ao
        fim de D-1, iterando em ORDEM ALFABÉTICA de ticker (R9).

        Para cada posição: se `Low[data] ≤ stop_price` → fecha por 'stop' (preço
        de saída = stop_price exato); elif `High[data] ≥ take_price` → fecha por
        'take' (preço = take_price exato). Ambiguidade stop E take no mesmo dia:
        prioridade STOP (worst-case conservador, R1). Ticker sem barra em `data`:
        loga aviso e MANTÉM a posição aberta (R8).

        Executa o fechamento (caixa, TradeRegistro, `notificar_fechamento`,
        remoção da posição) e devolve os trades gerados no dia.
        """
        c = self._config
        trades: list[TradeRegistro] = []
        for ticker in sorted(self._posicoes_internas):
            pos = self._posicoes_internas[ticker]
            df = self._get_precos(ticker, self._data_inicio, data)
            barra = self._barra_exata(df, data)
            if barra is None:
                self._adicionar_aviso(
                    "sem_dados_intraday", data, ticker,
                    "sem barra intraday em D; posição mantida aberta",
                )
                continue
            low = float(barra["Low"])
            high = float(barra["High"])
            if low <= pos.stop_price:
                preco_saida, motivo = pos.stop_price, "stop"
            elif high >= pos.take_price:
                preco_saida, motivo = pos.take_price, "take"
            else:
                continue
            trade = self._fechar_posicao(pos, preco_saida, data, motivo)
            del self._posicoes_internas[ticker]
            self._orq.notificar_fechamento(ticker, _para_sp(data))
            trades.append(trade)
        # `c` referenciado só para deixar claro que custos vêm do config no
        # helper `_fechar_posicao`; nenhuma constante mágica aqui.
        del c
        return trades

    # ── Passo 3 — fechamentos do ORQUESTRADOR (prazo/reversão) (R1/R4) ─────────

    def _executar_fechamento_orquestrador(
        self, fech, data_decisao: pd.Timestamp
    ) -> Optional[TradeRegistro]:
        """Executa um `FechamentoOrdem` (prazo/reversão) na abertura do próximo
        pregão dentro da janela (Open[D+1]); se não houver próximo pregão, força
        por Close[data_decisao] com aviso, preservando o motivo (R1 Passo 3/R4).

        INVARIANTE (R1 Passo 3): um ticker fechado por stop/take no Passo 1 já foi
        removido das posições internas e notificado ao ORQUESTRADOR — logo NÃO
        pode reaparecer aqui. Se aparecer, é violação de contrato → `RuntimeError`
        com mensagem clara (ticker + data).
        """
        ticker = fech.ticker
        if ticker not in self._posicoes_internas:
            raise RuntimeError(
                f"ORQUESTRADOR pediu fechamento de {ticker!r} em "
                f"{pd.Timestamp(data_decisao).date()}, mas o PROGRAM não tem essa "
                "posição — provável stop/take intraday no mesmo dia já a fechou no "
                "Passo 1. Invariante R1/Passo 3 violada."
            )
        pos = self._posicoes_internas[ticker]
        data_efetiva = self._proximo_pregao_na_janela(data_decisao)

        preco_saida: Optional[float] = None
        data_saida = data_decisao
        if data_efetiva is not None:
            df = self._get_precos(ticker, self._data_inicio, data_efetiva)
            barra = self._barra_exata(df, data_efetiva)
            if barra is not None:
                preco_saida = float(barra["Open"])
                data_saida = data_efetiva
            else:
                self._adicionar_aviso(
                    "fechamento_sem_open", data_decisao, ticker,
                    f"sem Open em {pd.Timestamp(data_efetiva).date()}; forçado por "
                    f"Close[{pd.Timestamp(data_decisao).date()}]",
                )
        else:
            self._adicionar_aviso(
                "fechamento_sem_proximo_pregao", data_decisao, ticker,
                "sem próximo pregão na janela; forçado por Close",
            )

        if preco_saida is None:
            preco_saida, data_saida = self._preco_forcado_close(
                ticker, data_decisao, pos
            )

        trade = self._fechar_posicao(pos, preco_saida, data_saida, fech.motivo)
        del self._posicoes_internas[ticker]
        self._orq.notificar_fechamento(ticker, _para_sp(data_saida))
        return trade

    def _preco_forcado_close(
        self, ticker: str, data: pd.Timestamp, pos: PosicaoInterna
    ) -> tuple[float, pd.Timestamp]:
        """Preço de fechamento forçado = Close[data] (ou carry-forward ≤ data;
        preço de entrada em último caso, com aviso). Retorna (preco, data)."""
        df = self._get_precos(ticker, self._data_inicio, data)
        close = self._close_marcacao(df, data)
        if close is None:
            self._adicionar_aviso(
                "fechamento_sem_preco", data, ticker,
                "sem Close para forçar fechamento; usando preço de entrada",
            )
            close = pos.preco_entrada
        return close, data

    # ── Passo 4 — nova ordem (sizing, custos, caixa) (R1/R3/R6) ────────────────

    def _executar_nova_ordem(
        self, ordem, equity_hoje: float
    ) -> Optional[PosicaoInterna]:
        """Executa uma `Ordem` do ORQUESTRADOR: sizing em `equity_hoje` (MTM do
        fim de D-1, R2/R6), preço de entrada = Open[data_execucao], custo de
        entrada de 0.4% do notional (R3), checagem de caixa (não fura o caixa,
        R1) e abertura da posição com stop/take (R4). Notifica o ORQUESTRADOR ao
        fim (R1 Passo 4). Devolve a `PosicaoInterna` ou None se a ordem foi pulada.
        """
        c = self._config
        ticker = ordem.ticker
        data_exec = _para_naive(ordem.data_execucao).normalize()

        df = self._get_precos(ticker, self._data_inicio, data_exec)
        barra = self._barra_exata(df, data_exec)
        if barra is None:
            self._adicionar_aviso(
                "sem_dados_execucao", data_exec, ticker,
                f"sem Open em {data_exec.date()}; ordem não executada",
            )
            return None
        preco_entrada = float(barra["Open"])
        if preco_entrada <= 0:
            self._adicionar_aviso(
                "preco_invalido", data_exec, ticker,
                f"Open inválido ({preco_entrada}); ordem não executada",
            )
            return None

        qtd = int((c.sizing_pct * equity_hoje) / preco_entrada)
        if qtd == 0:
            self._adicionar_aviso(
                "preco_alto_qtd_zero", data_exec, ticker,
                f"sizing {c.sizing_pct:.0%}×{equity_hoje:.2f}/{preco_entrada:.2f} "
                "< 1 ação; ordem pulada",
            )
            return None

        notional = qtd * preco_entrada
        custo_entrada = c.custo_perna * notional
        if self._caixa_livre < notional + custo_entrada:
            self._adicionar_aviso(
                "caixa_insuficiente", data_exec, ticker,
                f"caixa {self._caixa_livre:.2f} < notional+custo "
                f"{notional + custo_entrada:.2f}; ordem pulada",
            )
            return None

        self._caixa_livre -= notional + custo_entrada
        pos = PosicaoInterna(
            ticker=ticker,
            setor=ordem.setor,
            qtd=qtd,
            preco_entrada=preco_entrada,
            data_entrada=data_exec,
            stop_price=(1 - c.stop_pct) * preco_entrada,
            take_price=(1 + c.take_pct) * preco_entrada,
            custo_entrada=custo_entrada,
            y_pred=ordem.y_pred,
            score_econ=ordem.score_econ,
            rank=ordem.rank,
        )
        self._posicoes_internas[ticker] = pos
        self._orq.notificar_execucao(
            ticker, ordem.setor, preco_entrada, _para_sp(data_exec)
        )
        return pos

    # ── R8 — fechamento forçado no fim do backtest ─────────────────────────────

    def _forcar_fechamento_fim_backtest(
        self, data_fim: pd.Timestamp
    ) -> list[TradeRegistro]:
        """Fecha TODAS as posições ainda abertas pelo Close[data_fim], motivo
        'fim_backtest', em ordem alfabética de ticker (R8/R9). Custo de saída
        aplicado normalmente; ORQUESTRADOR notificado."""
        trades: list[TradeRegistro] = []
        for ticker in sorted(self._posicoes_internas):
            pos = self._posicoes_internas[ticker]
            df = self._get_precos(ticker, self._data_inicio, data_fim)
            close = self._close_marcacao(df, data_fim)
            if close is None:
                self._adicionar_aviso(
                    "fim_backtest_sem_preco", data_fim, ticker,
                    "sem Close no fim do backtest; usando preço de entrada",
                )
                close = pos.preco_entrada
            trade = self._fechar_posicao(pos, close, data_fim, "fim_backtest")
            del self._posicoes_internas[ticker]
            self._orq.notificar_fechamento(ticker, _para_sp(data_fim))
            trades.append(trade)
        return trades

    # ── Helpers de fechamento / avisos / calendário ────────────────────────────

    def _fechar_posicao(
        self,
        pos: PosicaoInterna,
        preco_saida: float,
        data_saida: pd.Timestamp,
        motivo: str,
    ) -> TradeRegistro:
        """Fecha uma posição: calcula custo de saída (0.4% do notional de saída,
        R3), P&L bruto/líquido, atualiza `caixa_livre` (+ qtd×preço − custo) e
        devolve o `TradeRegistro`. NÃO remove a posição nem notifica — o chamador
        controla a ordem dessas ações (difere entre stop/take, prazo e fim)."""
        c = self._config
        qtd = pos.qtd
        custo_saida = c.custo_perna * preco_saida * qtd
        pnl_bruto = (preco_saida - pos.preco_entrada) * qtd
        pnl_liquido = pnl_bruto - pos.custo_entrada - custo_saida
        self._caixa_livre += qtd * preco_saida - custo_saida
        return TradeRegistro(
            ticker=pos.ticker,
            setor=pos.setor,
            qtd=qtd,
            preco_entrada=pos.preco_entrada,
            preco_saida=preco_saida,
            data_entrada=pos.data_entrada,
            data_saida=data_saida,
            motivo=motivo,
            custo_entrada=pos.custo_entrada,
            custo_saida=custo_saida,
            pnl_bruto=pnl_bruto,
            pnl_liquido=pnl_liquido,
            y_pred_entrada=pos.y_pred,
            score_econ_entrada=pos.score_econ,
            rank_entrada=pos.rank,
            dias_uteis_ate_saida=self._dias_uteis_ate_saida(
                pos.data_entrada, data_saida
            ),
        )

    def _dias_uteis_ate_saida(
        self, data_entrada: pd.Timestamp, data_saida: pd.Timestamp
    ) -> int:
        """Nº de pregões da B3 no intervalo (data_entrada, data_saida], usando o
        calendário do backtest. 0 quando entrada e saída caem no mesmo pregão
        (ex.: stop no dia da entrada)."""
        cal = self._calendario
        de = pd.Timestamp(data_entrada).normalize()
        ds = pd.Timestamp(data_saida).normalize()
        return int(((cal > de) & (cal <= ds)).sum())

    def _adicionar_aviso(
        self, tipo: str, data: pd.Timestamp, ticker: str, detalhe: str
    ) -> None:
        """Registra um aviso estruturado (R8). Não interrompe o backtest."""
        self._avisos.append(
            {"tipo": tipo, "data": data, "ticker": ticker, "detalhe": detalhe}
        )

    def _proximo_pregao_na_janela(
        self, data: pd.Timestamp
    ) -> Optional[pd.Timestamp]:
        """Próximo pregão da B3 estritamente após `data`, dentro da janela do
        backtest (R1 Passo 3). None se não houver (ex.: `data` = último pregão)."""
        cal = self._calendario
        posteriores = cal[cal > pd.Timestamp(data).normalize()]
        return posteriores[0] if len(posteriores) else None

    # ── Loop principal (R1) ────────────────────────────────────────────────────

    def rodar_backtest(
        self,
        data_inicio: pd.Timestamp,
        data_fim: pd.Timestamp,
    ) -> ResultadoBacktest:
        """Roda o backtest entre `data_inicio` e `data_fim` inclusive.

        Fluxo por dia útil D (R1): (0) `equity_hoje` = mark-to-market do fim de
        D-1 (`_mtm_fim_dia(D-1)`) — calculado ANTES de qualquer leitura intraday
        de D (anti-lookahead, R2); no 1º dia = `capital_inicial`. (1) detecção
        intraday de stop/take; (2) `orq.decidir(D, equity_hoje)` — `equity_hoje`
        ainda é do fim de D-1, intencional; (3) execução dos fechamentos por
        prazo/reversão; (4) execução das novas ordens (sizing sobre o mesmo
        `equity_hoje` de D-1); (5) registro do equity do fim de D.

        Ao fim, fecha todas as posições remanescentes por Close[data_fim]
        (motivo 'fim_backtest', R8). Determinístico (R9).
        """
        # Estado inicial (R7). Datas do PROGRAM são naive (normalizadas a midnight).
        self._data_inicio = _para_naive(data_inicio).normalize()
        self._data_fim = _para_naive(data_fim).normalize()
        self._caixa_livre = self._config.capital_inicial
        self._posicoes_internas = {}
        self._trades_fechados = []
        self._equity_diario = []
        self._avisos = []
        self._calendario = self._calendario_bmf(self._data_inicio, self._data_fim)
        cal = self._calendario

        for i, dia in enumerate(cal):
            # Passo 0 — equity_hoje = MTM do fim de D-1 (R2). 1º dia = capital.
            if i == 0:
                equity_hoje = self._config.capital_inicial
            else:
                equity_hoje = self._mtm_fim_dia(cal[i - 1])

            # Passo 1 — stop/take intraday (antes de decidir).
            self._trades_fechados.extend(self._detectar_stop_take(dia))

            # Passo 2 — decisão do ORQUESTRADOR (equity_hoje = fim de D-1).
            dec = self._orq.decidir(_para_sp(dia), equity_hoje)

            # Passo 3 — fechamentos por prazo/reversão, na ordem entregue (R9).
            for fech in dec.fechamentos:
                trade = self._executar_fechamento_orquestrador(fech, dia)
                if trade is not None:
                    self._trades_fechados.append(trade)

            # Passo 4 — novas ordens, na ordem entregue (R9).
            for ordem in dec.novas_ordens:
                self._executar_nova_ordem(ordem, equity_hoje)

            # Passo 5 — registra equity do fim de D.
            equity_fim = self._mtm_fim_dia(dia)
            self._equity_diario.append((dia, equity_fim))

        # R8 — fecha o que sobrou por Close[data_fim].
        self._trades_fechados.extend(
            self._forcar_fechamento_fim_backtest(self._data_fim)
        )

        return self._montar_resultado()

    def _montar_resultado(self) -> ResultadoBacktest:
        """Empacota o estado final num `ResultadoBacktest` (objeto de dados)."""
        colunas = [f.name for f in dataclasses.fields(TradeRegistro)]
        if self._trades_fechados:
            trades_df = pd.DataFrame(
                [dataclasses.asdict(t) for t in self._trades_fechados]
            )[colunas]
        else:
            trades_df = pd.DataFrame(columns=colunas)

        if self._equity_diario:
            equity_series = pd.Series(
                data=[v for _, v in self._equity_diario],
                index=pd.DatetimeIndex([d for d, _ in self._equity_diario]),
                name="equity",
            )
        else:
            equity_series = pd.Series(dtype=float, name="equity")

        return ResultadoBacktest(
            trades=trades_df,
            equity_diario=equity_series,
            avisos=list(self._avisos),
            config=self._config,
            data_inicio=self._data_inicio,
            data_fim=self._data_fim,
            n_dias_uteis=len(self._calendario),
            n_trades=len(self._trades_fechados),
            capital_final=self._caixa_livre,
        )
