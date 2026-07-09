"""ORQUESTRADOR (JEMPO) — coordena JOURNAL/ECON/MATH&ML, toma a decisão final
de compra, aplica gestão de risco da carteira e opera o circuit-breaker de
drawdown.

Posição no pipeline:

    JOURNAL → ECON → MATH&ML → ORQUESTRADOR → PROGRAM
      dados   score   ranking    (este)        backtest

Fronteiras (decisões arquiteturais travadas):
- NÃO calcula P&L nem custos — é do PROGRAM.
- NÃO busca dados brutos e NÃO chama o JOURNAL diretamente. Toda informação
  chega via `MathMLAgent.prever_universo`, `EconAgent.avaliar` e
  `config.tickers_ativos`.
- NÃO retreina o MATH&ML.

Além dos records do contrato público (`OrchestratorConfig`, `Ordem`,
`FechamentoOrdem`, `DecisaoDia`) e do record interno `PosicaoAberta`, expõe o
`OrchestratorAgent`. Nesta etapa o agente tem construtor, bookkeeping de
posições (`notificar_execucao`/`notificar_fechamento`) e introspecção
(`status`). A lógica de decisão (drawdown/seleção/timing/`decidir`) entra nas
etapas seguintes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import time
from typing import Callable, Literal, Optional

import pandas as pd
from pandas.tseries.offsets import BusinessDay

import config

# Alias em escopo de módulo p/ o default do __init__ — evita confusão com o
# parâmetro `config: OrchestratorConfig` (mesmo nome do módulo).
_TICKERS_ATIVOS_DEFAULT = config.tickers_ativos


# ── Configuração (imutável) ───────────────────────────────────────────────────


@dataclass(frozen=True)
class OrchestratorConfig:
    """Parâmetros do ORQUESTRADOR. Todos os defaults vêm das regras travadas
    do desafio (entrada top-N, sizing equal-weight, stop/take, circuit-breaker)."""

    # Filtros de entrada
    score_econ_min: float = 0.30
    volume_relativo_min: float = 1.5

    # Limites de carteira
    max_posicoes: int = 3
    max_por_setor: int = 2
    sizing_pct: float = 0.15  # equal weight

    # Regras de saída
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.15
    prazo_max_dias_uteis: int = 5
    score_reversao: float = -0.30

    # Circuit-breaker de drawdown
    dd_janela_dias_uteis: int = 21
    dd_limite: float = 0.10
    dd_pausa_dias_uteis: int = 5

    # Timing (corte da B3 para decidir D+1 vs D+2)
    hora_corte_b3: time = time(17, 5)


# ── Records de output (contrato público) ──────────────────────────────────────


@dataclass
class Ordem:
    """Uma ordem de compra proposta por `decidir`. O ORQUESTRADOR não executa —
    o PROGRAM executa e chama `notificar_execucao` de volta."""

    ticker: str
    setor: str
    data_decisao: pd.Timestamp
    data_execucao: pd.Timestamp  # D+1 (padrão) ou D+2 (notícia pós-17h05 de D-1)
    rank: int
    y_pred: float
    score_econ: float
    volume_relativo: float
    sizing_pct: float
    motivo_execucao_atrasada: Optional[str]  # "noticia_pos_17h" se D+2, senão None


@dataclass
class FechamentoOrdem:
    """Fechamento de posição detectado pelo ORQUESTRADOR. Cobre apenas `prazo` e
    `reversao`; stop/take são detectados pelo PROGRAM (dependem de intraday)."""

    ticker: str
    motivo: Literal["prazo", "reversao"]
    data_gatilho: pd.Timestamp


@dataclass
class DecisaoDia:
    """Retorno de `decidir` — decisão do dia D. Não efetiva nada; o PROGRAM
    executa e notifica de volta."""

    data: pd.Timestamp
    novas_ordens: list[Ordem]
    fechamentos: list[FechamentoOrdem]
    pausado: bool
    motivo_pausa: Optional[str]  # "drawdown" ou None
    dd_corrente: float  # sempre reportado, para debug
    posicoes_abertas_snapshot: list[str]  # tickers das posições abertas


# ── Record interno (NÃO exposto) ──────────────────────────────────────────────


@dataclass
class PosicaoAberta:
    """Estado interno de uma posição aberta. stop/take são calculados sobre o
    preço REAL de execução (informados a `notificar_execucao`)."""

    ticker: str
    setor: str
    preco_entrada: float
    data_execucao: pd.Timestamp
    stop_price: float  # preco_entrada × (1 - stop_loss_pct)
    take_price: float  # preco_entrada × (1 + take_profit_pct)
    prazo_max: pd.Timestamp  # data_execucao + prazo_max_dias_uteis (dias úteis)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _exigir_aware(ts: pd.Timestamp, nome: str) -> None:
    """Levanta ValueError se `ts` não for timezone-aware.

    Defesa do §12: qualquer Timestamp em argumento público do ORQUESTRADOR deve
    ser tz-aware (America/Sao_Paulo). Naive é bug do chamador (PROGRAM)."""
    if ts is None or getattr(ts, "tzinfo", None) is None:
        raise ValueError(
            f"{nome} deve ser timezone-aware (America/Sao_Paulo); recebeu {ts!r}"
        )


# ── Agente ────────────────────────────────────────────────────────────────────


class OrchestratorAgent:
    """Coordena os agentes do JEMPO e aplica gestão de risco da carteira.

    Consome apenas `math_ml.prever_universo`, `econ.avaliar` e (na etapa de
    `decidir`) `config.tickers_ativos`. Não calcula P&L/custos, não chama o
    JOURNAL diretamente e não retreina o MATH&ML.
    """

    def __init__(
        self,
        journal,
        econ,
        math_ml,
        config: OrchestratorConfig,
        tickers_ativos: Callable[[pd.Timestamp], list[str]] = _TICKERS_ATIVOS_DEFAULT,
    ) -> None:
        """
        Args:
            journal: provedor de dados. Aceito por simetria arquitetural do
                pipeline, mas NUNCA chamado pelo ORQUESTRADOR — o setor chega
                via `prever_universo` (em `decidir`) e via `notificar_execucao`.
            econ: EconAgent. Usado só em `decidir` para checar reversão de sinal
                em posições abertas (`avaliar(ticker, data_limite=...)`).
            math_ml: MathMLAgent, via `prever_universo(tickers, data_limite)`.
            config: OrchestratorConfig imutável.
            tickers_ativos: resolvedor de survivorship `data → list[str]`.
                Default = `config.tickers_ativos` (produção). Injetável para
                testes determinísticos (stub com lista fixa). É a ÚNICA
                dependência de dados fora de MATH&ML/ECON, e é função pura de
                configuração (não é o JOURNAL).
        """
        self._journal = journal
        self._econ = econ
        self._math_ml = math_ml
        self._config = config
        self._tickers_ativos = tickers_ativos
        self._posicoes: dict[str, PosicaoAberta] = {}
        self._equity_series: list[tuple[pd.Timestamp, float]] = []
        self._pausado_ate: Optional[pd.Timestamp] = None
        self._ultima_data_decidida: Optional[pd.Timestamp] = None
        self._dd_corrente: float = 0.0

    # ── notificações do PROGRAM (bookkeeping) ─────────────────────────────────

    def notificar_execucao(
        self,
        ticker: str,
        setor: str,
        preco_execucao: float,
        data_execucao: pd.Timestamp,
    ) -> None:
        """Registra a posição após o PROGRAM executar uma `Ordem` de `decidir`.

        stop/take/prazo são calculados sobre o preço REAL de execução — D+1
        pode ter slippage vs o preço de decisão.

        Args:
            ticker: papel executado.
            setor: setor da posição. Vem da `Ordem` retornada por `decidir`,
                NÃO do JOURNAL (o ORQUESTRADOR não consulta `get_setor`).
            preco_execucao: preço real de execução (> 0).
            data_execucao: data da execução (tz-aware America/Sao_Paulo).

        Raises:
            ValueError: se já houver posição aberta em `ticker` (dupla
                notificação — bug do PROGRAM); se `data_execucao` for naive; ou
                se `preco_execucao <= 0`.
        """
        _exigir_aware(data_execucao, "data_execucao")
        if preco_execucao <= 0:
            raise ValueError(
                f"preco_execucao deve ser > 0; recebeu {preco_execucao!r}"
            )
        if ticker in self._posicoes:
            raise ValueError(
                f"posição já aberta em {ticker!r} — dupla notificação do PROGRAM"
            )
        c = self._config
        preco = float(preco_execucao)
        self._posicoes[ticker] = PosicaoAberta(
            ticker=ticker,
            setor=setor,
            preco_entrada=preco,
            data_execucao=data_execucao,
            stop_price=preco * (1 - c.stop_loss_pct),
            take_price=preco * (1 + c.take_profit_pct),
            prazo_max=data_execucao + BusinessDay(c.prazo_max_dias_uteis),
        )

    def notificar_fechamento(
        self, ticker: str, data_fechamento: pd.Timestamp
    ) -> None:
        """Remove a posição após o PROGRAM fechá-la (por qualquer motivo).

        Args:
            ticker: papel fechado.
            data_fechamento: data do fechamento (tz-aware America/Sao_Paulo).

        Raises:
            ValueError: se não houver posição aberta em `ticker` (bug do
                PROGRAM); ou se `data_fechamento` for naive.
        """
        _exigir_aware(data_fechamento, "data_fechamento")
        if ticker not in self._posicoes:
            raise ValueError(
                f"sem posição aberta em {ticker!r} — fechamento inválido do PROGRAM"
            )
        del self._posicoes[ticker]

    # ── introspecção ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Snapshot do estado interno para logs/debug.

        Introspecção pura — não é decisão e não muta estado. Retorna cópias
        (nada de referências mutáveis ao estado interno).
        """
        return {
            "n_posicoes_abertas": len(self._posicoes),
            "tickers": list(self._posicoes.keys()),
            "dd_corrente": self._dd_corrente,
            "pausado_ate": self._pausado_ate,
            "ultima_data_decidida": self._ultima_data_decidida,
            "n_equity_pontos": len(self._equity_series),
        }

    # ── circuit-breaker de drawdown ───────────────────────────────────────────

    def _atualizar_pausa(self, data: pd.Timestamp, equity_hoje: float) -> float:
        """Registra a equity do dia e atualiza o circuit-breaker de drawdown.

        Corresponde aos passos 2-3 do fluxo de `decidir` (§7). Assume que
        `decidir` já validou `data` (tz-aware, monotônica) e `equity_hoje`
        (> 0) — este método não revalida.

        Mecânica:
        - Append `(data, equity_hoje)` em `_equity_series` (append-only).
        - **Janela = os últimos `dd_janela_dias_uteis` (21) pontos, incluindo
          t.** Se após o append houver < 21 pontos, o circuit-breaker fica
          INATIVO: `dd_corrente = 0.0` e a pausa não é atualizada (primeiros
          20 dias do backtest não têm circuit-breaker).
        - Com janela completa: `pico = max(janela)`,
          `dd = (pico - equity_hoje) / pico`.
        - Nova pausa só se `dd > dd_limite` (estrito) **e** não já pausado
          (`_pausado_ate is None` ou `data >= _pausado_ate`) →
          `_pausado_ate = data + BusinessDay(dd_pausa_dias_uteis)`.
        - **Não estende:** durante pausa ativa (`data < _pausado_ate`) um novo
          gatilho é ignorado — a pausa não "rola".

        NÃO toca `_ultima_data_decidida` (responsabilidade do `decidir`).

        Returns:
            float: o `dd_corrente` do dia (0.0 se janela incompleta).
        """
        c = self._config
        self._equity_series.append((data, float(equity_hoje)))

        if len(self._equity_series) < c.dd_janela_dias_uteis:
            self._dd_corrente = 0.0
            return 0.0

        janela = [v for _, v in self._equity_series[-c.dd_janela_dias_uteis:]]
        pico = max(janela)
        dd = (pico - float(equity_hoje)) / pico
        self._dd_corrente = dd

        ja_pausado = self._pausado_ate is not None and data < self._pausado_ate
        if dd > c.dd_limite and not ja_pausado:
            self._pausado_ate = data + BusinessDay(c.dd_pausa_dias_uteis)
        return dd

    # ── resolução de data de execução (D+1 vs D+2) ────────────────────────────

    def _resolver_data_execucao(
        self, row: pd.Series, data: pd.Timestamp
    ) -> tuple[pd.Timestamp, Optional[str]]:
        """Resolve a data de execução de uma candidata (§4.6 / §7 passo 8.1).

        Regra do corte da B3: a decisão às 10h de D usa dados até o fechamento
        de D-1. Se a notícia mais recente saiu **após 17h05 de D-1**, o candle
        de D-1 já estava contaminado quando ela chegou → a execução é adiada
        para D+2. Caso contrário (inclusive notícia exatamente às 17h05, ou sem
        notícia), executa em D+1.

        `data` pode chegar com hora arbitrária (típico das 10h); normalizamos
        para midnight (a "abertura") antes de subtrair/somar dias úteis. Todas
        as saídas ficam na mesma tz de `data` (sem round-trip para UTC).
        `BusinessDay` já pula sábado/domingo.

        Args:
            row: linha do DataFrame do MATH&ML; usa `data_noticia_mais_recente`
                (tz-aware SP ou NaT).
            data: data da decisão D (tz-aware).

        Returns:
            (data_execucao, motivo_execucao_atrasada) — motivo é
            "noticia_pos_17h" em D+2, None em D+1.
        """
        c = self._config
        base = data.normalize()  # midnight, preserva tz — "abertura" de D
        corte_d_menos_1 = (base - BusinessDay(1)).replace(
            hour=c.hora_corte_b3.hour, minute=c.hora_corte_b3.minute
        )
        data_noticia = row["data_noticia_mais_recente"]
        if pd.isna(data_noticia) or data_noticia <= corte_d_menos_1:
            return base + BusinessDay(1), None
        return base + BusinessDay(2), "noticia_pos_17h"

    # ── seleção de ordens (pool + top-N dinâmico + limite setorial) ───────────

    def _selecionar_ordens(
        self, df: pd.DataFrame, data: pd.Timestamp
    ) -> list[Ordem]:
        """Aplica a regra de entrada travada (§4.1) sobre o DataFrame do
        MATH&ML e devolve as novas ordens do dia.

        Passos (§7 passo 7-8):
        1. `slots_livres = max_posicoes - len(_posicoes)`; se 0, retorna [].
        2. Pool = `score_econ > score_econ_min & volume_relativo >
           volume_relativo_min`, ordenado por `rank` ascendente. NaN reprova
           naturalmente (`NaN > x == False`) — sem tratamento manual.
        3. Loop em ordem de rank: pula ticker já em posição e setor saturado
           (abertas + a comprar ≥ max_por_setor); senão cria `Ordem`,
           incrementa o contador setorial e decrementa `slots_livres`. Para
           quando `slots_livres == 0`.

        NÃO efetiva nada e NÃO resolve reversão/fechamento — só seleção de
        entradas. `data_execucao` de cada ordem vem de `_resolver_data_execucao`.

        Args:
            df: saída de `MathMLAgent.prever_universo` (contrato de 8 colunas).
            data: data da decisão D (tz-aware; validada pelo `decidir`).

        Returns:
            list[Ordem] com no máximo `slots_livres` itens.
        """
        c = self._config
        slots_livres = c.max_posicoes - len(self._posicoes)
        if slots_livres <= 0:
            return []

        pool = df[
            (df["score_econ"] > c.score_econ_min)
            & (df["volume_relativo"] > c.volume_relativo_min)
        ].sort_values("rank")

        setores_alocados: Counter = Counter(
            p.setor for p in self._posicoes.values()
        )
        ordens: list[Ordem] = []
        for _, row in pool.iterrows():
            if slots_livres <= 0:
                break
            ticker = row["ticker"]
            setor = row["setor"]
            if ticker in self._posicoes:
                continue
            if setores_alocados[setor] >= c.max_por_setor:
                continue
            data_execucao, motivo = self._resolver_data_execucao(row, data)
            ordens.append(
                Ordem(
                    ticker=ticker,
                    setor=setor,
                    data_decisao=data,
                    data_execucao=data_execucao,
                    rank=int(row["rank"]),
                    y_pred=float(row["y_pred"]),
                    score_econ=float(row["score_econ"]),
                    volume_relativo=float(row["volume_relativo"]),
                    sizing_pct=c.sizing_pct,
                    motivo_execucao_atrasada=motivo,
                )
            )
            setores_alocados[setor] += 1
            slots_livres -= 1
        return ordens

    # ── fechamentos por prazo / reversão (§7 passo 4) ─────────────────────────

    def _verificar_fechamentos(self, data: pd.Timestamp) -> list[FechamentoOrdem]:
        """Detecta fechamentos de posições abertas por PRAZO ou REVERSÃO.

        Para cada posição aberta, na prioridade **prazo > reversão** (§4.4):
        - Se `data >= pos.prazo_max`: fecha por `"prazo"`.
        - Senão, consulta `econ.avaliar(ticker, data_limite=data)`; se
          `score_total < score_reversao` (-0.30, estrito): fecha por `"reversao"`.
        - Senão: nada para essa posição.

        Se o prazo venceu, o ECON **não** é chamado para essa posição — economiza
        a chamada e garante que prazo prevaleça sobre reversão no mesmo dia.

        stop/take NÃO são avaliados aqui: dependem de intraday (Low/High) e são
        do PROGRAM (§4.4). O ORQUESTRADOR não vê intraday.

        Assume que `decidir` já validou `data` (tz-aware). `econ.avaliar` é
        sempre chamado com `data_limite=data` explícito — defesa anti-lookahead
        do §10 (armadilha 2).

        Args:
            data: data da decisão D (tz-aware).

        Returns:
            list[FechamentoOrdem] — no máximo uma por posição aberta.
        """
        c = self._config
        fechamentos: list[FechamentoOrdem] = []
        for pos in self._posicoes.values():
            if data >= pos.prazo_max:
                fechamentos.append(FechamentoOrdem(pos.ticker, "prazo", data))
                continue
            score = self._econ.avaliar(pos.ticker, data_limite=data)
            if score.score_total < c.score_reversao:
                fechamentos.append(FechamentoOrdem(pos.ticker, "reversao", data))
        return fechamentos

    # ── método principal ──────────────────────────────────────────────────────

    def decidir(self, data: pd.Timestamp, equity_hoje: float) -> DecisaoDia:
        """Decisão do dia D. Chamado pelo PROGRAM uma vez por dia útil.

        NÃO efetiva nada: apenas devolve a `DecisaoDia`. O PROGRAM executa as
        ordens e chama `notificar_execucao`/`notificar_fechamento` de volta.

        Fluxo (ordem estrita do §7):
        1. Valida inputs.
        2. Marca a chamada (`_ultima_data_decidida = data`).
        3. `_atualizar_pausa` (registra equity, atualiza circuit-breaker).
        4. `_verificar_fechamentos` (prazo + reversão via ECON).
        5. Se pausado por drawdown → retorna SEM chamar o MATH&ML.
        6. `config.tickers_ativos(data)` → `math_ml.prever_universo(...)`.
        7. `_selecionar_ordens` (pool + top-N + limite setorial).
        8. Retorna a `DecisaoDia` completa.

        **Divisão de responsabilidade com o PROGRAM (§4.4):** os `fechamentos`
        aqui cobrem só prazo/reversão. stop/take são detectados pelo PROGRAM via
        intraday, e o PROGRAM tem a última palavra — se stop/take dispararam no
        dia D, ele ignora o `FechamentoOrdem` de prazo/reversão do mesmo ticker
        (preserva a prioridade stop > take > prazo > reversão).

        **Premissa anti-lookahead (§10 armadilha 1):** o ORQUESTRADOR pressupõe
        que `equity_hoje` é marcação de fechamento e NÃO depende de decisões
        tomadas no próprio dia D — responsabilidade do PROGRAM.

        Args:
            data: data da decisão D (tz-aware America/Sao_Paulo).
            equity_hoje: capital corrente injetado pelo PROGRAM (> 0). Fonte
                única de verdade sobre equity; o ORQUESTRADOR só faz bookkeeping.

        Returns:
            DecisaoDia com novas ordens, fechamentos, estado da pausa e dd.

        Raises:
            ValueError: se `data` for naive; se `equity_hoje <= 0`; ou se `data`
                não for estritamente posterior à última data decidida (chamada
                repetida ou fora de ordem — bug de loop do PROGRAM).
        """
        # 1 — validar inputs
        _exigir_aware(data, "data")
        if equity_hoje <= 0:
            raise ValueError(f"equity_hoje deve ser > 0; recebeu {equity_hoje!r}")
        if (
            self._ultima_data_decidida is not None
            and data <= self._ultima_data_decidida
        ):
            raise ValueError(
                f"data {data} não é posterior à última decidida "
                f"{self._ultima_data_decidida} — decidir repetido ou fora de "
                "ordem (bug do PROGRAM)"
            )

        # 2 — marca a chamada (antes do resto: a chamada aconteceu, mesmo que um
        #     passo posterior levante)
        self._ultima_data_decidida = data

        # 3 — drawdown + circuit-breaker (usa o dd retornado, sem reler o estado)
        dd_corrente = self._atualizar_pausa(data, equity_hoje)

        # 4 — fechamentos (prazo + reversão)
        fechamentos = self._verificar_fechamentos(data)

        # 5 — pausado por drawdown: bloqueia novas entradas (posições abertas
        #     seguem suas regras). Condição do §7 passo 5: independe do dd do dia
        #     — a pausa dura os 5du inteiros mesmo se o drawdown se recuperar.
        pausado = self._pausado_ate is not None and data < self._pausado_ate
        if pausado:
            return DecisaoDia(
                data=data,
                novas_ordens=[],
                fechamentos=fechamentos,
                pausado=True,
                motivo_pausa="drawdown",
                dd_corrente=dd_corrente,
                posicoes_abertas_snapshot=list(self._posicoes.keys()),
            )

        # 6 — consulta o MATH&ML sobre o universo ativo em D
        universo = self._tickers_ativos(data)
        df = self._math_ml.prever_universo(universo, data_limite=data)

        # 7 — seleção de novas ordens (universo/df vazios → [] naturalmente)
        novas_ordens = self._selecionar_ordens(df, data)

        # 8 — decisão completa
        return DecisaoDia(
            data=data,
            novas_ordens=novas_ordens,
            fechamentos=fechamentos,
            pausado=False,
            motivo_pausa=None,
            dd_corrente=dd_corrente,
            posicoes_abertas_snapshot=list(self._posicoes.keys()),
        )
