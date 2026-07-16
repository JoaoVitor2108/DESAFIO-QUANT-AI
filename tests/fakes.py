"""Fakes ESTRITOS para os testes do PROGRAM (BacktestEngine).

Espelham o padrão determinístico de `tests/test_orchestrator.py` e replicam os
CONTRATOS TRAVADOS dos agentes reais — em particular a exigência de timestamps
tz-aware `America/Sao_Paulo` nas fronteiras públicas:

- `FakeJournalBacktest.get_precos` levanta `ValueError` se receber limites naive
  (replica `_validate_aware` do JOURNAL) e devolve DataFrame com índice tz-aware
  SP (replica a saída real). Ticker desconhecido → `DadoIndisponivel`.
- `FakeOrquestrador.decidir/notificar_execucao/notificar_fechamento` levantam
  `ValueError` se receberem data naive (replica `_exigir_aware` do ORQUESTRADOR).

Com Fakes estritos, um call-site do engine que esqueça o adaptador `_para_sp`
FALHA o teste em vez de passar silenciosamente em "naive-land".
"""

from __future__ import annotations

import pandas as pd

from agents.journal import DadoIndisponivel
from agents.orchestrator import DecisaoDia, FechamentoOrdem, Ordem

FUSO = "America/Sao_Paulo"


# ── Builders determinísticos ──────────────────────────────────────────────────


def preco_df(
    datas: list[str],
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float] | None = None,
) -> pd.DataFrame:
    """Monta um DataFrame OHLCV indexado por datas NAIVE (o Fake localiza em SP
    na entrega). Todas as listas devem ter o mesmo comprimento."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in datas])
    n = len(idx)
    vol = volume if volume is not None else [1_000_000.0] * n
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def mk_ordem(
    ticker: str,
    data_execucao: pd.Timestamp,
    setor: str = "Energia",
    rank: int = 1,
    y_pred: float = 0.05,
    score_econ: float = 0.5,
    data_decisao: pd.Timestamp | None = None,
) -> Ordem:
    """Ordem do contrato do ORQUESTRADOR com defaults enxutos para os testes."""
    return Ordem(
        ticker=ticker,
        setor=setor,
        data_decisao=data_decisao if data_decisao is not None else data_execucao,
        data_execucao=data_execucao,
        rank=rank,
        y_pred=y_pred,
        score_econ=score_econ,
        volume_relativo=2.0,
        sizing_pct=0.15,
        motivo_execucao_atrasada=None,
    )


def mk_fechamento(
    ticker: str, motivo: str, data_gatilho: pd.Timestamp
) -> FechamentoOrdem:
    """FechamentoOrdem (motivo em {'prazo','reversao'})."""
    return FechamentoOrdem(ticker=ticker, motivo=motivo, data_gatilho=data_gatilho)


def mk_decisao(
    data: pd.Timestamp,
    novas_ordens: list[Ordem] | None = None,
    fechamentos: list[FechamentoOrdem] | None = None,
) -> DecisaoDia:
    """DecisaoDia do contrato do ORQUESTRADOR, campos de risco neutros."""
    return DecisaoDia(
        data=data,
        novas_ordens=novas_ordens or [],
        fechamentos=fechamentos or [],
        pausado=False,
        motivo_pausa=None,
        dd_corrente=0.0,
        posicoes_abertas_snapshot=[],
    )


# ── Fakes estritos ────────────────────────────────────────────────────────────


class FakeJournalBacktest:
    """Provedor de preços determinístico. Replica o contrato do JOURNAL real:
    exige limites tz-aware e devolve índice tz-aware SP. Ticker desconhecido
    levanta `DadoIndisponivel` (como o yfinance vazio no real)."""

    def __init__(
        self,
        dados: dict[str, pd.DataFrame] | None = None,
        setores: dict[str, str] | None = None,
    ) -> None:
        # Armazena com índice naive; localiza em SP na entrega.
        self._dados = {t: df.copy() for t, df in (dados or {}).items()}
        self._setores = setores or {}
        self.chamadas: list[tuple] = []  # (ticker, data_inicio, data_fim)

    def get_precos(
        self,
        ticker: str,
        data_inicio: pd.Timestamp,
        data_fim: pd.Timestamp,
    ) -> pd.DataFrame:
        if getattr(data_inicio, "tzinfo", None) is None or getattr(
            data_fim, "tzinfo", None
        ) is None:
            raise ValueError(
                "FakeJournalBacktest.get_precos exige timestamps tz-aware "
                f"(recebeu data_inicio={data_inicio!r}, data_fim={data_fim!r})"
            )
        self.chamadas.append((ticker, data_inicio, data_fim))
        if ticker not in self._dados or self._dados[ticker].empty:
            raise DadoIndisponivel(f"FakeJournalBacktest: sem dados para {ticker!r}")
        df = self._dados[ticker].copy()
        df.index = df.index.tz_localize(FUSO)  # replica saída tz-aware SP do real
        mask = (df.index >= data_inicio) & (df.index <= data_fim)
        return df.loc[mask]

    def get_setor(self, ticker: str) -> str:
        return self._setores.get(ticker, "SetorMock")


class FakeOrquestrador:
    """Playback determinístico de `DecisaoDia` por data. Não roda MATH&ML/ECON.
    Replica a exigência de tz-aware das fronteiras públicas do ORQUESTRADOR e
    registra todas as chamadas (com um log unificado e ordenado) para asserts."""

    def __init__(self, cronograma: dict[pd.Timestamp, DecisaoDia] | None = None) -> None:
        for chave in (cronograma or {}):
            if getattr(chave, "tzinfo", None) is None:
                raise ValueError(
                    "FakeOrquestrador: chaves do cronograma devem ser tz-aware SP "
                    f"(recebeu {chave!r})"
                )
        self._cronograma = cronograma or {}
        self.decidir_calls: list[tuple] = []      # (data, equity_hoje)
        self.execucoes: list[tuple] = []          # (ticker, setor, preco, data)
        self.fechamentos_notificados: list[tuple] = []  # (ticker, data)
        self.log: list[tuple] = []                # sequência unificada e ordenada

    def decidir(self, data: pd.Timestamp, equity_hoje: float) -> DecisaoDia:
        if getattr(data, "tzinfo", None) is None:
            raise ValueError(f"decidir exige data tz-aware; recebeu {data!r}")
        self.decidir_calls.append((data, equity_hoje))
        self.log.append(("decidir", data, equity_hoje))
        dec = self._cronograma.get(data)
        if dec is None:
            return mk_decisao(data)
        return dec

    def notificar_execucao(
        self,
        ticker: str,
        setor: str,
        preco_execucao: float,
        data_execucao: pd.Timestamp,
    ) -> None:
        if getattr(data_execucao, "tzinfo", None) is None:
            raise ValueError(
                f"notificar_execucao exige data_execucao tz-aware; recebeu {data_execucao!r}"
            )
        self.execucoes.append((ticker, setor, preco_execucao, data_execucao))
        self.log.append(("execucao", ticker, setor, preco_execucao, data_execucao))

    def notificar_fechamento(
        self, ticker: str, data_fechamento: pd.Timestamp
    ) -> None:
        if getattr(data_fechamento, "tzinfo", None) is None:
            raise ValueError(
                f"notificar_fechamento exige data_fechamento tz-aware; recebeu {data_fechamento!r}"
            )
        self.fechamentos_notificados.append((ticker, data_fechamento))
        self.log.append(("fechamento", ticker, data_fechamento))

    def status(self) -> dict:
        return {
            "n_decidir": len(self.decidir_calls),
            "n_execucoes": len(self.execucoes),
            "n_fechamentos": len(self.fechamentos_notificados),
        }
