"""Smoke test manual do ORQUESTRADOR — 10 dias sequenciais fora de teste unitário.

Roda `decidir()` para 10 dias úteis com cenários variados (pool vazio, top-N por
setor, slot único, reversão, D+2 por notícia >17h05, fechamento por prazo),
simulando o loop do PROGRAM (`notificar_execucao`/`notificar_fechamento`).
Serve como sanity check antes do commit e como demo executável para a banca.

Fakes locais (NÃO importa de tests/ — script de demonstração, não código de
produção). Rode com: `python scripts/smoke_test_orquestrador.py`.

Nota de coerência: uma posição só sai de `_posicoes` quando o PROGRAM chama
`notificar_fechamento`, o que ocorre DEPOIS do `decidir`. Logo, um slot liberado
por fechamento no dia D só fica disponível no dia D+1. O roteiro abaixo respeita
isso — daí a ordem dos cenários.
"""

import sys
from pathlib import Path

# adiciona a raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from agents.econ import ScoreEcon
from agents.orchestrator import OrchestratorAgent, OrchestratorConfig

FUSO = "America/Sao_Paulo"

_COLS_MATHML = [
    "ticker", "y_pred", "score_econ", "tem_evento", "rank",
    "volume_relativo", "data_noticia_mais_recente", "setor",
]

SETOR = {
    "PETR4.SA": "Energia",
    "ITUB4.SA": "Bancos",
    "VALE3.SA": "Mineração",
    "MGLU3.SA": "Varejo",
    "WEGE3.SA": "Bens de Capital",
}


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


# ── Fakes locais ──────────────────────────────────────────────────────────────


class FakeJournal:
    """Trivial: argumento do __init__, nunca chamado pelo ORQUESTRADOR."""


def _mk_score(score_total: float) -> ScoreEcon:
    return ScoreEcon(
        ticker="X", data_referencia=_ts("2024-01-01"), score_total=score_total,
        comp_noticia=score_total, comp_saude_financeira=0.0, comp_setorial=0.0,
        comp_macro=0.0, confianca=0.8, tem_evento=True, n_noticias=1,
        justificativa="smoke", modelo="fake",
    )


class FakeEcon:
    """avaliar(ticker, data_limite) → ScoreEcon com reversões agendadas por data."""

    def __init__(self, reversoes: dict[pd.Timestamp, set[str]]):
        # {data (midnight): {tickers que revertem naquele dia}}
        self._reversoes = reversoes

    def avaliar(self, ticker, data_limite):
        revertidos = self._reversoes.get(data_limite.normalize(), set())
        return _mk_score(-0.5 if ticker in revertidos else 0.0)


class FakeMathML:
    """prever_universo(tickers, data_limite) → DataFrame pré-fabricado por data."""

    def __init__(self, df_por_data: dict[pd.Timestamp, pd.DataFrame]):
        self._df_por_data = df_por_data

    def prever_universo(self, tickers, data_limite):
        if data_limite in self._df_por_data:
            return self._df_por_data[data_limite].copy()
        return pd.DataFrame(columns=_COLS_MATHML)


def _row(ticker, rank, data_noticia=pd.NaT):
    """Linha do contrato de 8 colunas. score/volume passam nos filtros."""
    return {
        "ticker": ticker, "y_pred": -0.01 * rank, "score_econ": 0.9,
        "tem_evento": True, "rank": rank, "volume_relativo": 2.0,
        "data_noticia_mais_recente": data_noticia, "setor": SETOR[ticker],
    }


def _df(rows):
    return pd.DataFrame(rows, columns=_COLS_MATHML)


def _tickers_ativos_stub(data):
    return list(SETOR.keys())


# ── Impressão ─────────────────────────────────────────────────────────────────


def _print_dia(n, dec):
    pausa = f"pausado={dec.pausado}"
    if dec.pausado:
        pausa += f" ({dec.motivo_pausa})"
    print(f"\n=== Dia {n} ({dec.data.date()}) ===")
    print(f"Estado: dd={dec.dd_corrente * 100:.2f}%, {pausa}, "
          f"{len(dec.posicoes_abertas_snapshot)} posições abertas "
          f"{dec.posicoes_abertas_snapshot}")
    if dec.novas_ordens:
        print("Novas ordens:")
        for o in dec.novas_ordens:
            flag = ("  [D+2: notícia >17h05]"
                    if o.motivo_execucao_atrasada else "")
            print(f"  {o.ticker:<9} ({o.setor:<15}) sizing={o.sizing_pct:.0%}"
                  f"  exec={o.data_execucao.date()}  rank={o.rank}{flag}")
    else:
        print("Novas ordens: (nenhuma)")
    if dec.fechamentos:
        print("Fechamentos:")
        for f in dec.fechamentos:
            print(f"  {f.ticker:<9} motivo={f.motivo}")
    else:
        print("Fechamentos: (nenhum)")


# ── Cenário ───────────────────────────────────────────────────────────────────


def main():
    dias = pd.bdate_range("2024-03-04", periods=10, tz=FUSO)  # 04..15/03 (úteis)

    # Reversões agendadas (dia → tickers): VALE3 no dia 5, ITUB4 no dia 6.
    # Liberam slots para os cenários seguintes (o slot só abre no dia após o
    # fechamento, pois o PROGRAM notifica DEPOIS do decidir).
    reversoes = {
        dias[4]: {"VALE3.SA"},   # 08/03 → carteira cai de 3 para 2
        dias[5]: {"ITUB4.SA"},   # 11/03 → carteira cai de 2 para 1
    }

    # Cenários de MATH&ML por dia (dias sem entrada = df vazio).
    cenarios = {
        dias[2]: _df([_row("PETR4.SA", 1), _row("ITUB4.SA", 2),
                      _row("VALE3.SA", 3)]),                    # 3 setores distintos
        dias[3]: _df([_row("MGLU3.SA", 1)]),                   # carteira cheia → 0
        dias[6]: _df([_row("VALE3.SA", 1),                     # 1 pos aberta → 2 slots
                      _row("WEGE3.SA", 2,                       # D+2 (notícia >17h05)
                           data_noticia=_ts("2024-03-11 18:00"))]),
        dias[9]: _df([_row("MGLU3.SA", 1)]),                   # 1 slot livre → 1
    }

    journal = FakeJournal()
    econ = FakeEcon(reversoes)
    math_ml = FakeMathML(cenarios)
    agent = OrchestratorAgent(journal, econ, math_ml, OrchestratorConfig(),
                              tickers_ativos=_tickers_ativos_stub)

    print("SMOKE TEST — ORQUESTRADOR (10 dias sequenciais)")
    print("Universo:", ", ".join(f"{t} [{s}]" for t, s in SETOR.items()))

    for i, d in enumerate(dias, start=1):
        dec = agent.decidir(d, equity_hoje=100_000.0)
        _print_dia(i, dec)
        # loop do PROGRAM: fecha o que decidiu fechar, executa as novas ordens
        for f in dec.fechamentos:
            agent.notificar_fechamento(f.ticker, dec.data)
        for o in dec.novas_ordens:
            agent.notificar_execucao(o.ticker, o.setor, 100.0, o.data_execucao)

    print("\n=== status() final ===")
    st = agent.status()
    for k, v in st.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
