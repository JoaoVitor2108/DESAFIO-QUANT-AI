"""Smoke test v2: mais tickers para ver se IC=0.50 era artefato de amostra pequena."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config as cfg
from agents.journal import JournalAgent
from agents.math_ml import MathMLAgent, make_econ_mock


def main():
    inicio = pd.Timestamp("2023-01-02", tz=cfg.FUSO)
    fim    = pd.Timestamp("2023-12-29", tz=cfg.FUSO)

    tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA",
               "B3SA3.SA", "PRIO3.SA", "RENT3.SA", "WEGE3.SA"]

    print(f"-> JOURNAL + {len(tickers)} tickers, range {inicio.date()} a {fim.date()}")
    journal = JournalAgent()

    print("-> dataset calibracao...")
    ag1 = MathMLAgent(journal=journal,
                      econ_mock=make_econ_mock(journal, ic_alvo=0.0, seed=42,
                                               universo=tickers))
    ds1 = ag1.construir_dataset(tickers, inicio, fim)
    print(f"   amostra: {len(ds1)} linhas, {ds1.data.nunique()} datas, {ds1.ticker.nunique()} tickers")

    print("-> mock IC=0.15...")
    mock = make_econ_mock(journal, ic_alvo=0.15, seed=42, universo=tickers,
                          amostra_calibracao=ds1[["data","ticker","y"]])
    print(f"   alpha={mock.alpha:.3f}, ic_realizado={mock.ic_realizado:.3f}")

    print("-> dataset final + treino...")
    agent = MathMLAgent(journal=journal, econ_mock=mock)
    dataset = agent.construir_dataset(tickers, inicio, fim)
    agent.treinar(dataset, data_treino_fim=fim)
    print(f"   n_platau:  {agent.cv_report.get('n_platau')}")
    print(f"   n_argmax:  {agent.cv_report.get('n_argmax')}")

    print("-> avaliando IC...")
    out = agent.avaliar_ic(dataset)
    print(f"   IC_total:  {out['IC_total']:+.4f}")
    print(f"   IC_evento: {out['IC_evento']:+.4f}")
    print(f"   IC95:      [{out['IC95'][0]:+.4f}, {out['IC95'][1]:+.4f}]")
    print(f"   Baselines:")
    for k, v in out["baselines"].items():
        print(f"     {k}: {v:+.4f}")
    print(f"   GAP_total:  {out['GAP_total']:+.4f}")
    print(f"   GAP_evento: {out['GAP_evento']:+.4f}")
    print(f"   n: {out['n']}")
    print(f"   beta_fallback count: {dataset['beta_fallback'].sum()}/{len(dataset)}")


if __name__ == "__main__":
    main()
