"""Smoke test: MATH&ML contra dados REAIS do JOURNAL + mock do ECON.

Rode uma vez para confirmar que o pipeline funciona end-to-end fora dos testes.
NÃO é o run de sensibilidade final — é só o smoke.
"""
import sys
from pathlib import Path

# adiciona a raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config as project_config  # constantes globais (FUSO, períodos, universo)
from agents.journal import JournalAgent
from agents.math_ml import MathMLAgent, make_econ_mock


def main():
    # Range pequeno pra ser rápido: 6 meses de 2023
    inicio = pd.Timestamp("2023-01-02", tz=project_config.FUSO)
    fim    = pd.Timestamp("2023-06-30", tz=project_config.FUSO)
    tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]  # 3 do core do IBOV

    print("→ inicializando JOURNAL (real, com cache em disco)...")
    journal = JournalAgent()

    print("→ montando amostra de calibração do mock (subset do range)...")
    agent_temp = MathMLAgent(
        journal=journal,
        econ_mock=make_econ_mock(journal, ic_alvo=0.0, seed=42, universo=tickers),
        # config omitido → usa MathMLConfig() com defaults
    )
    ds_calib = agent_temp.construir_dataset(tickers, inicio, fim)
    print(f"  amostra: {len(ds_calib)} linhas, {ds_calib.data.nunique()} datas")

    print("→ criando mock 'meta' (ic_alvo=0.15) calibrado contra a amostra...")
    mock = make_econ_mock(
        journal, ic_alvo=0.15, seed=42, universo=tickers,
        amostra_calibracao=ds_calib[["data", "ticker", "y"]],
    )
    print(f"  alpha={mock.alpha:.3f}, ic_realizado={mock.ic_realizado:.3f}")

    print("→ construindo dataset final + treinando...")
    agent = MathMLAgent(journal=journal, econ_mock=mock)
    dataset = agent.construir_dataset(tickers, inicio, fim)
    agent.treinar(dataset, data_treino_fim=fim)
    print(f"  n_estimators escolhido (platô): {agent.cv_report.get('n_platau')}")
    print(f"  n_estimators argmax (comparação): {agent.cv_report.get('n_argmax')}")

    print("→ avaliando IC + baselines + GAP...")
    out = agent.avaliar_ic(dataset)
    print(f"  IC_total:  {out['IC_total']:+.4f}")
    print(f"  IC_evento: {out['IC_evento']:+.4f}")
    print(f"  IC95:      [{out['IC95'][0]:+.4f}, {out['IC95'][1]:+.4f}] "
          f"({out['IC95_method']})")
    print(f"  Baselines:")
    for k, v in out["baselines"].items():
        print(f"    {k}: {v:+.4f}")
    print(f"  GAP_total:  {out['GAP_total']:+.4f}")
    print(f"  GAP_evento: {out['GAP_evento']:+.4f}")
    print(f"  n: {out['n']}")

    print("→ ranking do universo em uma data final...")
    ranking = agent.prever_universo(tickers, data_limite=fim)
    print(ranking.to_string(index=False))

    print("\n✅ smoke test OK — pipeline funciona com dados reais do JOURNAL.")


if __name__ == "__main__":
    main()