"""Diagnóstico dirigido: por que o GBM não extrai `score_econ`?

Rebuild só o modo `meta` do run oficial (survivorship por dia, cache warm) e
treina 4 variantes do modelo sobre o MESMO dataset, comparando IC_evento_OOS,
GAP vs. B1, degeneração (np.unique(y_pred)) e o ganho de `score_econ`.

Variantes:
  V1_baseline : platô natural, sem peso         → replica o run oficial (piso)
  V2_argmax   : n_estimators=31 (argmax do meta) → H2 isolada (só platô)
  V3_n200     : n_estimators=200                 → força bruta em H2
  V4_weighted : n_estimators=200 + peso 5x evento → H1 (+H2)

Não gera relatório nem commita. Só imprime tabela no stdout.
Uso: GDELT_THROTTLE_SECONDS=12 python scripts/diagnostico_gbm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as cfg
from agents.journal import JournalAgent
from agents.math_ml import MathMLAgent, make_econ_mock, FEATURES


def _weight_evento_5x(row):
    return 5.0 if row["tem_evento"] else 1.0


def _ic(a, b):
    if len(a) < 3:
        return float("nan")
    ic, _ = spearmanr(a, b)
    return float(ic)  # NaN honesto se predição constante


def main():
    print("→ inicializando JOURNAL (cache warm esperado)...")
    journal = JournalAgent()

    treino_ini = pd.Timestamp("2020-01-02", tz=cfg.FUSO)
    treino_fim = cfg.FIM_TREINO
    oos_ini = pd.Timestamp("2024-01-02", tz=cfg.FUSO)
    oos_fim = cfg.FIM_BACKTEST

    # ds_prep p/ calibração (mesma chamada do run oficial: survivorship por dia)
    print("→ dataset de calibração (ds_prep, mock ic_alvo=0)...")
    ag_prep = MathMLAgent(journal=journal,
                          econ_mock=make_econ_mock(journal, ic_alvo=0.0, seed=42))
    ds_prep = ag_prep.construir_dataset(None, treino_ini, oos_fim)

    print("→ mock meta (ic_alvo=0.15)...")
    mock = make_econ_mock(journal, ic_alvo=0.15, seed=42,
                          amostra_calibracao=ds_prep[["data", "ticker", "y"]])
    print(f"  alpha={mock.alpha:.4f}, ic_realizado={mock.ic_realizado:.4f}")

    print("→ dataset final (meta)...")
    dataset = MathMLAgent(journal=journal, econ_mock=mock).construir_dataset(
        None, treino_ini, oos_fim)
    df_oos = dataset[(dataset["data"] >= oos_ini) & (dataset["data"] <= oos_fim)]
    evt = df_oos["tem_evento"].to_numpy(dtype=bool)
    df_oos_ev = df_oos[evt]
    X_oos = df_oos[FEATURES].to_numpy()
    y_oos = df_oos["y"].to_numpy()
    print(f"  {len(dataset)} linhas total, {len(df_oos)} OOS, "
          f"{int(evt.sum())} eventos OOS")

    ic_b1 = _ic(df_oos_ev["score_econ"].to_numpy(), df_oos_ev["y"].to_numpy())
    print(f"\n📌 REFERÊNCIA: B1 (score_econ sozinho) IC_evento_OOS = {ic_b1:+.4f}")

    variantes = [
        ("V1_baseline", {}),
        ("V2_argmax", {"n_estimators_override": 31}),
        ("V3_n200", {"n_estimators_override": 200}),
        ("V4_weighted", {"n_estimators_override": 200,
                         "sample_weight": _weight_evento_5x}),
    ]

    linhas = []
    for nome, kw in variantes:
        print(f"\n{'=' * 60}\nTREINANDO: {nome}\n{'=' * 60}")
        agent = MathMLAgent(journal=journal, econ_mock=mock)
        agent.treinar(dataset, data_treino_fim=treino_fim, **kw)
        y_pred = agent.model.predict(X_oos)
        ic_total = _ic(y_pred, y_oos)
        ic_evento = _ic(y_pred[evt], y_oos[evt])
        n_unique = int(np.unique(np.round(y_pred, 6)).size)
        imp = pd.Series(agent.model.feature_importances_, index=FEATURES)
        linhas.append({
            "variante": nome,
            "n_est": len(agent.model.estimators_),
            "IC_total": ic_total,
            "IC_evento": ic_evento,
            "GAP_vs_B1": ic_evento - ic_b1,
            "n_unique_pred": n_unique,
            "ganho_score_econ": float(imp.get("score_econ", 0.0)),
        })
        print(f"  n_est={linhas[-1]['n_est']}, IC_evento={ic_evento:+.4f}, "
              f"n_unique={n_unique}, ganho_score_econ={linhas[-1]['ganho_score_econ']:.4f}")

    print(f"\n{'=' * 60}\nRESUMO COMPARATIVO\n{'=' * 60}\n")
    df = pd.DataFrame(linhas)
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:+.4f}"))
    print(f"\n📌 B1 IC_evento = {ic_b1:+.4f} (piso a bater)")
    print("\n🎯 INTERPRETAÇÃO:")
    print("  - V4 bate B1 (IC_evento≳B1) → dilução (H1) domina. Fix: sample_weight.")
    print("  - V2/V3 batem B1 sem peso → platô (H2) sozinho basta.")
    print("  - Nenhuma bate B1 → problema arquitetural (H4).")
    print("  - n_unique_pred < 100 → modelo colapsou p/ quase-constante.")
    print("  - ganho_score_econ > 0.15 → GBM está usando a feature-chave.")


if __name__ == "__main__":
    main()
