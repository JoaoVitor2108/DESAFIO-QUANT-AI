"""Testes determinísticos do gate de custo Haiku vs Sonnet.

Cobrem apenas as funções PURAS novas do gate (block bootstrap por data, ΔIC
pareado, critério de decisão, atribuição de blocos). Nada toca a API — os
helpers de IC/beta/target já são testados em tests/test_econ_calibration.py.
"""

import numpy as np
import pandas as pd
import pytest

import calibration.gate_custo_haiku_vs_sonnet as gate
from calibration.gate_custo_haiku_vs_sonnet import (
    _Checkpoint,
    _avaliar_com_retry,
    adicionar_blocos,
    bootstrap_delta_ic_pareado,
    bootstrap_ic_bloco,
    decidir_modelo,
)

FUSO = "America/Sao_Paulo"


def _ts(s):
    return pd.Timestamp(s, tz=FUSO)


# ── Atribuição de blocos (5 dias úteis) ───────────────────────────────────────


def test_adicionar_blocos_agrupa_5_dias_uteis():
    """Datas dentro de 5 pregões caem no mesmo bloco; +5 dias úteis → próximo bloco."""
    datas = [_ts("2025-07-01"), _ts("2025-07-04"), _ts("2025-07-08")]
    df = pd.DataFrame({"data": datas})
    out = adicionar_blocos(df, "data")
    # 07-01 e 07-04 (≤4 dias úteis de distância) → bloco 0; 07-08 (5 úteis) → bloco 1
    assert out.loc[0, "bloco"] == out.loc[1, "bloco"] == 0
    assert out.loc[2, "bloco"] == 1


# ── Block bootstrap por data ──────────────────────────────────────────────────


def test_bootstrap_bloco_por_data():
    """Sanity + determinismo: com score == y (correlação perfeita), o IC de ponto e
    todo reamostra-por-bloco dão 1.0, então IC95 = [1, 1]. Mesma seed = idêntico."""
    df = pd.DataFrame({
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "bloco": [0, 0, 0, 1, 1, 1],
    })
    r1 = bootstrap_ic_bloco(df, "score", "y", n_iter=2000, seed=7)
    r2 = bootstrap_ic_bloco(df, "score", "y", n_iter=2000, seed=7)
    assert r1["ic"] == pytest.approx(1.0)
    assert r1["ic95_low"] == pytest.approx(1.0)
    assert r1["ic95_high"] == pytest.approx(1.0)
    assert r1["n_blocos"] == 2
    # determinismo byte-a-byte
    assert (r1["ic95_low"], r1["ic95_high"]) == (r2["ic95_low"], r2["ic95_high"])


def test_ic_conjunto_preserva_pareamento():
    """ΔIC pareado usa os MESMOS blocos reamostrados para os dois modelos. Se
    score_haiku == score_sonnet, cada iteração dá ic_sonnet - ic_haiku == 0
    exatamente → ΔIC de ponto 0 e IC95 [0, 0]. Um bootstrap NÃO pareado (índices
    independentes) quase nunca daria 0 exato."""
    rng = np.random.default_rng(0)
    n = 40
    scores = rng.normal(size=n)
    y = rng.normal(size=n)
    df = pd.DataFrame({
        "score_haiku": scores,
        "score_sonnet": scores,  # idênticos → pareamento força Δ=0
        "y": y,
        "bloco": np.repeat(np.arange(8), 5),
    })
    r = bootstrap_delta_ic_pareado(df, "score_haiku", "score_sonnet", "y",
                                   n_iter=2000, seed=11)
    assert r["delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["ic95_low"] == pytest.approx(0.0, abs=1e-12)
    assert r["ic95_high"] == pytest.approx(0.0, abs=1e-12)


# ── Critério de decisão ───────────────────────────────────────────────────────


def test_criterio_decisao_gap_grande_vai_para_sonnet():
    """ΔIC > 0.05 → Sonnet."""
    assert decidir_modelo(0.06) == "sonnet"
    assert decidir_modelo(0.051) == "sonnet"


def test_criterio_decisao_zona_cinza_vai_para_haiku():
    """0.03 ≤ ΔIC ≤ 0.05 → Haiku (default custo-benefício); ΔIC < 0.03 → Haiku."""
    assert decidir_modelo(0.05) == "haiku"   # limite superior da zona cinza
    assert decidir_modelo(0.04) == "haiku"   # meio da zona cinza
    assert decidir_modelo(0.03) == "haiku"   # limite inferior da zona cinza
    assert decidir_modelo(0.02) == "haiku"   # abaixo da zona cinza
    assert decidir_modelo(-0.10) == "haiku"  # Sonnet pior


# ── Retry com backoff (sem API) ───────────────────────────────────────────────


class _FakeScore:
    score_total = 0.5


class _FakeAgent:
    """Agent falso: falha nas primeiras `falhas` chamadas, depois retorna score."""

    def __init__(self, falhas):
        self.falhas = falhas
        self.chamadas = 0

    def avaliar(self, ticker, data, noticias_override=None):
        self.chamadas += 1
        if self.chamadas <= self.falhas:
            raise RuntimeError("erro transitório simulado")
        return _FakeScore()


def test_avaliar_com_retry_reevalua_ate_sucesso(monkeypatch):
    monkeypatch.setattr(gate.time, "sleep", lambda s: None)  # não dormir de verdade
    ev = {"ticker": "PETR4.SA", "data": _ts("2025-10-10"), "noticias": []}
    agent = _FakeAgent(falhas=2)  # falha 2x → sucesso na 3ª tentativa
    score = _avaliar_com_retry(agent, ev, max_tentativas=3)
    assert score.score_total == 0.5
    assert agent.chamadas == 3


def test_avaliar_com_retry_desiste_apos_max(monkeypatch):
    monkeypatch.setattr(gate.time, "sleep", lambda s: None)
    ev = {"ticker": "PETR4.SA", "data": _ts("2025-10-10"), "noticias": []}
    agent = _FakeAgent(falhas=99)
    with pytest.raises(RuntimeError):
        _avaliar_com_retry(agent, ev, max_tentativas=3)
    assert agent.chamadas == 3  # exatamente max_tentativas, não mais


# ── Checkpoint incremental / resume (sem API) ─────────────────────────────────


def test_checkpoint_resume_pula_ja_feitos(tmp_path):
    path = tmp_path / "inter.csv"
    pd.DataFrame([{
        "evento_id": 0, "ticker": "PETR4.SA", "data": "2025-10-10T00:00:00-03:00",
        "y_realizado": 0.01, "modelo": "claude-haiku-4-5-20251001", "score": 0.3,
        "latencia_llm_s": 1.2, "tokens_in": 100, "tokens_out": 20, "custo_usd": 0.0002,
    }], columns=_Checkpoint.COLS).to_csv(path, index=False)

    cp = _Checkpoint(path)
    assert cp.ja_feito("claude-haiku-4-5-20251001", 0)
    assert not cp.ja_feito("claude-sonnet-4-6", 0)      # outro modelo, mesmo evento
    assert not cp.ja_feito("claude-haiku-4-5-20251001", 1)  # outro evento

    # registrar novo + flush deve persistir e ser lido numa nova instância.
    cp.registrar({
        "evento_id": 1, "ticker": "VALE3.SA", "data": "2025-10-11T00:00:00-03:00",
        "y_realizado": -0.01, "modelo": "claude-haiku-4-5-20251001", "score": -0.2,
        "latencia_llm_s": 1.0, "tokens_in": 90, "tokens_out": 15, "custo_usd": 0.0001,
    })
    cp.flush()
    recarregado = _Checkpoint(path)
    assert recarregado.ja_feito("claude-haiku-4-5-20251001", 1)
