"""
Testes do MathMLAgent — determinísticos e OFFLINE (sem rede, sem API Anthropic).

O JOURNAL é substituído por um fake fixturizado (`FakeJournal`) que devolve
preços/fundamentos/macro/Ibov sintéticos; o ECON entra via `make_econ_mock`.
Execute: pytest tests/test_math_ml.py -v
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from config import FUSO
from agents.journal import Fundamentals
from agents.econ import ScoreEcon
from agents.math_ml import (
    MathMLAgent,
    MathMLConfig,
    make_econ_mock,
    PurgedTimeSeriesSplit,
    LookaheadError,
    FEATURES,
    _block_bootstrap_ic,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


def _dias_uteis(inicio: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=inicio, periods=n, tz=FUSO)


def _frame_precos(idx: pd.DatetimeIndex, close_raw: np.ndarray,
                  volume: np.ndarray | None = None) -> pd.DataFrame:
    if volume is None:
        volume = np.full(len(idx), 1_000_000.0)
    return pd.DataFrame(
        {
            "Open": close_raw,
            "High": close_raw,
            "Low": close_raw,
            "Close": close_raw,
            "Close_raw": close_raw,
            "Volume": volume,
            "flag_qualidade": [""] * len(idx),
        },
        index=idx,
    )


class FakeJournal:
    """JOURNAL offline: devolve séries pré-montadas, recortadas em data_limite.

    `fundamentals[ticker]` pode ser um `Fundamentals` fixo OU um callable
    `(data_limite) -> Fundamentals` (para testar variação temporal sem tocar o
    contrato do JOURNAL real).
    """

    def __init__(self, precos: dict[str, pd.DataFrame], ibov: pd.DataFrame,
                 fundamentals: dict | None = None,
                 macro: dict[str, pd.Series] | None = None,
                 setor: str = "Setor Teste"):
        self._precos = precos
        self._ibov = ibov
        self._fund = fundamentals or {}
        self._macro = macro
        self._setor = setor

    def get_precos(self, ticker, data_inicio, data_limite, preencher_gaps=False):
        df = self._ibov if ticker == "^BVSP" else self._precos[ticker]
        return df.loc[(df.index >= data_inicio) & (df.index <= data_limite)].copy()

    def get_fundamentals(self, ticker, data_limite):
        f = self._fund.get(ticker)
        if callable(f):
            return f(data_limite)
        if f is not None:
            return f
        return Fundamentals(ticker=ticker, data_referencia=data_limite,
                            setor=self._setor)

    def get_macro(self, data_limite):
        if self._macro is None:
            idx = pd.bdate_range(end=data_limite, periods=60, tz=FUSO)
            return {
                "selic_meta": pd.Series(np.full(60, 11.0), index=idx),
                "ptax_usdbrl": pd.Series(np.full(60, 5.0), index=idx),
            }
        return {k: v.loc[v.index <= data_limite] for k, v in self._macro.items()}

    def get_setor(self, ticker):
        return self._setor


def _random_walk(idx, seed, start=100.0, sigma=0.02):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, sigma, len(idx))
    return start * np.exp(np.cumsum(rets))


def _journal_random(tickers, n_dias=320, inicio="2020-01-01", seed0=0,
                    ibov_flat=False):
    idx = _dias_uteis(inicio, n_dias)
    precos = {}
    for i, t in enumerate(tickers):
        precos[t] = _frame_precos(idx, _random_walk(idx, seed0 + i + 1))
    ibov_vals = (np.full(n_dias, 1000.0) if ibov_flat
                 else _random_walk(idx, seed0 + 999, start=1000.0, sigma=0.01))
    ibov = _frame_precos(idx, ibov_vals)
    return FakeJournal(precos, ibov), idx, tickers


def _amostra(idx_slice, tickers) -> pd.DataFrame:
    """DataFrame ('data','ticker') p/ calibrar o mock (y é recomputado internamente)."""
    return pd.DataFrame([(d, t) for t in tickers for d in idx_slice],
                        columns=["data", "ticker"])


def _cfg(**kw):
    base = dict(n_splits_cv=2, embargo=5,
                gb_params=dict(max_depth=3, learning_rate=0.1, subsample=1.0,
                               n_estimators=40, random_state=42))
    base.update(kw)
    return MathMLConfig(**base)


# ── 1. Alvo beta-ajustado ─────────────────────────────────────────────────────


def test_alvo_beta_ajustado():
    # Ação = 2x o Ibov (beta≈2) + um salto idiossincrático conhecido na janela fwd.
    idx = _dias_uteis("2020-01-01", 300)
    rng = np.random.default_rng(7)
    r_ibov = rng.normal(0, 0.01, len(idx))
    r_acao = 2.0 * r_ibov.copy()
    t_pos = 270
    r_acao[t_pos + 1] += 0.03  # salto idiossincrático fora da janela do beta
    ibov_px = 1000.0 * np.exp(np.cumsum(r_ibov))
    acao_px = 100.0 * np.exp(np.cumsum(r_acao))
    jr = FakeJournal({"AAAA": _frame_precos(idx, acao_px)},
                     _frame_precos(idx, ibov_px))
    agent = MathMLAgent(journal=jr, econ_mock=make_econ_mock(jr, ic_alvo=0.0))

    t = idx[t_pos]
    info = agent._alvo("AAAA", t)
    assert info["beta"] == pytest.approx(2.0, abs=0.05)   # ddof consistente (polyfit)
    assert info["beta_fallback"] is False
    r_acao_fwd = acao_px[t_pos + 5] / acao_px[t_pos] - 1
    r_ibov_fwd = ibov_px[t_pos + 5] / ibov_px[t_pos] - 1
    esperado = r_acao_fwd - info["beta"] * r_ibov_fwd
    assert info["y"] == pytest.approx(esperado, abs=1e-9)
    assert info["y"] == pytest.approx(0.03, abs=0.01)


# ── 2. Beta fallback com poucos dados ─────────────────────────────────────────


def test_beta_fallback_poucos_dados():
    idx = _dias_uteis("2020-01-01", 60)  # < min_obs_beta
    jr = FakeJournal({"AAAA": _frame_precos(idx, _random_walk(idx, 1))},
                     _frame_precos(idx, _random_walk(idx, 2, start=1000.0)))
    agent = MathMLAgent(journal=jr, econ_mock=make_econ_mock(jr, ic_alvo=0.0))
    info = agent._alvo("AAAA", idx[50])
    assert info["beta"] == 1.0
    assert info["beta_fallback"] is True


# ── 3. _assert_no_lookahead cobre 4 fontes ────────────────────────────────────


def test_assert_no_lookahead_cobre_4_fontes():
    jr, idx, tickers = _journal_random(["AAAA", "BBBB"], n_dias=300)
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=tickers))
    t = idx[270]
    fut = idx[290]
    # linha legítima não levanta
    feats = agent._montar_features("AAAA", t, verificar_lookahead=True)
    assert set(FEATURES).issubset(feats.keys())

    # 1) preço futuro
    with pytest.raises(LookaheadError):
        agent._assert_no_lookahead(t, precos=pd.DataFrame({"Close_raw": [1.0]}, index=[fut]))
    # 2) fundamento (DT_RECEB) futuro
    with pytest.raises(LookaheadError):
        agent._assert_no_lookahead(t, fund=Fundamentals(
            ticker="AAAA", data_referencia=t, data_recebimento_cvm=fut))
    # 3) macro com índice futuro
    with pytest.raises(LookaheadError):
        agent._assert_no_lookahead(t, macro={"selic_meta": pd.Series([11.0], index=[fut])})
    # 4) ScoreEcon com data_referencia futura
    with pytest.raises(LookaheadError):
        agent._assert_no_lookahead(t, score=ScoreEcon(
            ticker="AAAA", data_referencia=fut, score_total=0.0, comp_noticia=0.0,
            comp_saude_financeira=0.0, comp_setorial=0.0, comp_macro=0.0,
            confianca=0.5, tem_evento=True, n_noticias=1,
            justificativa="x", modelo="m"))


# ── 4. Label incompleto é dropado ─────────────────────────────────────────────


def test_label_incompleto_dropado():
    jr, idx, tickers = _journal_random(["AAAA"], n_dias=280)
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=tickers))
    ds = agent.construir_dataset(["AAAA"], idx[255], idx[-1])
    assert ds["data"].max() <= idx[-1 - agent.config.horizonte]
    assert ds["y"].notna().all()


# ── 5. Purged split: sem overlap + embargo ────────────────────────────────────


def test_purged_split_sem_overlap():
    n = 200
    datas = np.arange(n)
    pts = PurgedTimeSeriesSplit(n_splits=4, horizonte=5, embargo=5)
    folds = list(pts.split(n, datas))
    assert len(folds) >= 1
    for tr, te in folds:
        assert datas[tr].max() + 5 + 5 <= datas[te].min()
        assert set(tr).isdisjoint(set(te))


# ── 6. Fundamento None imputado cross-sectional ───────────────────────────────


def test_fundamental_none_imputado_cross_sectional():
    idx = _dias_uteis("2020-01-01", 300)
    tickers = ["AAAA", "BBBB", "CCCC"]
    precos = {t: _frame_precos(idx, _random_walk(idx, i + 1))
              for i, t in enumerate(tickers)}
    ibov = _frame_precos(idx, _random_walk(idx, 50, start=1000.0))
    fund = {
        "AAAA": Fundamentals(ticker="AAAA", data_referencia=idx[-1], pl=10.0, pvp=2.0),
        "BBBB": Fundamentals(ticker="BBBB", data_referencia=idx[-1], pl=20.0, pvp=4.0),
        "CCCC": Fundamentals(ticker="CCCC", data_referencia=idx[-1], pl=None, pvp=None),
    }
    jr = FakeJournal(precos, ibov, fundamentals=fund)
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=tickers))
    ds = agent.construir_dataset(tickers, idx[270], idx[272])
    cc = ds[ds["ticker"] == "CCCC"]
    assert len(cc) > 0
    assert (cc["pl"] == 15.0).all()              # mediana do dia entre {10,20}
    assert cc["fundamental_imputado"].all()
    assert not ds[ds["ticker"] == "AAAA"]["fundamental_imputado"].any()


# ── 7. Os dois zeros do ECON ──────────────────────────────────────────────────


def test_dois_zeros_do_econ():
    jr, idx, tickers = _journal_random(["AAAA"], n_dias=300)
    t = idx[270]

    def _se(conf):
        return ScoreEcon(ticker="AAAA", data_referencia=t, score_total=0.0,
                         comp_noticia=0.0, comp_saude_financeira=0.0,
                         comp_setorial=0.0, comp_macro=0.0, confianca=conf,
                         tem_evento=True, n_noticias=2,
                         justificativa="t", modelo="m")

    a1 = MathMLAgent(journal=jr, econ_mock=lambda *_: _se(0.6))
    a2 = MathMLAgent(journal=jr, econ_mock=lambda *_: _se(0.0))
    assert a1._montar_features("AAAA", t)["econ_degradado"] is False
    assert a2._montar_features("AAAA", t)["econ_degradado"] is True


# ── 8. Mock calibra IC + distribuição unimodal ────────────────────────────────


def test_mock_calibra_ic():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(8)], n_dias=320)
    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0,
                          amostra_calibracao=_amostra(idx[260:300], tickers),
                          universo=tickers)
    assert abs(mock.ic_realizado - 0.15) < 0.02
    # sem reescala bimodal: existem scores de magnitude pequena (5º pct < 0.3)
    scores = np.abs([mock(tk, d).score_total
                     for tk in tickers for d in idx[260:300]])
    assert np.percentile(scores, 5) < 0.3
    assert mock.fallback_minimo_universo_count == 0


# ── 9. Mock ruído: pipeline não inventa sinal ─────────────────────────────────


def test_mock_ruido_degrada():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(8)], n_dias=320)
    mock = make_econ_mock(jr, ic_alvo=0.0, prob_evento=1.0,
                          amostra_calibracao=_amostra(idx[260:300], tickers),
                          universo=tickers)
    assert abs(mock.ic_realizado) < 0.05
    agent = MathMLAgent(journal=jr, econ_mock=mock, config=_cfg())
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    corte = idx[280]
    agent.treinar(ds[ds["data"] <= corte], data_treino_fim=corte)
    oos = ds[ds["data"] > idx[286]]  # embargo entre treino e teste
    res = agent.avaliar_ic(oos)
    assert abs(res["IC_total"]) < 0.15
    for v in res["baselines"].values():       # baselines também ~0 com ruído
        assert abs(v) < 0.3


# ── 10. prever_universo ordena ────────────────────────────────────────────────


def test_prever_universo_ordena():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(6)], n_dias=320)
    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0,
                          amostra_calibracao=_amostra(idx[260:300], tickers),
                          universo=tickers)
    agent = MathMLAgent(journal=jr, econ_mock=mock, config=_cfg())
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    agent.treinar(ds, data_treino_fim=idx[-1 - agent.config.horizonte])
    out = agent.prever_universo(tickers, idx[300])
    assert {"ticker", "y_pred", "score_econ", "tem_evento", "rank"}.issubset(out.columns)
    assert out["y_pred"].is_monotonic_decreasing
    assert list(out["rank"]) == list(range(1, len(out) + 1))


# ── 11. Walk-forward não vaza ─────────────────────────────────────────────────


def test_walk_forward_nao_vaza():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(5)], n_dias=400)
    mock = make_econ_mock(jr, ic_alvo=0.1, prob_evento=1.0,
                          amostra_calibracao=_amostra(idx[260:320], tickers),
                          universo=tickers)
    agent = MathMLAgent(journal=jr, econ_mock=mock, config=_cfg())
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    painel = agent.walk_forward(ds, idx[330], idx[-1], freq="MS")
    assert len(painel) > 0
    assert {"data", "ticker", "y_pred", "y_real"}.issubset(painel.columns)
    for fold in agent._wf_folds:
        assert fold["treino_fim_ord"] + agent.config.horizonte + agent.config.embargo \
            <= fold["teste_inicio_ord"]


# ── 12. Checagem de sinal das features ────────────────────────────────────────


def test_sinal_features_smoke():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(8)], n_dias=340)
    mock = make_econ_mock(jr, ic_alvo=0.35, prob_evento=1.0,
                          amostra_calibracao=_amostra(idx[260:320], tickers),
                          universo=tickers)
    agent = MathMLAgent(journal=jr, econ_mock=mock, config=_cfg())
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    agent.treinar(ds, data_treino_fim=idx[-1 - agent.config.horizonte])
    imp = agent.importancia_features()
    assert imp.iloc[0]["feature"] == "score_econ"
    linha = imp[imp["feature"] == "score_econ"].iloc[0]
    assert linha["sinal_observado"] >= 0
    assert bool(linha["sinal_invertido"]) is False


# ── 13. Mock sem fallback fora da amostra (z(y) dinâmico generaliza) ───────────


def test_mock_sem_fallback_fora_amostra():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(8)], n_dias=345)
    D1 = idx[230:290]          # calibração
    D2 = idx[295:335]          # avaliação, DISJUNTA de D1
    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0,
                          amostra_calibracao=_amostra(D1, tickers), universo=tickers)
    agent = MathMLAgent(journal=jr, econ_mock=mock)
    scores, ys = [], []
    for t in D2:
        for tk in tickers:
            y = agent._alvo(tk, t)["y"]
            if not np.isnan(y):
                scores.append(mock(tk, t).score_total)
                ys.append(y)
    ic_d2 = spearmanr(scores, ys).statistic
    assert mock.fallback_minimo_universo_count == 0       # nunca caiu no fallback
    assert abs(ic_d2 - 0.15) < 0.06                       # propriedade generaliza (OOS do alpha)


# ── 14. Block bootstrap é mais largo que i.i.d. em painel correlacionado ───────


def test_block_bootstrap_intervalo_mais_largo():
    rng = np.random.default_rng(0)
    n_dias, n_tk = 60, 8
    datas = np.repeat(np.arange(n_dias), n_tk)
    eps_dia = np.repeat(rng.normal(0, 1, n_dias), n_tk)   # choque comum do dia
    y_real = eps_dia + rng.normal(0, 0.1, len(datas))
    y_pred = y_real + rng.normal(0, 0.5, len(datas))      # correlacionado com y
    block = _block_bootstrap_ic(y_pred, y_real, datas, n_boot=500)
    block_w = block["ic_p97_5"] - block["ic_p2_5"]
    # i.i.d. de pares (inline, só para comparação)
    r2 = np.random.default_rng(42)
    n = len(y_real)
    ics = [spearmanr(y_pred[s], y_real[s]).statistic
           for s in (r2.integers(0, n, n) for _ in range(500))]
    iid_w = np.percentile(ics, 97.5) - np.percentile(ics, 2.5)
    assert block_w >= 1.3 * iid_w


# ── 15. crescimento_lucro_yoy valor correto ───────────────────────────────────


def test_crescimento_lucro_yoy_valor_correto():
    idx = _dias_uteis("2020-01-01", 300)
    t = idx[280]
    corte = t - pd.Timedelta(days=200)

    def fund_aaaa(dl):  # 120 recente, 100 há um ano → +20%
        return Fundamentals(ticker="AAAA", data_referencia=dl,
                            lucro_liquido=120.0 if dl >= corte else 100.0)

    def fund_bbbb(dl):  # virada prejuízo(-100) → lucro(50): (50-(-100))/100 = 1.5
        return Fundamentals(ticker="BBBB", data_referencia=dl,
                            lucro_liquido=50.0 if dl >= corte else -100.0)

    precos = {t_: _frame_precos(idx, _random_walk(idx, i + 1))
              for i, t_ in enumerate(["AAAA", "BBBB"])}
    jr = FakeJournal(precos, _frame_precos(idx, _random_walk(idx, 9, start=1000.0)),
                     fundamentals={"AAAA": fund_aaaa, "BBBB": fund_bbbb})
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=["AAAA", "BBBB"]))
    assert agent._montar_features("AAAA", t)["crescimento_lucro_yoy"] == pytest.approx(0.20)
    assert agent._montar_features("BBBB", t)["crescimento_lucro_yoy"] == pytest.approx(1.5)


# ── 16. Fallback de imputação usa só o treino ─────────────────────────────────


def test_fallback_imputacao_usa_so_treino():
    jr, idx, tickers = _journal_random(["AAAA"], n_dias=60)
    agent = MathMLAgent(journal=jr, econ_mock=make_econ_mock(jr, ic_alvo=0.0))
    d_treino = [ts("2020-01-06"), ts("2020-01-07")]
    d_oos_extremo = ts("2024-01-08")
    d_oos_nan = ts("2024-02-08")
    rows = []
    for d in d_treino:                       # treino: mom_12_1 ∈ {1,3} → mediana 2.0
        for v in (1.0, 3.0):
            r = {f: 0.0 for f in FEATURES}; r["mom_12_1"] = v; r["data"] = d
            rows.append(r)
    for _ in range(2):                       # OOS com valores EXTREMOS (puxam global)
        r = {f: 0.0 for f in FEATURES}; r["mom_12_1"] = 100.0; r["data"] = d_oos_extremo
        rows.append(r)
    for _ in range(2):                       # OOS com a coluna toda-NaN naquele dia
        r = {f: 0.0 for f in FEATURES}; r["mom_12_1"] = np.nan; r["data"] = d_oos_nan
        rows.append(r)
    df = pd.DataFrame(rows)
    medianas_treino = df[df["data"].isin(d_treino)][FEATURES].median()  # mom=2.0
    out = agent._imputar_cross_sectional(df, medianas_treino=medianas_treino)
    nan_dia = out[out["data"] == d_oos_nan]["mom_12_1"]
    assert (nan_dia == 2.0).all()            # mediana do TREINO
    assert (nan_dia != 51.5).all()           # NÃO a mediana global (median[1,3,100,100])


# ── 17. Baselines competitivos corretos ───────────────────────────────────────


def test_baselines_estao_corretos():
    idx = _dias_uteis("2020-01-01", 320)
    tickers = [f"T{i}" for i in range(8)]
    precos = {}
    for i, tk in enumerate(tickers):
        d = 0.0005 * (i + 1)                         # drift distinto por ticker
        precos[tk] = _frame_precos(idx, 100.0 * np.exp(d * np.arange(len(idx))))
    ibov = _frame_precos(idx, np.full(len(idx), 1000.0))   # plano → beta fallback
    jr = FakeJournal(precos, ibov)
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=tickers),
                        config=_cfg())
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    agent.treinar(ds, data_treino_fim=idx[-1 - agent.config.horizonte])
    res = agent.avaliar_ic(ds)
    bl = res["baselines"]
    assert bl["B2_mom_12_1_total"] > 0.10
    max_b = max(bl["B1_score_econ_total"], bl["B2_mom_12_1_total"], bl["B3_intercepto_total"])
    assert res["GAP_total"] == pytest.approx(res["IC_total"] - max_b)


# ── 18. Survivorship via tickers=None ─────────────────────────────────────────


def test_survivorship_via_tickers_None(monkeypatch):
    idx = _dias_uteis("2020-01-01", 320)
    tickers = ["AAAA", "BBBB"]
    precos = {t: _frame_precos(idx, _random_walk(idx, i + 1))
              for i, t in enumerate(tickers)}
    ibov = _frame_precos(idx, _random_walk(idx, 99, start=1000.0))
    jr = FakeJournal(precos, ibov)
    saida_a = idx[290]

    def fake_ativos(t):
        return ["AAAA", "BBBB"] if t <= saida_a else ["BBBB"]

    monkeypatch.setattr("agents.math_ml.tickers_ativos", fake_ativos)
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=tickers))
    ds = agent.construir_dataset(None, idx[270], idx[-6])
    assert ds[ds["ticker"] == "AAAA"]["data"].max() <= saida_a
    assert (ds["ticker"] == "BBBB").sum() > (ds["ticker"] == "AAAA").sum()


# ── 19. beta_contra_setor=True levanta NotImplementedError ─────────────────────


def test_beta_contra_setor_levanta_not_implemented():
    jr, idx, tickers = _journal_random(["AAAA"], n_dias=300)
    agent = MathMLAgent(journal=jr, econ_mock=make_econ_mock(jr, ic_alvo=0.0),
                        config=MathMLConfig(beta_contra_setor=True))
    with pytest.raises(NotImplementedError):
        agent.construir_dataset(["AAAA"], idx[270], idx[-1])


# ── 20. Features de macro com valor correto ───────────────────────────────────


def test_features_macro_valor_correto():
    idx = _dias_uteis("2020-01-01", 300)
    t = idx[280]
    macro_idx = pd.bdate_range(end=t, periods=40, tz=FUSO)
    selic = pd.Series(np.linspace(10.0, 13.0, 40), index=macro_idx)
    ptax = pd.Series(np.linspace(5.0, 5.5, 40), index=macro_idx)
    jr = FakeJournal({"AAAA": _frame_precos(idx, _random_walk(idx, 1))},
                     _frame_precos(idx, _random_walk(idx, 2, start=1000.0)),
                     macro={"selic_meta": selic, "ptax_usdbrl": ptax})
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=["AAAA"]))
    feats = agent._montar_features("AAAA", t)
    assert feats["selic_nivel"] == pytest.approx(13.0)
    assert feats["selic_var_21d"] == pytest.approx(selic.iloc[-1] - selic.iloc[-22])
    assert feats["cambio_var_21d"] == pytest.approx(ptax.iloc[-1] / ptax.iloc[-22] - 1)


# ── 24. Pré-fetch: chamadas únicas (guardião anti n+1) ────────────────────────


def _eq(a, b):
    """Comparação nan-safe até 1e-8."""
    if a is None and b is None:
        return
    if isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b)
        return
    assert a == pytest.approx(b, abs=1e-8) if isinstance(a, (int, float)) else a == b


def test_prefetch_chamadas_unicas():
    """Guardião contra regressão a n+1 no CAMINHO DE PRODUÇÃO.

    IMPORTANTE: o CountingFakeJournal retorna Fundamentals com
    `data_recebimento_cvm` POPULADO → ativa o modo INDEXED do
    `_prefetch_fundamentos` (âncoras mensais + `_fund_em_t`), que é o
    comportamento de produção. O modo passthrough (fixtures sem DT_RECEB) NÃO é
    coberto aqui — ele só ocorre em unit tests com FakeJournal simplificado,
    nunca em produção.
    """
    # Fundamentos COM DT_RECEB (antes da janela) → modo 'indexed' do pré-fetch.
    idx = _dias_uteis("2021-06-01", 700)
    tickers = ["AAA", "BBB"]
    precos = {tk: _frame_precos(idx, _random_walk(idx, i + 1))
              for i, tk in enumerate(tickers)}
    ibov = _frame_precos(idx, _random_walk(idx, 99, start=1000.0))
    dt_receb = ts("2022-06-01")
    fund = {tk: Fundamentals(ticker=tk, data_referencia=dt_receb,
                             data_recebimento_cvm=dt_receb, pl=10.0, pvp=2.0,
                             lucro_liquido=100.0) for tk in tickers}

    chamadas = {"get_precos": 0, "get_macro": 0, "get_fundamentals": 0,
                "get_retornos_setor": 0}

    class Counting(FakeJournal):
        def get_precos(self, *a, **k):
            chamadas["get_precos"] += 1
            return super().get_precos(*a, **k)

        def get_macro(self, *a, **k):
            chamadas["get_macro"] += 1
            return super().get_macro(*a, **k)

        def get_fundamentals(self, *a, **k):
            chamadas["get_fundamentals"] += 1
            return super().get_fundamentals(*a, **k)

        def get_retornos_setor(self, *a, **k):  # não deve ser chamado
            chamadas["get_retornos_setor"] += 1
            return {"retorno_medio": None, "n_tickers": 0, "tickers": [], "setor": "x"}

    jr = Counting(precos, ibov, fundamentals=fund)
    inicio, fim = ts("2023-06-01"), ts("2023-09-30")
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=tickers))
    ds = agent.construir_dataset(tickers, inicio, fim)

    assert len(ds) > 0
    assert chamadas["get_precos"] <= len(tickers) + 1        # precos/ticker + ^BVSP
    assert chamadas["get_macro"] <= 1                        # 1 chamada cobre a série
    # Fundamentos: âncoras mensais cobrindo [inicio − 365d, fim] (por causa do YoY).
    # Limite = n_tickers × (n_meses + folga), bem abaixo do n+1 real (~n × 250).
    n_meses = int(np.ceil(((fim - inicio).days + 365) / 30))
    assert chamadas["get_fundamentals"] <= len(tickers) * (n_meses + 4)
    assert chamadas["get_retornos_setor"] == 0               # não é feature
    # SEM o refator, get_precos/get_macro/get_fundamentals estariam em milhares.


# ── 25. Pré-fetch: invariância de valores (cache == sem-cache) ────────────────


def test_prefetch_invariancia_valores():
    jr, idx, tickers = _journal_random(["AAAA", "BBBB", "CCCC"], n_dias=340)
    agent = MathMLAgent(journal=jr,
                        econ_mock=make_econ_mock(jr, ic_alvo=0.0, universo=tickers))
    di, dfim = idx[300], idx[320]
    cache = agent._prefetch(tickers, di, dfim)
    for ticker in tickers:
        for t in idx[300:315]:
            a_c = agent._alvo(ticker, t, cache=cache)
            a_n = agent._alvo(ticker, t, cache=None)
            _eq(a_c["y"], a_n["y"])
            _eq(a_c["beta"], a_n["beta"])
            assert a_c["beta_fallback"] == a_n["beta_fallback"]
            f_c = agent._montar_features(ticker, t, cache=cache)
            f_n = agent._montar_features(ticker, t, cache=None)
            assert set(f_c) == set(f_n)
            for k in f_n:
                _eq(f_c[k], f_n[k])


# ── 21. Estrutura de saída do avaliar_ic ──────────────────────────────────────


def test_avaliar_ic_estrutura_de_saida():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(8)], n_dias=320)
    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0,
                          amostra_calibracao=_amostra(idx[260:300], tickers),
                          universo=tickers)
    agent = MathMLAgent(journal=jr, econ_mock=mock, config=_cfg())
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    agent.treinar(ds, data_treino_fim=idx[-1 - agent.config.horizonte])
    out = agent.avaliar_ic(ds)

    for k in ["IC_total", "IC_evento", "IC95", "IC95_method",
              "baselines", "GAP_total", "GAP_evento", "n"]:
        assert k in out, f"chave ausente: {k}"
    assert len(out["IC95"]) == 2
    assert out["IC95"][0] <= out["IC95"][1]
    assert out["IC95_method"] == "block_bootstrap_by_date"
    bl = out["baselines"]
    for k in ["B1_score_econ_total", "B2_mom_12_1_total", "B3_intercepto_total"]:
        assert k in bl
    if ds["tem_evento"].sum() > 30:
        for k in ["B1_score_econ_evento", "B2_mom_12_1_evento", "B3_intercepto_evento"]:
            assert k in bl
    assert np.isfinite(out["GAP_total"])


# ── 26. Fast-path de calibração usa o MESMO z(y) do runtime (ddof consistente) ─


def test_mock_fastpath_calibracao_consistente():
    """Fast-path (amostra com 'y') deve calibrar o MESMO z(y) que o runtime injeta.

    Regressão: `pandas.Series.std()` (ddof=1) na calibração fast-path divergia do
    `numpy.ndarray.std()` (ddof=0) em `_z_info` (runtime) → `mock.ic_realizado`
    não batia com o IC efetivamente injetado no dataset (erro ∝ 1/|U|, grande p/
    universo pequeno). Universo de 3 tickers amplifica o efeito.
    """
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(3)], n_dias=320)
    # dataset base só para extrair a coluna 'y' (ativa o fast-path da calibração)
    base = MathMLAgent(journal=jr,
                       econ_mock=make_econ_mock(jr, ic_alvo=0.0, prob_evento=1.0,
                                                universo=tickers))
    ds_base = base.construir_dataset(tickers, idx[256], idx[-1])

    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0, universo=tickers,
                          amostra_calibracao=ds_base[["data", "ticker", "y"]])
    agent = MathMLAgent(journal=jr, econ_mock=mock)
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    evt = ds["tem_evento"].to_numpy(bool)
    ic_injetado = spearmanr(ds.loc[evt, "score_econ"], ds.loc[evt, "y"]).statistic

    # o IC reportado pela calibração tem que casar com o IC realmente injetado
    assert mock.ic_realizado == pytest.approx(ic_injetado, abs=0.01)


# ── 27. GAP_evento é NaN quando não há eventos p/ baseline (3 ≤ n ≤ 30) ────────


def test_gap_evento_nan_sem_baseline_de_evento():
    """GAP_evento deve ser NaN quando faltam eventos p/ um baseline de evento.

    Regressão: `ic_evento` sai com ≥3 eventos, mas os baselines de evento só com
    >30. No meio-termo (3–30), `GAP_evento` comparava contra baseline ausente
    (fallback 0.0), superestimando o edge. Agora reporta NaN (comparação inviável),
    enquanto GAP_total (que sempre tem B3_intercepto=0) segue definido.
    """
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(8)], n_dias=320)
    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0,
                          amostra_calibracao=_amostra(idx[260:300], tickers),
                          universo=tickers)
    agent = MathMLAgent(journal=jr, econ_mock=mock, config=_cfg())
    ds = agent.construir_dataset(tickers, idx[256], idx[-1])
    agent.treinar(ds, data_treino_fim=idx[-1 - agent.config.horizonte])

    # força poucos eventos (5), no intervalo 3 ≤ n ≤ 30
    poucos = ds.copy()
    poucos["tem_evento"] = False
    poucos.iloc[:5, poucos.columns.get_loc("tem_evento")] = True
    assert 3 <= int(poucos["tem_evento"].sum()) <= 30
    out = agent.avaliar_ic(poucos)

    assert not np.isnan(out["IC_evento"])   # ic_evento ainda é calculado (≥3)
    assert np.isnan(out["GAP_evento"])      # mas sem baseline de evento → indefinido
    assert np.isfinite(out["GAP_total"])    # total intacto (B3_intercepto=0)


# ── 22. Mock com |U|<3 → neutro+degradado ─────────────────────────────────────


def test_mock_universo_insuficiente_retorna_degradado():
    jr, idx, tickers = _journal_random(["AAAA", "BBBB"], n_dias=320)  # só 2 tickers
    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0, universo=tickers)
    se = mock("AAAA", idx[300])  # y válido, mas |U|=2 < 3
    assert se.score_total == 0.0
    assert se.confianca == 0.0
    assert se.tem_evento is True
    assert "universo_insuficiente" in se.avisos
    assert mock.fallback_minimo_universo_count == 1


# ── 23. Mock com y indisponível (sem t+5) → neutro+degradado ──────────────────


def test_mock_y_indisponivel_retorna_degradado():
    jr, idx, tickers = _journal_random([f"T{i}" for i in range(8)], n_dias=320)
    mock = make_econ_mock(jr, ic_alvo=0.15, prob_evento=1.0, universo=tickers)
    se = mock("T0", idx[-1])  # última data → sem janela forward → y=NaN
    assert se.score_total == 0.0
    assert se.confianca == 0.0
    assert se.tem_evento is True
    assert "y_indisponivel" in se.avisos
