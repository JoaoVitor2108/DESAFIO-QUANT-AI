"""
MATH&ML — terceiro agente do JEMPO.

Combina o `score_total` do ECON (sinal qualitativo da notícia) com features
quantitativas CRUAS do JOURNAL e treina um GradientBoosting raso que prevê o
retorno IDIOSSINCRÁTICO (beta-ajustado) de 5 dias úteis e ranqueia o universo. A
seleção final (top-N, limites setoriais) é do ORQUESTRADOR — aqui só se prevê e
ranqueia.

────────────────────────────────────────────────────────────────────────────────
NOTA DE DESIGN — como consumir o ECON (evitar colinearidade) [decisão Opção A]

Saúde financeira, momento setorial e macro entram como FEATURES CRUAS vindas
direto do JOURNAL (get_fundamentals / get_macro) — NUNCA os componentes `comp_*`
do ECON (duplicação de sinal proibida). O `score_total` (impacto da notícia) é a
contribuição central do ECON e a única usada como feature.

────────────────────────────────────────────────────────────────────────────────
PRINCÍPIO — hipótese antes de padrão

Cada feature mapeia 1-para-1 numa hipótese da literatura, com direção de sinal
esperada (`SINAL_ESPERADO`) e cálculo anti-lookahead. `importancia_features`
cruza o efeito observado com o esperado e sinaliza inversões (red flag de overfit).

────────────────────────────────────────────────────────────────────────────────
ANTI-LOOKAHEAD

- Feature em `t` usa só dados `<= t` (todas as janelas terminam em `t`).
- Label usa `t+1..t+5` (forward, legítimo); linhas sem `t+5` são DROPADAS.
- Split com PURGE + EMBARGO (López de Prado): treino cujo `[t, t+h]` invade o
  teste é removido, mais um gap de `embargo` dias úteis.
- Imputação só com estatística cross-sectional do próprio dia (nunca futura/global).
- `_assert_no_lookahead` na saída de `prever` (defesa em profundidade).

MOCK do ECON (fase sem ANTHROPIC_API_KEY): `make_econ_mock` injeta um sinal
CONTROLADO calibrado a um `ic_alvo` — para provar que o pipeline reconhece sinal
quando há e medir a sensibilidade da estratégia à qualidade do ECON. O `y` futuro
entra APENAS na geração do `score_econ` (fixture), nunca como feature do modelo.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

from config import (
    FUSO,
    INICIO_WARMUP,
    INICIO_TREINO,
    FIM_TREINO,
    INICIO_BACKTEST,
    FIM_BACKTEST,
    tickers_ativos,
)

logger = logging.getLogger(__name__)


class LookaheadError(AssertionError):
    """Levantado quando alguma fonte de feature contém dado posterior a `t`."""


# ── Vetor de features (§4.8 da especificação) ─────────────────────────────────
# `crescimento_lucro_yoy` substitui o antigo `surpresa_lucro` (era YoY de lucro
# TTM, não SUE clássico de PEAD). `pead_window` foi removida (não testamos PEAD
# clássico — ver _montar_features). `dias_desde_resultado` permanece como regime.
FEATURES = [
    "mom_12_1", "rev_1m",
    "dias_desde_resultado", "crescimento_lucro_yoy",
    "pl", "pvp", "roe", "margem", "divida_ebitda",
    "volume_relativo",
    "selic_nivel", "selic_var_21d", "cambio_var_21d",
    "score_econ", "econ_confianca", "econ_n_noticias",
]

# Flags = METADADOS da linha (não são features do modelo).
FLAGS = ["beta_fallback", "fundamental_imputado", "econ_degradado"]

# Direção teórica esperada do efeito (None = condicional/gate, sem sinal fixo).
SINAL_ESPERADO = {
    "mom_12_1": +1, "rev_1m": -1,
    "crescimento_lucro_yoy": +1,
    "pl": -1, "pvp": -1, "roe": +1, "margem": +1, "divida_ebitda": -1,
    "volume_relativo": +1,
    "score_econ": +1,
}


@dataclass
class MathMLConfig:
    """Hiperparâmetros, janelas e períodos. Defaults lêem as constantes de config.py."""
    warmup_inicio: pd.Timestamp = INICIO_WARMUP
    treino_inicio: pd.Timestamp = INICIO_TREINO
    treino_fim: pd.Timestamp = FIM_TREINO
    oos_inicio: pd.Timestamp = INICIO_BACKTEST
    oos_fim: pd.Timestamp = FIM_BACKTEST
    horizonte: int = 5              # dias úteis do alvo e do embargo
    janela_beta: int = 252
    min_obs_beta: int = 200
    janela_momentum: int = 252
    skip_momentum: int = 21
    janela_volume: int = 20
    janela_macro: int = 21
    n_splits_cv: int = 5
    embargo: int = 5
    max_posicoes: int = 3           # consumido pelo ORQUESTRADOR, não aqui
    max_por_setor: int = 2
    beta_contra_setor: bool = False  # §3 refinamento opcional (default OFF)
    gb_params: dict = field(default_factory=lambda: dict(
        max_depth=3, learning_rate=0.05, subsample=0.8,
        n_estimators=500, random_state=42))


@dataclass
class _DatasetCache:
    """Cache em memória das chamadas externas — uma chamada por fonte por run.

    Elimina o n+1 do `construir_dataset` (uma chamada ao JOURNAL por ticker×dia).
    NÃO inclui `retornos_setor` (não é consumido por nenhuma feature) nem `setores`.

    - precos:      dict[ticker, DataFrame] cobrindo o range estendido.
    - ibov:        DataFrame do ^BVSP no range estendido.
    - fundamentos: dict[ticker, (modo, dados)] — modo "indexed" (lista de
                   Fundamentals ordenada por DT_RECEB) ou "passthrough" (None):
                   fixtures sem DT_RECEB caem no passthrough (chamada por t),
                   preservando invariância; dados reais (com DT_RECEB) usam a lista.
    - macro:       dict[nome, Series] — séries completas cortadas em fetch_fim
                   (uma única `get_macro`, que já devolve a série warmup→data).
    """
    precos: dict = field(default_factory=dict)
    ibov: object = None
    fundamentos: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)


# ── Split temporal com purge + embargo (López de Prado) ───────────────────────


class PurgedTimeSeriesSplit:
    """TimeSeriesSplit com purge + embargo para labels de horizonte `h`.

    Garante que nenhuma amostra de treino conheça o futuro do teste: mantém no
    treino apenas amostras cujo `[t, t+h]` termina, com folga de `embargo` dias
    úteis, antes do início do bloco de teste.
    """

    def __init__(self, n_splits: int = 5, horizonte: int = 5, embargo: int = 5):
        self.n_splits = n_splits
        self.horizonte = horizonte
        self.embargo = embargo

    def split(self, n_samples: int, datas):
        """`datas`: ordinal de dia útil de cada amostra (mesma unidade do horizonte)."""
        datas = np.asarray(datas)
        gap = self.horizonte + self.embargo
        unique = np.unique(datas)
        if len(unique) <= self.n_splits + 1:
            return
        blocos = np.array_split(unique, self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            teste_ords = blocos[i]
            teste_min = teste_ords.min()
            teste_idx = np.where((datas >= teste_min) & (datas <= teste_ords.max()))[0]
            treino_idx = np.where(datas + gap <= teste_min)[0]
            if len(treino_idx) == 0 or len(teste_idx) == 0:
                continue
            yield treino_idx, teste_idx


# ── Mock estruturado do ECON ──────────────────────────────────────────────────


class _MockHandle:
    """Callable f(ticker, data_limite) -> ScoreEcon com diagnósticos AO VIVO.

    `fundamental`-free; expõe `alpha`, `ic_alvo`, `ic_realizado`, `n_amostra` e a
    propriedade `fallback_minimo_universo_count` (atualizada em runtime).
    """

    def __init__(self, fn, diagnostics, set_cache_fn=None, **attrs):
        self._fn = fn
        self._diag = diagnostics
        self._set_cache_fn = set_cache_fn
        for k, v in attrs.items():
            setattr(self, k, v)

    def __call__(self, ticker, data_limite):
        return self._fn(ticker, data_limite)

    def set_cache(self, cache) -> None:
        """Injeta o _DatasetCache do construir_dataset para o z(y) cross-sectional
        ler da memória (evita o n+1 dentro do mock). Sem cache → caminho normal."""
        if self._set_cache_fn is not None:
            self._set_cache_fn(cache)

    @property
    def fallback_minimo_universo_count(self) -> int:
        return self._diag["fallback_minimo_universo_count"]


def make_econ_mock(journal, ic_alvo: float = 0.15, seed: int = 42,
                   prob_evento: float = 0.15,
                   amostra_calibracao: Optional[pd.DataFrame] = None,
                   universo: Optional[list] = None,
                   janela_beta: int = 252, min_obs_beta: int = 200,
                   horizonte: int = 5) -> _MockHandle:
    """Mock estruturado do ECON com z(y) cross-sectional DINÂMICO (sem fallback tanh).

    Em dia com evento injeta `score = clamp(alpha·z(y) + sqrt(1-alpha²)·ruído, -1, 1)`,
    distribuição UNIMODAL (sem reescala bimodal) para favorecer a transferência
    mock → ECON real sem retreino. `z(y)` é a padronização cross-sectional do alvo
    realizado (§3) calculada SOB DEMANDA olhando o universo do dia. `alpha` é
    auto-calibrado por busca binária até `Spearman(score, y) ≈ ic_alvo` na
    `amostra_calibracao` — usada APENAS para calibrar, nunca consultada em runtime
    (em runtime o `y` é recomputado via journal). O `y` futuro entra SÓ aqui
    (fixture); no deploy real o ECON entra de verdade e este mock some.

    Parâmetros:
      amostra_calibracao : DataFrame com colunas ('data','ticker') [e 'y' opcional];
                           se 'y' ausente, é recomputado via `_alvo`.
      universo           : lista de tickers do cross-section por dia. Se None, usa
                           `config.tickers_ativos(t)`.
    """
    from agents.econ import ScoreEcon

    _agent = MathMLAgent(journal=journal, econ_mock=lambda *a, **k: None,
                         config=MathMLConfig(janela_beta=janela_beta,
                                             min_obs_beta=min_obs_beta,
                                             horizonte=horizonte))
    diagnostics = {"fallback_minimo_universo_count": 0}
    universo_fixo = list(universo) if universo is not None else None
    _estado = {"cache": None}   # _DatasetCache injetado via set_cache (opcional)

    def _y(ticker, data):
        try:
            return _agent._alvo(ticker, data, cache=_estado["cache"])["y"]
        except Exception:
            return np.nan

    def _universo_dia(t):
        if universo_fixo is not None:
            return universo_fixo
        try:
            return tickers_ativos(t)
        except Exception:
            return []

    # cache do cross-section de y por data (evita O(|U|²) por dia).
    _yU_cache: dict = {}

    def _y_universo(t):
        chave = pd.Timestamp(t)
        if chave not in _yU_cache:
            yU = {}
            for tk in _universo_dia(t):
                yv = _y(tk, t)
                if not np.isnan(yv):
                    yU[tk] = yv
            _yU_cache[chave] = yU
        return _yU_cache[chave]

    def _z_info(ticker, t):
        """z(y) cross-sectional DINÂMICO — recomputa y do dia via journal.

        Retorna (z, motivo). `motivo` ∈ {None, "y_indisponivel",
        "universo_insuficiente"}: ou o z-score cross-sectional acontece, ou não há
        sinal a injetar (caller degrada). NÃO há cálculo alternativo de z.
        """
        y_ti = _y(ticker, t)
        if np.isnan(y_ti):
            return None, "y_indisponivel"          # sem janela forward (t+5)
        vals = np.array(list(_y_universo(t).values()), dtype=float)
        if len(vals) < 3:
            return None, "universo_insuficiente"    # |U(t)|<3 → z indefinido
        mu, sd = float(vals.mean()), float(vals.std())
        return (y_ti - mu) / (sd + 1e-12), None

    def _ruido(ticker, data):
        h = _stable_seed("ruido", ticker, pd.Timestamp(data).date(), seed)
        return float(np.random.default_rng(h).normal())

    def _score(alpha, z, r):
        return float(np.clip(alpha * z + np.sqrt(max(0.0, 1 - alpha ** 2)) * r, -1.0, 1.0))

    def _conf(ticker, data):
        return float(np.random.default_rng(
            _stable_seed("cf", ticker, pd.Timestamp(data).date(), seed)).uniform(0.5, 0.9))

    # ── Calibração de alpha (uma vez; nunca consultada em runtime) ──
    cal = []  # (z, y, ruído)
    if amostra_calibracao is not None and len(amostra_calibracao) > 0:
        df = amostra_calibracao
        if "y" in df.columns:
            # FAST-PATH: y do cross-section já fornecido → z(y) por data IN-MEMORY,
            # sem recomputar via journal (evita n+1 na calibração). Mesmos valores
            # (o y da amostra veio do mesmo `_alvo`).
            d = df.dropna(subset=["y"]).copy()
            # ddof=0 (populacional) casa com `_z_info` no runtime (numpy .std());
            # pandas .std() default é ddof=1 e enviesaria o z(y) da calibração.
            d["_z"] = d.groupby("data")["y"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
            for _, row in d.iterrows():
                cal.append((float(row["_z"]), float(row["y"]),
                            _ruido(row["ticker"], row["data"])))
        else:
            # amostra sem y (ex.: fixtures de teste) → recomputa z via journal/cache.
            for _, row in df.iterrows():
                tk, t = row["ticker"], row["data"]
                z, _motivo = _z_info(tk, t)   # não incrementa o contador
                yv = _y(tk, t)
                if z is None or pd.isna(yv):
                    continue
                cal.append((z, float(yv), _ruido(tk, t)))

    def _ic(alpha):
        if len(cal) < 3:
            return 0.0
        scores = [_score(alpha, z, r) for z, _, r in cal]
        ys = [y for _, y, _ in cal]
        ic, _ = spearmanr(scores, ys)
        return 0.0 if np.isnan(ic) else ic

    alpha = 0.0
    ic_realizado = 0.0
    if cal and ic_alvo > 0:
        lo_a, hi_a = 0.0, 1.0
        for _ in range(40):
            alpha = 0.5 * (lo_a + hi_a)
            ic_realizado = _ic(alpha)
            if abs(ic_realizado - ic_alvo) < 0.005:
                break
            if ic_realizado < ic_alvo:
                lo_a = alpha
            else:
                hi_a = alpha
    else:
        ic_realizado = _ic(0.0) if cal else 0.0

    def _tem_evento(ticker, data):
        if prob_evento >= 1.0:
            return True
        h = _stable_seed("evt", ticker, pd.Timestamp(data).date(), seed)
        return np.random.default_rng(h).random() < prob_evento

    def _se(ticker, data, score, conf, tem_evento, n):
        return ScoreEcon(
            ticker=ticker, data_referencia=pd.Timestamp(data),
            score_total=score, comp_noticia=score,
            comp_saude_financeira=0.0, comp_setorial=0.0, comp_macro=0.0,
            confianca=conf, tem_evento=tem_evento, n_noticias=n,
            justificativa="mock estruturado", modelo="econ_mock")

    def _degradado(ticker, data, aviso):
        """Evento que não pôde ser avaliado → neutro+degradado (conf=0,
        tem_evento=True). O agente marca como econ_degradado=True (dois-zeros)."""
        return ScoreEcon(
            ticker=ticker, data_referencia=pd.Timestamp(data),
            score_total=0.0, comp_noticia=0.0, comp_saude_financeira=0.0,
            comp_setorial=0.0, comp_macro=0.0, confianca=0.0,
            tem_evento=True, n_noticias=1, noticias_hashes=[],
            justificativa=f"degradado: {aviso}", modelo="econ_mock",
            avisos=[aviso])

    def f(ticker, data_limite):
        if not _tem_evento(ticker, data_limite):
            return _se(ticker, data_limite, 0.0, 0.0, False, 0)
        z, motivo = _z_info(ticker, data_limite)
        if motivo == "universo_insuficiente":
            diagnostics["fallback_minimo_universo_count"] += 1
            return _degradado(ticker, data_limite, "universo_insuficiente")
        if motivo is not None or z is None:        # y indisponível (sem t+5)
            return _degradado(ticker, data_limite, "y_indisponivel")
        score = _score(alpha, z, _ruido(ticker, data_limite))
        return _se(ticker, data_limite, score, _conf(ticker, data_limite), True, 2)

    return _MockHandle(f, diagnostics,
                       set_cache_fn=lambda c: _estado.__setitem__("cache", c),
                       alpha=alpha, ic_alvo=ic_alvo,
                       ic_realizado=ic_realizado, n_amostra=len(cal))


# ── Agente ────────────────────────────────────────────────────────────────────


class MathMLAgent:
    def __init__(self, journal, econ=None, econ_mock: Optional[Callable] = None,
                 config: Optional[MathMLConfig] = None):
        if (econ is None) == (econ_mock is None):
            raise ValueError("Forneça exatamente um entre `econ` e `econ_mock`.")
        self.journal = journal
        self._econ = econ
        self._econ_mock = econ_mock
        self.config = config or MathMLConfig()
        self.model: Optional[GradientBoostingRegressor] = None
        self.cv_report: dict = {}
        self._train_medians: Optional[pd.Series] = None
        self._medianas_treino: Optional[pd.Series] = None
        self._wf_folds: list = []

    # ---- acesso ao ECON (real ou mock) ----
    def _score_econ(self, ticker, data_limite):
        if self._econ is not None:
            return self._econ.avaliar(ticker, data_limite)
        return self._econ_mock(ticker, data_limite)

    # ---- preços / Ibov alinhado ----
    def _precos(self, ticker, data_limite):
        c = self.config
        buffer = int((c.janela_momentum + c.horizonte) * 1.7) + 10
        inicio = max(c.warmup_inicio, data_limite - pd.Timedelta(days=buffer * 2))
        return self.journal.get_precos(ticker, inicio, data_limite + pd.Timedelta(days=15))

    def _ibov_alinhado(self, idx_acao, data_limite):
        ibov = self.journal.get_precos(
            "^BVSP", idx_acao.min(), data_limite + pd.Timedelta(days=15))["Close_raw"]
        return ibov.reindex(idx_acao).ffill()

    # ---- pré-fetch (Fase 1): uma chamada por fonte no range inteiro ----
    def _prefetch(self, tickers, data_inicio, data_fim) -> _DatasetCache:
        """Pre-fetch único de todas as fontes externas para o range inteiro,
        reduzindo O(n_tickers × n_dias × n_fontes) → O(n_tickers) chamadas.

        Range estendido cobre as janelas históricas (beta/momentum) e o label
        forward. Valores são idênticos aos do caminho por-linha: a Fase 2 só
        fatia `<= t` e roda a MESMA lógica.
        """
        c = self.config
        janela_max = max(c.janela_beta, c.janela_momentum + c.skip_momentum) + 30
        fetch_inicio = max(c.warmup_inicio,
                           data_inicio - pd.Timedelta(days=janela_max * 2))
        fetch_fim = data_fim + pd.Timedelta(days=c.horizonte * 2 + 30)
        logger.info("_prefetch: range estendido %s..%s (%d tickers)",
                    fetch_inicio.date(), fetch_fim.date(), len(tickers))

        # Fundamentos só precisam cobrir [data_inicio − 365d, data_fim] (YoY),
        # range bem mais curto que a janela de preços (beta/momentum ~2 anos).
        fund_inicio = max(c.warmup_inicio, data_inicio - pd.Timedelta(days=365 + 30))

        cache = _DatasetCache()
        cache.ibov = self.journal.get_precos("^BVSP", fetch_inicio, fetch_fim)
        # get_macro(t) já devolve a SÉRIE completa warmup→t; uma chamada basta.
        cache.macro = self.journal.get_macro(fetch_fim)
        for ticker in tickers:
            cache.precos[ticker] = self.journal.get_precos(ticker, fetch_inicio, fetch_fim)
            cache.fundamentos[ticker] = self._prefetch_fundamentos(
                ticker, fund_inicio, fetch_fim)
        return cache

    def _prefetch_fundamentos(self, ticker, fetch_inicio, fetch_fim):
        """Coleta Fundamentals em âncoras mensais (fundamentos mudam ~4x/ano).

        Retorna ("indexed", lista_ordenada_por_DT_RECEB) quando há DT_RECEB
        (dados reais) — cada doc é o "mais recente" em alguma âncora e é dedupado.
        Retorna ("passthrough", None) quando nenhum doc tem DT_RECEB (fixtures de
        teste): aí `_fund_em_t` chama `get_fundamentals(ticker, t)` por linha,
        preservando invariância com o caminho não-cacheado.
        """
        ancoras = pd.date_range(fetch_inicio, fetch_fim, freq="MS", tz=FUSO)
        ancoras = ancoras.append(pd.DatetimeIndex([fetch_fim]))  # garante o fim
        vistos = {}
        algum_dt_receb = False
        for ancora in ancoras:
            try:
                f = self.journal.get_fundamentals(ticker, ancora)
            except Exception as e:
                logger.warning("get_fundamentals(%s, %s) falhou: %s", ticker, ancora, e)
                continue
            dt = getattr(f, "data_recebimento_cvm", None)
            if dt is not None:
                algum_dt_receb = True
                vistos[dt] = f
        if not algum_dt_receb:
            return ("passthrough", None)
        return ("indexed", sorted(vistos.values(), key=lambda f: f.data_recebimento_cvm))

    @staticmethod
    def _fund_em_t(fundamentos_entry, journal, ticker, t):
        """Último Fundamental com DT_RECEB <= t (modo indexed), ou
        `get_fundamentals(ticker, t)` direto (modo passthrough / fixtures)."""
        modo, dados = fundamentos_entry
        if modo == "passthrough":
            return journal.get_fundamentals(ticker, t)
        candidatos = [f for f in dados if f.data_recebimento_cvm <= t]
        return candidatos[-1] if candidatos else None

    # ---- acessores cache-aware (fast-path opcional; sem cache OU ticker fora do
    # cache → caminho por-chamada, garantindo mesmos valores e sem KeyError) -----
    def _precos_raw(self, ticker, t, cache):
        if cache is not None and ticker in cache.precos:
            return cache.precos[ticker]
        return self._precos(ticker, t)

    def _ibov_series(self, idx_acao, data_limite, cache):
        if cache is not None and cache.ibov is not None:
            return cache.ibov["Close_raw"].reindex(idx_acao).ffill()
        return self._ibov_alinhado(idx_acao, data_limite)

    def _macro_em_t(self, t, cache):
        if cache is not None and cache.macro:
            return {k: s[s.index <= t] for k, s in cache.macro.items()}
        return self.journal.get_macro(t)

    def _fund_at(self, ticker, t, cache):
        if cache is not None and ticker in cache.fundamentos:
            return self._fund_em_t(cache.fundamentos[ticker], self.journal, ticker, t)
        return self.journal.get_fundamentals(ticker, t)

    @staticmethod
    def _pos(idx, t):
        loc = idx.searchsorted(t, side="right") - 1
        if loc < 0 or idx[loc] != t:
            # t pode não ser dia útil exato: usa o último pregão <= t
            if loc < 0:
                raise KeyError(f"data {t} antes do início da série")
        return loc

    # ---- alvo beta-ajustado (§3) ----
    def _alvo(self, ticker, t, cache=None):
        c = self.config
        ate = t + pd.Timedelta(days=15)
        precos = self._precos_raw(ticker, t, cache)
        precos = precos[precos.index <= ate]
        cr = precos["Close_raw"]
        idx = cr.index
        ibov = self._ibov_series(idx, ate, cache)
        pos = self._pos(idx, t)

        # Beta defasado: OLS de r_acao em r_ibov nos 252 du que terminam em t.
        ini = max(0, pos - c.janela_beta + 1)
        r_i = cr.iloc[ini:pos + 1].pct_change().to_numpy()
        r_m = ibov.iloc[ini:pos + 1].pct_change().to_numpy()
        mask = ~(np.isnan(r_i) | np.isnan(r_m))
        r_i, r_m = r_i[mask], r_m[mask]
        if len(r_i) < c.min_obs_beta or np.var(r_m) <= 0:
            beta, beta_fallback = 1.0, True
        else:
            # OLS slope (ddof consistente, sem o viés cov[ddof=1]/var[ddof=0]).
            slope, _ = np.polyfit(r_m, r_i, deg=1)
            beta, beta_fallback = float(slope), False

        # Retornos forward de 5 dias úteis (t+1..t+5).
        if pos + c.horizonte >= len(idx):
            return {"y": np.nan, "beta": beta, "beta_fallback": beta_fallback}
        r_i_fwd = cr.iloc[pos + c.horizonte] / cr.iloc[pos] - 1
        r_m_fwd = ibov.iloc[pos + c.horizonte] / ibov.iloc[pos] - 1
        y = float(r_i_fwd - beta * r_m_fwd)
        return {"y": y, "beta": beta, "beta_fallback": beta_fallback}

    # ---- features de uma linha (§4) ----
    def _montar_features(self, ticker, t, verificar_lookahead=False, cache=None):
        c = self.config
        precos = self._precos_raw(ticker, t, cache)
        precos = precos[precos.index <= t]
        cr = precos["Close_raw"]
        vol = precos["Volume"]
        idx = cr.index
        pos = self._pos(idx, t)

        def _ret(off):
            j = pos - off
            return float(cr.iloc[pos] / cr.iloc[j] - 1) if j >= 0 else np.nan

        # MOMENTUM 12-1 e REVERSÃO 1m
        mom = (float(cr.iloc[pos - c.skip_momentum] / cr.iloc[pos - c.janela_momentum] - 1)
               if pos - c.janela_momentum >= 0 else np.nan)
        rev = _ret(21)
        # VOLUME relativo
        if pos - c.janela_volume >= 0:
            media_vol = vol.iloc[pos - c.janela_volume:pos].mean()
            volume_rel = float(vol.iloc[pos] / media_vol) if media_vol > 0 else np.nan
        else:
            volume_rel = np.nan

        # FUNDAMENTOS (anti-lookahead nativo via DT_RECEB no JOURNAL)
        fund = self._fund_at(ticker, t, cache)
        recv = getattr(fund, "data_recebimento_cvm", None)
        dias_result = float((t - recv).days) if recv is not None else np.nan
        # crescimento_lucro_yoy: crescimento anual do lucro líquido TTM. NÃO é PEAD
        # clássico (que exige SUE = surpresa vs consenso de analistas, indisponível);
        # é feature de growth (literatura de value/growth), sinal teórico +.
        lucro = getattr(fund, "lucro_liquido", None)
        fund_ant = self._fund_at(ticker, t - pd.Timedelta(days=365), cache)
        lucro_ant = getattr(fund_ant, "lucro_liquido", None)
        if lucro is not None and lucro_ant not in (None, 0):
            # divide por |antigo| para preservar o sinal na virada prejuízo→lucro.
            crescimento_lucro_yoy = float((lucro - lucro_ant) / abs(lucro_ant))
        else:
            crescimento_lucro_yoy = np.nan

        # MACRO (séries já cortadas em t pelo JOURNAL — ou do cache, mesma coisa)
        macro = self._macro_em_t(t, cache)
        selic = macro.get("selic_meta")
        ptax = macro.get("ptax_usdbrl")
        selic_nivel = float(selic.iloc[-1]) if selic is not None and len(selic) else np.nan
        selic_var = (float(selic.iloc[-1] - selic.iloc[-1 - c.janela_macro])
                     if selic is not None and len(selic) > c.janela_macro else 0.0)
        cambio_var = (float(ptax.iloc[-1] / ptax.iloc[-1 - c.janela_macro] - 1)
                      if ptax is not None and len(ptax) > c.janela_macro else 0.0)

        # SINAL DA NOTÍCIA (ECON)
        se = self._score_econ(ticker, t)
        econ_degradado = bool(se.tem_evento and se.confianca == 0)

        if verificar_lookahead:
            self._assert_no_lookahead(t, precos=precos, fund=fund, macro=macro, score=se)

        return {
            "mom_12_1": mom, "rev_1m": rev,
            "dias_desde_resultado": dias_result,
            "crescimento_lucro_yoy": crescimento_lucro_yoy,
            "pl": _num(getattr(fund, "pl", None)),
            "pvp": _num(getattr(fund, "pvp", None)),
            "roe": _num(getattr(fund, "roe", None)),
            "margem": _num(getattr(fund, "margem_liquida", None)),
            "divida_ebitda": _num(getattr(fund, "divida_liquida_ebitda", None)),
            "volume_relativo": volume_rel,
            "selic_nivel": selic_nivel, "selic_var_21d": selic_var,
            "cambio_var_21d": cambio_var,
            "score_econ": float(se.score_total),
            "econ_confianca": float(se.confianca),
            "econ_n_noticias": float(se.n_noticias),
            # metadados (não-features)
            "beta_fallback": False,  # preenchido no dataset
            "fundamental_imputado": getattr(fund, "pl", None) is None
            or getattr(fund, "pvp", None) is None,
            "econ_degradado": econ_degradado,
            "tem_evento": bool(se.tem_evento),
        }

    def _assert_no_lookahead(self, t, *, precos=None, fund=None, macro=None,
                             score=None):
        """Defesa em profundidade: não confia só nos contratos do JOURNAL/ECON.

        Levanta `LookaheadError` se preços, fundamentos (DT_RECEB), qualquer série
        macro ou o `ScoreEcon` (data_referencia) expuserem dado posterior a `t`.
        """
        if precos is not None and len(precos) > 0:
            if precos.index.max() > t:
                raise LookaheadError(f"precos.max={precos.index.max()} > t={t}")

        if fund is not None and getattr(fund, "data_recebimento_cvm", None) is not None:
            if fund.data_recebimento_cvm > t:
                raise LookaheadError(
                    f"fundamentos.DT_RECEB={fund.data_recebimento_cvm} > t={t}")

        if macro is not None:
            for nome, serie in macro.items():
                if hasattr(serie, "index") and len(serie) > 0 and serie.index.max() > t:
                    raise LookaheadError(f"macro[{nome}].max={serie.index.max()} > t={t}")

        if score is not None and getattr(score, "data_referencia", None) is not None:
            if score.data_referencia > t:
                raise LookaheadError(
                    f"ScoreEcon.data_referencia={score.data_referencia} > t={t}")

    # ---- construção do dataset ----
    def construir_dataset(self, tickers, data_inicio, data_fim) -> pd.DataFrame:
        """Painel (data, ticker) com features §4, alvo §3 e flags; labels incompletos
        dropados. Se `tickers` for None, usa tickers_ativos(t) por dia (produção)."""
        if self.config.beta_contra_setor:
            raise NotImplementedError(
                "beta_contra_setor=True não é suportado: get_retornos_setor entrega "
                "dict agregado, não série diária para estimar beta. Use o default "
                "(beta vs Ibov).")

        # FASE 1 — pré-fetch (uma chamada por fonte). Para tickers=None (produção
        # com survivorship), pré-busca a UNIÃO dos tickers ativos no range.
        cal_datas = pd.date_range(data_inicio, data_fim, freq="D", tz=FUSO)
        if tickers is not None:
            tickers_fetch = list(tickers)
        else:
            tickers_fetch = sorted({tk for d in cal_datas for tk in tickers_ativos(d)})
        cache = self._prefetch(tickers_fetch, data_inicio, data_fim)
        # Injeta o cache no mock do ECON (se em uso) para o z(y) cross-sectional
        # ler da memória em vez de refazer chamadas ao JOURNAL.
        if self._econ_mock is not None and hasattr(self._econ_mock, "set_cache"):
            self._econ_mock.set_cache(cache)

        # FASE 2 — montagem em memória (zero I/O). Calendário = pregões do cache.
        idx_ibov = cache.ibov.index
        cal = idx_ibov[(idx_ibov >= data_inicio) & (idx_ibov <= data_fim)]
        linhas = []
        for t in cal:
            universo = tickers if tickers is not None else tickers_ativos(t)
            for ticker in universo:
                # Ordem deliberada: y PRIMEIRO; linhas com y=NaN (sem t+5) saem
                # ANTES de _montar_features → o mock do ECON nunca é chamado para
                # um t sem janela forward (e não monta feature de linha descartada).
                try:
                    alvo = self._alvo(ticker, t, cache=cache)
                except Exception:
                    continue
                if np.isnan(alvo["y"]):
                    continue  # label incompleto (sem t+5) → dropar
                feats = self._montar_features(ticker, t, cache=cache)
                feats["beta_fallback"] = alvo["beta_fallback"]
                feats["data"] = t
                feats["ticker"] = ticker
                feats["y"] = alvo["y"]
                linhas.append(feats)
        if not linhas:
            return pd.DataFrame(columns=["data", "ticker", *FEATURES, *FLAGS,
                                         "tem_evento", "y"])
        df = pd.DataFrame(linhas)
        # Fallback de imputação isolado ao TREINO (nunca vê o futuro): mediana por
        # feature computada só no subset `data <= treino_fim`, antes de imputar.
        treino = df[df["data"] <= self.config.treino_fim]
        medianas_treino = (treino[FEATURES].median() if len(treino)
                           else df[FEATURES].median())
        self._medianas_treino = medianas_treino
        df = self._imputar_cross_sectional(df, medianas_treino=medianas_treino)
        df["ord"] = df["data"].map(
            {d: i for i, d in enumerate(sorted(df["data"].unique()))})
        cols = ["data", "ticker", "ord", *FEATURES, *FLAGS, "tem_evento", "y"]
        return df[cols].sort_values(["data", "ticker"]).reset_index(drop=True)

    def _imputar_cross_sectional(self, df: pd.DataFrame,
                                 medianas_treino=None) -> pd.DataFrame:
        """Imputa NaN de features com a mediana cross-sectional do PRÓPRIO dia.

        Para colunas inteiramente NaN num dia (ex.: macro indisponível), o fallback
        usa `medianas_treino` (subset de treino), NUNCA a mediana global do dataset
        completo — que veria o futuro. A flag `fundamental_imputado` é setada antes,
        em `_montar_features`.
        """
        df = df.copy()
        for feat in FEATURES:
            med = df.groupby("data")[feat].transform("median")
            df[feat] = df[feat].fillna(med)
        if medianas_treino is not None:
            df[FEATURES] = df[FEATURES].fillna(medianas_treino)
        df[FEATURES] = df[FEATURES].fillna(0.0)
        return df

    # ---- treino + validação interna ----
    def treinar(self, dataset, data_treino_fim=None) -> None:
        c = self.config
        fim = data_treino_fim or c.treino_fim
        df = dataset[dataset["data"] <= fim].copy()
        if df.empty:
            raise ValueError("dataset de treino vazio para o corte informado.")
        X = df[FEATURES].to_numpy()
        y = df["y"].to_numpy()
        ords = df["ord"].to_numpy() if "ord" in df else \
            df["data"].map({d: i for i, d in enumerate(sorted(df["data"].unique()))}).to_numpy()
        self._train_medians = df[FEATURES].median()

        info = self._escolher_n_estimators(X, y, ords)
        n_est = info["n_estimators"]
        params = {**c.gb_params, "n_estimators": n_est}
        self.model = GradientBoostingRegressor(**params).fit(X, y)
        self.cv_report = info
        argmax = info.get("n_argmax")
        if argmax and abs(argmax - n_est) > 0.20 * max(argmax, 1):
            logger.warning("n_estimators platô=%d diverge >20%% do argmax=%d",
                           n_est, argmax)

    def _escolher_n_estimators(self, X, y, ords) -> dict:
        """Escolhe n_estimators por REGRA DE PLATÔ (não argmax — evita pescar pico
        de ruído da validação). Reporta argmax e platô para auditoria."""
        c = self.config
        n_max = c.gb_params.get("n_estimators", 500)
        pts = PurgedTimeSeriesSplit(n_splits=c.n_splits_cv, horizonte=c.horizonte,
                                    embargo=c.embargo)
        folds = list(pts.split(len(y), ords))
        linhas_ic = []
        for tr, te in folds:
            if len(tr) < 10 or len(te) < 5:
                continue
            m = GradientBoostingRegressor(**{**c.gb_params, "n_estimators": n_max})
            m.fit(X[tr], y[tr])
            ic_stages = np.array([
                (lambda ic: 0.0 if np.isnan(ic) else ic)(spearmanr(pred, y[te])[0])
                for pred in m.staged_predict(X[te])])
            linhas_ic.append(ic_stages)
        if not linhas_ic:
            return {"n_estimators": n_max, "n_argmax": n_max, "n_platau": n_max,
                    "metodo": "sem_cv"}
        ic_mat = np.vstack(linhas_ic)
        mean_ic = ic_mat.mean(axis=0)
        n_argmax = int(mean_ic.argmax()) + 1
        n_platau = self._n_estimators_platau(ic_mat)
        return {"n_estimators": n_platau, "n_argmax": n_argmax,
                "n_platau": n_platau, "metodo": "platau", "n_folds": len(linhas_ic)}

    @staticmethod
    def _n_estimators_platau(ic_mat: np.ndarray, tol: float = 0.5) -> int:
        """Menor n cujo IC médio de validação está dentro de `tol` desvios-padrão
        do pico — o início do platô (mais regularizado que o argmax)."""
        mean_ic = ic_mat.mean(axis=0)
        std_at_peak = ic_mat[:, mean_ic.argmax()].std()
        threshold = mean_ic.max() - tol * std_at_peak
        candidatos = np.where(mean_ic >= threshold)[0]
        return int(candidatos[0]) + 1

    def avaliar_ic(self, dataset, mascara=None, n_boot: int = 1000) -> dict:
        if self.model is None:
            raise RuntimeError("modelo não treinado.")
        df = dataset if mascara is None else dataset[mascara]
        df = df.copy()
        pred = self.model.predict(df[FEATURES].to_numpy())
        y = df["y"].to_numpy()
        ic_total = _spearman(pred, y)
        evt = df["tem_evento"].to_numpy(dtype=bool)
        ic_evento = _spearman(pred[evt], y[evt]) if evt.sum() >= 3 else np.nan
        boot = _block_bootstrap_ic(pred, y, df["data"].to_numpy(), n_boot)
        baselines = self._calcular_baselines(df)
        max_total = _max_baseline(baselines, "_total")
        max_evento = _max_baseline(baselines, "_evento")
        return {
            "IC_total": ic_total,
            "IC_evento": ic_evento,
            "IC95": (boot["ic_p2_5"], boot["ic_p97_5"]),
            "IC95_method": "block_bootstrap_by_date",
            "baselines": baselines,
            "GAP_total": ic_total - max_total,
            "GAP_evento": (ic_evento - max_evento) if not np.isnan(ic_evento) else np.nan,
            "n": len(df),
        }

    def _calcular_baselines(self, df) -> dict:
        """Três baselines competitivos para o GAP (total e no subset de evento):
        B1 = só score_econ (sem ML); B2 = só mom_12_1 (competidor óbvio em equity);
        B3 = intercepto puro (IC de um constante é 0 — piso absoluto)."""
        y = df["y"].to_numpy()
        out = {
            "B1_score_econ_total": _spearman(df["score_econ"].to_numpy(), y),
            "B2_mom_12_1_total": _spearman(df["mom_12_1"].to_numpy(), y),
            "B3_intercepto_total": 0.0,
        }
        sub = df[df["tem_evento"].astype(bool)]
        if len(sub) > 30:
            ys = sub["y"].to_numpy()
            out["B1_score_econ_evento"] = _spearman(sub["score_econ"].to_numpy(), ys)
            out["B2_mom_12_1_evento"] = _spearman(sub["mom_12_1"].to_numpy(), ys)
            out["B3_intercepto_evento"] = 0.0
        return out

    # ---- predição ----
    def prever(self, ticker, data_limite) -> float:
        if self.model is None:
            raise RuntimeError("modelo não treinado.")
        feats = self._montar_features(ticker, data_limite, verificar_lookahead=True)
        x = self._linha_para_X(feats)
        return float(self.model.predict(x)[0])

    def prever_universo(self, tickers, data_limite) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("modelo não treinado.")
        linhas = []
        for ticker in tickers:
            try:
                feats = self._montar_features(ticker, data_limite,
                                              verificar_lookahead=True)
            except LookaheadError:
                raise
            except Exception:
                continue
            feats["ticker"] = ticker
            linhas.append(feats)
        if not linhas:
            return pd.DataFrame(columns=["ticker", "y_pred", "score_econ",
                                         "tem_evento", "rank"])
        df = pd.DataFrame(linhas)
        df["data"] = data_limite  # único dia → mediana cross-sectional do universo
        # imputação cross-sectional do dia; fallback nas medianas de treino
        df = self._imputar_cross_sectional(df, medianas_treino=self._train_medians)
        df["y_pred"] = self.model.predict(df[FEATURES].to_numpy())
        out = df[["ticker", "y_pred", "score_econ", "tem_evento"]].copy()
        out = out.sort_values("y_pred", ascending=False).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def _linha_para_X(self, feats: dict) -> np.ndarray:
        s = pd.Series({f: feats.get(f, np.nan) for f in FEATURES}, dtype=float)
        if self._train_medians is not None:
            s = s.fillna(self._train_medians)
        s = s.fillna(0.0)
        return s.to_numpy().reshape(1, -1)

    # ---- backtest walk-forward (janela expansiva) ----
    def walk_forward(self, dataset, data_inicio_oos, data_fim_oos,
                     freq="MS") -> pd.DataFrame:
        """Retreino em cada fronteira (`freq`, default 'MS' = início de mês; aceita
        'W', 'QS', etc.). Retreinos mais frequentes custam mais sem ganho garantido
        — para o backtest oficial, manter mensal."""
        c = self.config
        self._wf_folds = []
        ord_de_data = {d: i for i, d in enumerate(sorted(dataset["data"].unique()))}
        fronteiras = pd.date_range(data_inicio_oos, data_fim_oos, freq=freq, tz=FUSO)
        if len(fronteiras) == 0:
            fronteiras = pd.DatetimeIndex([data_inicio_oos])
        passo = fronteiras.freq or pd.tseries.frequencies.to_offset(freq)
        paineis = []
        for ini_mes in fronteiras:
            fim_mes = ini_mes + passo
            teste = dataset[(dataset["data"] >= ini_mes) & (dataset["data"] < fim_mes)]
            if teste.empty:
                continue
            teste_ord = teste["ord"].min() if "ord" in teste else \
                ord_de_data[teste["data"].min()]
            # purge+embargo: treino só com label completo até (teste - h - embargo)
            corte_ord = teste_ord - c.horizonte - c.embargo
            treino = dataset[dataset["ord"] <= corte_ord] if "ord" in dataset else \
                dataset[dataset["data"].map(ord_de_data) <= corte_ord]
            if len(treino) < 20:
                continue
            self.treinar(treino, data_treino_fim=treino["data"].max())
            pred = self.model.predict(teste[FEATURES].to_numpy())
            painel = teste[["data", "ticker", "score_econ", "tem_evento", "y"]].copy()
            painel = painel.rename(columns={"y": "y_real"})
            painel["y_pred"] = pred
            paineis.append(painel)
            self._wf_folds.append({
                "teste_inicio_ord": int(teste_ord),
                "treino_fim_ord": int(treino["ord"].max()),
            })
        if not paineis:
            return pd.DataFrame(columns=["data", "ticker", "y_pred", "y_real",
                                         "score_econ", "tem_evento"])
        return pd.concat(paineis, ignore_index=True)

    # ---- interpretabilidade ----
    def importancia_features(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("modelo não treinado.")
        ganho = self.model.feature_importances_
        # sinal observado = correlação da feature com a previsão no treino
        # (proxy de dependência parcial); cruzado com SINAL_ESPERADO.
        linhas = []
        for i, feat in enumerate(FEATURES):
            esperado = SINAL_ESPERADO.get(feat)
            sinal_obs = self._sinal_observado(i)
            invertido = (esperado is not None and sinal_obs != 0
                         and np.sign(sinal_obs) != np.sign(esperado))
            linhas.append({
                "feature": feat, "ganho": float(ganho[i]),
                "sinal_esperado": esperado, "sinal_observado": sinal_obs,
                "sinal_invertido": bool(invertido),
            })
        return pd.DataFrame(linhas).sort_values("ganho", ascending=False).reset_index(drop=True)

    def _sinal_observado(self, i: int) -> float:
        """Sinal do efeito da feature i: varia a feature em ± e olha a previsão média."""
        base = (self._train_medians.to_numpy() if self._train_medians is not None
                else np.zeros(len(FEATURES))).reshape(1, -1).astype(float)
        alto, baixo = base.copy(), base.copy()
        alto[0, i] += 1.0
        baixo[0, i] -= 1.0
        delta = self.model.predict(alto)[0] - self.model.predict(baixo)[0]
        return float(np.sign(delta)) if abs(delta) > 1e-12 else 0.0


# ── utilidades ────────────────────────────────────────────────────────────────


def _stable_seed(*parts) -> int:
    """Seed reprodutível entre processos (hash() do Python é salgado por PYTHONHASHSEED)."""
    raw = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _num(v):
    return float(v) if v is not None else np.nan


def _spearman(a, b) -> float:
    if len(a) < 3:
        return np.nan
    ic, _ = spearmanr(a, b)
    return float(ic) if not np.isnan(ic) else 0.0


def _max_baseline(baselines: dict, sufixo: str) -> float:
    # Sem baseline p/ o sufixo → NaN (comparação inviável), NÃO 0.0: um piso 0
    # inflaria o GAP no subset de evento com poucas amostras (3 ≤ n ≤ 30, quando
    # ic_evento existe mas os baselines de evento não). GAP_total é imune: sempre
    # tem B3_intercepto=0.0 na lista.
    vals = [v for k, v in baselines.items() if k.endswith(sufixo) and not np.isnan(v)]
    return max(vals) if vals else np.nan


def _block_bootstrap_ic(y_pred, y_real, datas, n_boot: int = 1000, rng=None) -> dict:
    """IC95 por bootstrap de BLOCOS = reamostragem com reposição de DATAS inteiras.

    Cada data sorteada entra com TODOS os tickers daquele dia, preservando a
    correlação cross-sectional e quebrando a dependência serial (labels de 5d se
    sobrepõem). Substitui o bootstrap i.i.d. de pares, que subestima a incerteza.
    """
    rng = rng or np.random.default_rng(42)
    datas = np.asarray(datas)
    datas_unicas = pd.unique(datas)
    n_dates = len(datas_unicas)
    if n_dates < 10:
        return {"ic_mean": np.nan, "ic_p2_5": np.nan, "ic_p97_5": np.nan,
                "n_dates": int(n_dates), "aviso": "n_dates<10 — bootstrap inviável"}
    idx_por_data = {i: np.where(datas == d)[0] for i, d in enumerate(datas_unicas)}
    ics = np.empty(n_boot)
    for b in range(n_boot):
        amostra = rng.integers(0, n_dates, n_dates)
        idx = np.concatenate([idx_por_data[i] for i in amostra])
        ic, _ = spearmanr(y_pred[idx], y_real[idx])
        ics[b] = 0.0 if np.isnan(ic) else ic
    return {"ic_mean": float(np.mean(ics)),
            "ic_p2_5": float(np.percentile(ics, 2.5)),
            "ic_p97_5": float(np.percentile(ics, 97.5)),
            "n_dates": int(n_dates)}


# ── script de sensibilidade (§9) — NÃO executado na fase offline ──────────────


if __name__ == "__main__":  # pragma: no cover
    # Gancho do entregável §9/§12: roda o pipeline histórico LIVE (yfinance/BCB/CVM
    # via JOURNAL) variando o ic_alvo do mock e tabula IC total/evento + GAP. Exige
    # rede e cache de disco do JOURNAL — executar como passo separado, fora dos
    # testes offline. Esqueleto:
    #
    #   from agents.journal import JournalAgent
    #   journal = JournalAgent()
    #   universo = tickers_ativos(INICIO_BACKTEST)
    #   amostra = pd.DataFrame(<(data,ticker) curados do TREINO>)
    #   for ic in (0.0, 0.10, 0.15, 0.20):
    #       mock = make_econ_mock(journal, ic_alvo=ic,
    #                             amostra_calibracao=amostra, universo=universo)
    #       agent = MathMLAgent(journal=journal, econ_mock=mock)
    #       ds = agent.construir_dataset(None, INICIO_TREINO, FIM_BACKTEST)
    #       painel = agent.walk_forward(ds, INICIO_BACKTEST, FIM_BACKTEST)
    #       ... # IC OOS total/evento, IC95(block), GAP vs max baseline -> mathml_cv.md
    print("Script de sensibilidade live — execute manualmente (requer rede).")
