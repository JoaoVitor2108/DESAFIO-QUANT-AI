"""Smoke test end-to-end do JEMPO — 4 agentes reais em cadeia, dados reais.

Objetivo
--------
Rodar o sistema inteiro (JOURNAL + ECON-mock + MATH&ML + ORQUESTRADOR) com
dados reais numa janela histórica, verificando que os agentes conversam entre
si sem exceção e que o pipeline de dados (yfinance, CVM, BCB, cache local)
funciona ponta a ponta. Serve de demo executável para a banca.

Escopo
------
É SMOKE TEST, não backtest. Verifica integração e plausibilidade das decisões.
NÃO calcula P&L, retorno, Sharpe, drawdown real, custos, Monte Carlo ou
qualquer métrica de performance — isso é do PROGRAM (próximo agente). Aqui o
PROGRAM é trivial: equity constante e execução simulada pelo fechamento do
JOURNAL no dia de execução.

Escolhas travadas
-----------------
- Janela: 1º mar → 30 abr 2024 (~40 dias úteis), dentro do range do OOS
  2024-2025. Valida os dados históricos antes da rodada oficial.
- ECON: mock estruturado `make_econ_mock` (agents/math_ml.py) no modo "meta"
  (IC alvo 0.15). Grátis, determinístico, sem API. O mock é um callable
  `f(ticker, data_limite)`; o ORQUESTRADOR espera `.avaliar(...)` → usamos um
  adapter minimal LOCAL (não modificamos o módulo).
- PROGRAM mock: equity constante em R$ 100.000,00 durante todo o loop.
  Execução simulada usa o fechamento do JOURNAL no `data_execucao` da Ordem.
  Zero P&L, zero custos.
- Universo: `config.tickers_ativos(data)` real, cardinalidade natural (22-24).

Rode com: `python scripts/smoke_test_e2e.py`

ACHADOS PARA O BACKTEST OFICIAL (não bloqueiam este smoke test)
---------------------------------------------------------------
1. JBSS3.SA: saída registrada em 2025-06-06 no UNIVERSO_HISTORICO.
   Deveria estar ATIVA em mar-abr/2024, mas yfinance falha
   consistentemente. Investigar causa da falha retroativa do yfinance
   para essa ticker (reestruturação societária conhecida no período?
   cobertura descontinuada?) e se tickers_ativos está filtrando
   corretamente.
2. ELET3.SA: yfinance notoriamente flaky para essa ticker desde a
   privatização de 2022. ELET6 (preferencial) é mais líquida. Validar
   se ELET3 pertence mesmo ao universo do período ou se deveria ser
   substituída/removida.
3. Calendário de feriados B3 ausente. pd.bdate_range e BusinessDay usam
   seg-sex sem conhecimento de feriados nacionais. Em 2024-03-29
   (Sexta-feira Santa, B3 fechada) o smoke test chamou decidir() mesmo
   assim, usando fechamento de quinta e resolvendo data_execucao para a
   segunda seguinte. Para o backtest oficial: usar pandas_market_calendars
   ou equivalente com calendário 'BMF' tanto no loop de dias úteis quanto
   no BusinessDay interno do ORQUESTRADOR. Investigar impacto em:
   (a) datas de decisão, (b) resolução D+1/D+2 do ORQUESTRADOR,
   (c) cálculo de prazo_max no notificar_execucao, (d) cálculo de janela
   do circuit-breaker.
4. Reversão same-day observada (ABEV3, 07/03). Posição aberta e fechada
   no mesmo dia via reversão. É consequência determinística do contrato
   + eventos independentes por dia do mock. No ECON real esperamos
   raridade (notícias contraditórias same-day são incomuns). Se o
   backtest oficial mostrar frequência relevante (ex.: acima de 10% das
   ordens fechando same-day), avaliar cool-off mínimo de 1du após entrada
   antes de checar reversão. Não implementado — preserva o contrato §4.4
   original.
5. Funil evento→volume no smoke test. A análise agregada dos 43 dias
   mostrou que 15% dos ticker-dias tiveram evento, 43% dos eventos
   passaram score > 0.30, mas 91% das candidatas de score foram cortadas
   pelo filtro volume_relativo > 1.5. Causa: o mock gera eventos como
   Bernoulli independente de volume, quebrando a correlação natural que
   Karpoff 1987 identifica em dados reais (notícia relevante causa surto
   de volume). No backtest oficial com ECON real, a correlação natural
   evento↔volume vai reduzir o corte substancialmente. Não é limitação
   da estratégia; é limitação do mock que só aparece em análise agregada.
"""

import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

# adiciona a raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from agents import journal as journal_mod
from agents.journal import DadoIndisponivel, JournalAgent
from agents.math_ml import MathMLAgent, make_econ_mock
from agents.orchestrator import OrchestratorAgent, OrchestratorConfig

# ── Monkeypatch do TTL do cache (decisão do usuário, Bloco 2) ─────────────────
# O _DiskCache do JOURNAL expira em 24h e apaga o arquivo no próximo get(). O
# relógio do ambiente (2026-07) está >24h à frente de TODO o cache já gravado
# (~1 mês atrás), então SEM este patch cada get vira download yfinance ao vivo
# (não-determinístico, dependente de rede). Preço histórico 2020-2024 não muda,
# então honrar o cache existente é seguro para um smoke test. Patch é de RUNTIME
# (não edita agents/journal.py) e cobre todas as instâncias de _DiskCache.
journal_mod._DiskCache.TTL = timedelta(days=3650)

# ── Constantes ────────────────────────────────────────────────────────────────

FUSO = "America/Sao_Paulo"

DATA_INICIO = pd.Timestamp("2024-03-01", tz=FUSO)
DATA_FIM = pd.Timestamp("2024-04-30 23:59:59", tz=FUSO)
EQUITY_INICIAL = 100_000.0

# Janela de treino do MATH&ML: 2 anos (decisão do usuário, Bloco 2). Smoke test
# não precisa de modelo defensável; janela curta = menos fetch. O modelo "de
# verdade" (walk-forward mensal) é do backtest oficial do PROGRAM, não daqui.
# treino_fim = 2023-12-31 mantém mar-abr/2024 OOS.
TREINO_INICIO = pd.Timestamp("2022-01-01", tz=FUSO)
TREINO_FIM = pd.Timestamp("2023-12-31 23:59:59", tz=FUSO)
DATA_SANITY = pd.Timestamp("2024-03-04", tz=FUSO)

# Modo do mock do ECON: "meta" = IC alvo 0.15 (§3 das escolhas travadas).
MODO_MOCK = "meta"
IC_ALVO = 0.15

# Contrato de saída do MATH&ML (8 colunas) — usado em asserts de sanidade.
_COLS_MATHML = [
    "ticker", "y_pred", "score_econ", "tem_evento", "rank",
    "volume_relativo", "data_noticia_mais_recente", "setor",
]

_DIAS_PT = {
    0: "segunda-feira", 1: "terça-feira", 2: "quarta-feira", 3: "quinta-feira",
    4: "sexta-feira", 5: "sábado", 6: "domingo",
}


class _EconMockAdapter:
    """Adapter minimal: o ORQUESTRADOR chama `econ.avaliar(ticker, data_limite)`,
    mas `make_econ_mock` devolve um callable `f(ticker, data_limite)`. Este wrapper
    expõe `.avaliar` delegando ao handle — sem tocar em nenhum módulo de agente."""

    def __init__(self, handle):
        self._handle = handle

    def avaliar(self, ticker, data_limite):
        return self._handle(ticker, data_limite)


def _preco_execucao(journal, ticker, data_execucao):
    """Preço de execução simulado = Close (ajustado) do JOURNAL EXATAMENTE em
    `data_execucao`. Retorna None (→ SKIP da ordem, sem inventar preço) se o
    JOURNAL não tiver dado naquele dia (feriado / delisting / gap). Janela curta
    de 7 dias só para robustez contra flakiness de janela de 1 dia do yfinance;
    o match é pela DATA exata de execução, nunca por 'último disponível'."""
    try:
        df = journal.get_precos(
            ticker, data_execucao - pd.Timedelta(days=7), data_execucao)
    except DadoIndisponivel:
        return None
    if df is None or df.empty:
        return None
    alvo = df[df.index.date == data_execucao.date()]
    if alvo.empty:
        return None
    preco = float(alvo["Close"].iloc[0])
    return preco if pd.notna(preco) and preco > 0 else None


def _instrumentar_cache(journal) -> dict:
    """Envolve journal._cache.get com contador de hit (retorno != None) vs miss
    (None). Reporta se o cache está sendo honrado (esperado ~100% pós-TTL patch)."""
    stats = {"hit": 0, "miss": 0}
    orig_get = journal._cache.get

    def counting_get(method, params):
        v = orig_get(method, params)
        if v is None:
            stats["miss"] += 1
        else:
            stats["hit"] += 1
        return v

    journal._cache.get = counting_get
    return stats


def main() -> None:
    print("smoke test iniciando")

    # sanidade do ambiente: universo do primeiro dia útil da janela
    u = config.tickers_ativos(DATA_INICIO)
    print(f"sanidade: tickers_ativos({DATA_INICIO.date()}) = {len(u)} tickers")

    # ── 1. JOURNAL real (config default do projeto) ──────────────────────────
    journal = JournalAgent()
    cache_stats = _instrumentar_cache(journal)

    # ── 2+3. Bootstrap: dataset (uncalibrated) só para extrair (data,ticker,y) ─
    # A calibração de α precisa de (data,ticker,y); esse y só existe após um
    # build. Bootstrap com mock não-calibrado (α=0) produz o MESMO y (y = retorno
    # forward, independe de α). Cronometra o _prefetch — é o passo de fetch real.
    print(f"\n[bootstrap] construir_dataset {TREINO_INICIO.date()}..{TREINO_FIM.date()} "
          f"(survivorship, tickers=None)")
    boot_handle = make_econ_mock(journal, ic_alvo=0.0)
    boot_agent = MathMLAgent(journal=journal, econ_mock=boot_handle)

    _pf = {}
    _orig_pf = boot_agent._prefetch

    def _timed_pf(*a, **k):
        t0 = time.perf_counter()
        r = _orig_pf(*a, **k)
        _pf["t"] = time.perf_counter() - t0
        return r

    boot_agent._prefetch = _timed_pf

    t0 = time.perf_counter()
    ds_boot = boot_agent.construir_dataset(None, TREINO_INICIO, TREINO_FIM)
    t_build_boot = time.perf_counter() - t0
    amostra = ds_boot[["data", "ticker", "y"]].copy()
    print(f"[bootstrap] prefetch={_pf.get('t', float('nan')):.1f}s  "
          f"build_total={t_build_boot:.1f}s  "
          f"linhas={len(ds_boot)}  tickers={ds_boot['ticker'].nunique()}  "
          f"dias={ds_boot['data'].nunique()}")

    # ── 2b. Mock CALIBRADO (meta, IC≈0.15) reaproveitando o dataset como amostra ─
    econ_handle = make_econ_mock(journal, ic_alvo=IC_ALVO, amostra_calibracao=amostra)
    econ = _EconMockAdapter(econ_handle)
    print(f"[mock] modo={MODO_MOCK} ic_alvo={econ_handle.ic_alvo} "
          f"ic_realizado={econ_handle.ic_realizado:+.3f} "
          f"alpha={econ_handle.alpha:.3f} n_amostra={econ_handle.n_amostra}")

    # ── 3b. MATH&ML real com o mock calibrado; rebuild (cache-hit) + treino ────
    math_ml = MathMLAgent(journal=journal, econ_mock=econ_handle)
    t0 = time.perf_counter()
    ds_train = math_ml.construir_dataset(None, TREINO_INICIO, TREINO_FIM)
    t_build_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    math_ml.treinar(ds_train, data_treino_fim=TREINO_FIM)
    t_fit = time.perf_counter() - t0
    print(f"[treino] rebuild(cache)={t_build_train:.1f}s  fit_GBM={t_fit:.1f}s  "
          f"linhas_treino={len(ds_train)}")

    # ── FIX (Bloco 3): limpa o cache do TREINO injetado no mock ───────────────
    # construir_dataset injeta o _DatasetCache da janela de treino no mock via
    # set_cache (para acelerar o z(y) cross-sectional). Esse cache termina em
    # ~treino_fim+buffer (≈2024-02-09) e é COMPARTILHADO com o serve-time (o
    # ORQUESTRADOR usa o mesmo mock). Sem limpar, toda avaliação em mar-abr/2024
    # cai fora do range → _alvo devolve y=NaN → ScoreEcon degradado (score 0) →
    # pool sempre vazio → 0 ordens. set_cache(None) devolve o mock ao caminho
    # normal (busca ao vivo via JOURNAL, disk cache cobre 2024). API pública do
    # mock — nenhum agente é modificado.
    econ_handle.set_cache(None)
    print("[fix] mock.set_cache(None): serve-time volta a buscar preços ao vivo")

    # ── Sanidade: prever_universo devolve DataFrame com 8 colunas ─────────────
    u_sanity = config.tickers_ativos(DATA_SANITY)
    df_pred = math_ml.prever_universo(u_sanity, DATA_SANITY)
    print(f"\n[sanidade] prever_universo({len(u_sanity)} tickers, {DATA_SANITY.date()}) "
          f"-> shape={df_pred.shape}")
    print(f"[sanidade] colunas == contrato (8): "
          f"{list(df_pred.columns) == _COLS_MATHML}  {list(df_pred.columns)}")

    # ── 4. ORQUESTRADOR real com os 4 agentes em cadeia ──────────────────────
    orq = OrchestratorAgent(
        journal=journal,
        econ=econ,
        math_ml=math_ml,
        config=OrchestratorConfig(),
        tickers_ativos=config.tickers_ativos,
    )
    print("\n4 agentes instanciados com sucesso "
          "(JOURNAL + ECON-mock + MATH&ML + ORQUESTRADOR)")
    print(f"cache: hit={cache_stats['hit']} miss={cache_stats['miss']} "
          f"(hit-rate={cache_stats['hit'] / max(1, cache_stats['hit'] + cache_stats['miss']):.1%})")
    print(f"universo ativo em {DATA_INICIO.date()}: {len(u)} tickers -> {sorted(u)}")

    # ── 5. Loop diário + funil diagnóstico ───────────────────────────────────
    # Instrumentação read-only do funil (não exposto na DecisaoDia): envolve
    # _selecionar_ordens lendo o MESMO df do prever_universo e os MESMOS
    # thresholds do config. NÃO altera filtros nem agentes — só mede cada
    # estágio: evento → score>0.30 → (score>0.30 ∧ volume>1.5) = pool final.
    funil: dict = {}
    _orig_sel = orq._selecionar_ordens

    def _sel_instrumentado(df, data):
        c = orq._config
        cond_score = df["score_econ"] > c.score_econ_min
        cond_vol = df["volume_relativo"] > c.volume_relativo_min
        funil[pd.Timestamp(data)] = {
            "n_df": len(df),
            "n_evento": int(df["tem_evento"].sum()) if len(df) else 0,
            "n_score_ok": int(cond_score.sum()) if len(df) else 0,
            "n_pool_final": int((cond_score & cond_vol).sum()) if len(df) else 0,
            "n_score_ok_vol_cut": int((cond_score & ~cond_vol).sum()) if len(df) else 0,
        }
        return _orig_sel(df, data)

    orq._selecionar_ordens = _sel_instrumentado

    # ── Wrappers de contagem + anti-lookahead (§6) ───────────────────────────
    # Envolvem as DUAS interfaces ORQUESTRADOR→agente (prever_universo e
    # avaliar), contando cada chamada e registrando o data_limite recebido. A
    # cada dia o loop seta `auditoria["dia"]`; toda chamada deve ter
    # data_limite == dia da decisão (armadilha anti-lookahead §10). Read-only.
    auditoria = {
        "dia": None,
        "mathml_calls": [],       # data_limite de cada prever_universo
        "econ_calls": [],         # (ticker, data_limite) de cada avaliar
        "violacoes": [],          # (interface, detalhe) se data_limite != dia
    }

    _orig_prever = math_ml.prever_universo
    _orig_avaliar = econ.avaliar

    def _prever_audit(tickers, data_limite):
        auditoria["mathml_calls"].append(data_limite)
        if auditoria["dia"] is not None and data_limite != auditoria["dia"]:
            auditoria["violacoes"].append(
                ("MATH&ML.prever_universo",
                 f"data_limite={data_limite} != dia={auditoria['dia']}"))
        return _orig_prever(tickers, data_limite)

    def _avaliar_audit(ticker, data_limite):
        auditoria["econ_calls"].append((ticker, data_limite))
        if auditoria["dia"] is not None and data_limite != auditoria["dia"]:
            auditoria["violacoes"].append(
                ("ECON.avaliar",
                 f"ticker={ticker} data_limite={data_limite} != dia={auditoria['dia']}"))
        return _orig_avaliar(ticker, data_limite)

    math_ml.prever_universo = _prever_audit
    econ.avaliar = _avaliar_audit

    max_pos = OrchestratorConfig().max_posicoes
    print(f"\n{'=' * 72}")
    print("Bloco 4 — loop diário COM execução simulada (PROGRAM mock)")
    print(f"Fluxo/dia (ordem estrita): a) decidir  b) print §5  c) notificar_")
    print(f"execucao(preço=Close do JOURNAL em data_execucao)  d) notificar_")
    print(f"fechamento. Equity constante R$ {EQUITY_INICIAL:,.2f}, sem P&L/custos.")
    print(f"Agora as posições ACUMULAM (até {max_pos}); ao atingir {max_pos} os slots")
    print(f"zeram e novas candidatas não viram ordem. Fechamentos por prazo/")
    print(f"reversão passam a disparar sobre posições realmente abertas.")
    print(f"{'=' * 72}")

    # Contadores de auditoria (Bloco 4)
    n_dias = n_dias_pool = n_ordens = n_cb = n_d2 = 0
    n_fech_prazo = n_fech_reversao = 0
    n_exec = 0
    pico_posicoes = 0
    skips_exec: list = []       # (ticker, data_execucao) sem preço → SKIP
    setores_ordens: Counter = Counter()

    dias = pd.bdate_range(DATA_INICIO, DATA_FIM, tz=FUSO)
    for data in dias:
        # a) decisão do dia (auditoria["dia"] = referência anti-lookahead)
        auditoria["dia"] = data
        dec = orq.decidir(data, equity_hoje=EQUITY_INICIAL)

        n_dias += 1
        universo = config.tickers_ativos(data)
        fu = funil.get(pd.Timestamp(data),
                       {"n_df": 0, "n_evento": 0, "n_score_ok": 0,
                        "n_pool_final": 0, "n_score_ok_vol_cut": 0})
        pool_n = fu["n_pool_final"]
        if pool_n > 0:
            n_dias_pool += 1
        n_ordens += len(dec.novas_ordens)
        for f in dec.fechamentos:
            if f.motivo == "prazo":
                n_fech_prazo += 1
            elif f.motivo == "reversao":
                n_fech_reversao += 1
        if dec.pausado:
            n_cb += 1
        for o in dec.novas_ordens:
            setores_ordens[o.setor] += 1
            if o.motivo_execucao_atrasada == "noticia_pos_17h":
                n_d2 += 1

        dia_nome = _DIAS_PT[data.weekday()]
        dd_pct = dec.dd_corrente * 100
        n_pos = len(dec.posicoes_abertas_snapshot)

        # funil diagnóstico do dia (universo → evento → score → pool)
        funil_txt = (f"funil[univ={len(universo)} df={fu['n_df']} "
                     f"evento={fu['n_evento']} score>0.30={fu['n_score_ok']} "
                     f"pool={fu['n_pool_final']} (score_ok∧vol≤1.5={fu['n_score_ok_vol_cut']})]")

        # b) print §5 (dia sem ordens nem fechamentos → 1 linha compacta)
        if not dec.novas_ordens and not dec.fechamentos:
            print(f"=== {data.date()} ({dia_nome}) === | "
                  f"dd={dd_pct:.2f}% pausado={dec.pausado} {n_pos} pos | "
                  f"{funil_txt} | ordens=0 fech=0")
        else:
            print(f"\n=== {data.date()} ({dia_nome}) ===")
            print(f"Estado: dd={dd_pct:.2f}%, pausado={dec.pausado}, "
                  f"{n_pos} posições abertas")
            print(f"Universo ativo: {len(universo)} tickers")
            print(f"Funil: {funil_txt}")
            print(f"Pool após filtros: {pool_n} candidatas")
            print(f"Novas ordens: {len(dec.novas_ordens)}")
            for o in dec.novas_ordens:
                print(f"  {o.ticker:<10} ({o.setor})  sizing={o.sizing_pct:.0%}  "
                      f"exec={o.data_execucao.date()}  rank={o.rank}  "
                      f"y_pred={o.y_pred:+.3f}  score_econ={o.score_econ:+.2f}")
            print(f"Fechamentos: {len(dec.fechamentos)}")
            for f in dec.fechamentos:
                print(f"  {f.ticker} — {f.motivo} (gatilho {f.data_gatilho.date()})")

        # c) execução simulada das novas ordens: preço = Close do JOURNAL no
        #    data_execucao. Sem preço → SKIP (não inventa preço).
        for o in dec.novas_ordens:
            preco = _preco_execucao(journal, o.ticker, o.data_execucao)
            if preco is None:
                skips_exec.append((o.ticker, o.data_execucao))
                print(f"  ⚠ SKIP execução {o.ticker}: JOURNAL sem Close em "
                      f"{o.data_execucao.date()} (feriado/gap/delisting)")
                continue
            orq.notificar_execucao(o.ticker, o.setor, preco, o.data_execucao)
            n_exec += 1
            print(f"  → exec {o.ticker} @ R$ {preco:,.2f} em "
                  f"{o.data_execucao.date()}")

        # d) fechamentos: bookkeeping (remove a posição do ORQUESTRADOR)
        for f in dec.fechamentos:
            orq.notificar_fechamento(f.ticker, f.data_gatilho)

        pico_posicoes = max(pico_posicoes, len(orq._posicoes))

    # ── status() final + auditoria (Bloco 4) ─────────────────────────────────
    print(f"\n{'=' * 72}")
    print("=== STATUS FINAL DO ORQUESTRADOR ===")
    st = orq.status()
    print(f"posições abertas ao final: {st['n_posicoes_abertas']} {st['tickers']}")
    print(f"dd_corrente={st['dd_corrente']:.2%}  pausado_ate={st['pausado_ate']}  "
          f"última_data_decidida={st['ultima_data_decidida'].date()}  "
          f"pontos_equity={st['n_equity_pontos']}")

    print(f"\n{'=' * 72}")
    print("=== SUMÁRIO DE AUDITORIA (§6) ===")
    print(f"Dias úteis processados: {n_dias}")
    print(f"Dias com pool não-vazio: {n_dias_pool}")
    print(f"Total de novas ordens: {n_ordens}")
    print(f"  - execução D+2 (noticia_pos_17h): {n_d2}  (esperado 0 — mock NaT)")
    print(f"  - executadas com preço real: {n_exec}")
    print(f"  - puladas (sem preço no JOURNAL): {len(skips_exec)} {skips_exec}")
    print(f"Total de fechamentos: {n_fech_prazo + n_fech_reversao}")
    print(f"  - por prazo: {n_fech_prazo}")
    print(f"  - por reversão: {n_fech_reversao}")
    print(f"Posições abertas ao final: {st['n_posicoes_abertas']}")
    print(f"Pico de posições simultâneas: {pico_posicoes} (limite max_posicoes={max_pos})")
    print(f"Dias em que o circuit-breaker foi acionado: {n_cb}  (equity constante)")
    print(f"Setores utilizados: {dict(setores_ordens)}")
    print("Nota: mock não popula data_noticia_mais_recente → NaT em toda a "
          "janela (timing sempre D+1).")

    print(f"\nChamadas aos agentes:")
    print(f"  - MATH&ML.prever_universo: {len(auditoria['mathml_calls'])}")
    print(f"  - ECON.avaliar (via mock, checagem de reversão): "
          f"{len(auditoria['econ_calls'])}")
    print(f"  - MATH&ML._prefetch (setup: bootstrap + treino): 2")
    print(f"  - JOURNAL cache no run inteiro: hit={cache_stats['hit']} "
          f"miss={cache_stats['miss']}")

    n_viol = len(auditoria["violacoes"])
    veredito = "OK" if n_viol == 0 else "FAIL"
    print(f"\nAuditoria anti-lookahead: {veredito}")
    print(f"  - Toda chamada ao MATH&ML tinha data_limite == data da decisão "
          f"({len(auditoria['mathml_calls'])} chamadas verificadas)")
    print(f"  - Toda chamada ao ECON tinha data_limite == data da decisão "
          f"({len(auditoria['econ_calls'])} chamadas verificadas)")
    if n_viol:
        print(f"  ⚠ {n_viol} VIOLAÇÕES detectadas:")
        for interface, detalhe in auditoria["violacoes"]:
            print(f"      {interface}: {detalhe}")

    # ── Funil agregado (diagnóstico: onde o funil aperta) ─────────────────────
    tot_evento = sum(v["n_evento"] for v in funil.values())
    tot_score = sum(v["n_score_ok"] for v in funil.values())
    tot_pool = sum(v["n_pool_final"] for v in funil.values())
    tot_vol_cut = sum(v["n_score_ok_vol_cut"] for v in funil.values())
    tot_df = sum(v["n_df"] for v in funil.values())
    print(f"\n=== FUNIL AGREGADO (43 dias) ===")
    print(f"Ticker-dias avaliados (linhas do prever_universo): {tot_df}")
    print(f"  → com evento (tem_evento=True): {tot_evento}"
          f"  ({tot_evento / max(1, tot_df):.1%} — prob_evento do mock ≈ 0.15)")
    print(f"  → passaram score_econ > 0.30: {tot_score}"
          f"  ({tot_score / max(1, tot_evento):.1%} dos eventos)")
    print(f"  → passaram score E volume>1.5 (pool final): {tot_pool}")
    print(f"  → tinham score_ok mas volume_relativo ≤ 1.5 (cortadas no volume): "
          f"{tot_vol_cut}")
    print(f"Leitura: dos {tot_score} com score_ok, {tot_pool} sobreviveram ao "
          f"volume e {tot_vol_cut} caíram — filtro de volume corta "
          f"{tot_vol_cut / max(1, tot_score):.0%} das candidatas de score.")

    print(f"\n{'=' * 72}")
    print("smoke test end-to-end concluído (JOURNAL + ECON-mock + MATH&ML + "
          "ORQUESTRADOR)")


if __name__ == "__main__":
    main()
