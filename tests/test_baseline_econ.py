"""Testes determinísticos do runner do baseline do ECON (Etapa 1).

Cobrem a lógica PURA da rodada: amostragem/deduplicação de eventos, agregação de
métricas (IC completo × IC limpo × B0 lexical) e diagnóstico. Nada toca a API —
o `journal` é um fake em memória e a etapa paga (`avaliar_eventos`) não é
exercida aqui. Execute: pytest tests/test_baseline_econ.py -v
"""
import pandas as pd
import pytest

from config import FUSO
from agents.sources.noticia import Noticia
from calibration.baseline_econ import (
    _confere_identidade,
    _foi_avaliado,
    _linha_retomada,
    _veredito,
    amostrar_eventos,
    calcular_metricas,
    diagnosticar,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


def _noticia(titulo: str, publicado: str) -> Noticia:
    return Noticia(titulo=titulo, conteudo="", url="http://x",
                   publicado_em=ts(publicado), fonte="Bloomberg",
                   peso_fonte=1.0, ticker="PETR4.SA")


class _JournalFake:
    """Journal em memória: devolve as notícias registradas por (ticker, dia) e um
    target fixo. Sem rede, sem yfinance."""

    def __init__(self, por_dia: dict, y: float = 0.01):
        self.por_dia = por_dia
        self.y = y

    def get_noticias(self, ticker, data_limite, lookback_days=7):
        return list(self.por_dia.get((ticker, str(data_limite.date())), []))


@pytest.fixture
def sem_rede(monkeypatch):
    """Neutraliza o calendário da B3 e o cálculo de target (ambos batem em rede)."""
    import calibration.baseline_econ as bl

    pregoes = [ts(f"2025-09-{d:02d} 23:59") for d in (1, 2, 3)]
    monkeypatch.setattr(bl, "calendario_pregoes", lambda i, f: pregoes)
    monkeypatch.setattr(bl, "tickers_ativos", lambda d: ["PETR4.SA"])
    monkeypatch.setattr(bl, "_retorno_excesso_5d",
                        lambda journal, t, d, ajuste_beta=False: journal.y)
    return pregoes


# ── Amostragem e deduplicação ─────────────────────────────────────────────────


class TestAmostrarEventos:
    def test_dedup_mantem_apenas_a_primeira_ocorrencia(self, sem_rede):
        """O lookback de 7du repete o MESMO conjunto de notícias em pregões
        vizinhos: com dedup, só o dia da chegada da notícia vira evento."""
        n = _noticia("Lucro recorde", "2025-09-01 10:00")
        journal = _JournalFake({("PETR4.SA", f"2025-09-{d:02d}"): [n] for d in (1, 2, 3)})

        eventos, diag = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"))

        assert len(eventos) == 1
        assert eventos[0]["data"].date() == pd.Timestamp("2025-09-01").date()
        assert diag["n_duplicados_descartados"] == 2

    def test_sem_dedup_mantem_todas_as_ocorrencias(self, sem_rede):
        n = _noticia("Lucro recorde", "2025-09-01 10:00")
        journal = _JournalFake({("PETR4.SA", f"2025-09-{d:02d}"): [n] for d in (1, 2, 3)})

        eventos, diag = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"),
                                         dedup=False)

        assert len(eventos) == 3
        assert diag["n_duplicados_descartados"] == 0

    def test_conjunto_novo_gera_evento_novo(self, sem_rede):
        """Chegou notícia nova no meio da janela → configuração diferente → evento."""
        a = _noticia("Lucro recorde", "2025-09-01 10:00")
        b = _noticia("Dívida cai", "2025-09-02 10:00")
        journal = _JournalFake({
            ("PETR4.SA", "2025-09-01"): [a],
            ("PETR4.SA", "2025-09-02"): [a, b],
            ("PETR4.SA", "2025-09-03"): [a, b],
        })

        eventos, _ = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"))

        assert [e["data"].date().day for e in eventos] == [1, 2]

    def test_dia_sem_noticia_nao_vira_evento(self, sem_rede):
        journal = _JournalFake({})
        eventos, diag = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"))
        assert eventos == []
        assert diag["n_sem_noticia"] == 3

    def test_evento_sem_target_e_descartado(self, sem_rede, monkeypatch):
        import calibration.baseline_econ as bl
        monkeypatch.setattr(bl, "_retorno_excesso_5d",
                            lambda journal, t, d, ajuste_beta=False: None)
        n = _noticia("Lucro recorde", "2025-09-01 10:00")
        journal = _JournalFake({("PETR4.SA", "2025-09-01"): [n]})

        eventos, diag = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"))

        assert eventos == []
        assert diag["n_sem_y"] == 1

    def test_marca_limpo_pela_data_da_noticia_nao_do_pregao(self, sem_rede):
        """Pregão de setembro/2025 com notícia de julho/2025 continua CONTAMINADO:
        o que o modelo pode ter visto no treino é o texto, não o dia do pregão."""
        velha = _noticia("Notícia de julho", "2025-07-15 10:00")
        journal = _JournalFake({("PETR4.SA", "2025-09-01"): [velha]})

        eventos, diag = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"))

        assert eventos[0]["limpo"] is False
        assert diag["n_limpos"] == 0

    def test_n_max_corta_a_amostra(self, sem_rede):
        journal = _JournalFake({
            ("PETR4.SA", f"2025-09-{d:02d}"): [_noticia(f"N{d}", f"2025-09-{d:02d} 10:00")]
            for d in (1, 2, 3)
        })

        eventos, _ = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"), n_max=2)

        assert len(eventos) == 2

    def test_evento_id_e_sequencial(self, sem_rede):
        journal = _JournalFake({
            ("PETR4.SA", f"2025-09-{d:02d}"): [_noticia(f"N{d}", f"2025-09-{d:02d} 10:00")]
            for d in (1, 2, 3)
        })
        eventos, _ = amostrar_eventos(journal, ts("2025-09-01"), ts("2025-09-03"))
        assert [e["evento_id"] for e in eventos] == [0, 1, 2]


# ── Métricas ──────────────────────────────────────────────────────────────────


def _df_sintetico(n: int = 30, sinal: int = 1) -> pd.DataFrame:
    """Metade contaminada (jul/2025), metade limpa (set/2025). `sinal=-1` inverte
    a relação score×retorno na metade limpa."""
    linhas = []
    for i in range(n):
        limpo = i >= n // 2
        data = ts("2025-09-01") if limpo else ts("2025-07-01")
        y = (i % 10) / 100.0
        score = y * (sinal if limpo else 1)
        linhas.append({"data": data + pd.Timedelta(days=i), "ticker": "PETR4.SA",
                       "y": y, "score": score, "score_lexical": score / 2,
                       "limpo": limpo})
    return pd.DataFrame(linhas)


class TestCalcularMetricas:
    def test_reporta_completo_e_limpo_separadamente(self):
        met = calcular_metricas(_df_sintetico())
        assert met["econ_completo"]["n"] == 30
        assert met["econ_limpo"]["n"] == 15
        assert met["econ_completo"]["n_blocos"] >= 2

    def test_ic_limpo_pode_divergir_do_completo(self):
        """Sinal invertido só na janela limpa: o IC completo esconde, o limpo mostra."""
        met = calcular_metricas(_df_sintetico(sinal=-1))
        assert met["econ_limpo"]["ic"] < 0
        assert met["econ_limpo"]["ic"] < met["econ_completo"]["ic"]

    def test_gap_e_a_diferenca_contra_o_lexical(self):
        met = calcular_metricas(_df_sintetico())
        assert met["gap_completo"] == pytest.approx(
            met["econ_completo"]["ic"] - met["lexical_completo"]["ic"])

    def test_cobertura_lexical_conta_scores_nao_nulos(self):
        """Cobertura baixa = léxico mudo no dataset e GAP sem significado."""
        df = _df_sintetico()
        df["score_lexical"] = 0.5
        df.loc[df.index[:15], "score_lexical"] = 0.0
        assert calcular_metricas(df)["cobertura_lexical"] == pytest.approx(0.5)

    def test_janela_limpa_vazia_nao_quebra(self):
        df = _df_sintetico()
        df["limpo"] = False
        met = calcular_metricas(df)
        assert pd.isna(met["econ_limpo"]["ic"])
        assert met["econ_limpo"]["n"] == 0


# ── Diagnóstico ───────────────────────────────────────────────────────────────


class TestDiagnosticar:
    def test_ignora_grupos_abaixo_do_piso(self):
        df = _df_sintetico()
        df.loc[df.index[:3], "ticker"] = "VALE3.SA"  # só 3 linhas → abaixo do piso
        assert "VALE3.SA" not in set(diagnosticar(df)["ic_por_ticker"]["grupo"])

    def test_ordena_do_pior_ic_para_o_melhor(self):
        df = _df_sintetico(n=40)
        df.loc[df.index[:20], "ticker"] = "RUIM.SA"
        df.loc[df.index[:20], "score"] = -df.loc[df.index[:20], "y"]
        tab = diagnosticar(df)["ic_por_ticker"]
        assert tab.iloc[0]["grupo"] == "RUIM.SA"
        assert tab.iloc[0]["ic"] < tab.iloc[-1]["ic"]

    def test_piores_casos_sao_discordancias_de_rank(self):
        """Score no topo com retorno no fundo tem de aparecer entre os piores."""
        df = _df_sintetico(n=40)
        df.loc[df.index[0], ["score", "y"]] = [1.0, -0.10]
        piores = diagnosticar(df)["piores_casos"]
        assert (piores["score"] == 1.0).any()


# ── Veredito contra a meta ────────────────────────────────────────────────────


class TestRetomadaDeCheckpoint:
    """`evento_id` é posicional. Se a amostragem mudar entre execuções (yfinance
    derruba um ticker, por exemplo), o id passa a apontar para outro ticker-dia —
    e colar o score de um evento em outro corromperia o IC em silêncio."""

    def _evento(self):
        return {"ticker": "PETR4.SA", "data": ts("2025-10-10 23:59"), "y": 0.01,
                "evento_id": 0, "noticias": [], "data_noticia": ts("2025-10-09"),
                "limpo": True, "score_lexical": 0.0}

    def test_identidade_confere_quando_ticker_e_data_batem(self):
        prev = {"ticker": "PETR4.SA", "data": "2025-10-10T23:59:00-03:00"}
        assert _confere_identidade(self._evento(), prev) is True

    def test_identidade_falha_com_ticker_diferente(self):
        prev = {"ticker": "VALE3.SA", "data": "2025-10-10T23:59:00-03:00"}
        assert _confere_identidade(self._evento(), prev) is False

    def test_identidade_falha_com_data_diferente(self):
        prev = {"ticker": "PETR4.SA", "data": "2025-10-13T23:59:00-03:00"}
        assert _confere_identidade(self._evento(), prev) is False

    def test_retomada_recupera_justificativa_persistida(self):
        prev = {"score": 0.4, "tokens_in": 100, "tokens_out": 20,
                "latencia_llm_s": 3.0, "custo_usd": 0.004, "confianca": 0.7,
                "tem_evento": True, "degradacao": None,
                "justificativa": "capex eleva geração de caixa"}
        linha = _linha_retomada(self._evento(), prev)
        assert linha["justificativa"] == "capex eleva geração de caixa"
        assert linha["confianca"] == pytest.approx(0.7)

    def test_alinhamento_detecta_amostra_deslocada(self):
        """Pré-voo: se a amostragem mudou, retomar reavaliaria eventos já pagos."""
        from calibration.baseline_econ import conferir_alinhamento

        class _CP:
            def ja_feito(self, modelo, eid):
                return eid == 0

            def linha_feita(self, modelo, eid):
                return {"ticker": "VALE3.SA", "data": "2025-10-10T23:59:00-03:00"}

        out = conferir_alinhamento([self._evento()], _CP())
        assert out["n_desalinhados"] == 1
        assert out["taxa"] == 0.0

    def test_alinhamento_ok_quando_amostra_e_a_mesma(self):
        from calibration.baseline_econ import conferir_alinhamento

        class _CP:
            def ja_feito(self, modelo, eid):
                return eid == 0

            def linha_feita(self, modelo, eid):
                return {"ticker": "PETR4.SA", "data": "2025-10-10T23:59:00-03:00"}

        out = conferir_alinhamento([self._evento()], _CP())
        assert out["n_desalinhados"] == 0
        assert out["taxa"] == 1.0

    def test_retomada_de_schema_antigo_nao_inventa_valor(self):
        """Sem as colunas novas, o campo fica vazio — o relatório mostra a lacuna."""
        prev = {"score": 0.4, "tokens_in": 100, "tokens_out": 20,
                "latencia_llm_s": 3.0, "custo_usd": 0.004}
        linha = _linha_retomada(self._evento(), prev)
        assert linha["justificativa"] == ""
        assert pd.isna(linha["confianca"])


class TestFoiAvaliado:
    """Distinguir FALHA de CACHE HIT: as duas gastam zero token, mas uma é lixo
    que enviesa o IC para zero e a outra é avaliação legítima."""

    def test_degradacao_registrada_e_falha(self):
        assert _foi_avaliado({"degradacao": "erro_api", "justificativa": "",
                              "tokens_in": 0}) is False

    def test_cache_hit_com_justificativa_e_avaliacao(self):
        assert _foi_avaliado({"degradacao": None, "tokens_in": 0,
                              "justificativa": "capex eleva caixa"}) is True

    def test_schema_antigo_cai_na_heuristica_de_tokens(self):
        assert _foi_avaliado({"degradacao": None, "justificativa": None,
                              "tokens_in": 2500}) is True
        assert _foi_avaliado({"degradacao": None, "justificativa": None,
                              "tokens_in": 0}) is False


class TestCaminhosPorVersao:
    """O checkpoint é chaveado por (modelo, evento_id) — SEM versão de prompt. Se
    duas iterações compartilharem arquivo, a segunda retoma os scores da primeira
    e reporta ΔIC = 0 falso. A separação tem que vir do caminho."""

    def test_versoes_diferentes_usam_arquivos_diferentes(self):
        from calibration.baseline_econ import caminho_checkpoint

        assert caminho_checkpoint("2026-06-econA") != caminho_checkpoint("2026-08-econB")

    def test_baseline_preserva_o_arquivo_ja_pago(self):
        """A Etapa 1 custou US$2,79; seu checkpoint não pode mudar de nome."""
        from calibration.baseline_econ import (
            INTERMEDIARIO_CSV, VERSAO_BASELINE, caminho_checkpoint,
        )

        assert caminho_checkpoint(VERSAO_BASELINE) == INTERMEDIARIO_CSV

    def test_nome_de_arquivo_e_seguro(self):
        from calibration.baseline_econ import caminho_checkpoint

        nome = caminho_checkpoint("2026-08/econ B").name
        assert "/" not in nome and " " not in nome

    def test_csv_completo_tambem_e_por_versao(self):
        from calibration.baseline_econ import caminho_completo

        assert caminho_completo("2026-06-econA") != caminho_completo("2026-08-econB")


class TestVeredito:
    def test_acima_da_meta(self):
        assert "SUFICIENTE" in _veredito(0.20)

    def test_faixa_intermediaria(self):
        assert "ACEITÁVEL" in _veredito(0.12)

    def test_abaixo_de_010(self):
        assert "ITERAÇÃO" in _veredito(0.05)

    def test_negativo_e_destacado(self):
        assert "NEGATIVO" in _veredito(-0.02)

    def test_nan_e_indeterminado(self):
        assert _veredito(float("nan")) == "INDETERMINADO"
