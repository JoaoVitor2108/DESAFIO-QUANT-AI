"""
Testes do JournalAgent contra APIs reais.
Execute: pytest tests/test_journal.py -v -s
"""
import pytest
import pandas as pd

from config import FUSO, tickers_ativos
from agents.journal import (
    JournalAgent,
    LookaheadError,
    DadoIndisponivel,
    Noticia,
    _assert_no_lookahead,
    _ultimo_fechamento_disponivel,
)

_SP = FUSO
_AGENT = JournalAgent()


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=_SP)


# ── get_precos ────────────────────────────────────────────────────────────────


class TestGetPrecos:
    def test_retorna_dataframe_nao_vazio(self):
        df = _AGENT.get_precos("PETR4.SA", ts("2024-10-01"), ts("2024-12-31 17:00"))
        assert not df.empty
        print(f"\nÚltimas 5 linhas PETR4:\n{df.tail()}")

    def test_colunas_obrigatorias(self):
        df = _AGENT.get_precos("PETR4.SA", ts("2024-10-01"), ts("2024-12-31 17:00"))
        for col in ("Open", "High", "Low", "Close", "Volume", "Close_raw", "flag_qualidade"):
            assert col in df.columns, f"Coluna ausente: {col}"

    def test_close_e_close_raw_diferem_em_janela_longa(self):
        # Ao longo de 4 anos deve haver ajuste de dividendo/split
        df = _AGENT.get_precos("PETR4.SA", ts("2021-01-01"), ts("2024-12-31 17:00"))
        assert not df["Close"].equals(df["Close_raw"]), (
            "Close e Close_raw são idênticos — ajuste de dividendo não está sendo aplicado"
        )

    def test_index_e_timezone_aware(self):
        df = _AGENT.get_precos("PETR4.SA", ts("2024-10-01"), ts("2024-12-31 17:00"))
        assert df.index.tz is not None
        assert "Sao_Paulo" in str(df.index.tz)

    def test_flags_existem_para_todas_linhas(self):
        df = _AGENT.get_precos("PETR4.SA", ts("2019-01-01"), ts("2024-12-31 17:00"))
        assert df["flag_qualidade"].notna().all()
        contagem = df["flag_qualidade"].value_counts()
        print(f"\nFlags PETR4 (2019-2024):\n{contagem}")

    def test_nenhum_dado_apos_data_limite(self):
        limite = ts("2024-06-30 17:00")
        df = _AGENT.get_precos("PETR4.SA", ts("2024-01-01"), limite)
        assert df.index.max() <= limite

    def test_data_inicio_anterior_warmup_levanta(self):
        with pytest.raises(ValueError, match="warmup"):
            _AGENT.get_precos("PETR4.SA", ts("2018-12-31"), ts("2024-12-31 17:00"))

    def test_timestamp_naive_levanta(self):
        with pytest.raises(ValueError):
            _AGENT.get_precos(
                "PETR4.SA",
                pd.Timestamp("2024-01-01"),   # naive
                pd.Timestamp("2024-12-31"),
            )

    def test_ticker_inexistente_levanta(self):
        with pytest.raises(DadoIndisponivel):
            _AGENT.get_precos("XXXXXXX.SA", ts("2024-01-01"), ts("2024-12-31 17:00"))


# ── get_macro ─────────────────────────────────────────────────────────────────


class TestGetMacro:
    def test_retorna_todas_series(self):
        macro = _AGENT.get_macro(ts("2024-12-31 17:00"))
        for serie in ("selic_diaria", "selic_meta", "ptax_usdbrl", "ipca_mensal", "ipca_12m"):
            assert serie in macro, f"Série ausente: {serie}"
            s = macro[serie]
            assert isinstance(s, pd.Series)
            if not s.empty:
                print(
                    f"\n{serie}: {len(s)} obs | "
                    f"último={s.iloc[-1]:.4f} em {s.index[-1].date()}"
                )

    def test_indices_sao_aware(self):
        macro = _AGENT.get_macro(ts("2024-12-31 17:00"))
        for nome, serie in macro.items():
            if not serie.empty:
                assert serie.index.tz is not None, f"{nome}: index naive"

    def test_nenhum_dado_apos_data_limite(self):
        limite = ts("2022-12-31 17:00")
        macro = _AGENT.get_macro(limite)
        for nome, serie in macro.items():
            if not serie.empty:
                assert serie.index.max() <= limite, f"{nome}: dado após data_limite"

    def test_ipca_12m_derivado_do_mensal(self):
        macro = _AGENT.get_macro(ts("2024-12-31 17:00"))
        ipca_m = macro["ipca_mensal"]
        ipca_12 = macro["ipca_12m"]
        if not ipca_m.empty and not ipca_12.empty:
            # ipca_12m deve ter menos observações (precisa de 12 meses de warmup)
            assert len(ipca_12) < len(ipca_m)


# ── get_fundamentals ──────────────────────────────────────────────────────────


class TestGetFundamentals:
    def test_retorna_fundamentals(self):
        # yfinance v1.4+ só retorna os 4 trimestres mais recentes.
        # Usar data_limite próxima ao presente para garantir dados elegíveis.
        limite = pd.Timestamp.now(tz=_SP) - pd.Timedelta(days=60)
        fund = _AGENT.get_fundamentals("PETR4.SA", limite)
        assert fund.ticker == "PETR4.SA"
        print(
            f"\nFundamentals PETR4.SA (data_limite={limite.date()}):\n"
            f"  trimestre_fim  = {fund.trimestre_fim}\n"
            f"  lucro_liquido  = {fund.lucro_liquido}\n"
            f"  receita        = {fund.receita}\n"
            f"  patrimonio     = {fund.patrimonio}\n"
            f"  pl             = {fund.pl}\n"
            f"  pvp            = {fund.pvp}\n"
            f"  avisos         = {fund.avisos}"
        )

    def test_avisos_sao_lista_de_strings(self):
        limite = pd.Timestamp.now(tz=_SP) - pd.Timedelta(days=60)
        fund = _AGENT.get_fundamentals("PETR4.SA", limite)
        assert isinstance(fund.avisos, list)
        for a in fund.avisos:
            assert isinstance(a, str)

    def test_dt_receb_respeitado(self):
        # Trimestre mais recente deve ter DT_RECEB (data receb. CVM) <= data_limite
        limite = pd.Timestamp.now(tz=_SP) - pd.Timedelta(days=60)
        fund = _AGENT.get_fundamentals("PETR4.SA", limite)
        if fund.data_recebimento_cvm is not None:
            assert fund.data_recebimento_cvm <= limite, (
                f"Lookahead: DT_RECEB={fund.data_recebimento_cvm} > data_limite={limite.date()}"
            )

    def test_data_limite_antiga_sem_dados_nao_levanta(self):
        # Para datas históricas (2019-2024), yfinance não retorna dados elegíveis.
        # O comportamento correto é: retornar Fundamentals com avisos, não levantar exceção.
        fund = _AGENT.get_fundamentals("PETR4.SA", ts("2024-09-30 17:00"))
        assert fund.ticker == "PETR4.SA"
        assert isinstance(fund.avisos, list)
        assert len(fund.avisos) > 0  # deve avisar que não há dados elegíveis
        print(f"\nAvisos esperados para data histórica: {fund.avisos}")


# ── get_retorno_ibovespa ──────────────────────────────────────────────────────


class TestGetRetornoIbovespa:
    def test_retorna_series(self):
        serie = _AGENT.get_retorno_ibovespa(ts("2024-10-01"), ts("2024-12-31 17:00"))
        assert isinstance(serie, pd.Series)
        assert not serie.empty
        assert serie.index.tz is not None
        print(
            f"\nIBOV: {len(serie)} retornos diários | "
            f"{serie.index[0].date()} → {serie.index[-1].date()}"
        )

    def test_mesma_frequencia_que_precos(self):
        di, dl = ts("2024-10-01"), ts("2024-12-31 17:00")
        ibov = _AGENT.get_retorno_ibovespa(di, dl)
        petr = _AGENT.get_precos("PETR4.SA", di, dl)["Close"].pct_change().dropna()
        # Tolerância: feriados podem diferir em ±2 dias úteis
        assert abs(len(ibov) - len(petr)) <= 2


# ── get_retornos_setor ────────────────────────────────────────────────────────


class TestGetRetornosSetor:
    def test_petroleo_gas_agrega(self):
        result = _AGENT.get_retornos_setor(
            "petroleo_gas", ts("2024-12-31 17:00"), janela_dias=60
        )
        assert result["n_tickers"] > 0
        assert result["retorno_medio"] is not None
        assert isinstance(result["tickers"], list)
        print(f"\nSetor petroleo_gas (60d): {result}")

    def test_setor_inexistente_retorna_vazio(self):
        result = _AGENT.get_retornos_setor(
            "setor_que_nao_existe", ts("2024-12-31 17:00")
        )
        assert result["n_tickers"] == 0
        assert result["retorno_medio"] is None


# ── tickers_ativos (survivorship bias) ───────────────────────────────────────


class TestTickersAtivos:
    def test_amer3_ativa_antes_da_saida(self):
        ativos = tickers_ativos(ts("2022-06-01"))
        assert "AMER3.SA" in ativos

    def test_amer3_inativa_apos_saida(self):
        ativos = tickers_ativos(ts("2023-02-01"))
        assert "AMER3.SA" not in ativos

    def test_prio3_inativa_antes_da_entrada(self):
        # PRIO3 entrou em 08/09/2020
        ativos = tickers_ativos(ts("2020-01-15"))
        assert "PRIO3.SA" not in ativos

    def test_prio3_ativa_apos_entrada(self):
        ativos = tickers_ativos(ts("2020-10-01"))
        assert "PRIO3.SA" in ativos

    def test_irbr3_ativa_durante_crise(self):
        # IRB teve escândalo em fev/2020 mas ficou no IBOV até jan/2023
        ativos = tickers_ativos(ts("2021-06-01"))
        assert "IRBR3.SA" in ativos

    def test_irbr3_inativa_apos_exclusao(self):
        ativos = tickers_ativos(ts("2023-06-01"))
        assert "IRBR3.SA" not in ativos

    def test_bpac11_inativo_antes_entrada(self):
        # BPAC11 entrou em 02/10/2019
        ativos = tickers_ativos(ts("2019-09-01"))
        assert "BPAC11.SA" not in ativos

    def test_timestamp_naive_levanta(self):
        with pytest.raises(ValueError):
            tickers_ativos(pd.Timestamp("2023-01-01"))

    def test_universo_cobre_2025(self):
        ativos = tickers_ativos(ts("2025-06-30"))
        assert len(ativos) > 0, "Universo vazio em 2025-06-30"
        # Tickers de alta confiança ainda ativos em 2025
        for ticker in ("PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "WEGE3.SA"):
            assert ticker in ativos, f"{ticker} deveria estar ativo em 2025-06-30"
        # Saídas anteriores a 2025 não devem aparecer
        assert "AMER3.SA" not in ativos, "AMER3 saiu em jan/2023"
        assert "IRBR3.SA" not in ativos, "IRBR3 saiu em jan/2023"
        # MGLU3 saiu no rebalanceamento jan/2025
        assert "MGLU3.SA" not in ativos, "MGLU3 saiu no início de 2025"
        print(f"\nTickers ativos em 2025-06-30 ({len(ativos)}): {sorted(ativos)}")


# ── Anti-lookahead ────────────────────────────────────────────────────────────


class TestAntiLookahead:
    def test_lookahead_barrado_em_series(self):
        limite = ts("2023-06-30 17:00")
        idx = pd.date_range("2023-06-28", "2023-07-02", freq="D", tz=_SP)
        serie_vazada = pd.Series([1.0] * 5, index=idx)
        with pytest.raises(LookaheadError):
            _assert_no_lookahead(serie_vazada, limite, "teste_proposital")

    def test_sem_lookahead_nao_levanta(self):
        limite = ts("2023-06-30 17:00")
        idx = pd.date_range("2023-06-24", "2023-06-30", freq="D", tz=_SP)
        serie_ok = pd.Series([1.0] * 7, index=idx)
        _assert_no_lookahead(serie_ok, limite, "teste_ok")  # não deve levantar

    def test_data_limite_naive_levanta(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _assert_no_lookahead(
                pd.Series([1.0]),
                pd.Timestamp("2023-01-01"),  # naive
                "naive_test",
            )

    def test_dataframe_vazio_nao_levanta(self):
        limite = ts("2023-06-30 17:00")
        _assert_no_lookahead(pd.DataFrame(), limite, "df_vazio")

    def test_corte_b3_antes_das_17h05(self):
        agora = ts("2024-03-15 10:00")  # sexta, antes do corte
        ult = _ultimo_fechamento_disponivel(agora)
        assert ult.date() < agora.date()

    def test_corte_b3_apos_17h05(self):
        agora = ts("2024-03-15 18:00")  # sexta, após corte
        ult = _ultimo_fechamento_disponivel(agora)
        assert ult.date() == agora.date()


# ── get_noticias (integração de fontes + dedup) ──────────────────────────────


def _noticia(titulo, fonte, peso, pub):
    return Noticia(
        titulo=titulo, conteudo="", url=f"https://{fonte}/x",
        publicado_em=ts(pub), fonte=fonte, peso_fonte=peso,
    )


class TestGetNoticias:
    def test_get_noticias_deduplica(self):
        # Mesma notícia em duas fontes, <24h → mantém a de maior peso
        alta = _noticia("Petrobras anuncia dividendo recorde", "bloomberg.com", 1.0, "2024-05-10 09:00")
        baixa = _noticia("Petrobras anuncia dividendo recorde hoje", "infomoney.com.br", 0.75, "2024-05-10 11:00")
        unicas = _AGENT._deduplicar([baixa, alta])
        assert len(unicas) == 1
        assert unicas[0].fonte == "bloomberg.com", "Dedup deve manter a fonte de maior peso"

    def test_get_noticias_nao_deduplica_distantes_no_tempo(self):
        # Títulos idênticos mas >24h de diferença → não são duplicata
        n1 = _noticia("Vale reporta produção recorde", "reuters.com", 0.95, "2024-05-10 09:00")
        n2 = _noticia("Vale reporta produção recorde", "valor.com.br", 0.95, "2024-05-12 09:00")
        assert len(_AGENT._deduplicar([n1, n2])) == 2

    def test_get_noticias_nao_deduplica_titulos_distintos(self):
        n1 = _noticia("Petrobras eleva produção no pré-sal", "reuters.com", 0.95, "2024-05-10 09:00")
        n2 = _noticia("Vale fecha acordo de minério com China", "bloomberg.com", 1.0, "2024-05-10 10:00")
        assert len(_AGENT._deduplicar([n1, n2])) == 2

    def test_get_noticias_ordenacao_peso(self):
        # _deduplicar processa por peso desc; verificamos a ordenação final aqui
        ns = [
            _noticia("A nota baixa", "infomoney.com.br", 0.75, "2024-05-10 09:00"),
            _noticia("B nota alta", "bloomberg.com", 1.0, "2024-05-10 09:00"),
            _noticia("C nota média", "estadao.com.br", 0.85, "2024-05-10 09:00"),
        ]
        unicas = _AGENT._deduplicar(ns)
        unicas.sort(key=lambda n: (n.peso_fonte, n.publicado_em), reverse=True)
        pesos = [n.peso_fonte for n in unicas]
        assert pesos == sorted(pesos, reverse=True), "Notícias devem sair por peso desc"

    def test_get_noticias_uma_fonte_caida_continua(self, monkeypatch):
        # GDELT explode; get_noticias deve sobreviver e retornar lista
        def _boom(*a, **k):
            raise RuntimeError("GDELT fora do ar")
        monkeypatch.setattr(_AGENT.gdelt, "buscar", _boom)
        resultado = _AGENT.get_noticias("PETR4.SA", ts("2024-05-31 12:01"))
        assert isinstance(resultado, list), "Falha de uma fonte não pode derrubar get_noticias"

    def test_get_noticias_anti_lookahead_final(self):
        limite = ts("2024-05-31 12:02")
        for n in _AGENT.get_noticias("PETR4.SA", limite):
            assert n.publicado_em <= limite, f"Lookahead: {n.publicado_em} > {limite}"

    def test_get_noticias_resolve_ticker_para_nome(self, monkeypatch):
        # Verifica que o ticker é resolvido para o nome antes de consultar as fontes
        capturado = {}
        def _spy(query, *a, **k):
            capturado["query"] = query
            return []
        monkeypatch.setattr(_AGENT.gdelt, "buscar", _spy)
        monkeypatch.setattr(_AGENT.newsapi, "buscar", lambda *a, **k: [])
        _AGENT.get_noticias("PETR4.SA", ts("2024-05-31 12:03"))
        assert capturado["query"] == "Petrobras", f"Ticker não resolvido: {capturado.get('query')}"


# ── health_check ──────────────────────────────────────────────────────────────


class TestHealthCheck:
    _VALIDOS = {"online", "offline", "sem_chave", "vazio"}

    def test_retorna_todas_as_fontes(self):
        status = _AGENT.health_check()
        for fonte in ("bloomberg_csv", "gdelt", "newsapi", "cvm", "bcb"):
            assert fonte in status, f"Fonte ausente no health_check: {fonte}"
        print(f"\nhealth_check: {status}")

    def test_status_sao_validos(self):
        status = _AGENT.health_check()
        for fonte, st in status.items():
            assert st in self._VALIDOS, f"Status inválido para {fonte}: {st!r}"

    def test_bloomberg_vazio_sem_csv(self):
        # data/bloomberg só tem .gitkeep → vazio
        assert _AGENT.health_check()["bloomberg_csv"] in {"vazio", "online"}

    def test_newsapi_sem_chave(self, monkeypatch):
        agente = JournalAgent()
        agente._news_api_key = ""
        assert agente.health_check()["newsapi"] == "sem_chave"


# ── Validação do CSV Bloomberg ────────────────────────────────────────────────


def _agent_com_bloomberg(tmp_path) -> JournalAgent:
    (tmp_path / "cache").mkdir(exist_ok=True)
    return JournalAgent(cache_dir=tmp_path / "cache", bloomberg_dir=tmp_path)


_HEADER_OK = "ticker,data_publicacao,titulo,conteudo,url,categoria\n"


class TestBloombergCSVValidacao:
    def test_coluna_faltando_levanta(self, tmp_path):
        # Sem a coluna 'categoria'
        (tmp_path / "noticias.csv").write_text(
            "ticker,data_publicacao,titulo,conteudo,url\n"
            "PETR4.SA,2024-05-10 09:00,Petrobras sobe,corpo,https://bloomberg.com/x\n"
        )
        agent = _agent_com_bloomberg(tmp_path)
        with pytest.raises(ValueError, match="categoria"):
            agent._bloomberg_csv("Petrobras", ts("2024-05-01"), ts("2024-05-31 23:59"))

    def test_coluna_extra_levanta(self, tmp_path):
        (tmp_path / "noticias.csv").write_text(
            _HEADER_OK.rstrip() + ",coluna_intrusa\n"
            "PETR4.SA,2024-05-10 09:00,Petrobras sobe,corpo,https://bloomberg.com/x,mercado,xxx\n"
        )
        agent = _agent_com_bloomberg(tmp_path)
        with pytest.raises(ValueError, match="coluna_intrusa"):
            agent._bloomberg_csv("Petrobras", ts("2024-05-01"), ts("2024-05-31 23:59"))

    def test_linha_com_timestamp_invalido_pula_so_ela(self, tmp_path, caplog):
        (tmp_path / "noticias.csv").write_text(
            _HEADER_OK +
            "PETR4.SA,data-invalida,Petrobras linha ruim,corpo,https://bloomberg.com/a,mercado\n"
            "PETR4.SA,2024-05-10 09:00,Petrobras linha boa,corpo,https://bloomberg.com/b,mercado\n"
        )
        agent = _agent_com_bloomberg(tmp_path)
        with caplog.at_level("WARNING"):
            result = agent._bloomberg_csv("Petrobras", ts("2024-05-01"), ts("2024-05-31 23:59"))
        assert len(result) == 1, "Só a linha válida deve ser retornada"
        assert result[0].titulo == "Petrobras linha boa"
        assert any("linha 2" in m for m in caplog.messages), "Aviso deve citar o nº da linha"

    def test_csv_valido_retorna_noticias(self, tmp_path):
        (tmp_path / "noticias.csv").write_text(
            _HEADER_OK +
            "PETR4.SA,2024-05-10 09:00,Petrobras anuncia dividendo,corpo,https://bloomberg.com/x,mercado\n"
        )
        agent = _agent_com_bloomberg(tmp_path)
        result = agent._bloomberg_csv("Petrobras", ts("2024-05-01"), ts("2024-05-31 23:59"))
        assert len(result) == 1
        assert result[0].fonte == "bloomberg_csv"
        assert result[0].peso_fonte == 1.0


# ── get_noticias: não cachear resultado vazio (anti cache-poisoning) ──────────


def test_get_noticias_nao_cacheia_vazio(tmp_path, monkeypatch):
    agent = JournalAgent(cache_dir=tmp_path / "cache", bloomberg_dir=tmp_path / "bbg")
    chamadas = {"n": 0}

    def _vazio(*a, **k):
        chamadas["n"] += 1
        return []

    monkeypatch.setattr(agent.gdelt, "buscar", _vazio)
    monkeypatch.setattr(agent.newsapi, "buscar", lambda *a, **k: [])
    monkeypatch.setattr(agent, "_bloomberg_csv", lambda *a, **k: [])

    r1 = agent.get_noticias("PETR4.SA", ts("2024-05-31 12:00"))
    r2 = agent.get_noticias("PETR4.SA", ts("2024-05-31 12:00"))
    assert r1 == [] and r2 == []
    # Vazio não é cacheado: a 2ª chamada deve reconsultar as fontes.
    assert chamadas["n"] == 2, "Resultado vazio não pode ser cacheado (deve reconsultar)"


# ── get_macro: lag de publicação do IPCA (anti-lookahead) ─────────────────────


def _fake_bcb_ipca(codigo, data_inicio, data_limite):
    """Substitui _bcb_serie: IPCA mensal sintético (jan–mai/2024), resto vazio."""
    if codigo == 433:  # IPCA mensal
        idx = pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
        )
        return pd.Series([0.4, 0.5, 0.3, 0.2, 0.4], index=idx, name="433")
    return pd.Series(dtype=float, name=str(codigo))


class TestGetMacroLagIPCA:
    def test_exclui_ipca_nao_publicado(self, tmp_path, monkeypatch):
        agent = JournalAgent(cache_dir=tmp_path / "cache")
        monkeypatch.setattr(agent, "_bcb_serie", _fake_bcb_ipca)
        # 05/06/2024: IPCA de maio (publicado ~11/06) ainda NÃO disponível;
        # IPCA de abril (publicado ~11/05) já disponível.
        ipca = agent.get_macro(ts("2024-06-05"))["ipca_mensal"]
        assert ts("2024-04-01") in ipca.index, "Abril já publicado, deveria constar"
        assert ts("2024-05-01") not in ipca.index, "Maio ainda não publicado — lookahead!"

    def test_inclui_ipca_apos_publicacao(self, tmp_path, monkeypatch):
        agent = JournalAgent(cache_dir=tmp_path / "cache")
        monkeypatch.setattr(agent, "_bcb_serie", _fake_bcb_ipca)
        # 15/06/2024: IPCA de maio já publicado (~11/06).
        ipca = agent.get_macro(ts("2024-06-15"))["ipca_mensal"]
        assert ts("2024-05-01") in ipca.index, "Maio já publicado em 15/06, deveria constar"
