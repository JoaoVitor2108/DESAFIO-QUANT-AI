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

    def test_lag_45_dias_respeitado(self):
        # Trimestre mais recente elegível deve satisfazer (fim + 45d) <= data_limite
        limite = pd.Timestamp.now(tz=_SP) - pd.Timedelta(days=60)
        fund = _AGENT.get_fundamentals("PETR4.SA", limite)
        if fund.trimestre_fim is not None:
            from config import LAG_FUNDAMENTALS_DIAS
            assert fund.trimestre_fim + pd.Timedelta(days=LAG_FUNDAMENTALS_DIAS) <= limite, (
                "Trimestre dentro do lag de 45 dias — lookahead nos fundamentals!"
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
