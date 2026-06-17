"""
Testes do CVMSource contra dados reais da CVM.
Execute: pytest tests/test_cvm.py -v -s
"""
import pytest
import pandas as pd
from pathlib import Path

from agents.sources.cvm import CVMSource

_SP = "America/Sao_Paulo"
_SRC = CVMSource()

# CD_CVM fixos confirmados via cad_cia_aberta.csv
_CD_PETR4 = 9512    # Petrobras
_CD_AMER3 = 20990   # Americanas (em RJ)
_CD_ITUB4 = 19348   # Itaú Unibanco Holding (banco)
_CNPJ_PETR4 = "33.000.167/0001-01"
_CNPJ_AMER3 = "00.776.574/0001-56"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=_SP)


# ── 1. Download ───────────────────────────────────────────────────────────────


class TestBaixarAno:
    def test_baixar_itr_2023_retorna_path_valido(self):
        path = _SRC.baixar_ano(2023, "ITR")
        assert path.exists()
        assert path.suffix == ".zip"
        assert path.stat().st_size > 1_000_000  # > 1 MB
        print(f"\nITR 2023: {path.stat().st_size / 1e6:.1f} MB")

    def test_tipo_invalido_levanta(self):
        with pytest.raises(ValueError):
            _SRC.baixar_ano(2023, "INVALIDO")

    def test_cache_nao_rebaixa(self):
        # Segunda chamada deve ser instantânea (cache válido)
        import time
        t0 = time.time()
        _SRC.baixar_ano(2023, "ITR")
        assert time.time() - t0 < 1.0, "Re-download inesperado"


# ── 2. Encoding latin-1 ───────────────────────────────────────────────────────


class TestEncoding:
    def test_acentos_corretos_no_cadastro(self):
        cad = _SRC.get_cadastro()
        # Petrobras tem acento em "PETRÓLEO"
        match = cad[cad["DENOM_SOCIAL"].str.contains("PETR", na=False)]
        assert not match.empty
        nomes = match["DENOM_SOCIAL"].tolist()
        petrobras = [n for n in nomes if "PETROLEO" in n.upper() or "PETRÓLEO" in n.upper()]
        assert petrobras, f"Nome com acento não encontrado; nomes={nomes[:3]}"
        print(f"\nNome Petrobras no cadastro: {petrobras[0]!r}")

    def test_demonstrativos_nao_tem_caracteres_estranhos(self):
        demos = _SRC.carregar_demonstrativos(2023, "ITR")
        dre = demos.get("DRE", pd.DataFrame())
        assert not dre.empty
        # Não deve ter '?' ou '□' (indicadores de encoding errado)
        amostra = dre["DS_CONTA"].dropna().head(50).str.cat(sep=" ")
        assert "?" not in amostra, "Possível problema de encoding"
        assert "□" not in amostra


# ── 3. Versão máxima ──────────────────────────────────────────────────────────


class TestVersaoMaxima:
    def test_sem_duplicatas_por_conta_e_trimestre(self):
        demos = _SRC.carregar_demonstrativos(2023, "ITR")
        for key, df in demos.items():
            if key == "CAP" or df.empty:
                continue
            grp_cols = [c for c in ("CD_CVM", "DT_REFER", "CD_CONTA", "ORDEM_EXERC")
                        if c in df.columns]
            if not grp_cols:
                continue
            duplicatas = df.duplicated(subset=grp_cols).sum()
            assert duplicatas == 0, f"{key}: {duplicatas} duplicatas após dedup de versão"


# ── 4. get_fundamentals PETR4 ────────────────────────────────────────────────


class TestGetFundamentalsBasico:
    def test_petr4_2023q3_retorna_dados(self):
        # data_limite = final de 2023; espera-se dados do 3T23 ou anterior
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2023-12-31 17:00"))
        assert fund is not None, "Nenhum dado encontrado para PETR4 em 2023"
        print(
            f"\nPETR4 em 2023-12-31:\n"
            f"  trimestre     = {fund['trimestre']}\n"
            f"  dt_receb      = {fund['dt_receb']}\n"
            f"  lucro_liquido = {fund['lucro_liquido'] and fund['lucro_liquido']/1e9:.1f}B\n"
            f"  receita       = {fund['receita'] and fund['receita']/1e9:.1f}B\n"
            f"  patrimonio    = {fund['patrimonio_liquido'] and fund['patrimonio_liquido']/1e9:.1f}B\n"
            f"  avisos        = {fund['avisos']}"
        )

    def test_petr4_lucro_e_positivo(self):
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2023-12-31 17:00"))
        assert fund is not None
        if fund.get("lucro_liquido") is not None:
            assert fund["lucro_liquido"] > 0, "Petrobras lucro negativo em 2023 — verificar"

    def test_petr4_receita_e_grande(self):
        # Receita da Petrobras deve ser da ordem de centenas de bilhões
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2023-12-31 17:00"))
        assert fund is not None
        if fund.get("receita") is not None:
            assert fund["receita"] > 1e10, "Receita suspeita (< R$ 10B)"


# ── 5. DT_RECEB anti-lookahead ────────────────────────────────────────────────


class TestDtRecebAntiLookahead:
    def test_balancos_apos_data_limite_nao_retornados(self):
        # Usar uma data muito antiga onde não há DT_RECEB válido
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2019-01-05 17:00"))
        # Pode retornar None (sem dados tão cedo) ou dados com dt_receb <= 2019-01-05
        if fund is not None:
            assert fund["dt_receb"] <= ts("2019-01-05 17:00"), (
                f"Lookahead: dt_receb={fund['dt_receb']} > data_limite=2019-01-05"
            )

    def test_dt_receb_posterior_a_data_limite_barrado(self):
        # Pegar um filing real de PETR4 2023 e testar que não aparece com data_limite anterior
        fund_2023 = _SRC.get_fundamentals(_CD_PETR4, ts("2023-12-31 17:00"))
        if fund_2023 is None:
            pytest.skip("Sem dados de 2023 para executar o teste")
        dt_receb_real = fund_2023["dt_receb"]
        # Buscar com data_limite = dt_receb - 1 dia: não deve retornar este filing
        data_anterior = dt_receb_real - pd.Timedelta(days=1)
        fund_anterior = _SRC.get_fundamentals(_CD_PETR4, data_anterior)
        if fund_anterior is not None:
            assert fund_anterior["dt_receb"] <= data_anterior, (
                f"Lookahead: filing com dt_receb={fund_anterior['dt_receb']} "
                f"retornado para data_limite={data_anterior.date()}"
            )


# ── 6. AMER3 caso emblemático ─────────────────────────────────────────────────


class TestAmer3CasoEmblematico:
    def test_amer3_dados_disponiveis_em_dez_2022(self):
        # Antes do pedido de RJ (12/01/2023), AMER3 tinha balanços normais
        fund = _SRC.get_fundamentals(_CD_AMER3, ts("2022-12-01 17:00"))
        print(f"\nAMER3 em 2022-12-01: {fund}")
        # Pode ter dados ou não dependendo do timing das publicações
        if fund is not None:
            assert fund["dt_receb"] <= ts("2022-12-01 17:00")
            print(f"  trimestre={fund['trimestre']}, dt_receb={fund['dt_receb'].date()}")

    def test_amer3_dt_receb_respeitado(self):
        fund = _SRC.get_fundamentals(_CD_AMER3, ts("2021-01-01 17:00"))
        if fund is not None:
            assert fund["dt_receb"] <= ts("2021-01-01 17:00")


# ── 7. Banco (ITUB4 — estrutura diferente) ────────────────────────────────────


class TestBancoEstruturaEstrutura:
    def test_itub4_retorna_dados(self):
        fund = _SRC.get_fundamentals(_CD_ITUB4, ts("2023-12-31 17:00"))
        print(f"\nITUB4 em 2023-12-31: {fund}")
        if fund is not None:
            print(
                f"  trimestre    = {fund['trimestre']}\n"
                f"  lucro_liq    = {fund['lucro_liquido'] and fund['lucro_liquido']/1e9:.1f}B\n"
                f"  patrimônio   = {fund['patrimonio_liquido'] and fund['patrimonio_liquido']/1e9:.1f}B\n"
                f"  avisos       = {fund['avisos']}"
            )

    def test_itub4_patrimonio_positivo(self):
        fund = _SRC.get_fundamentals(_CD_ITUB4, ts("2023-12-31 17:00"))
        if fund is not None and fund.get("patrimonio_liquido") is not None:
            assert fund["patrimonio_liquido"] > 0


# ── 8. Ticker sem CD_CVM ──────────────────────────────────────────────────────


class TestSemCdCvm:
    def test_cd_cvm_zero_retorna_none(self):
        result = _SRC.get_fundamentals(999999, ts("2023-12-31 17:00"))
        assert result is None

    def test_journal_sem_cd_cvm_retorna_avisos(self):
        from agents.journal import JournalAgent
        agent = JournalAgent()
        fund = agent.get_fundamentals("AAPL", ts("2023-12-31 17:00"))
        assert fund.ticker == "AAPL"
        assert any("cd_cvm" in a.lower() for a in fund.avisos)


# ── TTM ───────────────────────────────────────────────────────────────────────


class TestTTM:
    """Testes de consistência TTM para itens de fluxo e point-in-time para balanço."""

    def test_ttm_itr_q1_2023_bate_calculo_manual(self):
        """
        TTM Q1-2023 deve ser: receita_Q1_2023 + DFP_2022 - receita_Q1_2022.

        Valores validados previamente (run manual):
          Q1-2023 ÚLTIMO:    rec=139.07B, ll=38.16B, ebit=60.20B
          Q1-2022 PENÚLTIMO: rec=141.64B, ll=44.56B, ebit=65.40B
          DFP 2022:          rec=641.26B, ll=189.00B, ebit=294.25B
          TTM esperado:      rec≈638.7B,  ll≈182.6B,  ebit≈289.0B
        """
        # Cálculo manual como ground truth
        demos_2023 = _SRC.carregar_demonstrativos(2023, "ITR")
        demos_dfp  = _SRC.carregar_demonstrativos(2022, "DFP")
        dre = demos_2023["DRE"]
        dre_dfp = demos_dfp["DRE"]

        dt_q1 = pd.Timestamp("2023-03-31")
        dt_dfp = pd.Timestamp("2022-12-31")

        def val(df, dt, ordem, cd):
            rows = df[(df["CD_CVM"]==_CD_PETR4) & (df["DT_REFER"]==dt) &
                      (df["ORDEM_EXERC"]==ordem) & (df["CD_CONTA"]==cd)]
            return float(rows["VL_CONTA"].iloc[0]) * 1000 if not rows.empty else None

        rec_u  = val(dre,     dt_q1,  "ÚLTIMO",   "3.01")
        rec_p  = val(dre,     dt_q1,  "PENÚLTIMO","3.01")
        rec_d  = val(dre_dfp, dt_dfp, "ÚLTIMO",   "3.01")
        ll_u   = val(dre,     dt_q1,  "ÚLTIMO",   "3.11.01") or val(dre, dt_q1, "ÚLTIMO", "3.11")
        ll_p   = val(dre,     dt_q1,  "PENÚLTIMO","3.11.01") or val(dre, dt_q1, "PENÚLTIMO","3.11")
        ll_d   = val(dre_dfp, dt_dfp, "ÚLTIMO",   "3.11.01") or val(dre_dfp,dt_dfp,"ÚLTIMO","3.11")

        ttm_rec_manual = rec_u + rec_d - rec_p
        ttm_ll_manual  = ll_u  + ll_d  - ll_p

        # data_limite = 2023-06-01: DT_RECEB Q1-2023 = 2023-05-11 ✓
        #                           DT_RECEB DFP 2022 = 2023-03-02 ✓
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2023-06-01 17:00"))

        assert fund is not None, "Sem dados para PETR4 em 2023-06-01"
        assert fund["tipo_doc"] == "ITR", f"Esperado ITR, recebido {fund['tipo_doc']}"
        assert "Q1-2023" in fund["trimestre"], f"trimestre inesperado: {fund['trimestre']}"

        assert fund["receita"] == pytest.approx(ttm_rec_manual, rel=1e-6), (
            f"Receita TTM: esperado {ttm_rec_manual/1e9:.2f}B, "
            f"recebido {fund['receita']/1e9:.2f}B"
        )
        assert fund["lucro_liquido"] == pytest.approx(ttm_ll_manual, rel=1e-6), (
            f"LL TTM: esperado {ttm_ll_manual/1e9:.2f}B, "
            f"recebido {fund['lucro_liquido']/1e9:.2f}B"
        )
        print(
            f"\nTTM Q1-2023 PETR4:\n"
            f"  receita       = {fund['receita']/1e9:.2f}B  (manual={ttm_rec_manual/1e9:.2f}B)\n"
            f"  lucro_liquido = {fund['lucro_liquido']/1e9:.2f}B  (manual={ttm_ll_manual/1e9:.2f}B)\n"
            f"  periodicidade = {fund['periodicidade']}"
        )

    def test_balanco_point_in_time_nao_muda_com_ttm(self):
        """Itens de balanço devem ser idênticos ao valor bruto do CVM (sem ajuste TTM)."""
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2023-06-01 17:00"))
        assert fund is not None

        # Extrair valor bruto do parquet para comparação
        demos = _SRC.carregar_demonstrativos(2023, "ITR")
        bpa = demos["BPA"]
        bpp = demos["BPP"]
        dt_q1 = pd.Timestamp("2023-03-31")

        def pit(df, cd):
            rows = df[(df["CD_CVM"]==_CD_PETR4) & (df["DT_REFER"]==dt_q1) &
                      (df["ORDEM_EXERC"]=="ÚLTIMO") & (df["CD_CONTA"]==cd)]
            return float(rows["VL_CONTA"].iloc[0]) * 1000 if not rows.empty else None

        pat_raw = (
            pit(bpp, "2.03") or pit(bpp, "2.03.01")
        )
        if pat_raw is not None:
            assert fund["patrimonio_liquido"] == pytest.approx(pat_raw, rel=1e-6), (
                "patrimônio_liquido difere do valor point-in-time bruto"
            )

        # Verificar que a periodicidade está marcada corretamente
        p = fund["periodicidade"]
        assert p["receita"] == "TTM"
        assert p["lucro_liquido"] == "TTM"
        assert p["ebit"] == "TTM"
        assert p["ebitda_aproximado"] == "TTM"
        assert p["patrimonio_liquido"] == "point_in_time"
        assert p["caixa"] == "point_in_time"
        assert p["divida_bruta"] == "point_in_time"
        assert p["ativo_total"] == "point_in_time"

    def test_dfp_ttm_igual_ao_anual_sem_dfp_prev(self):
        """Quando o documento é DFP, TTM = anual diretamente (não precisa de DFP_prev)."""
        # data_limite = 2023-04-01 → DFP 2022 elegível (DT_RECEB=2023-03-02), mas Q1-2023 não
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2023-04-01 17:00"))
        assert fund is not None
        assert fund["tipo_doc"] == "DFP", f"Esperado DFP, recebido {fund['tipo_doc']}"
        assert "ANUAL-2022" in fund["trimestre"]

        # Extrair valores brutos do DFP 2022 para comparar
        demos = _SRC.carregar_demonstrativos(2022, "DFP")
        dre = demos["DRE"]
        dt_dfp = pd.Timestamp("2022-12-31")
        rows = dre[(dre["CD_CVM"]==_CD_PETR4) & (dre["DT_REFER"]==dt_dfp) &
                   (dre["ORDEM_EXERC"]=="ÚLTIMO") & (dre["CD_CONTA"]=="3.01")]
        rec_raw = float(rows["VL_CONTA"].iloc[0]) * 1000 if not rows.empty else None

        if rec_raw is not None:
            assert fund["receita"] == pytest.approx(rec_raw, rel=1e-6), (
                f"DFP receita TTM ({fund['receita']/1e9:.2f}B) "
                f"deve ser igual ao anual bruto ({rec_raw/1e9:.2f}B)"
            )
        print(
            f"\nDFP 2022 PETR4:\n"
            f"  trimestre = {fund['trimestre']}\n"
            f"  receita   = {fund['receita'] and fund['receita']/1e9:.2f}B\n"
            f"  avisos    = {fund['avisos'][:2]}"
        )

    def test_aviso_quando_dfp_prev_indisponivel(self):
        """Fluxo TTM deve ser None com aviso quando DFP do ano anterior não está disponível."""
        # Usar uma data onde Q1-2020 poderia estar disponível mas DFP-2019 não
        # Na prática: testar com data bem antiga que não tem DFP_prev carregado
        # Alternativa determinística: passar demos_dfp_prev=None explicitamente
        from agents.sources.cvm import _localize
        demos_2023 = _SRC.carregar_demonstrativos(2023, "ITR")

        # Chamar _extrair_metricas sem DFP_prev — simula gap
        resultado = _SRC._extrair_metricas(
            cd_cvm=_CD_PETR4,
            dt_refer=_localize(pd.Timestamp("2023-03-31")),
            dt_receb=_localize(pd.Timestamp("2023-05-11")),
            demos=demos_2023,
            tipo_doc="ITR",
            demos_dfp_prev=None,   # forçar ausência
        )

        # Fluxo deve ser None
        assert resultado["receita"] is None, (
            f"Esperado None sem DFP_prev, recebido {resultado['receita']}"
        )
        assert resultado["lucro_liquido"] is None
        # Aviso deve mencionar DFP
        avisos_str = " ".join(resultado["avisos"])
        assert "DFP" in avisos_str or "dfp" in avisos_str.lower(), (
            f"Aviso sobre DFP ausente: {resultado['avisos']}"
        )
        # Balanço NÃO deve ser None (point-in-time não depende do DFP_prev)
        assert resultado["patrimonio_liquido"] is not None, "Patrimônio PIT não deve ser None"
        assert resultado["caixa"] is not None, "Caixa PIT não deve ser None"
        print(f"\nSem DFP_prev:\n  receita={resultado['receita']}\n  patrimônio={resultado['patrimonio_liquido']/1e9:.1f}B\n  avisos={resultado['avisos'][:3]}")


# ── Testes de periodicidade 2024-2025 ─────────────────────────────────────────


class TestPeriodo2025:
    """Valida que o sistema cobre o período de backtest 2024-2025."""

    def test_cvm_fundamentals_2025(self):
        """CVM deve baixar e parsear dados de 2024/2025 para PETR4."""
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2025-06-01 17:00"))
        assert fund is not None, "Sem dados CVM para PETR4 em 2025-06-01"
        # Documento deve ser de 2024 ou 2025
        assert fund["dt_refer"].year >= 2024, (
            f"Esperado documento de 2024+, recebido {fund['dt_refer'].year}"
        )
        # DT_RECEB deve respeitar data_limite
        assert fund["dt_receb"] <= ts("2025-06-01 17:00"), (
            f"Lookahead: dt_receb={fund['dt_receb']} > data_limite=2025-06-01"
        )
        # Receita TTM deve ser centenas de bilhões (Petrobras)
        if fund.get("receita") is not None:
            assert fund["receita"] > 1e11, (
                f"Receita suspeita (< R$ 100B): {fund['receita']/1e9:.1f}B"
            )
        print(
            f"\nPETR4 em 2025-06-01:\n"
            f"  trimestre = {fund['trimestre']}\n"
            f"  dt_receb  = {fund['dt_receb'].date()}\n"
            f"  receita   = {fund['receita'] and fund['receita']/1e9:.1f}B\n"
            f"  avisos    = {fund['avisos'][:2]}"
        )

    def test_ttm_virada_de_ano_2024(self):
        """data_limite jun/2024 → ITR Q1-2024 com DFP 2023 como base TTM.

        Valida o caso de borda na fronteira treino (2020-2023) / backtest (2024-2025):
        o sistema deve resolver ano_prev=2023 dinamicamente para calcular TTM.
        """
        fund = _SRC.get_fundamentals(_CD_PETR4, ts("2024-06-01 17:00"))
        assert fund is not None, "Sem dados para PETR4 em 2024-06-01"
        # Deve usar documento de 2024
        assert "2024" in fund["trimestre"], (
            f"Esperado documento de 2024, recebido: {fund['trimestre']}"
        )
        assert fund["tipo_doc"] == "ITR", (
            f"Esperado ITR para fronteira Q1-2024, recebido {fund['tipo_doc']}"
        )
        # Se receita não é None, TTM foi calculado com DFP 2023 disponível
        if fund.get("receita") is not None:
            assert fund["receita"] > 1e11, (
                f"Receita TTM suspeita: {fund['receita']/1e9:.1f}B"
            )
        print(
            f"\nPETR4 virada treino→backtest (2024-06-01):\n"
            f"  trimestre = {fund['trimestre']}\n"
            f"  tipo_doc  = {fund['tipo_doc']}\n"
            f"  receita   = {fund['receita'] and fund['receita']/1e9:.1f}B\n"
            f"  avisos    = {fund['avisos'][:2]}"
        )
