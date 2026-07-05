from __future__ import annotations

import pandas as pd

FUSO = "America/Sao_Paulo"
LAG_FUNDAMENTALS_DIAS = 45  # mantido para fallback yfinance; CVM usa DT_RECEB
HORARIO_CORTE_B3 = "17:05"

# ── Periodização do sistema ───────────────────────────────────────────────────
# Warmup:          2019          (momentum usa 252 dias úteis)
# Calibração ECON: 2020-2021
# Treino MATH&ML:  2020-2023     (4 anos, inclui regime adverso 2022 e recuperação 2023)
# Backtest OOS:    2024-2025     (dados consolidados, condições de mercado recentes)
INICIO_WARMUP            = pd.Timestamp("2019-01-01",          tz=FUSO)
INICIO_CALIBRACAO_ECON   = pd.Timestamp("2020-01-01",          tz=FUSO)
FIM_CALIBRACAO_ECON      = pd.Timestamp("2021-12-31 23:59:59", tz=FUSO)
INICIO_TREINO            = pd.Timestamp("2020-01-01",          tz=FUSO)
FIM_TREINO               = pd.Timestamp("2023-12-31 23:59:59", tz=FUSO)
INICIO_BACKTEST          = pd.Timestamp("2024-01-01",          tz=FUSO)
FIM_BACKTEST             = pd.Timestamp("2025-12-31 23:59:59", tz=FUSO)


def _ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str, tz=FUSO)


# Universo histórico: composição do IBOV ao longo de 2019-2025.
# entrada=None  → ticker já estava no índice em 01/01/2019.
# saida=None    → ticker ainda estava no índice em 31/12/2025.
# cd_cvm        → código CVM (fonte: cad_cia_aberta.csv). Usado pelo CVMSource.
# cnpj          → CNPJ da empresa (redundante, facilita joins com CVM).
# Todos os timestamps são timezone-aware em America/Sao_Paulo.
UNIVERSO_HISTORICO: dict[str, dict] = {
    # ── petroleo_gas ─────────────────────────────────────────────────────────
    "PETR4.SA": {
        "setor": "petroleo_gas",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 9512,
        "cnpj": "33.000.167/0001-01",
    },
    "PRIO3.SA": {
        "setor": "petroleo_gas",
        "entrada": _ts("2020-09-08"),
        "saida": None,
        "confianca": "alta",
        "fonte": "PetroRio ingressou no IBOV em 08/09/2020 (Money Times)",
        "cd_cvm": 22187,
        "cnpj": "10.629.105/0001-68",
    },
    # ── mineracao ─────────────────────────────────────────────────────────────
    "VALE3.SA": {
        "setor": "mineracao",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 4170,
        "cnpj": "33.592.510/0001-54",
    },
    "CMIN3.SA": {
        "setor": "mineracao",
        "entrada": _ts("2021-02-18"),
        "saida": None,
        "confianca": "alta",
        "fonte": "IPO CSN Mineração 18/02/2021",
        "cd_cvm": 25585,
        "cnpj": "08.902.291/0001-15",
    },
    # ── industria ─────────────────────────────────────────────────────────────
    "WEGE3.SA": {
        "setor": "industria",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "WEG no IBOV desde 2016",
        "cd_cvm": 5410,
        "cnpj": "84.429.695/0001-11",
    },
    "GGBR4.SA": {
        "setor": "industria",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019 (siderurgia/metalurgia)",
        "cd_cvm": 3980,
        "cnpj": "33.611.500/0001-19",
    },
    # ── bancos ────────────────────────────────────────────────────────────────
    "ITUB4.SA": {
        "setor": "bancos",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 19348,
        "cnpj": "60.872.504/0001-23",
    },
    "BBDC4.SA": {
        "setor": "bancos",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 906,
        "cnpj": "60.746.948/0001-12",
    },
    "BBAS3.SA": {
        "setor": "bancos",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 1023,
        "cnpj": "00.000.000/0001-91",
    },
    "BPAC11.SA": {
        "setor": "bancos",
        "entrada": _ts("2019-10-02"),
        "saida": None,
        "confianca": "alta",
        "fonte": "BTG Pactual entrou no IBOV em 02/10/2019",
        "cd_cvm": 22616,
        "cnpj": "30.306.294/0001-45",
    },
    # ── alimentos_bebidas ─────────────────────────────────────────────────────
    "ABEV3.SA": {
        "setor": "alimentos_bebidas",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 23264,
        "cnpj": "07.526.557/0001-00",
    },
    "JBSS3.SA": {
        "setor": "alimentos_bebidas",
        "entrada": None,
        "saida": _ts("2025-06-06"),
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 20575,
        "cnpj": "02.916.265/0001-60",
    },
    # ── energia_eletrica ──────────────────────────────────────────────────────
    # ELET3: Privatizada jun/2022; renomeada para AXIA ENERGIA S.A. na CVM, mas
    # CD_CVM=2437 e CNPJ permanecem os mesmos. Histórico CVM contínuo.
    "ELET3.SA": {
        "setor": "energia_eletrica",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019; privatizada jun/2022 → AXIA ENERGIA",
        "cd_cvm": 2437,
        "cnpj": "00.001.180/0001-26",
    },
    "EGIE3.SA": {
        "setor": "energia_eletrica",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 17329,
        "cnpj": "02.474.103/0001-19",
    },
    # ── papel_celulose ────────────────────────────────────────────────────────
    "SUZB3.SA": {
        "setor": "papel_celulose",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019; fusão Suzano+Fibria concluída jan/2019",
        "cd_cvm": 13986,
        "cnpj": "16.404.287/0001-55",
    },
    "KLBN11.SA": {
        "setor": "papel_celulose",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 12653,
        "cnpj": "89.637.490/0001-45",
    },
    # ── telecom ───────────────────────────────────────────────────────────────
    "VIVT3.SA": {
        "setor": "telecom",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 17671,
        "cnpj": "02.558.157/0001-62",
    },
    # ── saude ─────────────────────────────────────────────────────────────────
    "RDOR3.SA": {
        "setor": "saude",
        "entrada": _ts("2021-09-06"),
        "saida": None,
        "confianca": "alta",
        "fonte": "Inclusão no IBOV 06/09/2021; IPO 10/12/2020",
        "cd_cvm": 24821,
        "cnpj": "06.047.087/0001-39",
    },
    # ── varejo ────────────────────────────────────────────────────────────────
    "LREN3.SA": {
        "setor": "varejo",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 8133,
        "cnpj": "92.754.738/0001-62",
    },
    # TODO: verificar data exata de exclusão — estimativa rebalanceamento jan/2025
    "MGLU3.SA": {
        "setor": "varejo",
        "entrada": None,
        "saida": _ts("2025-01-06"),
        "confianca": "baixa",
        "fonte": "B3 composição jan/2019; excluído IBOV ~jan/2025 (queda ~98% desde pico)",
        "cd_cvm": 22470,
        "cnpj": "47.960.950/0001-21",
    },
    # Survivorship bias — padrão "colapso abrupto": Americanas entrou em RJ em
    # jan/2023. O modelo treina com AMER3 durante a janela ativa e não projeta
    # além de saida. AMER3 (B2W renomeada) começou a negociar em 19/07/2021;
    # LAME4 foi incorporada e deslistada em 24/01/2022.
    "AMER3.SA": {
        "setor": "varejo",
        "entrada": _ts("2021-07-19"),
        "saida": _ts("2023-01-12"),
        "confianca": "alta",
        "fonte": "AMER3 início 19/07/2021; recuperação judicial 12/01/2023",
        "cd_cvm": 20990,
        "cnpj": "00.776.574/0001-56",
    },
    # ── construcao ────────────────────────────────────────────────────────────
    "CYRE3.SA": {
        "setor": "construcao",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
        "cd_cvm": 14460,
        "cnpj": "73.178.600/0001-18",
    },
    # ── tecnologia ────────────────────────────────────────────────────────────
    "TOTS3.SA": {
        "setor": "tecnologia",
        "entrada": _ts("2020-01-06"),
        "saida": None,
        "confianca": "alta",
        "fonte": "Totvs entrou no IBOV em 06/01/2020",
        "cd_cvm": 19992,
        "cnpj": "53.113.791/0001-22",
    },
    # ── outros ────────────────────────────────────────────────────────────────
    # Survivorship bias — padrão "lenta agonia": IRB revelou fraude contábil em
    # fev/2020 e caiu ~80%, mas só foi excluído do IBOV no rebalanceamento de
    # jan/2023. Demonstra que ticker permanece no universo até a saída formal,
    # mesmo durante crise prolongada — não julgamento qualitativo.
    "IRBR3.SA": {
        "setor": "outros",
        "entrada": None,
        "saida": _ts("2023-01-02"),
        "confianca": "media",
        "fonte": "Excluído IBOV jan/2023; escândalo contábil fev/2020",
        "cd_cvm": 24180,
        "cnpj": "33.376.989/0001-91",
    },
}


def tickers_ativos(data: pd.Timestamp) -> list[str]:
    """Retorna tickers cuja janela [entrada, saida] contém `data`.

    Levanta ValueError se `data` for naive (sem timezone).
    """
    if data.tzinfo is None:
        raise ValueError(
            f"tickers_ativos requer timestamp timezone-aware; recebeu naive: {data}"
        )
    result = []
    for ticker, info in UNIVERSO_HISTORICO.items():
        entrada = info["entrada"] if info["entrada"] is not None else INICIO_WARMUP
        saida = info["saida"] if info["saida"] is not None else FIM_BACKTEST
        if entrada <= data <= saida:
            result.append(ticker)
    return result


# Mapa setor → [tickers] derivado do universo completo (não filtrado por data).
# Para iteração time-aware, usar tickers_ativos(data) filtrado por setor.
MAPA_SETORES: dict[str, list[str]] = {}
for _ticker, _info in UNIVERSO_HISTORICO.items():
    MAPA_SETORES.setdefault(_info["setor"], []).append(_ticker)


# ── Fontes de notícia ─────────────────────────────────────────────────────────
# Whitelist de domínios de imprensa econômica reconhecida → peso de confiança.
# Apenas artigos destes domínios são aceitos; o peso entra no ranking e na
# deduplicação (mantém-se a versão da fonte de maior peso). Casamento por
# substring na URL/domínio (cobre www. e subdomínios).
WHITELIST_FONTES: dict[str, float] = {
    "bloomberg.com": 1.00,
    "reuters.com": 0.95,
    "valor.globo.com": 0.95,
    "valor.com.br": 0.95,
    "broadcast.com.br": 0.90,
    "economia.estadao.com.br": 0.85,
    "estadao.com.br": 0.85,
    "exame.com": 0.75,
    "infomoney.com.br": 0.75,
    "moneytimes.com.br": 0.70,
}

# Tabela ticker → nome da empresa para montar queries de notícia.
# Buscar pelo nome reconhecido retorna mais resultados que pelo ticker cru.
# get_noticias resolve o ticker para o nome antes de consultar as fontes.
TICKER_PARA_NOME: dict[str, str] = {
    "PETR4.SA": "Petrobras",
    "PRIO3.SA": "PRIO Petrorio",
    "VALE3.SA": "Vale",
    "CMIN3.SA": "CSN Mineração",
    "WEGE3.SA": "WEG",
    "GGBR4.SA": "Gerdau",
    "ITUB4.SA": "Itaú Unibanco",
    "BBDC4.SA": "Bradesco",
    "BBAS3.SA": "Banco do Brasil",
    "BPAC11.SA": "BTG Pactual",
    "ABEV3.SA": "Ambev",
    "JBSS3.SA": "JBS",
    "ELET3.SA": "Eletrobras",
    "EGIE3.SA": "Engie Brasil",
    "SUZB3.SA": "Suzano",
    "KLBN11.SA": "Klabin",
    "VIVT3.SA": "Vivo Telefônica Brasil",
    "RDOR3.SA": "Rede D'Or",
    "LREN3.SA": "Lojas Renner",
    "MGLU3.SA": "Magazine Luiza",
    "AMER3.SA": "Americanas",
    "CYRE3.SA": "Cyrela",
    "TOTS3.SA": "Totvs",
    "IRBR3.SA": "IRB Brasil Re",
}
