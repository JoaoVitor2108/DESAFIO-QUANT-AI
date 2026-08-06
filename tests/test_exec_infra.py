"""Testes da infra de execução paga compartilhada (sem rede, sem API).

Checkpoint e retry já são exercidos via reexport em `tests/test_gate_custo.py`;
aqui cobrimos o que é novo: o pré-fetch de macro, cujo contrato crítico é NÃO
vazar dados posteriores à `data_limite` de cada evento.
"""
import pandas as pd
import pytest

from config import FUSO
from calibration.exec_infra import (
    Checkpoint,
    MacroIndisponivel,
    custo_da_chamada,
    prefetch_macro,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=FUSO)


class _JournalMacro:
    """Journal falso que conta idas à 'rede' e devolve a série até data_limite."""

    def __init__(self, vazia: bool = False):
        self.chamadas = 0
        self.vazia = vazia

    def get_macro(self, data_limite):
        self.chamadas += 1
        idx = pd.date_range("2024-01-01", "2025-12-31", freq="D", tz=FUSO)
        if self.vazia:
            return {"selic_diaria": pd.Series(dtype=float)}
        serie = pd.Series(range(len(idx)), index=idx, dtype=float)
        return {
            "selic_diaria": serie[serie.index <= data_limite],
            "ipca_12m": pd.Series(dtype=float),  # série vazia coexiste
        }


class TestPrefetchMacro:
    def test_uma_unica_ida_a_rede_para_muitos_eventos(self):
        """É o ponto do pré-fetch: 500 datas distintas não podem virar 500 buscas
        da série inteira (foi assim que a rodada levou rate-limit do BCB)."""
        j = _JournalMacro()
        prefetch_macro(j, ts("2025-12-31"))

        for dia in range(1, 20):
            j.get_macro(ts(f"2025-09-{dia:02d}"))

        assert j.chamadas == 1

    def test_fatia_nunca_ultrapassa_a_data_limite(self):
        """ANTI-LOOKAHEAD: o range maior é só I/O; o payload vê só o passado."""
        j = _JournalMacro()
        prefetch_macro(j, ts("2025-12-31"))

        corte = ts("2025-06-15")
        macro = j.get_macro(corte)

        assert macro["selic_diaria"].index.max() <= corte

    def test_fatia_bate_com_a_busca_direta(self):
        """A troca tem de ser transparente: mesma data_limite, mesma série."""
        direto = _JournalMacro().get_macro(ts("2025-06-15"))["selic_diaria"]

        j = _JournalMacro()
        prefetch_macro(j, ts("2025-12-31"))
        fatiado = j.get_macro(ts("2025-06-15"))["selic_diaria"]

        pd.testing.assert_series_equal(direto, fatiado)

    def test_serie_vazia_nao_quebra_o_fatiamento(self):
        j = _JournalMacro()
        prefetch_macro(j, ts("2025-12-31"))
        assert j.get_macro(ts("2025-06-15"))["ipca_12m"].empty

    def test_aborta_quando_a_serie_exigida_vem_vazia(self):
        """Seguir sem macro mudaria o payload no meio da rodada — falha alto."""
        with pytest.raises(MacroIndisponivel):
            prefetch_macro(_JournalMacro(vazia=True), ts("2025-12-31"))

    def test_devolve_a_funcao_original_para_restaurar(self):
        j = _JournalMacro()
        original = prefetch_macro(j, ts("2025-12-31"))
        j.get_macro = original
        j.get_macro(ts("2025-06-15"))
        assert j.chamadas == 2


class TestCustoDaChamada:
    def test_usa_o_usage_real_do_sdk(self):
        c = custo_da_chamada({"input_tokens": 1_000_000, "output_tokens": 0},
                             "claude-haiku-4-5-20251001")
        assert c == pytest.approx(1.00)

    def test_sem_chamada_custo_zero(self):
        """Cache hit do EconAgent não gera chamada — custo verdadeiro é zero."""
        assert custo_da_chamada(None, "claude-haiku-4-5-20251001") == 0.0


class TestCheckpointSchema:
    def test_colunas_incluem_o_que_o_diagnostico_precisa(self):
        """Sem justificativa/confiança persistidas, uma retomada perde exatamente
        o material do diagnóstico de prompt da Etapa 2."""
        for col in ("justificativa", "confianca", "tem_evento", "degradacao"):
            assert col in Checkpoint.COLS

    def test_retomada_le_arquivo_com_schema_antigo(self, tmp_path):
        """Checkpoints gravados antes da extensão de schema têm de continuar
        legíveis — senão uma rodada interrompida vira lixo."""
        path = tmp_path / "antigo.csv"
        pd.DataFrame([{
            "evento_id": 0, "ticker": "PETR4.SA", "data": "2025-10-10T00:00:00-03:00",
            "y_realizado": 0.01, "modelo": "m", "score": 0.3, "latencia_llm_s": 1.2,
            "tokens_in": 100, "tokens_out": 20, "custo_usd": 0.0002,
        }]).to_csv(path, index=False)

        cp = Checkpoint(path)

        assert cp.ja_feito("m", 0)
        assert "justificativa" not in cp.linha_feita("m", 0)
