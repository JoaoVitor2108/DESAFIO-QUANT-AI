"""
MATH&ML — terceiro agente do JEMPO (stub; implementação futura).

Combina o score qualitativo do ECON com features quantitativas e treina o modelo
(GradientBoosting) que gera o sinal de seleção.

────────────────────────────────────────────────────────────────────────────────
NOTA DE DESIGN — como consumir o ECON (evitar colinearidade) [decisão Opção A]

O ECON entrega um `ScoreEcon` com:
  - `score_total`  → impacto da NOTÍCIA no retorno em excesso (5 dias úteis).
                     É o sinal que o ECON tem autoridade para julgar. USAR como feature.
  - `comp_saude_financeira`, `comp_setorial`, `comp_macro` → CONTEXTO que o ECON
                     considerou para calibrar a leitura da notícia, NÃO parcelas
                     somadas ao score.

REGRA ao montar o conjunto de features do MATH&ML:
  1. Saúde financeira, momento setorial e macro entram como FEATURES CRUAS
     INDEPENDENTES vindas direto do JOURNAL (get_fundamentals / get_retornos_setor
     / get_macro) — NÃO os componentes do ECON. As features cruas são objetivas,
     auditáveis e não custam inferência de LLM.
  2. Os componentes do ECON (comp_*) servem para INTERPRETABILIDADE e, no máximo,
     como features OPCIONAIS — e, se incluídos, validados por cross-validation para
     confirmar que NÃO duplicam o sinal já presente nas features cruas do JOURNAL
     (risco de colinearidade que prejudica o GradientBoosting).
  3. `score_total` (impacto da notícia) é a contribuição central do ECON — não há
     redundância com as features cruas, pois nenhuma delas mede o efeito da notícia.

TODO: implementar pipeline de features + treino walk-forward (treino 2020-2023,
backtest OOS 2024-2025) conforme a estratégia.
"""
