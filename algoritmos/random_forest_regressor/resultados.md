# Análise de Resultados Modelo Random Forest

## Significado das Métricas

| Sigla       | Nome Completo                   | Descrição                                                                                                                       |
| ----------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **MAE**     | _Mean Absolute Error_           | Erro médio absoluto — mede, em metros, o quanto as previsões diferem dos valores reais. Valores menores indicam maior precisão. |
| **RMSE**    | _Root Mean Squared Error_       | Raiz do erro quadrático médio — penaliza mais fortemente erros grandes. Ideal para avaliar estabilidade do modelo.              |
| **R²**      | _Coeficiente de Determinação_   | Mede o quanto o modelo explica a variância dos dados reais. Varia entre 0 e 1; quanto maior, melhor o ajuste.                   |
| **Holdout** | Divisão simples de treino/teste | Parte dos dados é usada para treino e outra para teste, permitindo avaliação direta da generalização.                           |
| **K-Fold**  | Validação Cruzada               | Divide os dados em _k_ partes (folds), treinando e testando várias vezes. Reduz viés e aumenta robustez da avaliação.           |

---

## Configurações Avaliadas

Foram testadas **cinco configurações distintas** do algoritmo **Random Forest Regressor**, variando número de árvores, profundidade máxima e tipo de validação:

| Modelo    | Estratégia | Árvores | Profundidade Máx. | MAE Teste | RMSE Teste | R² Teste | Observações                                             |
| --------- | ---------- | ------- | ----------------- | --------- | ---------- | -------- | ------------------------------------------------------- |
| **RF #1** | Holdout    | 100     | — (sem limite)    | 760.2     | 1298.6     | 0.62     | Modelo base (sem restrição de profundidade)             |
| **RF #2** | Holdout    | 200     | 30                | 749.7     | 1439.9     | 0.53     | Mais profundo, avalia overfitting e capacidade ampliada |
| **RF #3** | Holdout    | 300     | 15                | **857.9** | 1348.9     | 0.59     | Modelo robusto com grande diversidade de árvores        |
| **RF #4** | K-Fold (5) | 50      | 10                | **974.8** | —          | **0.49** | Modelo leve, prioriza eficiência e generalização        |
| **RF #5** | K-Fold (5) | 150     | 12                | **931.0** | —          | **0.53** | Validação cruzada com regularização estrutural          |

---

## Análise dos Resultados

### 🌲 RF #1 — Baseline (100 árvores, sem limite de profundidade)

Este modelo serviu como **referência base** para comparação.
Com R² ≈ **0.62** e MAE ≈ **760 m**, apresentou **excelente desempenho inicial**, capturando relações complexas entre as variáveis sem sinais fortes de sobreajuste.
A ausência de limitação de profundidade permitiu melhor ajuste às flutuações do congestionamento, embora com risco moderado de overfitting.

> **Features mais importantes:** população total (37%), via expressa (15%), mês (14%), hora do dia (13%).

---

### 🌳 RF #2 — Aumento de Profundidade (200 árvores, max_depth=30)

Com maior número de árvores e profundidade ampliada, o modelo buscou **aprimorar a capacidade de aprendizado**.
O R² caiu ligeiramente (0.53), sugerindo que **o aumento da complexidade trouxe overfitting**, refletido na discrepância entre erros de treino e teste.
Ainda assim, o MAE permaneceu próximo do baseline, mostrando **boa estabilidade do algoritmo**.

> **População total** continua como principal variável, reforçando sua influência no padrão de tráfego urbano.

---

### 🌲 RF #3 — Configuração Robusta (300 árvores, max_depth=15)

Com mais árvores, mas profundidade limitada, o modelo buscou **equilíbrio entre complexidade e generalização**.
Apresentou R² = 0.59 e MAE ≈ 858 m, com bom desempenho em estabilidade.
As importâncias de features ficaram mais distribuídas entre **população, via expressa e região**, demonstrando sensibilidade espacial do modelo.

> Essa configuração é **ideal para produção**, unindo desempenho consistente e menor variância entre execuções.

---

### 🌿 RF #4 — Modelo Leve (K-Fold, 50 árvores, max_depth=10)

Voltado à **eficiência computacional**, este modelo validado por 5 folds obteve R² médio de **0.49 ± 0.002** e MAE ≈ **975 m**.
Apesar de mais simples, manteve coerência de predição e **baixa dispersão entre folds**, indicando **boa estabilidade estatística**.
É o mais indicado para cenários de teste rápido ou baixa disponibilidade de hardware.

---

### 🌾 RF #5 — Regularização Estrutural (K-Fold, 150 árvores, max_depth=12)

A configuração intermediária com validação cruzada obteve **melhor desempenho médio entre os K-Fold**, com R² ≈ **0.53 ± 0.005** e MAE ≈ **931 m**.
Mostra que **a regularização via limitação de profundidade** e maior número de árvores resulta em **ótimo equilíbrio entre bias e variância**.
É o modelo mais robusto para validações amplas.

> **Top 3 features:** População total (37%), via expressa (24%), região (22%).

---

## Conclusões Gerais

- 🏆 **Melhor modelo:** RF #1 (100 árvores, sem limite de profundidade) — **R² = 0.62, MAE = 760 m**.
  → Alcança excelente precisão e forte correlação entre variáveis temporais e espaciais.

- ⚙️ Modelos com **K-Fold** apresentaram desempenho ligeiramente inferior, mas **maior estabilidade**, confirmando a robustez da abordagem.

- 🧩 **População total**, **via expressa** e **região** são consistentemente as variáveis mais importantes em todas as configurações.

- 🌐 Random Forest mantém desempenho superior ao KNN e comparável ao MLP, com **vantagem em interpretabilidade e estabilidade**.

---

## Próximos Passos

1. 🔍 Realizar _Grid Search_ para ajuste fino de `n_estimators`, `max_depth` e `min_samples_split`.
2. 🌳 Comparar resultados com **XGBoost**, que pode otimizar o uso de árvores sequenciais.
3. 🧠 Avaliar combinação de Random Forest com variáveis climáticas e eventos.
4. 📊 Explorar métricas adicionais como MAPE e erro percentual por faixa horária.

---

**Autor:** Alexandre Marques Tortoza Canoa
**Data:** Outubro de 2025
**Projeto:** _Pesquisa Aplicada — Previsão de Congestionamentos em São Paulo_
