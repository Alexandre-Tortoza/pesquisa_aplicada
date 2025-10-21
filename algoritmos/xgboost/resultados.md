# Análise de Resultados Modelo XGBoost

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

Foram testadas **cinco variações** do **XGBoost Regressor**, explorando combinações de profundidade, número de árvores, taxa de aprendizado e método de validação.

| Modelo     | Estratégia | Árvores | Profundidade | Learning Rate | MAE Teste | RMSE Teste | R² Teste | Observações                                       |
| ---------- | ---------- | ------- | ------------ | ------------- | --------- | ---------- | -------- | ------------------------------------------------- |
| **XGB #1** | Holdout    | 100     | 6            | 0.10          | 953.1     | 1468.7     | 0.51     | Modelo base com hiperparâmetros padrão            |
| **XGB #2** | Holdout    | 400     | 10           | 0.03          | **844.7** | **1327.3** | **0.60** | Profundo e estável, melhor resultado geral        |
| **XGB #3** | K-Fold (5) | 80      | 5            | 0.02          | 1124.3    | —          | 0.36     | Simples e regularizado, prioriza velocidade       |
| **XGB #4** | K-Fold (5) | 300     | 8            | 0.05          | **913.0** | —          | **0.55** | Alta estabilidade e generalização                 |
| **XGB #5** | Holdout    | 50      | 4            | 0.30          | 983.6     | 1507.5     | 0.49     | Leve e rápido, voltado à inferência em tempo real |

---

## Análise dos Resultados

### ⚙️ XGB #1 — Baseline (100 árvores, profundidade 6)

Modelo padrão com configuração balanceada.
Alcançou **R² = 0.51** e **MAE ≈ 953 m**, servindo como linha de base.
Mostra bom desempenho geral e baixo viés, com tempo de execução reduzido.
Ideal para prototipagem e comparações.

---

### 🌳 XGB #2 — Profundo e Preciso (400 árvores, max_depth=10)

O melhor modelo entre todos.
Com **R² = 0.60** e **MAE ≈ 845 m**, capturou **relações não-lineares complexas** entre variáveis regionais, temporais e populacionais.
A taxa de aprendizado reduzida (0.03) e o número maior de árvores suavizaram o aprendizado, evitando overfitting.
Excelente equilíbrio entre desempenho e estabilidade — **modelo mais indicado para produção.**

---

### 🌱 XGB #3 — Regularizado e Leve (80 árvores, max_depth=5)

Versão mais simples, priorizando eficiência e regularização L1/L2.
Obteve **R² = 0.36** e **MAE ≈ 1124 m**, com alta estabilidade entre folds.
Apesar do desempenho inferior, mostrou-se útil como **baseline otimizado para execução rápida**, ideal para comparação de tuning.

---

### 🌲 XGB #4 — Alta Estabilidade (300 árvores, max_depth=8)

Usando validação cruzada e subsampling agressivo, o modelo atingiu **R² médio = 0.55 ± 0.005** e **MAE ≈ 913 m**.
Mostra **consistência entre folds** e resistência a ruídos.
Indicado para aplicações em larga escala, onde estabilidade e previsibilidade são prioritárias.

---

### 🍃 XGB #5 — Modelo Rápido (50 árvores, max_depth=4)

Focado em **tempo de inferência**, o modelo alcançou **R² = 0.49**, com erro médio de **984 m**.
Desempenho competitivo considerando o custo computacional mínimo.
Adequado para cenários de atualização em tempo real ou dispositivos com restrição de hardware.

---

## Conclusões Gerais

- 🏆 **Melhor desempenho geral:** XGB #2 (400 árvores, profundidade 10, LR 0.03) — **R² = 0.60, MAE = 845 m**
  → Modelo ideal para implantação, equilibrando precisão e generalização.

- ⚙️ **Vantagem do XGBoost:** Combina interpretabilidade das árvores com poder de modelagem não-linear, **superando RF e MLP** em consistência geral.

- 🧩 **Variáveis mais relevantes (padrão entre execuções):**
  `pop_total`, `regiao_encoded` e `via_expressa_encoded` dominam a explicação da variância.

- 🚦 O XGBoost mostrou **melhor estabilidade e menor viés** que o Random Forest, além de convergência mais previsível.

---

## Próximos Passos

1. 🔧 Realizar _fine-tuning_ de `learning_rate`, `subsample` e `colsample_bytree` via _Grid Search_.
2. 📈 Avaliar impacto de features adicionais como clima e feriados.
3. 🧮 Explorar técnicas de _feature interaction_ para relações temporais (hora × dia da semana).
4. 🌐 Integrar modelo a pipelines de previsão em tempo real (API/streaming).

---

**Autor:** Alexandre Marques Tortoza Canoa
**Data:** Outubro de 2025
**Projeto:** _Pesquisa Aplicada — Previsão de Congestionamentos em São Paulo_
