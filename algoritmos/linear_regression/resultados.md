# Análise de Resultados – Modelos de Regressão Linear

## Significado das Métricas

| Sigla         | Nome Completo               | Descrição                                                                                                                                                          |
| ------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **MAE**       | Mean Absolute Error         | Erro médio absoluto em metros. Mede quanto, em média, as previsões diferem dos valores reais. Quanto menor, melhor.                                                |
| **RMSE**      | Root Mean Squared Error     | Raiz do erro quadrático médio. Penaliza mais fortemente grandes erros. Indica a estabilidade do modelo.                                                            |
| **R²**        | Coeficiente de Determinação | Mede o quanto o modelo explica a variância dos dados. Varia entre 0 e 1. Valores negativos indicam que o modelo piora as previsões em relação a uma média simples. |
| **Intercept** | Termo independente          | Valor previsto pelo modelo quando todas as variáveis independentes são zero. Representa o ponto de partida da linha de regressão.                                  |
| **Holdout**   | Validação simples           | Divide os dados em partes de treino e teste, avaliando diretamente a generalização.                                                                                |
| **K-Fold**    | Validação cruzada           | Divide os dados em k grupos e faz múltiplas rodadas de treino/teste, aumentando a robustez da avaliação.                                                           |

---

## Configurações Avaliadas

Foram testadas cinco variações da regressão linear, modificando intercepto, validação e variáveis utilizadas.

| Modelo        | Estratégia  | Intercepto | MAE Teste  | RMSE Teste | R² Teste  | Observação                                |
| ------------- | ----------- | ---------- | ---------- | ---------- | --------- | ----------------------------------------- |
| **Linear #1** | Holdout     | True       | **1379.7** | 1990.8     | **0.10**  | baseline com todas as variáveis           |
| **Linear #2** | Holdout     | False      | **2443.8** | 3133.9     | **-1.21** | sem intercepto, forte perda de desempenho |
| **Linear #3** | K-Fold (10) | True       | **1377.9** | -          | **0.10**  | avaliação cruzada estável                 |
| **Linear #4** | Holdout     | True       | **1379.7** | 1990.8     | **0.10**  | foco geográfico e populacional            |
| **Linear #5** | K-Fold (7)  | True       | **1377.9** | -          | **0.10**  | teste de robustez com mais folds          |

---

## Análise dos Resultados

### Linear #1 – Modelo Base com Intercepto

Este modelo é o baseline da regressão linear tradicional.  
Apresentou R² ≈ 0.10, indicando que o modelo explica apenas cerca de 10% da variância no tamanho dos congestionamentos.  
Os erros médios (MAE ≈ 1380 m) indicam previsões com diferença entre 1,3 e 2 km em relação aos valores reais.  
Os coeficientes mostram que a população total e a hora do dia aumentam o tamanho dos congestionamentos, enquanto a região e o tipo de via reduzem.  
Embora limitado em poder preditivo, o modelo é útil para análise interpretativa das relações lineares entre variáveis.

---

### Linear #2 – Sem Intercepto

Ao remover o intercepto, o desempenho caiu drasticamente (R² ≈ -1.21).  
Isso mostra que o termo independente é fundamental para representar o nível médio de congestionamento.  
Sem ele, o modelo não consegue ajustar a linha de regressão corretamente, gerando previsões sistematicamente enviesadas.  
O intercepto é, portanto, indispensável neste contexto.

---

### Linear #3 – Validação Cruzada (10 Folds)

Com validação cruzada, o modelo apresentou resultados idênticos ao baseline (R² médio = 0.10, MAE ≈ 1378 m).  
A baixa variação entre folds indica estabilidade estatística.  
O resultado reforça que a relação entre as variáveis e o congestionamento é quase linear, mas a linearidade simples não captura as interações complexas do tráfego.

---

### Linear #4 – Dimensão Espacial e Populacional

Mesmo reduzido às variáveis geográficas e populacionais, o desempenho foi semelhante ao modelo completo.  
Isso demonstra que a população e a região, isoladamente, não são preditores suficientes do tamanho dos congestionamentos.  
O modelo mostra limitações ao tentar representar padrões complexos de tráfego urbano com dependência linear direta.

---

### Linear #5 – K-Fold (7 Folds)

A ampliação da validação cruzada para 7 folds não trouxe ganhos de desempenho significativos.  
Os resultados se mantiveram estáveis, com R² ≈ 0.10, confirmando que a regressão linear é consistente, mas estruturalmente limitada com as variáveis disponíveis.  
Serve para confirmar a ausência de overfitting e a estabilidade dos coeficientes.

---

## Conclusões Gerais

1. **Baixa explicabilidade linear**  
   Todos os modelos apresentaram R² em torno de 0.10, o que demonstra que o fenômeno do congestionamento possui comportamento predominantemente não linear.

2. **Importância do intercepto**  
   A ausência do termo independente levou à queda acentuada de desempenho, indicando que ele é necessário para representar a média geral dos congestionamentos.

3. **Estabilidade estatística**  
   As diferenças entre holdout e K-Fold foram mínimas, evidenciando que os resultados são consistentes, mas o modelo é limitado por sua forma linear.

4. **Interpretação causal**  
   A regressão linear, apesar de simples, permite interpretar quais variáveis influenciam positivamente ou negativamente o congestionamento.  
   População e hora aumentam o valor previsto, enquanto tipo de via e região tendem a reduzir.

5. **Limitação teórica da abordagem**  
   O modelo linear não captura bem relações não lineares e interações complexas entre variáveis temporais e espaciais.  
   Modelos como Random Forest, XGBoost e MLP apresentam potencial superior para esse tipo de problema.

---

## Próximos Passos

- Incluir variáveis contextuais como clima, feriados e eventos.
- Avaliar regressões regularizadas (Ridge e Lasso).
- Comparar diretamente com modelos não lineares.
- Criar novas features que combinem variáveis temporais e regionais.

---

**Autor:** Alexandre Marques Tortoza Canoa  
**Data:** Outubro de 2025  
**Projeto:** Pesquisa Aplicada – Previsão de Congestionamentos em São Paulo
