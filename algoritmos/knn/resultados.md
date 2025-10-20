# Análise de Resultados Modelo KNN

## Significado das Métricas

| Sigla               | Nome Completo                     | Descrição                                                                                                                       |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **MAE**             | _Mean Absolute Error_             | Erro médio absoluto - mede, em metros, o quanto as previsões diferem dos valores reais. Valores menores indicam maior precisão. |
| **RMSE**            | _Root Mean Squared Error_         | Raiz do erro quadrático médio - penaliza mais fortemente erros grandes. Ideal para avaliar estabilidade do modelo.              |
| **R²**              | _Coeficiente de Determinação_     | Mede o quanto o modelo explica a variância dos dados reais. Varia entre 0 e 1, sendo valores mais altos melhores.               |
| **Holdout**         | Divisão simples de treino e teste | Parte dos dados é usada para treino e outra para teste, permitindo avaliação direta de generalização.                           |
| **K-Fold**          | Validação Cruzada                 | Divide os dados em _k_ partes (folds), treinando e testando várias vezes. Reduz viés e aumenta robustez da avaliação.           |
| **K (N_NEIGHBORS)** | Número de Vizinhos                | Quantos pontos mais próximos o modelo considera para estimar o valor alvo.                                                      |
| **Métrica**         | Tipo de Distância                 | Define como medir a "proximidade" entre observações (ex.: Minkowski, Manhattan, Chebyshev).                                     |

---

## Configurações Avaliadas

Foram testadas cinco configurações distintas do algoritmo K-Nearest Neighbors (KNN), variando estratégia de validação, métrica de distância e número de vizinhos:

| Modelo     | Estratégia  | Distância | Pesos    | K   | MAE Teste | RMSE Teste | R² Teste |
| ---------- | ----------- | --------- | -------- | --- | --------- | ---------- | -------- |
| **KNN #1** | Holdout     | Minkowski | Uniform  | 5   | 953.9     | 1504.2     | 0.49     |
| **KNN #2** | K-Fold (10) | Manhattan | Distance | 7   | **749.3** | —          | **0.56** |
| **KNN #3** | Holdout     | Euclidean | Distance | 3   | 793.9     | 1464.8     | 0.52     |
| **KNN #4** | K-Fold (5)  | Chebyshev | Uniform  | 8   | 974.9     | —          | 0.48     |
| **KNN #5** | Holdout     | Chebyshev | Uniform  | 6   | 966.1     | 1513.3     | 0.48     |

---

## Análise dos Resultados

### KNN #1 - Baseline com Minkowski

O modelo base apresentou um bom equilíbrio entre erro e explicação (R² = 0.49).  
O uso de pesos uniformes e distância Minkowski resultou em **boa estabilidade**, mas o modelo teve **leve overfitting** (melhor desempenho no treino).  
Serve como linha de referência para as demais variações.

---

### KNN #2 - Manhattan + Validação Cruzada (Melhor desempenho)

Este modelo apresentou **os melhores resultados gerais**:  
R² médio de **0.56** e MAE de **~749 m**, indicando que as previsões estão, em média, a menos de 1 km do valor real.  
A métrica **Manhattan (L1)** se mostrou mais adequada ao contexto urbano captando bem a dispersão de tráfego nas vias enquanto a validação cruzada com 10 folds reduziu viés.  
É a **configuração mais robusta e confiável** entre todas as testadas.

---

### KNN #3 - Ênfase Temporal e Espacial

Usando apenas variáveis de **tempo e localização**, o modelo atingiu R² = 0.52, próximo ao desempenho total.  
Isso demonstra que **região e horário** são fatores determinantes para prever congestionamentos, mesmo sem variáveis populacionais.  
Entretanto, o uso de poucos vizinhos (k=3) torna o modelo mais sensível ao ruído local, sugerindo leve sobreajuste.

---

### KNN #4 - Chebyshev com Validação K-Fold

Com R² = 0.48 e MAE ≈ 975 m, este modelo mostrou maior variação entre as previsões.  
A métrica **Chebyshev** mede apenas a maior diferença entre dimensões, o que o torna **mais sensível a outliers** e menos eficaz quando há grande dispersão geográfica.  
Ainda assim, manteve desempenho consistente em múltiplos folds.

---

### KNN #5 - Chebyshev Holdout (ênfase espacial)

Resultado similar ao anterior, com R² = 0.48 e erro médio de ~966 m.  
A combinação de **densidade populacional e região** manteve boa precisão espacial, mas o modelo não superou as abordagens com métricas mais equilibradas (como Manhattan ou Euclidean).  
Útil como comparação para efeitos de distribuição geográfica do tráfego.

---

## Conclusões Gerais

- **Melhor modelo:** KNN com distância _Manhattan_, pesos _distance_ e validação _K-Fold_ (Configuração #2).  
  -> Equilibra precisão e generalização, sendo ideal como modelo principal.

- **Fatores mais influentes:** Variáveis temporais e regionais (hora, dia da semana e região) explicam mais de 50% da variância no congestionamento.

- **Erro típico (~800-950 m):** Representa uma margem realista para previsões urbanas, dada a variabilidade do trânsito paulistano.

- **Limitação natural do KNN:** Os resultados se estabilizam em torno de R² ≈ 0.55, sugerindo que modelos não lineares (RF, XGBoost, MLP) podem capturar padrões mais complexos.

---

## Próximos Passos

1. Realizar comparação direta entre KNN, Random Forest, XGBoost e MLP.
2. Avaliar impacto de novas features, como clima e eventos locais.
3. Explorar versões espaciais do KNN (GeoKNN) com coordenadas geográficas reais.

---

**Autor:** Alexandre Marques Tortoza Canoa  
**Data:** Outubro de 2025  
**Projeto:** _Pesquisa Aplicada - Previsão de Congestionamentos em São Paulo_
