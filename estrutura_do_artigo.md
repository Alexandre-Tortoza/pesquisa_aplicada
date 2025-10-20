# **Previsão de Congestionamentos em São Paulo com Base no Crescimento Populacional**

### **Autor**

**Alexandre Marques Tortoza Canoa**
Departamento de Computação Aplicada — Universidade XYZ
📧 [alexandre.canoa@universidadexyz.br](mailto:alexandre.canoa@universidadexyz.br)

---

## **Resumo**

Este estudo propõe um modelo preditivo para estimar o tamanho dos congestionamentos na cidade de São Paulo a partir do crescimento populacional e de variáveis de tráfego urbano. Utilizando dados oficiais de estimativas populacionais por distrito e registros históricos de congestionamentos, foi desenvolvido um pipeline de análise que integra pré-processamento, normalização e modelagem com diferentes algoritmos supervisionados.
Os resultados indicam que modelos baseados em árvores, como **Random Forest** e **XGBoost**, apresentaram melhor desempenho geral (R² ≈ 0,62 e 0,51, respectivamente), sugerindo que o crescimento populacional é um fator relevante, mas interage fortemente com variáveis temporais e geográficas.

**Palavras-chave:** congestionamento, previsão, população, aprendizado de máquina, São Paulo, Random Forest.

---

## **1. Introdução**

O crescimento urbano acelerado de São Paulo nas últimas décadas tem gerado desafios complexos de mobilidade. A associação entre o aumento populacional e o agravamento dos congestionamentos viários é amplamente reconhecida, porém pouco explorada de forma quantitativa e preditiva.

Este trabalho busca responder à seguinte questão:

> **É possível prever a intensidade dos congestionamentos urbanos com base na dinâmica populacional e em variáveis temporais e espaciais?**

O objetivo principal é desenvolver e avaliar um modelo de *machine learning* capaz de estimar o tamanho médio dos congestionamentos, combinando dados demográficos e de tráfego.

---

## **2. Representação esquemática da solução**

A Figura 1 apresenta o **pipeline da solução** que abrange desde a aquisição de dados até a interpretação dos resultados.

```
+-------------------+
| Aquisição de Dados|
+-------------------+
         |
         v
+---------------------------+
| Pré-processamento         |
| - Limpeza de ruído        |
| - Normalização de textos  |
| - Mapeamento de regiões   |
+---------------------------+
         |
         v
+---------------------------+
| Integração População x    |
| Tráfego                   |
| (Merge por região e ano)  |
+---------------------------+
         |
         v
+---------------------------+
| Modelagem Preditiva       |
| (KNN, Linear, RF, XGB, MLP)|
+---------------------------+
         |
         v
+---------------------------+
| Avaliação e Interpretação |
| (Métricas, SHAP, Importância)|
+---------------------------+
```

📊 **Figura 1 — Fluxo geral da solução proposta.**
*(Inserir uma imagem: `pipeline_diagram.png` com as mesmas etapas acima.)*

---

## **3. Base de Dados**

### **3.1 Origem e Características**

Foram utilizados dois conjuntos principais:

* **População:** estimativas por distrito, sexo e faixa etária (fonte: IBGE/SEADE), processadas com o script `clear_population.py`.
* **Tráfego:** registros históricos de congestionamentos da CET/SP, tratados com o script `clear_traffic.py`.

Após o processamento, os dados foram integrados por **ano** e **região geográfica** (center, north, south, east, west).

### **3.2 Variáveis**

**Variáveis de entrada (features):**

* `pop_total`: população total estimada na região
* `hora_numeric`: hora em formato numérico
* `via_expressa_encoded`: codificação da via principal
* `regiao_encoded`: representação numérica da região
* `sexo_encoded`: sexo da população
* `dia_semana`: dia da semana (1–7)
* `mes`: mês (1–12)

**Variável-alvo (target):**

* `tamanho_congestionamento`: extensão média do congestionamento (em metros)

### **3.3 Qualidade e Balanceamento**

Os dados apresentaram boa cobertura geográfica (96 distritos) e temporal (2012–2020).
A distribuição entre regiões foi equilibrada, e os dados foram normalizados para reduzir diferenças de escala entre variáveis.

---

## **4. Pré-Processamento**

O pré-processamento envolveu as seguintes etapas:

1. **Normalização textual:** remoção de acentos e padronização (Unicode NFKD).
2. **Filtragem:** remoção de registros com idade < 20 anos e congestionamentos inválidos.
3. **Codificação categórica:** *Label Encoding* de regiões, vias e sexo.
4. **Normalização de escala:** *StandardScaler* aplicado antes de modelos sensíveis à distância (KNN, MLP).
5. **Análise de correlação:** cálculo dos coeficientes de Pearson e Spearman.
6. **Remoção de outliers:** exclusão de valores acima do 99º percentil.

📈 **Figura 2 — Matriz de correlação das variáveis numéricas.**
*(Inserir imagem: `correlation_matrix.png`.)*

---

## **5. Protocolo Experimental**

Foi adotado o método **Hold-Out (70% treino / 30% teste)** com *random state* fixo.
Essa escolha foi motivada pelo grande volume de dados (≈316 mil amostras) e pela necessidade de avaliar a generalização temporal e espacial dos modelos.

---

## **6. Modelagem**

### **6.1 Modelos Testados**

| Modelo               | Principais parâmetros              | R² (teste) |
| -------------------- | ---------------------------------- | ---------- |
| Regressão Linear     | `fit_intercept=True`               | 0.10       |
| KNN Regressor        | `k=5`                              | 0.53       |
| Random Forest        | `n_estimators=100`                 | **0.62**   |
| XGBoost              | `learning_rate=0.1`, `max_depth=6` | 0.51       |
| MLP (Neural Network) | camadas 100-50-25, ReLU            | 0.47       |

📊 **Tabela 1 — Modelos testados e desempenho (R²).**

### **6.2 Ajuste de Hiperparâmetros**

Os modelos foram ajustados com *Grid Search* e validação empírica, priorizando equilíbrio entre desempenho e interpretabilidade.

### **6.3 Resultados Visuais**

📉 **Figura 3 — Comparativo de desempenho no conjunto de teste (MAE, RMSE, R²).**
*(Inserir imagem: `rf_results_holdout.png` ou `xgboost_results_holdout.png`.)*

---

## **7. Métricas de Avaliação**

### **7.1 Métricas Utilizadas**

As métricas empregadas foram:

* **MAE (Erro Absoluto Médio)**
* **RMSE (Raiz do Erro Quadrático Médio)**
* **R² (Coeficiente de Determinação)**

### **7.2 Resultados Obtidos**

| Modelo        | MAE (teste) | RMSE (teste) | R² (teste) |
| ------------- | ----------- | ------------ | ---------- |
| Linear        | 1379.7      | 1990.9       | 0.10       |
| KNN           | 786.2       | 1443.6       | 0.53       |
| Random Forest | **760.2**   | **1298.6**   | **0.62**   |
| XGBoost       | 953.1       | 1468.6       | 0.51       |
| MLP           | 988.5       | 1531.6       | 0.47       |

📊 **Tabela 2 — Métricas de desempenho dos modelos (conjunto de teste).**

---

## **8. Interpretação dos Resultados**

### **8.1 Importância das Variáveis**

📈 **Figura 4 — Importância das variáveis no modelo Random Forest.**
*(Inserir imagem: `rf_feature_importance.png`.)*

O atributo `pop_total` foi o mais relevante (≈37%), seguido por `mes`, `hora_numeric` e `regiao_encoded`.
A variável `sexo` apresentou influência nula, indicando irrelevância direta para o fenômeno.

### **8.2 Análise SHAP**

📊 **Figura 5 — Análise SHAP: impacto de cada variável nas previsões.**
*(Inserir imagem: `rf_shap_analysis.png`.)*

A análise SHAP confirmou a relação não linear entre densidade populacional e horário de pico, reforçando que o crescimento populacional influencia de forma indireta, mediado por padrões temporais.

---

## **9. Discussão**

Os resultados mostraram que modelos baseados em árvores capturam melhor a interação entre variáveis populacionais e temporais.
O **Random Forest** foi o modelo mais equilibrado entre acurácia e interpretabilidade, seguido pelo **KNN**, que apresentou bom desempenho local, mas menor generalização.
Modelos lineares tiveram desempenho inferior, evidenciando que a relação entre população e congestionamento é não linear.

---

## **10. Conclusão**

O estudo confirmou que há uma relação mensurável entre o crescimento populacional e a intensidade dos congestionamentos em São Paulo.
Modelos de aprendizado de máquina, especialmente o **Random Forest**, mostraram-se eficazes na previsão da extensão dos congestionamentos.

**Trabalhos futuros:**

* Explorar modelos temporais (ex.: LSTM, Prophet);
* Incorporar variáveis socioeconômicas (renda média, frota veicular);
* Avaliar o impacto de políticas públicas sobre padrões de tráfego.

---

## **11. Agradecimentos**

Agradecimentos à **CET-SP** (Companhia de Engenharia de Tráfego de São Paulo) e à **Fundação SEADE** pelos dados utilizados nesta pesquisa.

---

## **12. Referências**

1. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
3. CET-SP. *Dados abertos de congestionamentos de São Paulo*, 2024.
4. Fundação SEADE. *Estimativas Populacionais por Distrito*, 2024.

---

### 🧩 **Resumo das imagens sugeridas**

| Figura       | Arquivo sugerido                                          | Seção                     | Descrição                             |
| ------------ | --------------------------------------------------------- | ------------------------- | ------------------------------------- |
| **Figura 1** | `pipeline_diagram.png`                                    | Representação esquemática | Fluxograma do pipeline completo       |
| **Figura 2** | `correlation_matrix.png`                                  | Pré-processamento         | Matriz de correlação das features     |
| **Figura 3** | `rf_results_holdout.png` ou `xgboost_results_holdout.png` | Modelagem                 | Comparativo de métricas entre modelos |
| **Figura 4** | `rf_feature_importance.png`                               | Interpretação             | Importância das variáveis             |
| **Figura 5** | `rf_shap_analysis.png`                                    | Interpretação             | Análise SHAP das previsões            |


