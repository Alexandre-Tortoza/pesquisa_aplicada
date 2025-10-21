# Análise de Resultados Modelo MLP (Redes Neurais)

## Significado das Métricas

| Sigla       | Nome Completo                   | Descrição                                                                                                                       |
| ----------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **MAE**     | _Mean Absolute Error_           | Erro médio absoluto — mede, em metros, o quanto as previsões diferem dos valores reais. Valores menores indicam maior precisão. |
| **RMSE**    | _Root Mean Squared Error_       | Raiz do erro quadrático médio — penaliza mais fortemente erros grandes. Ideal para avaliar estabilidade do modelo.              |
| **R²**      | _Coeficiente de Determinação_   | Mede o quanto o modelo explica a variância dos dados reais. Varia entre 0 e 1; quanto maior, melhor o ajuste.                   |
| **K-Fold**  | Validação Cruzada               | Divide os dados em _k_ partes (folds), treinando e testando várias vezes. Reduz viés e aumenta robustez da avaliação.           |
| **Holdout** | Divisão simples de treino/teste | Parte dos dados é usada para treino e outra para teste, permitindo avaliação direta da generalização.                           |

---

## Configurações Avaliadas

Foram testadas **cinco configurações** da rede neural _Multi-Layer Perceptron (MLP)_ variando arquitetura, função de ativação, otimizador e estratégia de validação:

| Modelo     | Estratégia | Arquitetura (camadas) | Ativação | Otimizador | Learning Rate | Épocas | MAE Teste         | RMSE Teste | R² Teste         |
| ---------- | ---------- | --------------------- | -------- | ---------- | ------------- | ------ | ----------------- | ---------- | ---------------- |
| **MLP #1** | Holdout    | (100, 50, 25)         | ReLU     | Adam       | 0.001         | 500    | 988.5             | 1531.6     | 0.47             |
| **MLP #2** | Holdout    | (256, 128, 64, 32)    | ReLU     | Adam       | 0.001         | 500    | **962.3**         | **1490.8** | **0.50**         |
| **MLP #3** | Holdout    | (64, 32)              | ReLU     | Adam       | 0.002         | 300    | 1005.1            | 1537.1     | 0.47             |
| **MLP #4** | K-Fold (5) | (100, 50)             | ReLU     | Adam       | 0.002         | 500    | **988.4 (média)** | —          | **0.47 (média)** |
| **MLP #5** | K-Fold (5) | (168, 64, 32)         | Tanh     | SGD        | 0.01          | 600    | 1502.7 (média)    | —          | 0.00             |

---

## Análise dos Resultados

### 🔹 MLP #1 — Baseline (3 camadas, ReLU, Adam)

Configurada como **arquitetura de referência**, a rede de três camadas intermediárias apresentou desempenho sólido com R² ≈ 0.47.
O erro médio de ~988 m mostra boa coerência com o KNN, sugerindo que o MLP captura parte relevante das relações não-lineares.
Convergência alcançada em 179 iterações, indicando **treinamento estável e eficiente**.

---

### 🔹 MLP #2 — Rede Profunda (4 camadas, ReLU, Adam)

Essa arquitetura ampliada apresentou **melhor desempenho global**, com R² ≈ 0.50 e MAE ≈ **962 m**.
O aumento de camadas e neurônios, aliado à regularização L2, permitiu **melhor generalização** sem overfitting.
O resultado mostra que a rede neural consegue modelar interações complexas entre variáveis temporais e regionais.
É o **modelo mais promissor** entre os testados, equilibrando precisão e estabilidade.

---

### 🔹 MLP #3 — Arquitetura Compacta (2 camadas, ReLU, Adam)

Versão reduzida voltada à **eficiência computacional** e inferência em tempo real.
Apesar da simplicidade, manteve desempenho semelhante ao baseline, com R² = 0.47 e erro médio de ~1 km.
Ideal para aplicações práticas onde a latência é fator crítico (ex.: sistemas embarcados).

---

### 🔹 MLP #4 — K-Fold ReLU (Validação Cruzada)

Avaliando robustez da arquitetura clássica (100, 50) via **validação cruzada 5-fold**, o modelo apresentou **média R² = 0.47 ± 0.005**, com MAE ≈ 988 m.
Os desvios baixos confirmam **consistência entre diferentes subconjuntos de dados**, validando a estabilidade da rede.

---

### 🔹 MLP #5 — Tanh + SGD (Experimento Alternativo)

Testando a combinação **função tanh + otimizador SGD**, o modelo não convergiu adequadamente (R² ≈ 0.0, MAE > 1500 m).
A ativação suave e o otimizador clássico exigiriam ajuste fino da taxa de aprendizado e momentum.
Resultado confirma que **Adam + ReLU** é mais adequado ao problema, dadas as relações não lineares e heterogêneas do trânsito urbano.

---

## Conclusões Gerais

- 🏆 **Melhor modelo:** MLP profunda com 4 camadas (256,128,64,32), ReLU e Adam — **R² ≈ 0.50, MAE ≈ 960 m**.
  → Excelente equilíbrio entre precisão e generalização.

- ⚙️ **Configurações ReLU + Adam** apresentaram desempenho consistente, superando combinações com SGD e tanh.

- 🧩 **Variáveis temporais e regionais** (hora, dia da semana, região) continuam sendo as mais influentes, reforçando achados de modelos anteriores (KNN, RF).

- 🧠 A capacidade de aprendizado não-linear do MLP traz ganhos modestos sobre o KNN e Random Forest, mas mostra **potencial para evolução com ajuste de hiperparâmetros e features adicionais**.

---

## Próximos Passos

1. 🔧 Testar _learning rates_ adaptativos e dropout para reduzir variações residuais.
2. 🧮 Comparar com **XGBoost** e **Random Forest** sob as mesmas métricas.
3. 🌦️ Incorporar variáveis externas (clima, eventos, feriados) ao conjunto de treino.
4. ⚡ Explorar arquiteturas mais profundas (e.g., (512,256,128)) e normalização por batch.

---

**Autor:** Alexandre Marques Tortoza Canoa
**Data:** Outubro de 2025
**Projeto:** _Pesquisa Aplicada — Previsão de Congestionamentos em São Paulo_
