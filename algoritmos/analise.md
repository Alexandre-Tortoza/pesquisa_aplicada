| Modelo     | Estratégia  | Distância | Pesos    | K   | MAE Teste | RMSE Teste | R² Teste |
| ---------- | ----------- | --------- | -------- | --- | --------- | ---------- | -------- |
| **KNN #1** | Holdout     | Minkowski | Uniform  | 5   | 953.9     | 1504.2     | 0.49     |
| **KNN #2** | K-Fold (10) | Manhattan | Distance | 7   | **749.3** | —          | **0.56** |
| **KNN #3** | Holdout     | Euclidean | Distance | 3   | 793.9     | 1464.8     | 0.52     |
| **KNN #4** | K-Fold (5)  | Chebyshev | Uniform  | 8   | 974.9     | —          | 0.48     |
| **KNN #5** | Holdout     | Chebyshev | Uniform  | 6   | 966.1     | 1513.3     | 0.48     |


| Modelo        | Estratégia  | Intercepto | MAE Teste  | RMSE Teste | R² Teste  | 
| ------------- | ----------- | ---------- | ---------- | ---------- | --------- | 
| **Linear #1** | Holdout     | True       | **1379.7** | 1990.8     | **0.10**  | 
| **Linear #2** | Holdout     | False      | **2443.8** | 3133.9     | **-1.21** | 
| **Linear #3** | K-Fold (10) | True       | **1377.9** | -          | **0.10**  |
| **Linear #4** | Holdout     | True       | **1379.7** | 1990.8     | **0.10**  |
| **Linear #5** | K-Fold (7)  | True       | **1377.9** | -          | **0.10**  |


| Modelo     | Estratégia | Arquitetura (camadas) | Ativação | Otimizador | Learning Rate | Épocas | MAE Teste         | RMSE Teste | R² Teste         |
| ---------- | ---------- | --------------------- | -------- | ---------- | ------------- | ------ | ----------------- | ---------- | ---------------- |
| **MLP #1** | Holdout    | (100, 50, 25)         | ReLU     | Adam       | 0.001         | 500    | 988.5             | 1531.6     | 0.47             |
| **MLP #2** | Holdout    | (256, 128, 64, 32)    | ReLU     | Adam       | 0.001         | 500    | **962.3**         | **1490.8** | **0.50**         |
| **MLP #3** | Holdout    | (64, 32)              | ReLU     | Adam       | 0.002         | 300    | 1005.1            | 1537.1     | 0.47             |
| **MLP #4** | K-Fold (5) | (100, 50)             | ReLU     | Adam       | 0.002         | 500    | **988.4 (média)** | —          | **0.47 (média)** |
| **MLP #5** | K-Fold (5) | (168, 64, 32)         | Tanh     | SGD        | 0.01          | 600    | 1502.7 (média)    | —          | 0.00             |


| Modelo    | Estratégia | Árvores | Profundidade Máx. | MAE Teste | RMSE Teste | R² Teste |
| --------- | ---------- | ------- | ----------------- | --------- | ---------- | -------- |
| **RF #1** | Holdout    | 100     | — (sem limite)    | 760.2     | 1298.6     | 0.62     |
| **RF #2** | Holdout    | 200     | 30                | 749.7     | 1439.9     | 0.53     |
| **RF #3** | Holdout    | 300     | 15                | **857.9** | 1348.9     | 0.59     |
| **RF #4** | K-Fold (5) | 50      | 10                | **974.8** | —          | **0.49** |
| **RF #5** | K-Fold (5) | 150     | 12                | **931.0** | —          | **0.53** |


| Modelo     | Estratégia | Árvores | Profundidade | Learning Rate | MAE Teste | RMSE Teste | R² Teste | 
| ---------- | ---------- | ------- | ------------ | ------------- | --------- | ---------- | -------- | 
| **XGB #1** | Holdout    | 100     | 6            | 0.10          | 953.1     | 1468.7     | 0.51     | 
| **XGB #2** | Holdout    | 400     | 10           | 0.03          | **844.7** | **1327.3** | **0.60** | 
| **XGB #3** | K-Fold (5) | 80      | 5            | 0.02          | 1124.3    | —          | 0.36     | 
| **XGB #4** | K-Fold (5) | 300     | 8            | 0.05          | **913.0** | —          | **0.55** | 
| **XGB #5** | Holdout    | 50      | 4            | 0.30          | 983.6     | 1507.5     | 0.49     |


