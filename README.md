#  Previsão de Congestionamentos em São Paulo com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.13.7-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Project-orange)]()
[![Made with](https://img.shields.io/badge/Made%20with-💡%20Machine%20Learning-brightgreen)]()

---

Este repositório apresenta uma **pesquisa aplicada** que utiliza **técnicas de aprendizado de máquina (Machine Learning)** para prever o **tamanho de congestionamentos** em **São Paulo**, com base em dados históricos de **tráfego urbano** e **população regional**.

O estudo busca entender como variáveis sociais e temporais influenciam o comportamento do trânsito em grandes centros urbanos.

---

## Estrutura do Projeto

```
pesquisa_aplicada/
│
├── rawData/                  # Dados brutos de entrada (população, trânsito)
├── clean/                    # Dados limpos e tratados
│
├── clear_population.py       # Limpeza e normalização dos dados populacionais
├── clear_traffic.py          # Limpeza e normalização dos dados de congestionamentos
│
├── preparedData/             # Dataset unificado pronto para modelagem
│
├── results/                  # Resultados e gráficos dos experimentos
│   ├── resultados_knn_*.json
│   ├── resultados_rf_*.json
│   ├── resultados_xgboost_*.json
│   ├── resultados_mlp_*.json
│   └── resultados_linear_*.json
│
└── estrutura_projeto.md      # Estrutura metodológica do artigo científico
```

---

## Objetivo

Explorar modelos de regressão supervisionada para prever o **tamanho do congestionamento (em metros)** com base em variáveis como:

* Região da cidade
* Horário e dia da semana
* Mês do ano
* Via expressa
* População total por sexo

---

## Pipeline Metodológico

### 1. **Aquisição e Integração de Dados**

Dados coletados de fontes públicas.
Durante o merge, a granularidade populacional foi simplificada para **total de homens e mulheres por região/ano**, reduzindo o dataset de **170 GB** para uma versão viável (~300k registros).

### 2. **Limpeza e Padronização**

Scripts automatizados realizam o tratamento:

* `clear_population.py` → limpa e mapeia distritos por região, remove acentos, filtra idades ≥ 20 anos e gera relatório.
* `clear_traffic.py` → valida datas e horários, normaliza textos, remove duplicatas e exporta dados limpos.

### 3. **Modelagem Preditiva**

Foram comparados cinco modelos:

| Modelo                | R² (Teste) | MAE (Teste) | Observações                        |
| --------------------- | ---------- | ----------- | ---------------------------------- |
| **KNN**               | 0.53       | 786.20      | Sensível à escala das features     |
| **Random Forest**     | **0.62**   | 760.17      | Melhor desempenho geral            |
| **XGBoost**           | 0.51       | 953.15      | Estável, leve overfitting          |
| **MLP Neural Net**    | 0.47       | 988.53      | Potencial para otimização          |
| **Linear Regression** | 0.10       | 1379.68     | Baixo ajuste para não linearidades |

> As análises incluem **gráficos SHAP**, **curvas de erro** e **importância das features** para interpretação dos resultados.

### 4. **Avaliação**

Foram usadas métricas de erro e explicabilidade:

* **MAE, RMSE, R²**
* **Feature Importance / Coeficientes**
* **Análise SHAP**

A variável **pop_total** foi a mais influente, seguida de **via_expressa_encoded** e **mês**, mostrando que o comportamento do trânsito é fortemente impactado por fatores demográficos e sazonais.

---

## Instalação e Execução

### 1. Clonar o Repositório

```bash
git clone https://github.com/Alexandre-Tortoza/pesquisa_aplicada.git
cd pesquisa_aplicada
```

### 2. Criar e Ativar Ambiente Virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
.venv\Scripts\activate     # Windows
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

*(Se o arquivo `requirements.txt` ainda não existe, pode ser criado com:)*

```bash
pip freeze > requirements.txt
```

### 4. Executar os Pipelines de Limpeza

```bash
python clear_population.py
python clear_traffic.py
```

### 5. Treinar e Avaliar Modelos

Os notebooks ou scripts de modelagem estão na pasta `results/` e podem ser executados em sequência para reproduzir os experimentos.

---

## Resultados Principais

* **Random Forest** obteve o melhor equilíbrio entre precisão e interpretabilidade.
* **XGBoost** confirmou estabilidade com bom ajuste geral.
* **MLP** apresentou potencial para capturar relações não lineares.
* A limpeza e unificação dos dados foram decisivas para a qualidade dos resultados.

---

## Tecnologias Utilizadas

| Categoria            | Ferramentas               |
| -------------------- | ------------------------- |
| Linguagem            | Python 3.13.7             |
| Manipulação de Dados | Pandas, NumPy             |
| Modelagem            | Scikit-learn, XGBoost     |
| Visualização         | Matplotlib, Seaborn, SHAP |
| Ambiente             | Neovim (Arch Linux) 🐧    |

---

## Estrutura Científica

Baseada no documento [`estrutura_projeto.md`](./estrutura_projeto.md), seguindo o formato clássico de artigos científicos:

1. **Introdução** – contextualização e motivação.
2. **Metodologia** – pipeline de dados e algoritmos.
3. **Resultados** – comparações quantitativas e visuais.
4. **Discussão** – implicações e limitações.
5. **Conclusão** – síntese e contribuições.

---


## Autor

**Alexandre Marques Tortoza Canoa**
[alexandre.tortoza@gmail.com](mailto:alexandre.tortoza@gmail.com)
[GitHub: Alexandre-Tortoza](https://github.com/Alexandre-Tortoza)
[alexandre-Tortoza.tech](https://alexandre-Tortoza.tech)
