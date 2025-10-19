#!/usr/bin/env python3
"""
    Previsão de Congestionamento de Trânsito usando MLPRegressor (Redes Neurais)
    
    Autor: Alexandre Marques Tortoza Canoa
    Versão do Python: 3.13.7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import json
from datetime import datetime

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("\uea6c  Para usar SHAP: pip install shap")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES GLOBAIS - ALTERE CONFORME NECESSÁRIO
# ============================================================================

class ConfigMLP:
    """Centraliza todas as configurações do experimento."""
    
    DATASET_PATH = "../../dataset/preparedData/dataset.csv"
    DELIMITER = ";"
    
    # Estratégia de Validação
    # Opções: 'holdout' ou 'kfold'
    VALIDATION_STRATEGY = 'holdout'
    HOLDOUT_TEST_SIZE = 0.2  # Proporção teste (0.2 = 80/20)
    KFOLD_N_SPLITS = 5  # Número de folds para validação cruzada
    RANDOM_STATE = 42
    
    # Hiperparâmetros do MLP
    HIDDEN_LAYER_SIZES = (100, 50, 25)  # Arquitetura: 3 camadas ocultas
    ACTIVATION = 'relu'  # 'relu', 'tanh', 'logistic', 'identity'
    SOLVER = 'adam'  # 'adam', 'sgd', 'lbfgs'
    LEARNING_RATE_INIT = 0.001  # Taxa de aprendizado inicial
    MAX_ITER = 500  # Número máximo de iterações
    EARLY_STOPPING = True  # Parar quando validação não melhorar
    VALIDATION_FRACTION = 0.1  # Fração para validação (early stopping)
    N_ITER_NO_CHANGE = 20  # Iterações sem melhora para parar
    ALPHA = 0.0001  # Regularização L2
    BATCH_SIZE = 'auto'  # Tamanho do batch (ou número específico)
    LEARNING_RATE = 'constant'  # 'constant', 'invscaling', 'adaptive'
    
    # Features e Target
    FEATURES = [
        'pop_total',            # População total (principal feature)
        'hora_numeric',         # Hora do dia (0-23)
        'via_expressa_encoded', # Via expressa (E, N, S, W)
        'regiao_encoded',       # Região (east, center, etc)
        'sexo_encoded',         # Sexo (Homens, Mulheres)
        'dia_semana',           # Dia da semana (0-6)
        'mes',                  # Mês (1-12)
    ]
    TARGET = 'tamanho_congestionamento'
    
    # SHAP
    SHAP_ENABLED = True
    SHAP_N_SAMPLES = 100  # Amostras para explicabilidade
    
    # Visualizações
    PLOT_RESULTS = True
    PLOT_SHAP = True
    PLOT_LEARNING_CURVES = True  # Curvas de aprendizado
    PLOT_LOSS_CURVE = True  # Histórico de loss
    
    # Logs
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_FILE = f"resultados_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


# ============================================================================
# FUNÇÕES DE CARREGAMENTO E PREPARAÇÃO DE DADOS
# ============================================================================

def load_data(filepath: str, delimiter: str = ";") -> pd.DataFrame:
    """
        Carrega o dataset do arquivo CSV.
        
        Args:
            filepath: Caminho do arquivo
            delimiter: Delimitador do CSV
            
        Returns:
            DataFrame com os dados carregados
    """
    try:
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                dataframe = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
                if ConfigMLP.VERBOSE:
                    print(f"\uf05d Dataset carregado: {dataframe.shape}")
                    print(f"  Encoding: {encoding}")
                    print(f"  Colunas: {list(dataframe.columns)}\n")
                return dataframe
            except UnicodeDecodeError:
                continue
    except FileNotFoundError:
        print(f"\uea87 Arquivo não encontrado: {filepath}")
        raise
    
    print(f"\uea87 Não foi possível carregar o arquivo com nenhum encoding")
    raise ValueError("Erro ao carregar arquivo")


def extract_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        Extrai features temporais das colunas 'data' e 'hora'.
        
        Args:
            dataframe: DataFrame com colunas 'data' e 'hora'
            
        Returns:
            DataFrame com novas colunas de features temporais
    """
    if ConfigMLP.VERBOSE:
        print("\uf017 Extraindo features temporais...")
    
    dataframe['datetime'] = pd.to_datetime(
        dataframe['data'] + ' ' + dataframe['hora'],
        format='%Y-%m-%d %H:%M:%S'
    )
    
    dataframe['hora_numeric'] = dataframe['datetime'].dt.hour
    dataframe['dia_semana'] = dataframe['datetime'].dt.dayofweek  # 0=segunda, 6=domingo
    dataframe['mes'] = dataframe['datetime'].dt.month
    dataframe['dia_mes'] = dataframe['datetime'].dt.day
    
    if ConfigMLP.VERBOSE:
        print("  \uf05d Features: hora_numeric, dia_semana, mes, dia_mes\n")
    
    return dataframe


def encode_categorical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        Codifica features categóricas em numéricas.
        
        Args:
            dataframe: DataFrame com features categóricas
            
        Returns:
            DataFrame com features codificadas
    """
    if ConfigMLP.VERBOSE:
        print("\uf15c Codificando features categóricas...")
    
    encoders = {}
    categorical_cols = ['via_expressa', 'regiao', 'sexo']
    
    for column in categorical_cols:
        if column in dataframe.columns:
            label_encoder = LabelEncoder()
            dataframe[f'{column}_encoded'] = label_encoder.fit_transform(
                dataframe[column].fillna('Unknown')
            )
            encoders[column] = label_encoder
            
            if ConfigMLP.VERBOSE:
                classes_dict = dict(zip(
                    label_encoder.classes_, 
                    label_encoder.transform(label_encoder.classes_)
                ))
                print(f"  \uf05d {column}: {classes_dict}")
    
    if ConfigMLP.VERBOSE:
        print()
    
    return dataframe


def prepare_data(filepath: str) -> pd.DataFrame:
    """
        Pipeline completo de preparação de dados.
        
        Args:
            filepath: Caminho do dataset
            
        Returns:
            DataFrame preparado
    """
    print("="*80)
    print("  \ueb03 CARREGAMENTO E PREPARAÇÃO DE DADOS")
    print("="*80)
    
    dataframe = load_data(filepath, delimiter=ConfigMLP.DELIMITER)
    
    print(f"\ueb2b Verificando valores ausentes iniciais:")
    missing = dataframe.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  \uf05d Nenhum valor ausente\n")
    
    dataframe = extract_datetime_features(dataframe)
    
    dataframe = encode_categorical_features(dataframe)
    
    print("\uebab Agregando dados por data/hora/via/região...")
    dataframe_aggregated = dataframe.groupby(
        ['data', 'hora', 'via_expressa', 'regiao']
    ).agg({
        'pop_total': 'sum', 
        'tamanho_congestionamento': 'first',  
        'hora_numeric': 'first',
        'dia_semana': 'first',
        'mes': 'first',
        'via_expressa_encoded': 'first',
        'regiao_encoded': 'first',
        'sexo_encoded': 'mean',
    }).reset_index()
    
    print(f"  \uf05d {len(dataframe)} → {len(dataframe_aggregated)} linhas\n")
    
    print(f"\ueb2b Valores ausentes após preparação:")
    missing = dataframe_aggregated.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  \uf05d Nenhum valor ausente\n")
    
    print(f"\ueb03 Estatísticas do target (tamanho_congestionamento):")
    print(f"  Mínimo: {dataframe_aggregated['tamanho_congestionamento'].min()}")
    print(f"  Máximo: {dataframe_aggregated['tamanho_congestionamento'].max()}")
    print(f"  Média: {dataframe_aggregated['tamanho_congestionamento'].mean():.2f}")
    print(f"  Mediana: {dataframe_aggregated['tamanho_congestionamento'].median():.2f}\n")
    
    return dataframe_aggregated


# ============================================================================
# FUNÇÕES DE TREINAMENTO
# ============================================================================

def train_mlp_holdout(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> dict:
    """
        Treina MLP com validação holdout (80/20).
        
        Args:
            X_train, X_test: Features de treino/teste (já escaladas)
            y_train, y_test: Target de treino/teste
            
        Returns:
            Dicionário com modelo, métricas e dados
    """
    print(f" Treinando MLP (hidden_layers={ConfigMLP.HIDDEN_LAYER_SIZES})...")
    
    model = MLPRegressor(
        hidden_layer_sizes=ConfigMLP.HIDDEN_LAYER_SIZES,
        activation=ConfigMLP.ACTIVATION,
        solver=ConfigMLP.SOLVER,
        learning_rate_init=ConfigMLP.LEARNING_RATE_INIT,
        max_iter=ConfigMLP.MAX_ITER,
        early_stopping=ConfigMLP.EARLY_STOPPING,
        validation_fraction=ConfigMLP.VALIDATION_FRACTION,
        n_iter_no_change=ConfigMLP.N_ITER_NO_CHANGE,
        alpha=ConfigMLP.ALPHA,
        batch_size=ConfigMLP.BATCH_SIZE,
        learning_rate=ConfigMLP.LEARNING_RATE,
        random_state=ConfigMLP.RANDOM_STATE,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'n_iterations': model.n_iter_,
    }
    
    print(f"  \uf05d Convergência em {model.n_iter_} iterações")
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': metrics,
    }


def train_mlp_kfold(X: np.ndarray, y: np.ndarray) -> dict:
    """
        Treina MLP com validação cruzada K-Fold.
        
        Args:
            X: Features (já escaladas)
            y: Target
            
        Returns:
            Dicionário com scores e estatísticas
    """
    print(f" Treinando MLP com {ConfigMLP.KFOLD_N_SPLITS}-Fold CV...")
    
    model = MLPRegressor(
        hidden_layer_sizes=ConfigMLP.HIDDEN_LAYER_SIZES,
        activation=ConfigMLP.ACTIVATION,
        solver=ConfigMLP.SOLVER,
        learning_rate_init=ConfigMLP.LEARNING_RATE_INIT,
        max_iter=ConfigMLP.MAX_ITER,
        early_stopping=ConfigMLP.EARLY_STOPPING,
        validation_fraction=ConfigMLP.VALIDATION_FRACTION,
        n_iter_no_change=ConfigMLP.N_ITER_NO_CHANGE,
        alpha=ConfigMLP.ALPHA,
        batch_size=ConfigMLP.BATCH_SIZE,
        learning_rate=ConfigMLP.LEARNING_RATE,
        random_state=ConfigMLP.RANDOM_STATE,
        verbose=False
    )
    
    kfold = KFold(
        n_splits=ConfigMLP.KFOLD_N_SPLITS, 
        shuffle=True, 
        random_state=ConfigMLP.RANDOM_STATE
    )
    
    scores_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    scores_mae = cross_val_score(
        model, X, y, cv=kfold, 
        scoring='neg_mean_absolute_error'
    )
    
    # Treinar modelo final com todos os dados para análise posterior
    model.fit(X, y)
    
    return {
        'model': model,
        'X': X,
        'y': y,
        'cv_r2_scores': scores_r2,
        'cv_mae_scores': -scores_mae,
        'cv_r2_mean': scores_r2.mean(),
        'cv_r2_std': scores_r2.std(),
        'cv_mae_mean': -scores_mae.mean(),
        'cv_mae_std': scores_mae.std(),
    }


def train_model(dataframe: pd.DataFrame) -> dict:
    """
        Pipeline de treinamento adaptado à estratégia de validação.
        
        Args:
            dataframe: DataFrame preparado
            
        Returns:
            Resultados do treinamento
    """
    print("="*80)
    print("   TREINAMENTO DO MODELO")
    print("="*80)
    
    missing_features = [
        feature for feature in ConfigMLP.FEATURES 
        if feature not in dataframe.columns
    ]
    if missing_features:
        print(f"\uea87 Features não encontradas: {missing_features}")
        print(f"   Colunas disponíveis: {list(dataframe.columns)}")
        raise ValueError("Features faltando no dataset")
    
    dataframe_clean = dataframe.dropna(
        subset=ConfigMLP.FEATURES + [ConfigMLP.TARGET]
    )
    print(f"\uf05d Dados limpos: {len(dataframe)} → {len(dataframe_clean)} linhas\n")
    
    X = dataframe_clean[ConfigMLP.FEATURES].values
    y = dataframe_clean[ConfigMLP.TARGET].values
    
    # CRÍTICO: Normalização é ESSENCIAL para redes neurais
    print(" Normalizando features (CRÍTICO para MLP)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  \uf05d Média: {X_scaled.mean(axis=0)}")
    print(f"  \uf05d Std: {X_scaled.std(axis=0)}\n")
    
    if ConfigMLP.VALIDATION_STRATEGY == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=ConfigMLP.HOLDOUT_TEST_SIZE,
            random_state=ConfigMLP.RANDOM_STATE
        )
        print(f"\uf05d Split: {len(X_train)} treino | {len(X_test)} teste\n")
        
        results = train_mlp_holdout(X_train, X_test, y_train, y_test)
        
    elif ConfigMLP.VALIDATION_STRATEGY == 'kfold':
        results = train_mlp_kfold(X_scaled, y)
    
    else:
        raise ValueError(
            f"Estratégia desconhecida: {ConfigMLP.VALIDATION_STRATEGY}"
        )
    
    results['scaler'] = scaler
    results['features'] = ConfigMLP.FEATURES
    results['dataframe'] = dataframe_clean
    
    return results


def print_metrics(results: dict) -> None:
    """Imprime métricas de forma formatada."""
    print("\n" + "="*80)
    print("  \ueb03 MÉTRICAS DO MODELO")
    print("="*80)
    
    if ConfigMLP.VALIDATION_STRATEGY == 'holdout':
        metrics = results['metrics']
        print(f"\n{'Métrica':<25} {'Treino':>12} {'Teste':>12}")
        print("-" * 50)
        print(f"{'MAE':<25} {metrics['train_mae']:>12.4f} {metrics['test_mae']:>12.4f}")
        print(f"{'RMSE':<25} {metrics['train_rmse']:>12.4f} {metrics['test_rmse']:>12.4f}")
        print(f"{'R² Score':<25} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}")
        print(f"{'Iterações':<25} {metrics['n_iterations']:>12}")
        
    elif ConfigMLP.VALIDATION_STRATEGY == 'kfold':
        print(f"\n{ConfigMLP.KFOLD_N_SPLITS}-Fold Cross Validation:")
        print("-" * 50)
        print(f"R² Scores: {results['cv_r2_scores']}")
        print(f"R² Média: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
        print(f"MAE Média: {results['cv_mae_mean']:.4f} (+/- {results['cv_mae_std']:.4f})")
    
    print()


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_results_holdout(results: dict) -> None:
    """Plota resultados para validação holdout."""
    if not ConfigMLP.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Real vs Previsto (Teste)
    ax = axes[0, 0]
    ax.scatter(
        results['y_test'], 
        results['y_test_pred'], 
        alpha=0.5, 
        s=20, 
        edgecolors='k', 
        linewidth=0.5
    )
    min_val = min(results['y_test'].min(), results['y_test_pred'].min())
    max_val = max(results['y_test'].max(), results['y_test_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Teste: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    textstr = (
        f"R² = {results['metrics']['test_r2']:.4f}\n"
        f"MAE = {results['metrics']['test_mae']:.2f}"
    )
    ax.text(
        0.05, 0.95, textstr, 
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Real vs Previsto (Treino)
    ax = axes[0, 1]
    ax.scatter(
        results['y_train'], 
        results['y_train_pred'], 
        alpha=0.5, 
        s=20, 
        edgecolors='k', 
        linewidth=0.5
    )
    min_val = min(results['y_train'].min(), results['y_train_pred'].min())
    max_val = max(results['y_train'].max(), results['y_train_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Treino: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    textstr = (
        f"R² = {results['metrics']['train_r2']:.4f}\n"
        f"MAE = {results['metrics']['train_mae']:.2f}"
    )
    ax.text(
        0.05, 0.95, textstr, 
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    
    # Distribuição de Erros
    ax = axes[1, 0]
    errors = results['y_test'] - results['y_test_pred']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Erro de Previsão')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição de Erros (Teste) - Média: {errors.mean():.2f}')
    ax.grid(True, alpha=0.3)
    
    # Análise de Resíduos
    ax = axes[1, 1]
    residuals = results['y_test'] - results['y_test_pred']
    ax.scatter(
        results['y_test_pred'], 
        residuals, 
        alpha=0.5, 
        s=20, 
        edgecolors='k', 
        linewidth=0.5
    )
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Valores Previstos')
    ax.set_ylabel('Resíduos')
    ax.set_title('Análise de Resíduos (Teste)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_results_holdout.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: mlp_results_holdout.png")
    plt.show()


def plot_results_kfold(results: dict) -> None:
    """Plota resultados para validação K-Fold."""
    if not ConfigMLP.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² por Fold
    ax = axes[0]
    folds = np.arange(1, len(results['cv_r2_scores']) + 1)
    ax.bar(
        folds, 
        results['cv_r2_scores'], 
        alpha=0.7, 
        color='skyblue', 
        edgecolor='black'
    )
    ax.axhline(
        y=results['cv_r2_mean'], 
        color='r', 
        linestyle='--', 
        linewidth=2, 
        label='Média'
    )
    ax.fill_between(
        folds, 
        results['cv_r2_mean'] - results['cv_r2_std'],
        results['cv_r2_mean'] + results['cv_r2_std'],
        alpha=0.2, 
        color='r'
    )
    ax.set_xlabel('Fold')
    ax.set_ylabel('R² Score')
    ax.set_title('Scores R² por Fold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # MAE por Fold
    ax = axes[1]
    ax.bar(
        folds, 
        results['cv_mae_scores'], 
        alpha=0.7, 
        color='lightcoral', 
        edgecolor='black'
    )
    ax.axhline(
        y=results['cv_mae_mean'], 
        color='r', 
        linestyle='--', 
        linewidth=2, 
        label='Média'
    )
    ax.fill_between(
        folds,
        results['cv_mae_mean'] - results['cv_mae_std'],
        results['cv_mae_mean'] + results['cv_mae_std'],
        alpha=0.2, 
        color='r'
    )
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('Scores MAE por Fold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('mlp_results_kfold.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: mlp_results_kfold.png")
    plt.show()


def plot_loss_curve(results: dict) -> None:
    """Plota a curva de loss durante o treinamento."""
    if not ConfigMLP.PLOT_LOSS_CURVE:
        return
    
    if ConfigMLP.VALIDATION_STRATEGY != 'holdout':
        print("\uea6c  Loss curve disponível apenas para validação holdout")
        return
    
    model = results['model']
    
    if not hasattr(model, 'loss_curve_'):
        print("\uea6c  Loss curve não disponível para este modelo")
        return
    
    print("\n" + "="*80)
    print("   CURVA DE LOSS (TREINAMENTO)")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(model.loss_curve_) + 1)
    ax.plot(iterations, model.loss_curve_, linewidth=2, color='blue', label='Training Loss')
    
    if hasattr(model, 'validation_scores_') and ConfigMLP.EARLY_STOPPING:
        # Convertendo validation scores para loss (inversamente proporcional)
        ax2 = ax.twinx()
        ax2.plot(
            iterations, 
            model.validation_scores_, 
            linewidth=2, 
            color='orange', 
            linestyle='--',
            label='Validation Score'
        )
        ax2.set_ylabel('Validation Score (R²)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
    
    ax.set_xlabel('Iterações')
    ax.set_ylabel('Training Loss', color='blue')
    ax.set_title(f'Curva de Loss - Convergência em {model.n_iter_} iterações')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.tick_params(axis='y', labelcolor='blue')
    
    plt.tight_layout()
    plt.savefig('mlp_loss_curve.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: mlp_loss_curve.png\n")
    plt.show()


def plot_network_architecture(results: dict) -> None:
    """Plota a arquitetura da rede neural."""
    print("\n" + "="*80)
    print("   ARQUITETURA DA REDE NEURAL")
    print("="*80)
    
    model = results['model']
    
    layers = [len(ConfigMLP.FEATURES)] + list(ConfigMLP.HIDDEN_LAYER_SIZES) + [1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layer_positions = np.linspace(0, 10, len(layers))
    max_neurons = max(layers)
    
    for layer_index, (x_pos, n_neurons) in enumerate(zip(layer_positions, layers)):
        y_positions = np.linspace(0, max_neurons, n_neurons + 2)[1:-1]
        
        for y_pos in y_positions:
            circle = plt.Circle(
                (x_pos, y_pos), 
                0.3, 
                color='skyblue', 
                ec='black', 
                linewidth=2
            )
            ax.add_patch(circle)
        
        # Conexões com próxima camada
        if layer_index < len(layers) - 1:
            next_y_positions = np.linspace(0, max_neurons, layers[layer_index + 1] + 2)[1:-1]
            
            for y1 in y_positions:
                for y2 in next_y_positions:
                    ax.plot(
                        [x_pos + 0.3, layer_positions[layer_index + 1] - 0.3],
                        [y1, y2],
                        'gray',
                        alpha=0.3,
                        linewidth=0.5
                    )
        
        # Labels
        if layer_index == 0:
            label = f'Input\n({n_neurons} features)'
        elif layer_index == len(layers) - 1:
            label = 'Output\n(1 target)'
        else:
            label = f'Hidden {layer_index}\n({n_neurons} neurons)'
        
        ax.text(
            x_pos, 
            max_neurons + 1, 
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, max_neurons + 3)
    ax.axis('off')
    ax.set_title(
        f'Arquitetura MLP: {ConfigMLP.HIDDEN_LAYER_SIZES}\n'
        f'Ativação: {ConfigMLP.ACTIVATION.upper()} | Solver: {ConfigMLP.SOLVER.upper()}',
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig('mlp_architecture.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: mlp_architecture.png\n")
    plt.show()


# ============================================================================
# FUNÇÕES SHAP
# ============================================================================

def explain_with_shap(results: dict) -> None:
    """Análise de explicabilidade com SHAP."""
    if not ConfigMLP.SHAP_ENABLED or not HAS_SHAP:
        if ConfigMLP.VERBOSE:
            print("\uea6c  SHAP desabilitado ou não disponível")
        return
    
    if ConfigMLP.VALIDATION_STRATEGY != 'holdout':
        print("\uea6c  SHAP disponível apenas para validação holdout")
        return
    
    print("\n" + "="*80)
    print("   ANÁLISE SHAP")
    print("="*80)
    print(f"Executando SHAP com {ConfigMLP.SHAP_N_SAMPLES} amostras...")
    print("  Aviso: KernelExplainer pode ser lento para redes neurais\n")
    
    n_samples = min(ConfigMLP.SHAP_N_SAMPLES, len(results['X_train']))
    X_sample = results['X_train'][:n_samples]
    X_test_sample = results['X_test'][:min(100, len(results['X_test']))]
    
    # KernelExplainer para redes neurais
    explainer = shap.KernelExplainer(results['model'].predict, X_sample)
    shap_values = explainer.shap_values(X_test_sample)
    
    print("\uf05d SHAP values calculados\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Summary Plot (beeswarm)
    plt.sca(axes[0, 0])
    shap.summary_plot(
        shap_values, 
        X_test_sample, 
        feature_names=results['features'], 
        show=False
    )
    axes[0, 0].set_title('Importância Global (SHAP)', fontweight='bold', fontsize=12)
    
    # Bar Plot (média absoluta)
    plt.sca(axes[0, 1])
    shap.summary_plot(
        shap_values, 
        X_test_sample, 
        feature_names=results['features'], 
        plot_type="bar", 
        show=False
    )
    axes[0, 1].set_title('Importância Média Absoluta', fontweight='bold', fontsize=12)
    
    # Dependence Plot (feature mais importante)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_index = np.argmax(mean_abs_shap)
    
    plt.sca(axes[1, 0])
    shap.dependence_plot(
        top_index, 
        shap_values, 
        X_test_sample,
        feature_names=results['features'], 
        ax=axes[1, 0], 
        show=False
    )
    
    # Tabela de Importâncias
    axes[1, 1].axis('off')
    importance_dataframe = pd.DataFrame({
        'Feature': results['features'],
        'SHAP Mean |Impact|': mean_abs_shap
    }).sort_values('SHAP Mean |Impact|', ascending=False)
    
    table_text = "RANKING SHAP\n" + "="*45 + "\n\n"
    for index, row in importance_dataframe.iterrows():
        table_text += (
            f"{row['Feature']:<25} {row['SHAP Mean |Impact|']:>10.4f}\n"
        )
    
    axes[1, 1].text(
        0.1, 0.9, table_text, 
        transform=axes[1, 1].transAxes,
        fontsize=10, 
        verticalalignment='top', 
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    )
    
    plt.tight_layout()
    plt.savefig('mlp_shap_analysis.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: mlp_shap_analysis.png\n")
    plt.show()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def save_results(results: dict, filename: str) -> None:
    """Salva resultados em JSON."""
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'strategy': ConfigMLP.VALIDATION_STRATEGY,
            'hidden_layer_sizes': list(ConfigMLP.HIDDEN_LAYER_SIZES),
            'activation': ConfigMLP.ACTIVATION,
            'solver': ConfigMLP.SOLVER,
            'learning_rate_init': ConfigMLP.LEARNING_RATE_INIT,
            'max_iter': ConfigMLP.MAX_ITER,
            'features': ConfigMLP.FEATURES,
        }
    }
    
    if ConfigMLP.VALIDATION_STRATEGY == 'holdout':
        results_to_save['metrics'] = {
            key: float(value) if not isinstance(value, int) else value
            for key, value in results['metrics'].items()
        }
    else:
        results_to_save['cv_scores'] = {
            'r2_mean': float(results['cv_r2_mean']),
            'r2_std': float(results['cv_r2_std']),
            'mae_mean': float(results['cv_mae_mean']),
            'mae_std': float(results['cv_mae_std']),
        }
    
    with open(filename, 'w') as file:
        json.dump(results_to_save, file, indent=2)
    
    print(f"\uf05d Resultados salvos: {filename}")


def main():
    """Pipeline principal."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  PREVISÃO DE CONGESTIONAMENTO COM MLP - PIPELINE MODULAR".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\n\ued7b CONFIGURAÇÕES:")
    print(f"   Dataset: {ConfigMLP.DATASET_PATH}")
    print(f"   Features: {ConfigMLP.FEATURES}")
    print(f"   Target: {ConfigMLP.TARGET}")
    print(f"   Validação: {ConfigMLP.VALIDATION_STRATEGY.upper()}")
    if ConfigMLP.VALIDATION_STRATEGY == 'holdout':
        print(f"   Test Size: {ConfigMLP.HOLDOUT_TEST_SIZE}")
    else:
        print(f"   N-Folds: {ConfigMLP.KFOLD_N_SPLITS}")
    print(f"   Arquitetura: {ConfigMLP.HIDDEN_LAYER_SIZES}")
    print(f"   Ativação: {ConfigMLP.ACTIVATION}")
    print(f"   Solver: {ConfigMLP.SOLVER}")
    print(f"   Learning Rate: {ConfigMLP.LEARNING_RATE_INIT}")
    print(f"   Max Iterations: {ConfigMLP.MAX_ITER}")
    print(f"   Early Stopping: {ConfigMLP.EARLY_STOPPING}")
    print(f"   SHAP: {'Habilitado' if ConfigMLP.SHAP_ENABLED else 'Desabilitado'}\n")
    
    try:
        dataframe = prepare_data(ConfigMLP.DATASET_PATH)
        
        results = train_model(dataframe)
        
        print_metrics(results)
        
        if ConfigMLP.VALIDATION_STRATEGY == 'holdout':
            plot_results_holdout(results)
        else:
            plot_results_kfold(results)
        
        plot_loss_curve(results)
        
        plot_network_architecture(results)
        
        explain_with_shap(results)
        
        if ConfigMLP.SAVE_RESULTS:
            save_results(results, ConfigMLP.RESULTS_FILE)
        
        print("\n" + "="*80)
        print("  \uf058 PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80 + "\n")
        
    except Exception as error:
        print(f"\n\uea87 ERRO: {str(error)}")
        raise


if __name__ == '__main__':
    main()