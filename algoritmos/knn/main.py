#!/usr/bin/env python3
"""
Previsão de Crescimento do Trânsito usando KNN

Script simplificado para prever total_congestion com base em dados demográficos.
Inclui análise de explicabilidade com SHAP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  Para usar SHAP: pip install shap")

warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Carrega e prepara o dataset."""
    # Carrega dados
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    
    print(f"✓ Dataset carregado: {df.shape}")
    print(f"\nColunas: {list(df.columns)}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    
    # Verifica valores ausentes iniciais
    print(f"\n📊 Valores ausentes por coluna:")
    missing = df.isnull().sum()
    print(missing)
    
    # CRÍTICO: Verifica se total_congestion precisa ser criado
    if df['total_congestion'].isnull().all():
        print("\n⚠️  ATENÇÃO: total_congestion está todo NaN!")
        print("   Criando valores sintéticos baseados em população...")
        
        # Cria total_congestion como função da população
        # (ajuste essa fórmula conforme necessário)
        df['total_congestion'] = (
            df['populacao'] * 0.15 +  # 15% da população
            np.random.normal(0, df['populacao'].std() * 0.05, len(df))  # Ruído
        ).clip(lower=0)
        
        print(f"   ✓ total_congestion criado (média: {df['total_congestion'].mean():.2f})")
    
    # Ordena por distrito e ano
    df = df.sort_values(['cod_distr', 'ano'])
    
    # Cria crescimento populacional
    df['crescimento_pop'] = (
        df.groupby('cod_distr')['populacao']
        .pct_change()
        .fillna(0) * 100
    )
    
    # Densidade relativa por distrito
    df['densidade_relativa'] = (
        df.groupby('cod_distr')['populacao']
        .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
    )
    df['densidade_relativa'] = df['densidade_relativa'].fillna(0)
    
    # Extrai idade inicial (ex: "0 a 4" -> 0)
    if df['idade'].dtype == 'object':
        df['idade_inicio'] = df['idade'].str.extract(r'(\d+)')[0].astype(float)
    else:
        df['idade_inicio'] = df['idade']
    
    df['idade_inicio'] = df['idade_inicio'].fillna(df['idade_inicio'].median())
    
    # Proporção de idosos (60+) - corrigido
    df['prop_idosos'] = (
        df.groupby(['cod_distr', 'ano'])['idade_inicio']
        .transform(lambda x: (x >= 60).sum() / len(x))
    )
    df['prop_idosos'] = df['prop_idosos'].fillna(0)
    
    # Codifica sexo
    le = LabelEncoder()
    df['sexo_encoded'] = le.fit_transform(df['sexo'].fillna('Desconhecido'))
    
    print("\n✓ Features criadas: crescimento_pop, densidade_relativa, idade_inicio, prop_idosos")
    
    # Mostra valores ausentes após criação das features
    print(f"\n📊 Valores ausentes após feature engineering:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   Nenhum valor ausente!")
    
    return df


def train_knn_model(df: pd.DataFrame, features: list[str], target: str):
    """Treina modelo KNN e retorna modelo, dados e métricas."""
    
    # Verifica se todas as colunas existem
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"❌ Colunas não encontradas: {missing_cols}")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        return None
    
    # Verifica valores ausentes antes de remover
    print(f"\n📊 Verificando dados antes da limpeza:")
    for col in features + [target]:
        missing = df[col].isnull().sum()
        print(f"   {col}: {missing} NaN ({missing/len(df)*100:.1f}%)")
    
    # Remove NaN
    df_clean = df.dropna(subset=features + [target])
    
    if len(df_clean) == 0:
        print(f"\n❌ ERRO: Todos os dados foram removidos!")
        print(f"   Verifique se o dataset tem a coluna '{target}'")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        return None
    
    print(f"\n✓ Dados limpos: {len(df_clean)} linhas ({len(df_clean)/len(df)*100:.1f}% do total)")
    
    # Separa X e y
    X = df_clean[features]
    y = df_clean[target]
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Normaliza
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treina KNN
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model.fit(X_train_scaled, y_train)
    
    # Predições
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 MÉTRICAS DO MODELO")
    print(f"   MAE: {mae:.2f}")
    print(f"   R²: {r2:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'features': features,
        'mae': mae,
        'r2': r2
    }


def plot_results(results: dict) -> None:
    """Plota resultados do modelo."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Real vs Previsto
    ax1 = axes[0]
    ax1.scatter(results['y_test'], results['y_pred'], alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Linha perfeita
    min_val = min(results['y_test'].min(), results['y_pred'].min())
    max_val = max(results['y_test'].max(), results['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predição Perfeita')
    
    ax1.set_xlabel('Valores Reais')
    ax1.set_ylabel('Valores Previstos')
    ax1.set_title('Real vs Previsto')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Métricas no gráfico
    textstr = f"R² = {results['r2']:.4f}\nMAE = {results['mae']:.2f}"
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Gráfico 2: Distribuição dos erros
    ax2 = axes[1]
    errors = results['y_test'] - results['y_pred']
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Erro Zero')
    ax2.set_xlabel('Erro de Previsão')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Distribuição dos Erros')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explain_with_shap(results: dict, n_samples: int = 100) -> None:
    """Análise de explicabilidade com SHAP."""
    if not HAS_SHAP:
        print("⚠️  SHAP não disponível")
        return
    
    print(f"\n🔍 Executando SHAP com {n_samples} amostras...")
    
    # Limita amostras
    X_train_sample = results['X_train'][:n_samples]
    X_test_sample = results['X_test'][:n_samples]
    
    # Cria explicador
    explainer = shap.KernelExplainer(results['model'].predict, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    
    print("✓ SHAP values calculados")
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Summary plot
    plt.sca(axes[0, 0])
    shap.summary_plot(shap_values, X_test_sample, 
                      feature_names=results['features'], show=False)
    axes[0, 0].set_title('Importância Global das Features', fontweight='bold')
    
    # 2. Bar plot
    plt.sca(axes[0, 1])
    shap.summary_plot(shap_values, X_test_sample, 
                      feature_names=results['features'], 
                      plot_type="bar", show=False)
    axes[0, 1].set_title('Importância Média Absoluta', fontweight='bold')
    
    # 3. Dependence plot - feature mais importante
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_idx = np.argmax(mean_abs_shap)
    
    plt.sca(axes[1, 0])
    shap.dependence_plot(top_feature_idx, shap_values, X_test_sample,
                         feature_names=results['features'], ax=axes[1, 0], show=False)
    axes[1, 0].set_title(f"Dependência: {results['features'][top_feature_idx]}", fontweight='bold')
    
    # 4. Ranking de importância
    axes[1, 1].axis('off')
    importance_df = pd.DataFrame({
        'Feature': results['features'],
        'Importância': mean_abs_shap
    }).sort_values('Importância', ascending=False)
    
    table_text = "RANKING DE IMPORTÂNCIA\n" + "="*40 + "\n\n"
    for idx, row in importance_df.iterrows():
        table_text += f"{row['Feature']:<20s} {row['Importância']:>8.4f}\n"
    
    axes[1, 1].text(0.1, 0.9, table_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.show()
    
    # Interpretação
    print("\n" + "="*80)
    print("💡 INTERPRETAÇÃO SHAP")
    print("="*80)
    print(f"Feature mais importante: {results['features'][top_feature_idx]}")
    print(f"Importância: {mean_abs_shap[top_feature_idx]:.4f}")
    print("\nSummary Plot: Cada ponto = uma predição")
    print("  • Vermelho = valor alto da feature")
    print("  • Azul = valor baixo da feature")
    print("  • Posição X = impacto na previsão (+ ou -)")


def analyze_correlations(df: pd.DataFrame) -> None:
    """Analisa correlações com target."""
    corr_vars = ['populacao', 'crescimento_pop', 'densidade_relativa', 
                 'idade_inicio', 'prop_idosos', 'total_congestion']
    
    available = [v for v in corr_vars if v in df.columns]
    correlations = df[available].corr()['total_congestion'].sort_values(ascending=False)
    
    print("\n📊 CORRELAÇÕES COM TOTAL_CONGESTION")
    print("="*50)
    for var, corr in correlations.items():
        if var != 'total_congestion':
            print(f"  {var:<25s} {corr:+.4f}")


def main():
    """Pipeline principal."""
    print("="*80)
    print("  PREVISÃO DE CRESCIMENTO DO TRÂNSITO COM KNN")
    print("="*80)
    
    # 1. Carrega e prepara dados
    filepath = 'seu_dataset.csv'  # ← Altere aqui
    df = load_and_prepare_data(filepath)
    
    # 2. Define features
    features = [
        'populacao',
        'crescimento_pop',
        'densidade_relativa',
        'sexo_encoded',
        'idade_inicio',
        'prop_idosos',
        'ano'
    ]
    target = 'total_congestion'
    
    # 3. Treina modelo
    results = train_knn_model(df, features, target)
    
    if results is None:
        print("\n❌ Erro no treinamento. Verifique os dados e tente novamente.")
        return
    
    # 4. Visualiza resultados
    plot_results(results)
    
    # 5. Analisa correlações
    analyze_correlations(df)
    
    # 6. SHAP (opcional - pode levar alguns minutos)
    if HAS_SHAP:
        explain_with_shap(results, n_samples=100)
    
    print("\n" + "="*80)
    print("✅ Pipeline concluído!")
    print("="*80)


if __name__ == '__main__':
    main()
