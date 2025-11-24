import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import storage
from google.cloud import bigquery
from io import BytesIO
from datetime import datetime, timedelta

# =============================================================================
# SEÇÃO DE AUTENTICAÇÃO E SEGURANÇA
# =============================================================================
# Verifica se o usuário está autenticado. Se não, bloqueia o acesso.
if not st.session_state.get('authenticated', False):
    st.error("🔒 Acesso negado. Por favor, faça o login para continuar.")
    st.stop()

# Botão de Logout será movido para o final da sidebar
# =============================================================================

# --- CONFIGURAÇÕES DO PAINEL E DO PROJETO ---
GCP_PROJECT_ID = "vaulted-zodiac-294702"                
MODEL_BUCKET = "rbbr-artifacts"                 
MODEL_BLOB = "models/elasticity/modelo_elasticidade_v7_hurdle_mensal.joblib"    
BQ_DATASET = "RBBR_DATA_SCIENCE"                 
BQ_BASE_TABLE = "DM_ELASTICITY_LGBM"         


@st.cache_resource
def load_model(project_id, bucket_name, blob_name):
    """Carrega o modelo hurdle do GCS usando os segredos do Streamlit."""
    try:
        from google.oauth2 import service_account
        
        # Converter o dicionário de credenciais para o formato correto
        credentials_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        
        storage_client = storage.Client(project=project_id, credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        model_file = BytesIO(blob.download_as_bytes())
        artefatos = joblib.load(model_file)
        
        # O modelo hurdle retorna um dicionário com os artefatos
        modelo_classificador = artefatos['modelo_classificador']
        modelo_regressor = artefatos['modelo_regressor']
        colunas_treino = artefatos['colunas_treino']
        colunas_categoricas = artefatos['colunas_categoricas']
        mapeamento_encoders = artefatos['mapeamento_encoders']
        
        return {
            'classificador': modelo_classificador,
            'regressor': modelo_regressor,
            'colunas_treino': colunas_treino,
            'colunas_categoricas': colunas_categoricas,
            'encoders': mapeamento_encoders
        }
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_data
def load_data(project_id, dataset, table):
    """Carrega os dados base do BigQuery."""
    try:
        from google.oauth2 import service_account
        
        # Converter o dicionário de credenciais para o formato correto
        credentials_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        
        query = f"""
            SELECT 
                NM_ITEM,
                COALESCE(NM_FAMILIA_ITEM, 'NAO IDENTIFICADO') AS NM_FAMILIA_ITEM,
                PRECO_ATUAL,
                PRECO_SIMULADO,
                VARIACAO_PERCENTUAL,
                VENDAS_PREVISTAS,
                UPDATED_DT
            FROM `{project_id}.{dataset}.{table}`
            WHERE VARIACAO_PERCENTUAL = 0
            ORDER BY UPDATED_DT DESC
            LIMIT 1000
        """
        bq_client = bigquery.Client(project=project_id, credentials=credentials)
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            st.warning("A consulta ao BigQuery não retornou dados. Verifique a tabela e a query.")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

def transform_with_unknown_handling(encoder, values):
    """
    Função auxiliar para transformar valores com tratamento de valores não vistos.
    Valores não vistos são mapeados para -1.
    """
    import numpy as np
    
    # Obter as classes conhecidas do encoder
    classes_known = set(encoder.classes_)
    
    # Converter para array numpy se necessário
    values_array = np.array(values)
    
    # Criar máscara para valores conhecidos
    mask_known = np.array([val in classes_known for val in values_array])
    
    # Inicializar array de resultados
    result = np.full(len(values_array), -1, dtype=int)
    
    # Transformar apenas valores conhecidos
    if mask_known.sum() > 0:
        result[mask_known] = encoder.transform(values_array[mask_known])
    
    return result

def preparar_features_para_modelo(df_produto, preco_simulado, artefatos_modelo, dados_base=None):
    """
    Prepara as features necessárias para o modelo hurdle.
    Esta função cria um DataFrame com todas as features necessárias baseado nos dados do produto.
    """
    import pandas as pd
    from datetime import datetime
    
    # Criar DataFrame base
    df_features = df_produto.copy()
    
    # Definir preço
    df_features['PRECO_MEDIO'] = preco_simulado
    
    # Estoque médio (não temos nos dados de simulação, usar 0 ou valor padrão)
    df_features['VL_ESTOQUE_MEDIO'] = 0
    
    # Data atual
    data_atual = datetime.now()
    df_features['DT_EMISSAO'] = pd.to_datetime(data_atual)
    df_features['ANO'] = data_atual.year
    df_features['MES'] = data_atual.month
    df_features['SEMANA_ANO'] = data_atual.isocalendar().week
    
    # Features de calendário/eventos
    mes = data_atual.month

    df_features['DIA_MULHERES_MES'] = 1 if mes == 3 else 0
    df_features['DIA_MAES_MES'] = 1 if mes == 5 else 0
    df_features['DIA_NAMORADOS_MES'] = 1 if mes == 6 else 0
    df_features['BLACK_FRIDAY_MES'] = 1 if mes == 11 else 0
    df_features['NATAL_MES'] = 1 if mes == 12 else 0
    df_features['CARNAVAL_MES'] = 1 if mes == 2 else 0
    df_features['SEMANA_CONSUMIDOR_MES'] = 1 if mes == 3 else 0
    df_features['VERAO_MES'] = 1 if mes in [12, 1, 2] else 0
    df_features['OUTONO_MES'] = 1 if mes in [3, 4, 5] else 0
    df_features['INVERNO_MES'] = 1 if mes in [6, 7, 8] else 0
    df_features['PRIMAVERA_MES'] = 1 if mes in [9, 10, 11] else 0
    df_features['FLAG_RUPTURA_DIARIA_MES'] = 0  # Assumindo que não há ruptura
    
    # Se temos dados base, usar lags e médias móveis
    # Caso contrário, usar valores padrão
    if dados_base is not None and len(dados_base) > 0:
        # Pegar o último registro do produto
        ultimo_registro = dados_base[dados_base['NM_ITEM'] == df_produto['NM_ITEM'].iloc[0]].iloc[-1] if len(dados_base) > 0 else None
        if ultimo_registro is not None:
            df_features['lag_vendas_1m'] = ultimo_registro.get('VENDAS_PREVISTAS', 0)
            df_features['lag_preco_1m'] = ultimo_registro.get('PRECO_ATUAL', preco_simulado)
            df_features['lag_estoque_1m'] = 0  # Não temos estoque nos dados de simulação
            df_features['lag_ruptura_1m'] = 0
            df_features['diff_preco_1m'] = preco_simulado - ultimo_registro.get('PRECO_ATUAL', preco_simulado)
            df_features['diff_estoque_1m'] = 0
            df_features['media_movel_vendas_3m'] = ultimo_registro.get('VENDAS_PREVISTAS', 0)
            df_features['media_movel_vendas_6m'] = ultimo_registro.get('VENDAS_PREVISTAS', 0)
            df_features['lag_vendas_12m'] = ultimo_registro.get('VENDAS_PREVISTAS', 0)
        else:
            # Valores padrão
            df_features['lag_vendas_1m'] = 0
            df_features['lag_preco_1m'] = preco_simulado
            df_features['lag_estoque_1m'] = 0
            df_features['lag_ruptura_1m'] = 0
            df_features['diff_preco_1m'] = 0
            df_features['diff_estoque_1m'] = 0
            df_features['media_movel_vendas_3m'] = 0
            df_features['media_movel_vendas_6m'] = 0
            df_features['lag_vendas_12m'] = 0
    else:
        # Valores padrão quando não há dados base
        df_features['lag_vendas_1m'] = 0
        df_features['lag_preco_1m'] = preco_simulado
        df_features['lag_estoque_1m'] = 0
        df_features['lag_ruptura_1m'] = 0
        df_features['diff_preco_1m'] = 0
        df_features['diff_estoque_1m'] = 0
        df_features['media_movel_vendas_3m'] = 0
        df_features['media_movel_vendas_6m'] = 0
        df_features['lag_vendas_12m'] = 0
    
    # Codificar variáveis categóricas
    encoders = artefatos_modelo['encoders']
    colunas_categoricas = artefatos_modelo['colunas_categoricas']
    
    for col in ['NM_ITEM', 'NM_FAMILIA_ITEM', 'NM_COR']:
        if col in df_features.columns and col in encoders:
            df_features[col] = transform_with_unknown_handling(
                encoders[col],
                df_features[col].values
            )
    
    # Garantir que todas as colunas categóricas estão presentes
    for col in colunas_categoricas:
        if col not in df_features.columns:
            df_features[col] = 0
    
    # Converter colunas categóricas para o tipo category
    for col in colunas_categoricas:
        if col in df_features.columns:
            try:
                # Tentar usar as categorias do treino se disponível
                max_val = int(df_features[col].max()) if pd.notna(df_features[col].max()) else 0
                df_features[col] = pd.Categorical(df_features[col], categories=range(max_val + 1))
            except Exception:
                df_features[col] = df_features[col].astype('category')
    
    # Selecionar apenas as colunas de treino
    colunas_treino = artefatos_modelo['colunas_treino']
    
    # Garantir que todas as colunas de treino existem antes de selecionar
    for col in colunas_treino:
        if col not in df_features.columns:
            # Se a coluna não existe, criar com valor padrão 0
            df_features[col] = 0
    
    # Selecionar apenas as colunas de treino na ordem correta
    df_final = df_features[colunas_treino].copy()
    
    return df_final

def generate_price_sensitivity_curve(df, selected_product, artefatos_modelo, num_points=30):
    """Gera dados para a curva de sensibilidade de preço usando modelo hurdle."""
    try:
        # Filtrar dados do produto selecionado
        product_data = df[df['NM_ITEM'] == selected_product].copy()
        if product_data.empty:
            return None
        
        # Obter preço atual
        current_price = product_data['PRECO_ATUAL'].iloc[0]
        
        # Calcular venda base usando o modelo com o preço atual
        # Isso garante que estamos usando a previsão do modelo, não os dados do BigQuery
        df_features_base = preparar_features_para_modelo(product_data, current_price, artefatos_modelo, df)
        modelo_class = artefatos_modelo['classificador']
        modelo_reg = artefatos_modelo['regressor']
        
        # Predição hurdle para o preço atual (situação base)
        prob_venda_base = modelo_class.predict_proba(df_features_base)[:, 1]
        qtd_venda_log_base = modelo_reg.predict(df_features_base)
        qtd_venda_real_base = np.expm1(qtd_venda_log_base)
        qtd_venda_real_base[qtd_venda_real_base < 0] = 0
        current_sales = prob_venda_base * qtd_venda_real_base
        current_sales_value = current_sales[0] if len(current_sales) > 0 else 0
        
        # Gerar range de preços (-50% a +50%)
        price_range = np.linspace(current_price * 0.5, current_price * 1.5, num_points)
        
        sensitivity_data = []
        
        for price in price_range:
            # Preparar features
            df_features = preparar_features_para_modelo(product_data, price, artefatos_modelo, df)
            
            # Predição hurdle
            prob_venda = modelo_class.predict_proba(df_features)[:, 1]
            qtd_venda_log = modelo_reg.predict(df_features)
            qtd_venda_real = np.expm1(qtd_venda_log)
            qtd_venda_real[qtd_venda_real < 0] = 0
            pred_final = prob_venda * qtd_venda_real
            pred_value = pred_final[0] if len(pred_final) > 0 else 0
            
            # Calcular percentual de variação das vendas
            # Se current_sales_value for 0, usar uma abordagem diferente
            if current_sales_value > 0:
                sales_change_percent = ((pred_value - current_sales_value) / current_sales_value) * 100
            else:
                # Se não há vendas base, mostrar crescimento absoluto
                sales_change_percent = pred_value * 100 if pred_value > 0 else 0
            
            sensitivity_data.append({
                'preco': price,
                'vendas': pred_value,
                'Percentual de Vendas': sales_change_percent
            })
        
        return pd.DataFrame(sensitivity_data)
    except Exception as e:
        st.error(f"Erro ao gerar curva de sensibilidade: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def predict_sales_with_price_change(df, selected_product, price_change_percent, artefatos_modelo):
    """Prediz vendas com mudança de preço usando modelo hurdle."""
    try:
        # Filtrar dados do produto selecionado
        product_data = df[df['NM_ITEM'] == selected_product].copy()
        if product_data.empty:
            return None
        
        # Obter preço atual
        current_price = product_data['PRECO_ATUAL'].iloc[0]
        
        # Calcular venda base usando o modelo com o preço atual
        # Isso garante consistência entre o gráfico e os KPIs
        df_features_base = preparar_features_para_modelo(product_data, current_price, artefatos_modelo, df)
        modelo_class = artefatos_modelo['classificador']
        modelo_reg = artefatos_modelo['regressor']
        
        # Predição hurdle para o preço atual (situação base)
        prob_venda_base = modelo_class.predict_proba(df_features_base)[:, 1]
        qtd_venda_log_base = modelo_reg.predict(df_features_base)
        qtd_venda_real_base = np.expm1(qtd_venda_log_base)
        qtd_venda_real_base[qtd_venda_real_base < 0] = 0
        current_sales_pred = prob_venda_base * qtd_venda_real_base
        current_sales = current_sales_pred[0] if len(current_sales_pred) > 0 else 0
        current_revenue = current_price * current_sales
        
        # Calcular novo preço
        new_price = current_price * (1 + price_change_percent / 100)
        
        # Preparar features para o novo preço
        df_features = preparar_features_para_modelo(product_data, new_price, artefatos_modelo, df)
        
        # Predição hurdle para o novo preço
        prob_venda = modelo_class.predict_proba(df_features)[:, 1]
        qtd_venda_log = modelo_reg.predict(df_features)
        qtd_venda_real = np.expm1(qtd_venda_log)
        qtd_venda_real[qtd_venda_real < 0] = 0
        predicted_sales_pred = prob_venda * qtd_venda_real
        predicted_sales_value = predicted_sales_pred[0] if len(predicted_sales_pred) > 0 else 0
        predicted_revenue = new_price * predicted_sales_value
        
        # Calcular métricas
        sales_change = predicted_sales_value - current_sales
        sales_change_percent = (sales_change / current_sales * 100) if current_sales > 0 else 0
        
        revenue_change = predicted_revenue - current_revenue
        revenue_change_percent = (revenue_change / current_revenue * 100) if current_revenue > 0 else 0
        
        return {
            'preco_atual': current_price,
            'preco_novo': new_price,
            'vendas_atuais': current_sales,
            'vendas_preditas': predicted_sales_value,
            'mudanca_vendas': sales_change,
            'mudanca_vendas_percent': sales_change_percent,
            'receita_atual': current_revenue,
            'receita_predita': predicted_revenue,
            'mudanca_receita': revenue_change,
            'mudanca_receita_percent': revenue_change_percent
        }
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# --- APLICAÇÃO STREAMLIT ---

# Configuração da página
st.set_page_config(
    page_title="Análise de Elasticidade de Preço - SR Fantástico",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para sidebar
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Background da sidebar */
    .css-1d391kg, .css-1cypcdb {
        background-color: #1a1a1a !important;
    }
    
    /* Seções da sidebar */
    .sidebar-section {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* Títulos de seção */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 500;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    
    .section-icon {
        font-size: 14px;
    }
    
    /* Header de filtros */
    .filters-header {
        padding: 5px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .filters-title {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .filters-separator {
        height: 2px;
        background-color: #404040;
    }
    
    /* Card do período */
    .period-card {
        background: #2C3E50;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 2px;
        margin-bottom: 10px;
        text-align: center;
        width: 100%;
    }
    
    .period-label {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
    }
    
    .period-value {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 400;
        color: #ffffff;
    }
    
    /* Container para centralizar inputs */
    .input-container {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    /* Cards de resumo */
    .summary-card {
        background: #2C3E50;
        border-radius: 8px;
        padding: 6px;
        margin-bottom: 5px;
        text-align: center;
        border-left: 3px solid;
        width: 100%;
    }
    
    .summary-card.current {
        border-left-color: #3b82f6;
    }
    
    .summary-card.new {
        border-left-color: #f97316;
    }
    
    .summary-card.variation {
        border-left-color: #ffffff;
    }
    
    .summary-card.variation.positive {
        border-left-color: #22c55e;
    }
    
    .summary-card.variation.positive .summary-value {
        color: #22c55e !important;
    }
    
    .summary-card.variation.negative {
        border-left-color: #ef4444;
    }
    
    .summary-card.variation.negative .summary-value {
        color: #ef4444 !important;
    }
    
    .summary-label {
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 2px;
        letter-spacing: 0.5px;
    }
    
    .summary-value {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
    }

    /* --- CÓDIGO CORRETO PARA CENTRALIZAR OS BLOCOS DE KPI --- */
    [data-testid="stMetric"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    [data-testid="stMetricLabel"] {
        text-align: center;
    }


</style>

""", unsafe_allow_html=True)

# Título principal
st.title("📊 Análise de Elasticidade de Preço")

# Carrega o modelo e os dados base
artefatos_modelo = load_model(GCP_PROJECT_ID, MODEL_BUCKET, MODEL_BLOB)
df = load_data(GCP_PROJECT_ID, BQ_DATASET, BQ_BASE_TABLE)

# A aplicação só continua se o modelo e os dados foram carregados com sucesso
if artefatos_modelo is not None and not df.empty:
    
    # Logo centralizado
    try:
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            st.image("logo.png", width=200)
    except:
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            st.markdown("### TANGLE TEEZER")
    
    # Header de Filtros
    st.sidebar.markdown("""
    <div class="filters-header">
        <div class="filters-title">
            <span class="icon">🔍</span>
            Filtros
        </div>
    </div>
    <div class="filters-separator"></div>
    """, unsafe_allow_html=True)
    
    # Período de previsão
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <div class="section-title">
            Período Atual
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calcular período de 15 dias automaticamente
    today = datetime.now()
    start_date = today
    end_date = today + timedelta(days=14)  # 15 dias incluindo hoje
    
    st.markdown("---")

    st.sidebar.markdown(f"""
    <div class="period-card">
        <div class="period-label">Quinzena</div>
        <div class="period-value">{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Linha separadora
    st.sidebar.markdown("""
    <div class="filters-separator"></div>
    """, unsafe_allow_html=True)
    
    # Seleção de produto
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <div class="section-title">
            Produto
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Container para centralizar o dropdown
    st.sidebar.markdown('<div class="input-container">', unsafe_allow_html=True)
    selected_product = st.sidebar.selectbox(
        "Escolha o produto:",
        options=df['NM_ITEM'].unique(),
        index=0,
        label_visibility="collapsed"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Linha separadora
    st.sidebar.markdown("""
    <div class="filters-separator"></div>
    """, unsafe_allow_html=True)
    
    # Input de preço com design melhorado
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <div class="section-title">
            Preço
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if selected_product:
        # Obter preço atual do produto selecionado
        product_data = df[df['NM_ITEM'] == selected_product]
        if not product_data.empty:
            current_price = product_data['PRECO_ATUAL'].iloc[0]
            
            # Input de preço no padrão Streamlit
            st.sidebar.markdown("""
            <div style="margin-bottom: 2px; font-family: 'Inter', sans-serif; font-size: 12px; color: #ffffff; font-weight: 500; text-align: left;">
                Novo preço (R$) <span style="background: #555555; color: #ffffff; border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; margin-left: 6px;">?</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Container para centralizar o input
            st.sidebar.markdown('<div class="input-container">', unsafe_allow_html=True)
            new_price = st.sidebar.number_input(
                "Preço (R$)",
                min_value=0.0,
                value=float(current_price),
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="price_input"
            )
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            # Calcular variação percentual
            price_change_percent = ((new_price - current_price) / current_price) * 100
            
            
            # Card do preço atual (azul)
            st.sidebar.markdown(f"""
            <div class="summary-card current">
                <div class="summary-label">Atual</div>
                <div class="summary-value">R$ {current_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card do novo preço (laranja)
            st.sidebar.markdown(f"""
            <div class="summary-card new">
                <div class="summary-label">Novo</div>
                <div class="summary-value">R$ {new_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card de variação com cores dinâmicas
            if price_change_percent > 0:
                # Positiva - verde
                variation_icon = "↗"
                variation_text = f"{variation_icon} +{price_change_percent:.1f}%"
                variation_class = "summary-card variation positive"
            elif price_change_percent < 0:
                # Negativa - vermelho
                variation_icon = "↘"
                variation_text = f"{variation_icon} {price_change_percent:.1f}%"
                variation_class = "summary-card variation negative"
            else:
                # Zero - branco
                variation_icon = "→"
                variation_text = f"{variation_icon} +0.0%"
                variation_class = "summary-card variation"
            
            st.sidebar.markdown(f"""
            <div class="{variation_class}">
                <div class="summary-label">Variação</div>
                <div class="summary-value">{variation_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Converter para percentual para usar nas funções existentes
            price_change = price_change_percent
    
    # Linha separadora
    st.sidebar.markdown("---")
    
    # Botão de Logout no final da sidebar
    if st.sidebar.button("Logout"):
        # Limpa todo o estado da sessão para deslogar o usuário
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun() # Reinicia a aplicação para voltar à tela de login
    
    # Calcular previsão com mudança de preço
    prediction = predict_sales_with_price_change(df, selected_product, price_change, artefatos_modelo)
    
    if prediction:
        # Gerar dados da curva de sensibilidade para o gráfico principal
        sensitivity_curve_data = generate_price_sensitivity_curve(df, selected_product, artefatos_modelo, 50)
        
        if sensitivity_curve_data is not None:
            # Gráfico principal: Preço (X) vs Percentual de Vendas (Y)
            fig_main = px.line(
                sensitivity_curve_data,
                x='preco',
                y='Percentual de Vendas',
                title=f"Crescimento X Preço - {selected_product}",
                markers=True,
                line_shape='spline'  # Curva suave
            )
            
            # Destacar ponto atual (preço atual) - sempre em 0% de crescimento
            current_price = prediction['preco_atual']
            
            fig_main.add_trace(go.Scatter(
                x=[current_price],
                y=[0],  # Situação atual sempre tem 0% de crescimento
                mode='markers',
                marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='white')),
                name='Situação Atual',
                hovertemplate='Preço=R$ %{x:.2f}<br>Crescimento=%{y:.2f}%<extra></extra>'
            ))
            
            # Destacar ponto com novo preço se houver mudança
            if price_change != 0:
                new_price = prediction['preco_novo']
                new_sales_change = prediction['mudanca_vendas_percent']
                
                # Encontrar o ponto correspondente na curva ou usar o valor calculado
                fig_main.add_trace(go.Scatter(
                    x=[new_price],
                    y=[new_sales_change],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='star', line=dict(width=2, color='white')),
                    name='Cenário Simulado',
                    hovertemplate='Preço=R$ %{x:.2f}<br>Crescimento=%{y:.2f}%<extra></extra>'
                ))
            
            # Configurar formatação do tooltip para a linha principal
            fig_main.update_traces(
                hovertemplate='Preço=R$ %{x:.2f}<br>Crescimento=%{y:.2f}%<extra></extra>',
                line=dict(width=3)
            )
            
            fig_main.update_layout(
                xaxis_title="Preço (R$)",
                yaxis_title="Crescimento Percentual (%)",
                showlegend=True,
                height=500,
                hovermode='x unified',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_main, use_container_width=True)
        
        st.markdown("---")
        
        # KPIs Principais
        st.header("📈 Indicadores Principais")
        
        # Organizar em 2 colunas centralizadas
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric(
                label="💰 Preço Atual",
                value=f"R$ {prediction['preco_atual']:.2f}",
                delta=f"R$ {prediction['preco_novo'] - prediction['preco_atual']:.2f}" if price_change != 0 else None
            )
        
        with col2:
            # Mostrar 0% se não há mudança de preço, senão mostrar o crescimento
            if price_change == 0:
                crescimento_value = "0%"
                delta = None
            else:
                crescimento_value = f"{prediction['mudanca_vendas_percent']:.1f}%"
                # Usar delta para colorir: positivo = verde, negativo = vermelho
                delta = f"{prediction['mudanca_vendas_percent']:.1f}%"
            
            st.metric(
                label="🎯 Crescimento",
                value=crescimento_value,
                delta=delta
            )
        
        
        # Rodapé
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>Previsão de Vendas com Machine Learning - SR Fantástico | Desenvolvido com Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
    )

else:
    st.error("🔴 Falha ao carregar modelo ou dados do BigQuery. Verifique as configurações e os logs.")