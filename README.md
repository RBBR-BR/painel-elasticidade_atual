# 📊 Painel de Elasticidade de Preço - SR Fantástico

Um painel interativo desenvolvido em Streamlit para análise de elasticidade de preço e previsão de vendas utilizando Machine Learning. O sistema permite simular mudanças de preço e visualizar o impacto previsto nas vendas através de modelos preditivos.

## 🎯 Funcionalidades

### Autenticação e Segurança
- **Sistema de login seguro** com autenticação via BigQuery
- **Redefinição obrigatória de senha** no primeiro acesso e após 15 dias
- **Validação de complexidade de senha** com requisitos de segurança
- **Hash de senhas** utilizando bcrypt para proteção dos dados

### Análise de Elasticidade
- **Simulação de mudanças de preço** em tempo real
- **Previsão de vendas** utilizando modelo de Machine Learning (XGBoost)
- **Visualização interativa** de curvas de sensibilidade de preço
- **Análise de impacto** em receita e volume de vendas
- **Indicadores principais (KPIs)** com métricas de crescimento

### Interface
- **Design moderno e responsivo** com sidebar personalizada
- **Gráficos interativos** utilizando Plotly
- **Filtros dinâmicos** por produto e período
- **Visualização de cenários** comparando situação atual vs. simulada

## 🏗️ Arquitetura

### Estrutura do Projeto

```
painel-elasticidade-streamlit-main/
├── auth.py                      # Módulo de autenticação e gerenciamento de usuários
├── login.py                     # Página principal de login
├── setup_initial_user.py        # Script de configuração inicial de usuário
├── logo.png                     # Logo da aplicação
├── pages/
│   ├── 1_Painel.py             # Painel principal de análise
│   └── 2_Reset_Password.py     # Página de redefinição de senha
└── requirements.txt            # Dependências do projeto
```

### Componentes Principais

#### `auth.py`
Módulo central de autenticação que gerencia:
- Conexão com BigQuery para armazenamento de usuários
- Hash e verificação de senhas com bcrypt
- Validação de login e controle de reset obrigatório
- Atualização de senhas e datas de reset

#### `login.py`
Interface de login que:
- Gerencia estado de autenticação da sessão
- Redireciona usuários conforme status (autenticado, reset obrigatório)
- Valida credenciais e controla acesso ao painel

#### `pages/1_Painel.py`
Painel principal que oferece:
- Carregamento de modelo ML do Google Cloud Storage
- Consulta de dados do BigQuery
- Engenharia de features (datas, feriados, sazonalidade)
- Predição de vendas com mudanças de preço
- Visualizações interativas de elasticidade

#### `pages/2_Reset_Password.py`
Página de redefinição de senha com:
- Validação em tempo real de requisitos de senha
- Checklist visual de complexidade
- Atualização segura no BigQuery

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Conta no Google Cloud Platform (GCP)
- Projeto BigQuery configurado
- Bucket no Google Cloud Storage para armazenar o modelo
- Credenciais de Service Account do GCP

### Passo a Passo

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd painel-elasticidade-streamlit-main
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as credenciais do GCP**

Crie o diretório `.streamlit` e o arquivo `secrets.toml`:

```bash
mkdir .streamlit
```

No arquivo `.streamlit/secrets.toml`, adicione:

```toml
[gcp_service_account]
type = "service_account"
project_id = "seu-projeto-id"
private_key_id = "sua-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "seu-service-account@projeto.iam.gserviceaccount.com"
client_id = "seu-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

5. **Configure o BigQuery**

Certifique-se de que a tabela `PAINEL_USERS` existe no dataset `RBBR_DATA_SCIENCE` com a seguinte estrutura:

```sql
CREATE TABLE `projeto.dataset.PAINEL_USERS` (
  USERNAME STRING,
  PASSWORD_HASH STRING,
  LAST_RESET_DATE TIMESTAMP,
  FIRST_LOGIN BOOL
);
```

6. **Configure o modelo no Cloud Storage**

Certifique-se de que o modelo está disponível no bucket `rbbr-artifacts` no caminho `models/elasticity/modelo_final_elasticidade.joblib`.

7. **Crie o usuário inicial**

Execute o script de configuração:

```bash
python setup_initial_user.py
```

Isso criará o usuário padrão:
- **Usuário**: `Dados`
- **Senha**: `changeme` (será necessário alterar no primeiro login)

## 📦 Dependências

O projeto utiliza as seguintes bibliotecas principais:

- **streamlit**: Framework web para criação da interface
- **pandas**: Manipulação e análise de dados
- **numpy**: Operações numéricas
- **scikit-learn**: Ferramentas de machine learning
- **xgboost**: Modelo de machine learning para previsão
- **plotly**: Gráficos interativos
- **google-cloud-bigquery**: Integração com BigQuery
- **google-cloud-storage**: Integração com Cloud Storage
- **bcrypt**: Hash seguro de senhas
- **joblib**: Carregamento de modelos ML

Para ver a lista completa, consulte `requirements.txt`.

## 🔧 Configuração

### Variáveis de Ambiente e Configurações

As principais configurações estão definidas nos arquivos:

**`auth.py`**:
- `GCP_PROJECT_ID`: ID do projeto GCP
- `BQ_DATASET`: Nome do dataset no BigQuery
- `BQ_USERS_TABLE`: Nome da tabela de usuários

**`pages/1_Painel.py`**:
- `GCP_PROJECT_ID`: ID do projeto GCP
- `MODEL_BUCKET`: Nome do bucket no Cloud Storage
- `MODEL_BLOB`: Caminho do modelo no bucket
- `BQ_DATASET`: Nome do dataset no BigQuery
- `BQ_BASE_TABLE`: Nome da tabela de dados de elasticidade

### Estrutura de Dados Esperada

A tabela `DM_ELASTICITY` no BigQuery deve conter as seguintes colunas:

- `NM_ITEM`: Nome do produto
- `PRECO_ATUAL`: Preço atual do produto
- `PRECO_SIMULADO`: Preço simulado
- `VARIACAO_PERCENTUAL`: Variação percentual de preço
- `VENDAS_PREVISTAS`: Vendas previstas
- `UPDATED_DT`: Data de atualização

## 🎮 Uso

### Iniciar a Aplicação

```bash
streamlit run login.py
```

A aplicação estará disponível em `http://localhost:8501`

### Fluxo de Uso

1. **Login**: Acesse a aplicação e faça login com suas credenciais
2. **Primeiro Acesso**: Se for o primeiro login, será obrigatório redefinir a senha
3. **Análise**: No painel, selecione um produto e ajuste o preço desejado
4. **Visualização**: Observe os gráficos e métricas de impacto nas vendas
5. **Simulação**: Compare diferentes cenários de preço em tempo real

### Recursos do Painel

- **Seleção de Produto**: Escolha o produto a ser analisado
- **Ajuste de Preço**: Defina o novo preço desejado
- **Gráfico de Elasticidade**: Visualize a curva de sensibilidade de preço
- **KPIs**: Acompanhe métricas de preço, receita e crescimento
- **Período**: Visualize o período de análise (quinzena atual)

## 🔒 Segurança

### Medidas Implementadas

- **Hash de Senhas**: Utilização de bcrypt com salt automático
- **Reset Obrigatório**: Senha deve ser alterada no primeiro acesso e a cada 15 dias
- **Validação de Complexidade**: Senhas devem atender critérios rigorosos:
  - Mínimo de 8 caracteres
  - Pelo menos uma letra minúscula
  - Pelo menos uma letra maiúscula
  - Pelo menos um número
  - Pelo menos um caractere especial
- **Autenticação de Sessão**: Controle de acesso baseado em estado de sessão
- **Credenciais Seguras**: Uso de Service Account do GCP com permissões mínimas necessárias

## 🛠️ Desenvolvimento

### Estrutura de Código

O projeto segue uma arquitetura modular:

- **Separação de responsabilidades**: Autenticação, interface e lógica de negócio em módulos distintos
- **Cache de recursos**: Utilização de `@st.cache_resource` e `@st.cache_data` para otimização
- **Tratamento de erros**: Validações e mensagens de erro apropriadas
- **Código limpo**: Funções bem documentadas e organizadas


## 📝 Licença

Este projeto é proprietário e desenvolvido para SR Fantástico.

## 👥 Suporte

Para questões, problemas ou sugestões, entre em contato com a equipe de desenvolvimento.

---

**Desenvolvido com ❤️ usando Streamlit e Machine Learning**

