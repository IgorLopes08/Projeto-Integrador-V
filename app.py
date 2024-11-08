import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_carousel import carousel

# Configurando a página para sempre usar o wide mode
#st.set_page_config(layout="wide")

# Alterando a cor de fundo do site, a cor das letras do cabeçalho e o tipo de fonte
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background-color: #8181FA;
            color: #8181FA;
            font-family: 'Arial', sans-serif;
        }
    </style>
    <div class="main-container">
    </style>
""", unsafe_allow_html=True)

# Título do aplicativo
st.write(f"<h1 style='color:white;'>Análise de Reviews da Amazon Alexa</h1>", unsafe_allow_html=True)

# Carregar a base de dados
base = pd.read_csv('data//amazon_alexa.tsv', sep='\t')  # Usando sep='\t' para arquivos TSV

# Título info gerais do banco de dados
st.write(f"<h2 style='color:white;'>Informações gerais sobre o banco de dados</h2>", unsafe_allow_html=True)

#Teste##################################################################################
# Dados textuais para o carrossel

texts = [
    f"1. ENTENDENDO OS DADOS: Nossa base de dados agrupa reviews extraídas de dentro do site da Amazon que avaliam uma versão de um dos seus principais produtos, a Alexa.",
    f"2. Os dados estão divididos em 3150 linhas e 5 colunas onde cada coluna armazena respectivamente um ID da avaliação, a nota que foi dada para o produto(rating), a loja em que o produto foi vendido (variation), a avaliação textual que cada cliente fez do produto comprado(verified_reviews) e por fim um valor booleano para feedback positivo ou negativo(feedback). ",
    f"3. Aqui, podemos visualizar melhor como de fato os dados estão distribuídos",
    f"4. Por fim, alguns dados gerais sobre os números obtidos em uma primeira análise dentro do banco de dados."
        ]

# Índice para rastrear a posição atual no carrossel
current_index = st.session_state.get('current_index', 0)
texto_atual= texts[current_index]

# Botões para navegação
col1, col2, col3 = st.columns([1,4,3])
with col1:
    # Botão de navegação para a esquerda
    if st.button("◀️", key="anterior") and current_index > 0:
        st.session_state['current_index'] = current_index - 1
        
with col2:
    # Exibir o texto atual
    titulo = texts[current_index]
    cor = "white"
    st.markdown(f"<h5 style='color:{cor}; text-align: center;'>{texts[current_index]}</h5>", unsafe_allow_html=True)
    if current_index == 2:
        st.write(base.head())
    elif current_index == 3:
        st.write(base.describe())

with col3:
    # Botão de navegação para a direita
    if st.button("▶️", key="proximo") and current_index < len(texts) - 1:
        st.session_state['current_index'] = current_index + 1

#Teste #############################################################################

# Título do aplicativo
st.title('Analisando os gráficos e histogramas do banco de dados')

st.write(f"<h3 style='color:white;'>Histogramas de rating e feedback</h3>", unsafe_allow_html=True)
# Visualizar histogramas para os atributos numéricos (rating e feedback)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(base['rating'], kde=True, ax=axs[0])
axs[0].set_title('Histograma do Rating')
sns.histplot(base['feedback'], kde=True, ax=axs[1])
axs[1].set_title('Histograma do Feedback')
st.pyplot(fig)

st.write(f"<h3 style='color:white;'>RATING:</h3>", unsafe_allow_html=True)
st.write(f"<h5 style='color:white;'>A média das avaliações é 4.46, indicando que, em geral, as avaliações são bastante positivas. O desvio padrão de 1.07 sugere que há alguma variação nas avaliações, mas a maioria está concentrada perto da nota máxima de 5. A mediana e o percentil de 75 mostram que muitas avaliações são 5, o que reforça a tendência de avaliações elevadas.</h5>", unsafe_allow_html=True)

st.write(f"<h3 style='color:white;'>FEEDBACK:</h3>", unsafe_allow_html=True)
st.write(f"<h5 style='color:white;'>A média do feedback é 0.92, e o valor máximo é 1.00, o que indica que o feedback é frequentemente positivo (ou seja, 1). O desvio padrão é baixo, o que sugere que há pouca variação no feedback, com a maioria dos valores sendo 1.</h5>", unsafe_allow_html=True)     

st.write(f"<h5 style='color:white;'>Esses dados fornecem uma visão geral de que a maioria das avaliações e feedbacks são altamente positivos, com a maioria dos registros recebendo as melhores classificações e feedbacks possíveis.</h5>", unsafe_allow_html=True)     

st.write("A maioria das notas (rating) foram boas, o que explica os valores 1 em feedback serem maiores também (levando em consideração que 1 representa um bom feedback e 0 um feedback ruim) e vice-versa.")

# Verificar o tamanho dos textos das revisões
base['review_length'] = base['verified_reviews'].fillna('').apply(len)
#st.write("### Tamanho dos textos das revisões:")
#st.write(base['review_length'].describe())

st.write("O conjunto de dados possui 3150 revisões, com tamanho médio de 132 caracteres e desvio padrão de 182, o que indica uma grande variação nos tamanhos. Enquanto 25% das revisões têm até 30 caracteres, a mediana, ou ponto médio, é de 74 caracteres, mostrando um tamanho típico das revisões, pois não é afetada por textos muito longos. Já o 75º percentil indica que 25% das revisões têm 165 caracteres ou mais. O tamanho máximo registrado é de 2851 caracteres, sugerindo que algumas revisões são muito detalhadas.")

# DataFrames de reviews positivas e negativas
df_positivos = base[base['rating'] > 3]
df_negativos = base[base['rating'] <= 3]

# Contagem das reviews positivas e negativas
quantidade_positivas = df_positivos.shape[0]
quantidade_negativas = df_negativos.shape[0]


# Dados para o gráfico de pizza
labels = ['Positivas', 'Negativas']
sizes = [quantidade_positivas, quantidade_negativas]
colors = ['#4CAF50', '#F44336']  # Verde para positivas, vermelho para negativas

# Criar o gráfico de pizza
fig, ax = plt.subplots(figsize=(10,4))
colors = ['purple','gray']
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=180, colors=colors)
ax.axis('equal')  # Garantir que o gráfico seja um círculo
ax.set_title('Gráfico de pizza para comparação de reviews')

st.write(f"<h3 style='color:white;'>Comparando reviews positivas e negativas</h3>", unsafe_allow_html=True)

# Exibir o gráfico no Streamlit
st.pyplot(fig)

# Exibir as quantidades no Streamlit
st.write(f"<h5 style='color:white;'>Esses dados fornecem uma visão geral de que a maioria das avaliações e feedbacks são altamente positivos, com a maioria dos registros recebendo as melhores classificações e feedbacks possíveis, tendo uma média de 120 caracteres em reviews positivas e 212 caracteres em reviews negativas.</h5>", unsafe_allow_html=True)     


#st.write("### Média de caracteres em reviews positivas:", df_positivos['review_length'].mean())
#st.write("### Média de caracteres em reviews negativas:", df_negativos['review_length'].mean())

# Gráfico de distribuição das avaliações
st.write(f"<h3 style='color:white;'>Visualizando a distribuição  das avaliações</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 4))
sns.countplot(x='rating', data=base, ax=ax)
ax.set_title('Distribuição das Avaliações')
st.pyplot(fig)
st.write(f"<h5 style='color:white;'>Já no gráfico acima podemos visualizar mais claramente a nota geral dada pelos consumidores entre 1 e 5, com 1 sendo a pior nota e 5 sendo a melhor nota, informação que reforça ainda mais a grande quantidade de avaliações positivas disnponiveis dentro do banco de dados</h5>", unsafe_allow_html=True)     

st.write(f"<h1 style='color:white;'>Aplicando algoritmo de análise de sentimentos</h1>", unsafe_allow_html=True)
# Inicializando o analisador VADER
analyzer = SentimentIntensityAnalyzer()

# Função para calcular o sentimento usando VADER
def get_vader_sentiment(text):
    if isinstance(text, str):
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score['compound']  # Usando a pontuação composta
    else:
        return None

# Aplicando o novo método de sentimento na coluna 'verified_reviews'
base['sentiment'] = base['verified_reviews'].apply(get_vader_sentiment)


# Função para classificar os sentimentos
def categorize_sentiment(polarity):
    if polarity >= 0.8:
        return 'Muito Bom'
    elif 0.5 <= polarity < 0.8:
        return 'Bom'
    elif 0.2 <= polarity < 0.5:
        return 'Médio'
    elif -0.2 < polarity < 0.2:
        return 'Neutro'
    elif -0.5 <= polarity <= -0.2:
        return 'Ruim'
    else:
        return 'Péssimo'

# Aplicando a classificação
base['sentiment_category'] = base['sentiment'].apply(categorize_sentiment)

# Contagem das categorias
category_counts = base['sentiment_category'].value_counts()

# Exibindo o título usando HTML estilizado
st.write("<h3 style='color:white;'>Classificação dos Sentimentos</h3>", unsafe_allow_html=True)

# Criando o histograma das categorias de sentimento
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(category_counts.index, category_counts.values, color='skyblue', edgecolor='black')
ax.set_title('Classificação dos Sentimentos')
ax.set_xlabel('Categorias de Sentimento')
ax.set_ylabel('Frequência')
plt.xticks(rotation=45)

# Exibindo o gráfico no Streamlit
st.image('data\\image.png', use_column_width=True)


st.write(f"<h5 style='color:white;'>Aplicando o vader sentiment dentro do banco de dados e categorizando a polaridade das classes desejadas, tivemos um resultado esperado de que a maioria das reviews seriam positivas, com as classes muito bom, bom e médio ultrapassando muito as avaliações neutras e negativas</h5>", unsafe_allow_html=True)

st.write("<h3 style='color:white;'>Entendendo a polarização dos dados</h3>", unsafe_allow_html=True)
# Criando um histograma para a distribuição de sentimento
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(base['sentiment'], bins=20, color='skyblue', edgecolor='black')
ax.set_title('Distribuição dos Sentimentos')
ax.set_xlabel('Sentimento')
ax.set_ylabel('Frequência')
st.pyplot(fig)

st.write(f"<h5 style='color:white;'>Nesse gráfico podemos ver mais claramente como as avaliações se comportaram com a classificação de polaridade aplicada, tendo as reviews separadas de 0 a 1, com 1 sendo uma ótima review e 0 sendo um péssima review</h5>", unsafe_allow_html=True)
      
# Nuvens de palavras para todas as reviews, negativas e positivas
def gerar_nuvem_palavras(texto, titulo):
    texto = texto.dropna().astype(str)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texto))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.write("### " + titulo)
    st.pyplot(fig)

st.write("<h3 style='color:white;'>Nuvens de palavras</h3>", unsafe_allow_html=True)

selecionar_nuvem = st.selectbox("Selecione o tipo de nuvem de palavra que deseja visualizar", ['Todas as reviews', 'Reviews negativas', 'Reviews positivas'])

if selecionar_nuvem == 'Todas as reviews':
    gerar_nuvem_palavras(base['verified_reviews'], 'Todas as Reviews')
elif selecionar_nuvem == 'Reviews negativas':
    gerar_nuvem_palavras(df_negativos['verified_reviews'], 'Reviews Negativas')
elif selecionar_nuvem == 'Reviews positivas':
    gerar_nuvem_palavras(df_positivos['verified_reviews'], 'Reviews Positivas')
    
st.write(f"<h5 style='color:white;'>Acima temos as nuvens de palavras de reviews positivos, negativos e também uma nuvem que colapsa as palavras encontradas tanto nos reviews positivos quanto nos negativos, dessa forma fica mais claro os assuntos e termos utilziados em cada tipo de avaliação</h5>", unsafe_allow_html=True)
