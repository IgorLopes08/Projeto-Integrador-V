import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title='Projeto integrador V')


# Título do aplicativo
st.title('Análise de Reviews da Amazon Alexa')
with st.container():
    # Carregar a base de dados
    base = pd.read_csv('data//amazon_alexa.tsv', sep='\t')  # Usando sep='\t' para arquivos TSV

    # Exibir informações sobre a base de dados
    # Título info gerais do banco de dados
    st.subheader('Análise geral de informações sobre o banco de dados')
    st.write("#### Shape da base de dados (Registros, Atributos):", base.shape)
    # Gerar a estatística para os atributos numéricos
    st.write("#### Estatísticas dos atributos numéricos:")
    st.write(base.describe())
    st.write('Rating: A média das avaliações é 4.46, indicando que, em geral, as avaliações são bastante positivas. O desvio padrão de 1.07 sugere que há alguma variação nas avaliações, mas a maioria está concentrada perto da nota máxima de 5. A mediana e o percentil de 75 mostram que muitas avaliações são 5, o que reforça a tendência de avaliações elevadas.')

    st.write('feedback: A média do feedback é 0.92, e o valor máximo é 1.00, o que indica que o feedback é frequentemente positivo (ou seja, 1). O desvio padrão é baixo, o que sugere que há pouca variação no feedback, com a maioria dos valores sendo 1.')        

    st.write('Esses dados fornecem uma visão geral de que a maioria das avaliações e feedbacks são altamente positivos, com a maioria dos registros recebendo as melhores classificações e feedbacks possíveis.')

    st.write("####  Primeiros registros da base de dados:")
    st.write(base.head())

# Título do aplicativo
st.title('Analisando os gráficos e histogramas do banco de dados')

# Visualizar histogramas para os atributos numéricos (rating e feedback)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(base['rating'], kde=True, ax=axs[0])
axs[0].set_title('Histograma do Rating')
sns.histplot(base['feedback'], kde=True, ax=axs[1])
axs[1].set_title('Histograma do Feedback')
st.pyplot(fig)

st.write("A maioria das notas (rating) foram boas, o que explica os valores 1 em feedback serem maiores também (levando em consideração que 1 representa um bom feedback e 0 um feedback ruim) e vice-versa.")

# Verificar o tamanho dos textos das revisões
base['review_length'] = base['verified_reviews'].fillna('').apply(len)
st.write("### Tamanho dos textos das revisões:")
st.write(base['review_length'].describe())

st.write("O conjunto de dados possui 3150 revisões, com tamanho médio de 132 caracteres e desvio padrão de 182, o que indica uma grande variação nos tamanhos. Enquanto 25% das revisões têm até 30 caracteres, a mediana, ou ponto médio, é de 74 caracteres, mostrando um tamanho típico das revisões, pois não é afetada por textos muito longos. Já o 75º percentil indica que 25% das revisões têm 165 caracteres ou mais. O tamanho máximo registrado é de 2851 caracteres, sugerindo que algumas revisões são muito detalhadas.")

# DataFrames de reviews positivas e negativas
df_positivos = base[base['rating'] > 3]
df_negativos = base[base['rating'] <= 3]
st.write("### Quantidade de reviews positivas:", df_positivos.shape[0])
st.write("### Quantidade de reviews negativas:", df_negativos.shape[0])
st.write("### Média de caracteres em reviews positivas:", df_positivos['review_length'].mean())
st.write("### Média de caracteres em reviews negativas:", df_negativos['review_length'].mean())

# Gráfico de distribuição das avaliações
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='rating', data=base, ax=ax)
ax.set_title('Distribuição das Avaliações')
st.pyplot(fig)

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

# Criando o histograma das categorias de sentimento
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(category_counts.index, category_counts.values, color='skyblue', edgecolor='black')
ax.set_title('Classificação dos Sentimentos')
ax.set_xlabel('Categorias de Sentimento')
ax.set_ylabel('Frequência')
ax.set_xticks(range(len(category_counts.index)))
ax.set_xticklabels(category_counts.index, rotation=45)
st.pyplot(fig)

# Criando um histograma para a distribuição de sentimento
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(base['sentiment'], bins=20, color='skyblue', edgecolor='black')
ax.set_title('Distribuição dos Sentimentos')
ax.set_xlabel('Sentimento')
ax.set_ylabel('Frequência')
st.pyplot(fig)
     
# Nuvens de palavras para todas as reviews, negativas e positivas
def gerar_nuvem_palavras(texto, titulo):
    texto = texto.dropna().astype(str)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texto))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.write("### " + titulo)
    st.pyplot(fig)

selecionar_nuvem = st.selectbox("Selecione o tipo de nuvem de palavra que deseja visualizar", ['Todas as reviews', 'Reviews negativas', 'Reviews positivas'])

if selecionar_nuvem == 'Todas as reviews':
    gerar_nuvem_palavras(base['verified_reviews'], 'Nuvem de Palavras para Todas as Reviews')
elif selecionar_nuvem == 'Reviews negativas':
    gerar_nuvem_palavras(df_negativos['verified_reviews'], 'Nuvem de Palavras para Reviews Negativas')
elif selecionar_nuvem == 'Reviews positivas':
    gerar_nuvem_palavras(df_positivos['verified_reviews'], 'Nuvem de Palavras para Reviews Positivas')
