import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to load the unbalanced dataset
@st.cache_data
def load_data_unbalanced():
    data = pd.read_csv('./data/youtube-comment-cleaned-sentiment.csv')
    return data

# Function to load balanced dataset
@st.cache_data
def load_data_balanced():
    data = pd.read_csv('./data/youtube-comment-cleaned-sentiment-balanced.csv')
    return data

# Main function
@st.cache_data
def main():
    data_unbalanced = load_data_unbalanced()
    data_balanced = load_data_balanced()

    st.subheader("Overview Dataset Unbalanced")
    st.write("Sekilas tentang dataset unbalanced yang digunakan oleh model unbalanced:")
    st.write(data_unbalanced)

    st.subheader("Overview Dataset Balanced")
    st.write("Sekilas tentang dataset balanced yang digunakan oleh model balanced:")
    st.write(data_balanced)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Sentimen Unbalanced")
        sentiment_count = data_unbalanced['sentiment_prediction'].value_counts()
        st.bar_chart(sentiment_count, color='#d2a4a7')

        csv_file = './data/youtube-comment-cleaned-sentiment.csv'
        df = pd.read_csv(csv_file)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='gist_heat_r').generate_from_text(' '.join(df['cleaned_stemmed']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    with col2:
        st.subheader("Distribusi Sentimen Balanced")
        sentiment_count = data_balanced['sentiment_prediction'].value_counts()
        st.bar_chart(sentiment_count, color='#d2a4a7')

        csv_file = './data/youtube-comment-cleaned-sentiment-balanced.csv'
        df = pd.read_csv(csv_file)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='gist_heat_r').generate_from_text(' '.join(df['cleaned_stemmed']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    st.subheader("Insights Berdasarkan Dataset:")
    st.markdown('''
        1. Dalam kasus ini, jika dataset tidak seimbang, alat analisis sentimen akan menerima lebih banyak variasi kata yang dapat diproses oleh model, namun model akan cenderung bias terhadap kategori sentimen yang memiliki jumlah kalimat terbanyak.  
        2. Namun, dalam kasus ini, jika dataset seimbang, alat analisis sentimen akan menerima variasi kata yang lebih sedikit untuk diproses oleh model, tetapi model akan lebih sedikit bias terhadap salah satu kategori sentimen karena jumlahnya sudah diseimbangkan.
    ''')

if __name__ == "__main__":
    main()