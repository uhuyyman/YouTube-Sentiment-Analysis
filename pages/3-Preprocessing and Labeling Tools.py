import streamlit as st
import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re
import ast
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and tokenizer for sentiment analysis
model = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student").to(device)
tokenizer = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")

# Function to clean text
def clean_comment(comment, replace_word):
    if pd.isna(comment):
        return ' '
    comment = str(comment)
    comment = re.sub(r'\n+', ' ', comment)
    comment = re.sub(r'http\S+|www\S+', ' ', comment, flags=re.MULTILINE)
    comment = re.sub(r'[^\w\s,]', ' ', comment)
    comment = re.sub(r'[^A-Za-z\s]', ' ', comment)
    comment = comment.lower()
    
    # Apply user-defined replace word dictionary
    for word, replace in replace_word.items():
        comment = re.sub(r'\b' + re.escape(word) + r'\b', replace, comment, flags=re.IGNORECASE)
    
    comment = re.sub(r'\b[a-zA-Z]\b', ' ', comment)
    comment = re.sub(r'(.)\1{2,}', r'\1\1', comment)
    comment = re.sub(r'\s+', ' ', comment).strip()
    
    return comment

# Function to tokenize text
def tokenize_comment(comment):
    return word_tokenize(comment)

# Function to remove stopwords
def stopwords_comment(tokens, additional_stopwords):
    cleaned_token = []
    all_stopwords = set(stopwords.words('english')).union(set(stopwords.words('indonesian'))).union(set(additional_stopwords))
    for token in tokens:
        if token not in all_stopwords:
            cleaned_token.append(token)
    return cleaned_token

# Stemmer setup
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# Function to stem text
def stemming_comment(tokens):
    return [stemmer.stem(token) for token in tokens]

# Function to clean and convert tokenized/stemmed text to a single string
def clean_and_convert(text):
    if isinstance(text, list):
        return ' '.join(text)
    elif isinstance(text, str):
        try:
            word_list = ast.literal_eval(text)
            if isinstance(word_list, list):
                return ' '.join(word_list)
            else:
                cleaned_text = re.sub(r"[\'\[\],]", "", text)
                return cleaned_text.strip()
        except (ValueError, SyntaxError):
            cleaned_text = re.sub(r"[\'\[\],]", "", text)
            return cleaned_text.strip()
    else:
        return ''

# Function to perform sentiment analysis using CUDA
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    labels = ["negatif", "netral", "positif"]
    highest_score_label = labels[probabilities.argmax()]
    return highest_score_label

if 'additional_stopwords' not in st.session_state:
    st.session_state['additional_stopwords'] = []

if 'replace_word' not in st.session_state:
    st.session_state['replace_word'] = {}

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

st.title("Preprocessing dan Labeling Menggunakan Custom Stopwords dan Replace Words")
st.markdown('''
    ### Petunjuk Penggunaan Tools Ini:  
        1. Kustomisasi Kata Pengganti (Opsional): Tambahkan kata-kata tertentu untuk diganti melalui sidebar.  
        2. Kustomisasi Stopwords (Opsional): Tambahkan stopwords Anda sendiri melalui sidebar.  
        3. Unggah File CSV: Pastikan file Anda memiliki kolom `comment`, lalu unggah.  
        4. Proses Data: Klik tombol 'Process Data' untuk memulai analisis.  
        5. Tunggu Hingga Selesai: Biarkan alat memproses dan menganalisis data Anda.  
        6. Unduh Hasil: Simpan data yang telah diproses dengan mengunduh hasilnya.  
''')

st.warning("Harap Tidak Mengganti tab lain di sidebar hingga analisis selesai!")

st.sidebar.subheader("Tambahkan Custom Replace Words")
replace_key = st.sidebar.text_input("Kata yang Akan Diganti (lowercase)")
replace_value = st.sidebar.text_input("Kata Pengganti (lowercase)")

if st.sidebar.button("Tambahkan Kata Pengganti"):
    if replace_key and replace_value:
        st.session_state['replace_word'][replace_key] = replace_value
    else:
        st.sidebar.warning("Kedua 'Kata yang Akan Diganti' dan 'Kata Pengganti' harus diisi!")

st.sidebar.subheader("Kata Pengganti Sebelum")
st.sidebar.write(st.session_state['replace_word'])

st.sidebar.subheader("Tambahkan Custom Stopwords")
additional_stopwords_input = st.sidebar.text_area(
    "Masukan stopwords tambahan dipisakan dengan koma", 
    value=", ".join(st.session_state['additional_stopwords'])
)

st.session_state['additional_stopwords'] = [word.strip() for word in additional_stopwords_input.split(',')] if additional_stopwords_input else []

uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'comment'", type=["csv"])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

if st.button("Process Data") and st.session_state['uploaded_file']:
    comment_df = pd.read_csv(st.session_state['uploaded_file'])

    st.subheader("Uploaded Data")
    st.write(comment_df)

    if 'comment' in comment_df.columns:
        st.subheader("Processing Data...")

        total_comments = len(comment_df)

        comment_df['cleaned_comment'] = comment_df['comment'].apply(lambda x: clean_comment(x, st.session_state['replace_word']))
        comment_df['tokenized_comment'] = comment_df['cleaned_comment'].apply(tokenize_comment)
        comment_df['stopwords_comment'] = comment_df['tokenized_comment'].apply(lambda tokens: stopwords_comment(tokens, st.session_state['additional_stopwords']))
        comment_df['stemmed_comment'] = comment_df['stopwords_comment'].apply(stemming_comment)
        comment_df['cleaned_stemmed'] = comment_df['stemmed_comment'].apply(clean_and_convert)
        comment_df = comment_df.drop_duplicates(subset='cleaned_stemmed')
        comment_df = comment_df.dropna(subset='cleaned_stemmed')

        st.subheader("Melakukan Analisis Sentimen...")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        sentiments = []
        for idx, comment in enumerate(tqdm(comment_df['cleaned_stemmed']), start=1):
            sentiment = sentiment_analysis(comment)
            sentiments.append(sentiment)
            progress_percentage = int((idx / total_comments) * 100)
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Processing: {idx} of {total_comments} ({progress_percentage}%)")

        comment_df['sentiment_prediction'] = sentiments

        st.subheader("Data yang Sudah Diproses dengan Sentimen")
        st.write(comment_df)

        st.subheader("Overview Dataset")
        col1, col2 = st.columns(2)
        with col1:
            sentiment_count = comment_df['sentiment_prediction'].value_counts()
            st.bar_chart(sentiment_count, color='#d2a4a7')
        with col2:
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='gist_heat_r').generate_from_text(' '.join(comment_df['cleaned_stemmed']))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        st.subheader("Download Data yang Diproses")
        csv = comment_df.to_csv(index=False)
        st.download_button(label="Download file CSV", data=csv, file_name='processed_sentiment.csv', mime='text/csv')

        progress_bar.progress(100)
        progress_text.text("Processing completed!")
    else:
        st.error("Tidak terdapat kolom 'comment' pada file yang diupload.")