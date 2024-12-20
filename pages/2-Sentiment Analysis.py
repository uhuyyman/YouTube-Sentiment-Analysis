import streamlit as st
import pickle
import pandas as pd

# Function to load the model
@st.cache_data
def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict sentiment
def predict_sentiment(model, input_text):
    prediction = model.predict([input_text])
    return prediction[0]

# Main function
def main():
    st.title("Tools Analisa Sentimen")

    model1_filename = './model/naive-bayes-model.pkl'
    model2_filename = './model/naive-bayes-model-balanced.pkl'
    model1 = load_model(model1_filename)
    model2 = load_model(model2_filename)
    
    st.success("Model Unbalanced dan Balanced berhasil dimuat!")
    
    input_text = st.text_input("Masukan Teks Untuk Dianalisa:")
    
    if input_text:
        sentiment_model1 = predict_sentiment(model1, input_text)
        st.info(f"Prediksi Model Naive Bayes Unbalanced: {sentiment_model1}")
        
        sentiment_model2 = predict_sentiment(model2, input_text)
        st.info(f"Prediksi Model Naive Bayes balanced: {sentiment_model2}")

if __name__ == '__main__':
    main()