import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def main():
    # Set up the title and description for the app
    st.title("Analisis Sentimen pada Komen YouTube")
    st.markdown("""
            <style>
            .video-container {
                text-align: center;  /* Center the video */
                margin: 20px 0;  /* Add some margin for spacing */
            }
            .video-container iframe {
                width: 80%;  /* Adjust the width of the video */
                max-width: 1000px;  /* Set max width to avoid being too large */
                height: 300px;  /* Set the height */
                border-radius: 10px;  /* Rounded corners */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Add a subtle shadow */
            }
            </style>
            <div class="video-container">
                <iframe src="https://www.youtube.com/embed/LjOxZjSujFI?si=eGfsNATXLnaG-4F_" frameborder="0" allowfullscreen></iframe>
                <iframe src="https://www.youtube.com/embed/4F2oOGDyWeY?si=OZiMuSrTU7utetxx" frameborder="0" allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("""
        Selamat datang di alat Analisis Sentimen Komentar YouTube! Aplikasi ini mengklasifikasikan komentar ke dalam tiga jenis:  
        - **Positif**  
        - **Netral**  
        - **Negatif**  

        Jelajahi dataset sentimen dan manfaatkan model machine learning untuk menganalisis sentimen dari komentar baru.
    """)
    st.markdown("""
        **Cara Penggunaan**: Gunakan sidebar untuk menjelajahi alat prediksi sentimen dan fitur-fitur lainnya.
    """)

if __name__ == "__main__":
    main()