import streamlit as st
import joblib

# Load model
model = joblib.load("model_logreg.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

st.title("Analisis Sentimen Program KKA")

st.write("Masukkan komentar YouTube")

komentar = st.text_area("Komentar")

if st.button("Prediksi"):

    data = tfidf.transform([komentar])

    hasil = model.predict(data)

    st.success("Hasil Prediksi : " + hasil[0])