import re
import string
import streamlit as st

# REPO HUGGING FACE (IndoBERT)
HF_MODEL = "safiirahsf/indobert-kka"

st.set_page_config(page_title="Sentimen KKA", page_icon="🧠", layout="centered")

# Preprocessing 
emoji_pattern = re.compile(
    "[" "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF\U0001FA70-\U0001FAFF]+", flags=re.UNICODE)


def preprocess_klasik(teks: str) -> str:
    """Preprocessing lengkap untuk model klasik"""
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#\w+", " ", teks)
    teks = emoji_pattern.sub(" ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = teks.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", teks).strip()


def preprocess_minimal(teks: str) -> str:
    """Preprocessing minimal untuk IndoBERT"""
    teks = re.sub(r"http\S+|@\w+", " ", str(teks))
    return re.sub(r"\s+", " ", teks).strip()


# Load model
@st.cache_resource
def load_klasik():
    import joblib
    model = joblib.load("model_logreg.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    return model, tfidf


@st.cache_resource
def load_bert():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    mdl.eval()
    return tok, mdl, torch


# UI
st.title("Analisis Sentimen Program KKA")
st.caption("Koding dan Kecerdasan Artifisial — klasifikasi komentar YouTube (positif / netral / negatif)")

pilihan = st.radio("Pilih model:", ["IndoBERT", "Logistic Regression"],
                   horizontal=True)

komentar = st.text_area("Masukkan komentar YouTube:", height=120,
                        placeholder="Contoh: Program KKA bagus, tapi fasilitas di daerah harus dipersiapkan dulu...")

warna = {"positif": "🟢", "netral": "🟡", "negatif": "🔴"}

if st.button("Prediksi", type="primary"):
    if not komentar.strip():
        st.warning("Tolong isi komentar dulu.")

    elif pilihan.startswith("IndoBERT"):
        with st.spinner("Memuat IndoBERT"):
            tok, mdl, torch = load_bert()
        enc = tok(preprocess_minimal(komentar), truncation=True, padding=True,
                  max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=1)[0]
        idx = int(torch.argmax(probs))
        label = mdl.config.id2label[idx]
        st.success(f"{warna.get(label, '')} Prediksi: **{label.upper()}**")
        st.bar_chart({mdl.config.id2label[i]: float(probs[i]) for i in range(len(probs))})

    else:
        model, tfidf = load_klasik()
        vek = tfidf.transform([preprocess_klasik(komentar)])
        label = model.predict(vek)[0]
        st.success(f"{warna.get(label, '')} Prediksi: **{label.upper()}**")
        if hasattr(model, "predict_proba"):
            st.bar_chart(dict(zip(model.classes_, model.predict_proba(vek)[0])))

st.divider()
st.caption("Proyek Akhir Text Mining — Perbandingan ML Klasik vs Transformer")
