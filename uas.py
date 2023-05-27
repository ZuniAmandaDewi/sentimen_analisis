import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load save model
loaded_model = pickle.load(open('model.sav', 'rb'))
# tfidf = TfidfVectorizer
# load_tf = TfidfVectorizer(decode_error='replace',vocabulary=set(pickle.load(open('tf_idf.sav','tb'))))
saved_vocabulary = pickle.load(open("tf_idf.sav", 'rb'))
cvec_tfidf = TfidfVectorizer(vocabulary=saved_vocabulary)

#judul halaman
st.title("Prediksi Postingan Twitter Tentang Resesi Ekonomi 2023")

clean_teks = st.text_input("Masukkan Kalimat Yang Ingin Di Cek")

hasil =''

if st.button("Prediksi"):
    tf = cvec_tfidf.fit_transform([clean_teks]).toarray()
    prediksi = loaded_model.predict(tf)

    if prediksi == 1 :
        hasil='Postingan Positif'
    else:
        hasil='Postingan Negatif'

st.success(hasil)
