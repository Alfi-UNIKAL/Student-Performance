import streamlit as st
import pandas as pd
import joblib

# --- Load model ---
try:
    with open("model_graduation.pkl", "rb") as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Model file 'model_graduation.pkl' tidak ditemukan.")
    st.stop()

# --- Judul Aplikasi ---
st.title("Prediksi Kategori Waktu Lulus Mahasiswa")

st.write("""
Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus **Tepat Waktu (On Time)** atau **Terlambat (Late)**
berdasarkan beberapa faktor seperti nilai ACT, SAT, rata-rata SMA, pendapatan orang tua, dan pendidikan orang tua.
Silakan isi data berikut:
""")

# --- Form Input Data ---
with st.form("prediction_form"):
    new_ACT = st.number_input("Nilai ACT Composite Score", min_value=0.0)
    new_SAT = st.number_input("Nilai SAT Total Score", min_value=0.0)
    new_GPA = st.number_input("Nilai Rata-rata SMA (GPA)", min_value=0.0, max_value=4.0)
    new_income = st.number_input("Pendapatan Orang Tua", min_value=0.0)
    new_education = st.number_input("Tingkat Pendidikan Orang Tua (dalam angka)", min_value=0.0)
    
    submitted = st.form_submit_button("Prediksi")

# --- Prediksi ---
if submitted:
    try:
        # Buat DataFrame dari input
        new_data_df = pd.DataFrame(
            [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
            columns=[
                'ACT composite score', 
                'SAT total score', 
                'high school gpa', 
                'parental income', 
                'parent_edu_numerical'
            ]
        )

        # Lakukan prediksi
        predicted_code = model.predict(new_data_df)[0]
        label_mapping = {1: 'On Time', 0: 'Late'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        # Tampilkan hasil
        st.success(f"Prediksi kategori masa studi: **{predicted_label}**")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
