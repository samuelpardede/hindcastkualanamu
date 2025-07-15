import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="HindCasting Kualanamu",
    page_icon="🌧️",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_assets():
    """Memuat model dan scaler yang sudah disimpan."""
    try:
        model = joblib.load('rf_me48_model.pkl')
        scaler_X = joblib.load('scaler_X_me48.pkl')
        scaler_y = joblib.load('scaler_y_me48.pkl')
        # Muat juga gambar feature importance
        try:
            importance_plot = Image.open('rf me48.png')
        except FileNotFoundError:
            importance_plot = None
        return model, scaler_X, scaler_y, importance_plot
    except FileNotFoundError:
        return None, None, None, None

# --- MEMUAT ASET ---
model, scaler_X, scaler_y, importance_plot = load_assets()

# --- TAMPILAN UTAMA ---
st.title("HindCasting Kualanamu 🌧️")
st.markdown("*Implementasi Model **Random Forest** pada Data Sinoptik ME 48*")
st.markdown("---")

if model is None:
    st.error("File model 'rf_me48_model.pkl' tidak ditemukan. Mohon jalankan skrip `train_rf_me48.py` terlebih dahulu.")
else:
    # --- UI INPUT PARAMETER ---
    st.sidebar.header("Input Variabel Prediktor")

    user_inputs = {}
    
    with st.sidebar.expander("Parameter Awan & Cuaca"):
        user_inputs["CLOUD_LOW_TYPE_CL"] = st.number_input("Jenis Awan Rendah (CL)", 0, 9, 5)
        user_inputs["CLOUD_LOW_MED_AMT_OKTAS"] = st.number_input("Jml Awan Rendah/Menengah (Oktas)", 0, 8, 3)
        user_inputs["CLOUD_MED_TYPE_CM"] = st.number_input("Jenis Awan Menengah (CM)", 0, 9, 7)
        user_inputs["CLOUD_HIGH_TYPE_CH"] = st.number_input("Jenis Awan Tinggi (CH)", 0, 9, 6)
        user_inputs["CLOUD_COVER_OKTAS_M"] = st.number_input("Tutupan Awan Total (Oktas)", 0, 8, 7)
        user_inputs["PRESENT_WEATHER_WW"] = st.number_input("Cuaca Saat Pengamatan (WW)", 0, 99, 10)
        
    with st.sidebar.expander("Parameter Suhu & Kondisi"):
# --- Di dalam bagian "with st.sidebar.expander("Parameter Suhu & Kondisi"):" ---

# Membuat pemetaan (mapping) untuk kode LAND_COND
        land_cond_map = {
            0: "0 - Permukaan Kering",
            1: "1 - Permukaan Basah",
            2: "2 - Permukaan Tergenang Air"
        }

# Membuat selectbox dengan keterangan yang benar
        user_inputs["LAND_COND"] = st.selectbox(
            "Kondisi Tanah (LAND_COND)",
            options=list(land_cond_map.keys()),
            format_func=lambda x: land_cond_map[x],
            index=1 # Default pilihan tetap 'Basah'
        )

        user_inputs["TEMP_DEWPOINT_C_TDTDTD"] = st.slider("Suhu Titik Embun (°C)", 20.0, 30.0, 24.5)
        user_inputs["TEMP_DRYBULB_C_TTTTTT"] = st.slider("Suhu Udara (°C)", 21.0, 36.0, 27.0)
        user_inputs["TEMP_WETBULB_C"] = st.slider("Suhu Bola Basah (°C)", 20.0, 30.0, 25.5)
        user_inputs["RELATIVE_HUMIDITY_PC"] = st.slider("Kelembapan Relatif (%)", 40, 100, 85)

    with st.sidebar.expander("Parameter Angin & Tekanan"):
        user_inputs["WIND_SPEED_FF"] = st.slider("Kecepatan Angin (m/s)", 0.0, 15.0, 4.0)
        user_inputs["PRESSURE_QFF_MB_DERIVED"] = st.slider("Tekanan QFF (mb)", 1000.0, 1020.0, 1009.5)
        user_inputs["PRESSURE_QFE_MB_DERIVED"] = st.slider("Tekanan QFE (mb)", 1000.0, 1020.0, 1008.6)
    
    # --- Tombol & Logika Prediksi ---
    if st.sidebar.button("Prediksi Curah Hujan", type="primary", use_container_width=True):
        
        # Urutkan input sesuai urutan fitur saat training
        features_order = scaler_X.feature_names_in_
        input_list = [user_inputs[feature] for feature in features_order]
        
        # Buat array numpy dan lakukan scaling
        input_array = np.array(input_list).reshape(1, -1)
        input_scaled = scaler_X.transform(input_array)

        # Lakukan prediksi
        prediction_scaled = model.predict(input_scaled)
        prediction_inv = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
        predicted_value = max(0, prediction_inv[0][0])

        # Tentukan kategori & warna
        if predicted_value < 0.5:
            kategori, delta_color = "Tidak Hujan", "off"
        elif 0.5 <= predicted_value <= 5:
            kategori, delta_color = "Hujan Ringan", "normal"
        elif 5 < predicted_value <= 10:
            kategori, delta_color = "Hujan Sedang", "normal"
        else: # > 10
            kategori, delta_color = "Hujan Lebat", "inverse"
            
        # Tampilkan Hasil
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Prediksi Curah Hujan (per 3 jam)", value=f"{predicted_value:.2f} mm")
        with col_res2:
            st.metric(label="Kategori", value=kategori, delta_color=delta_color)

        if importance_plot:
            st.markdown("---")
            st.subheader("Faktor Paling Berpengaruh (Feature Importance)")
            st.image(importance_plot, caption="Peringkat variabel yang paling dipertimbangkan oleh model Random Forest.", use_column_width=True)
    
    else:
        st.info("Silakan atur parameter di sebelah kiri dan klik tombol prediksi.")

    st.sidebar.markdown("---")
    st.sidebar.write("Dibuat oleh: Samuel F. Pardede")
