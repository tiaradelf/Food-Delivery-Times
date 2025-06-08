import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 🛠️ Page Config - WAJIB PALING ATAS
# ===============================
st.set_page_config(page_title="Prediksi Kategori Pengiriman Makanan", layout="wide")

# ===============================
# 1. Load & Preprocessing Data
# ===============================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Food_Delivery_Times.csv")

    # Drop kolom Order_ID
    if 'Order_ID' in df.columns:
        df = df.drop(columns=['Order_ID'])

    # Konversi target numerik jadi kategorikal
    bins = [0, 20, 40, df['Delivery_Time_min'].max()]
    labels = ['Cepat', 'Sedang', 'Lama']
    df['Delivery_Category'] = pd.cut(df['Delivery_Time_min'], bins=bins, labels=labels, include_lowest=True)

    X = df.drop(columns=['Delivery_Time_min', 'Delivery_Category'])
    y = df['Delivery_Category']

    # Encoding
    X_encoded = pd.get_dummies(X)

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return df, X_encoded, X_scaled, y, scaler, X.columns.tolist(), X_encoded.columns.tolist()

df, X_encoded, X_scaled, y, scaler, original_features, model_features = load_and_preprocess_data()

# ===============================
# 2. Train Model Sekali
# ===============================
model = RandomForestClassifier(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 3. Sidebar Navigasi
# ===============================
menu = st.sidebar.radio("Navigation", ["📊 EDA", "🔮 Prediksi", "📬 Contact"])

# ===============================
# 📊 EDA
# ===============================
if menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("🔍 5 Data Teratas")
    st.dataframe(df.head())

    st.subheader("🔗 Korelasi Numerik")
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu")
    st.pyplot(plt.gcf())

    st.subheader("📈 Histogram Kolom Numerik")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Pilih Kolom:", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

# ===============================
# 🔮 Prediksi
# ===============================
elif menu == "🔮 Prediksi":
    st.title("🔮 Prediksi Kategori Waktu Pengiriman")

    input_data = {}
    for col in original_features:
        if df[col].dtype == 'object':
            opt = df[col].unique().tolist()
            input_data[col] = st.selectbox(col, options=opt)
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_data[col] = st.number_input(col, value=mean_val, min_value=min_val, max_value=max_val)

    # Proses input user
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    if st.button("🎯 Prediksi"):
        pred = model.predict(input_scaled)[0]
        st.success(f"⏱️ Prediksi Kategori Pengiriman: **{pred}**")

        st.subheader("📊 Evaluasi Model")
        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        st.write("Akurasi:", round(acc, 2))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))

# ===============================
# 📬 Kontak & Penutup
# ===============================
elif menu == "📬 Contact":
    st.title("📬 Contact dan Penghargaan")

    st.write("Hubungi saya melalui:")
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/tiara-delfira/)"
    )
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/tiaradelf)"
    )
    st.write("📧 Email: delfiratiara7@gmail.com")

    st.divider()

    st.markdown("""
    <div style='text-align: center; font-size: 18px; margin-top: 20px;'>
        🌟 Terima kasih telah mengeksplorasi Project Data Science ini! 🌟<br>
        Semoga hasil analisis dan insight yang diberikan dapat bermanfaat dalam pengambilan keputusan bisnis yang lebih baik.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.header("🎨 Till Next Time")
    image_path = "ucapan.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Terima kasih telah menjelajahi streamlit ini!", use_container_width=True)
    else:
        st.warning("❗ Gambar 'ucapan.jpg' tidak ditemukan. Pastikan file berada di direktori yang sama.")
# ========== Credit di Footer ==========
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 14px; color: gray; margin-top: 10px;'>
        © 2025 Tiara Delfira - Informatics Engineer & Data Scientist Enthusiast<br>
        Developed with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True
)