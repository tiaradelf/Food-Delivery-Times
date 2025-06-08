import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ========== Konfigurasi halaman ==========
st.set_page_config(page_title="Prediksi Waktu Pengiriman Makanan", layout="wide")

# ========== Load & Preprocess Data ==========
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Food_Delivery_Times.csv")

    if "Order ID" in df.columns:
        df.drop(columns=["Order ID"], inplace=True)

    if "Time_taken (min)" not in df.columns:
        st.error("Kolom 'Time_taken (min)' tidak ditemukan dalam dataset.")
        st.stop()

    # Binning 'Time_taken (min)' menjadi kelas
    bins = [0, 20, 40, float('inf')]
    labels = ['Cepat', 'Sedang', 'Lama']
    df['Delivery_Class'] = pd.cut(df['Time_taken (min)'], bins=bins, labels=labels)

    X = df.drop(columns=['Time_taken (min)', 'Delivery_Class'])
    y = df['Delivery_Class']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_encoded = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return df, X_encoded, X_scaled, y_encoded, scaler, le, X.columns.tolist(), X_encoded.columns.tolist()

df, X_encoded, X_scaled, y_encoded, scaler, le, original_features, model_features = load_and_preprocess_data()

# ========== Training Model ==========
model = RandomForestClassifier(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# ========== Navigasi ==========
menu = st.sidebar.radio("Navigation", ["📊 EDA", "🔮 Prediksi", "📬 Contact"])

# ========== 📊 EDA ==========
if menu == "📊 EDA":
    st.title("📈 Exploratory Data Analysis")
    st.subheader("📌 5 Data Teratas")
    st.write(df.head())

    st.subheader("🔍 Korelasi Numerik")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Histogram")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Pilih kolom numerik", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

# ========== 🔮 Prediksi ==========
elif menu == "🔮 Prediksi":
    st.title("🔍 Prediksi Kelas Waktu Pengiriman")

    input_data = {}
    for col in original_features:
        if df[col].dtype == 'object':
            opt = df[col].unique().tolist()
            input_data[col] = st.selectbox(f"{col}", options=opt)
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_data[col] = st.number_input(f"{col}", value=mean_val, min_value=min_val, max_value=max_val)

    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    if st.button("🚀 Prediksi"):
        pred = model.predict(input_scaled)[0]
        label = le.inverse_transform([pred])[0]
        st.success(f"🕒 Prediksi Kategori Pengiriman: **{label}**")

        st.subheader("📊 Evaluasi Model")
        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        st.write("Akurasi:", round(acc, 2))
        st.text("Classification Report:")
        report = classification_report(y_test, y_pred_test, target_names=le.classes_)
        st.text(report)

# ========== 📬 Kontak ==========
elif menu == "📬 Contact":
    st.title("📫 Contact & Credits")
    st.write("Hubungi saya melalui tautan berikut:")

    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/tiara-delfira/)")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/tiaradelf)")
    st.write("📧 Email: delfiratiara7@gmail.com")

    st.divider()
    st.markdown(
        """
        <div style='text-align: center; font-size: 18px; margin-top: 20px;'>
            🌟 Terima kasih telah mengeksplorasi Project Data Science ini! 🌟<br>
            Semoga hasil analisis dan insight yang diberikan dapat bermanfaat dalam pengambilan keputusan bisnis yang lebih baik.
        </div>
        """, unsafe_allow_html=True
    )

    st.divider()
    st.header("🎨 Till Next Time")

    image_path = "ucapan.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Terima kasih telah menjelajahi aplikasi ini!", use_container_width=True)
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