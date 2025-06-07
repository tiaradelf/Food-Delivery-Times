import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import os

# =========================
# Fungsi load dan preprocess data
# =========================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Food_Delivery_Times.csv")

    target_col = "Weather"
    if target_col not in df.columns:
        st.error(f"Kolom target '{target_col}' tidak ditemukan di dataset!")
        st.stop()

    # Drop baris yang ada NaN di kolom target agar proses encoding lancar
    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_encoded = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return df_clean, X_encoded, X_scaled, y_encoded, scaler, le, X.columns.tolist(), X_encoded.columns.tolist()

# =========================
# Fungsi tampilkan contact page
# =========================
@st.cache_resource
def tampilkan_contact():
    st.title("Contact")
    st.write("Contact me through the following link:")

    # LinkedIn
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/tiara-delfira/)"
    )

    # GitHub
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/tiaradelf)"
    )

    # Email
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

    image_path = "ucapan.jpg"  # Pastikan file ini ada di folder utama Streamlit-mu

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Terima kasih telah menjelajahi streamlit ini!", use_container_width=True)
    else:
        st.warning("❗ Gambar 'ucapan.jpg' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    
    # Tambahan credit pembuat
    st.markdown(
        """
        <hr>
        <div style='text-align: center; font-size: 14px; color: gray; margin-top: 10px;'>
            © 2025 Tiara Delfira - Informatics Engineer & Data Scientist Enthusiast<br>
            Developed with ❤️ using Streamlit
        </div>
        """, unsafe_allow_html=True
    )

# =========================
# Halaman utama aplikasi
# =========================
def main():
    # Must be the first Streamlit command
    st.set_page_config(page_title="Food Delivery Weather Classifier", layout="wide")

    # Load data dan preprocess
    df, X_encoded, X_scaled, y_encoded, scaler, le, original_features, model_features = load_and_preprocess_data()

    # Train model
    model = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Sidebar menu
    menu = st.sidebar.radio("Navigasi", ["📊 EDA", "🔮 Prediksi", "📞 Contact"])

    if menu == "📊 EDA":
        st.title("🚚 Food Delivery Weather Prediction (EDA)")

        st.header("📈 Exploratory Data Analysis")
        st.subheader("📌 Tampilkan 5 data teratas")
        st.write(df.head())

        st.subheader("🔍 Korelasi Numerik (Heatmap)")
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())

        st.subheader("📊 Histogram")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Pilih kolom numerik", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Tidak ada kolom numerik untuk ditampilkan histogram.")

    elif menu == "🔮 Prediksi":
        st.title("🚚 Food Delivery Weather Prediction (Prediksi)")

        st.header("🧾 Input Data Prediksi")
        input_data = {}
        for col in original_features:
            if df[col].dtype == 'object':
                options = df[col].dropna().unique().tolist()
                input_data[col] = st.selectbox(f"{col}", options=options)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.number_input(f"{col}", value=mean_val, min_value=min_val, max_value=max_val)

        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
        input_scaled = scaler.transform(input_encoded)

        if st.button("🔍 Prediksi Weather"):
            pred = model.predict(input_scaled)[0]
            label = le.inverse_transform([pred])[0]
            st.success(f"🎯 Prediksi: **{label}**")

            st.subheader("📊 Evaluasi Model")
            y_pred_test = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred_test)
            st.write("Akurasi:", round(acc, 2))

            target_names = [str(c) for c in le.classes_]
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred_test, target_names=target_names))

    elif menu == "📞 Contact":
        tampilkan_contact()

if __name__ == "__main__":
    main()