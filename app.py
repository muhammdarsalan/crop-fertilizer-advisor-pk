import streamlit as st
import pickle
import numpy as np

# ── Load models ──────────────────────────────────────────
with open("models/crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("models/fertilizer_model.pkl", "rb") as f:
    fert_model = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Kissan AI 🌾", page_icon="🌾", layout="centered")

# ── Header ───────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #2e7d32;'>🌾 Kissan AI</h1>
    <p style='text-align: center; font-size: 18px; color: #555;'>
        Smart Crop & Fertilizer Advisor for Pakistani Farmers
    </p>
    <p style='text-align: center; font-size: 14px; color: #888;'>
        Built by a CS student from Peshawar who wants AI to help his own people 🇵🇰
    </p>
    <hr>
""", unsafe_allow_html=True)

st.info("📋 Fill in your soil and weather details below. Kissan AI will recommend the best crop to grow and the right fertilizer to use.")

st.markdown("### 🌡️ Weather & Soil Conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🧪 Nitrogen (N)", min_value=0, max_value=200, value=50,
                        help="Amount of Nitrogen in your soil")
    P = st.number_input("🧪 Phosphorus (P)", min_value=0, max_value=200, value=50,
                        help="Amount of Phosphorus in your soil")
    K = st.number_input("🧪 Potassium (K)", min_value=0, max_value=200, value=50,
                        help="Amount of Potassium in your soil")
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)

with col2:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("⚗️ Soil pH", min_value=0.0, max_value=14.0, value=6.5,
                         help="pH 6-7 is ideal for most crops")
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    moisture = st.number_input("🌱 Soil Moisture (%)", min_value=0, max_value=100, value=40)

st.markdown("### 🪨 Soil Type")
soil_type = st.selectbox("Select your soil type", encoders["soil"].classes_)

st.markdown("")

if st.button("🌱 Get My Recommendation", use_container_width=True, type="primary"):

    with st.spinner("Analyzing your soil conditions..."):

        crop_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        crop_prediction = crop_model.predict(crop_input)[0]

        soil_encoded = encoders["soil"].transform([soil_type])[0]
        crop_encoded = encoders["crop"].transform([crop_prediction])[0] \
            if crop_prediction in encoders["crop"].classes_ else 0

        fert_input = np.array([[temperature, humidity, moisture, soil_encoded,
                                crop_encoded, N, K, P]])
        fert_encoded = fert_model.predict(fert_input)[0]
        fert_prediction = encoders["fertilizer"].inverse_transform([fert_encoded])[0]

    st.markdown("---")
    st.markdown("## 🎯 Your Results")

    col3, col4 = st.columns(2)

    with col3:
        st.success(f"""
        ### ✅ Best Crop for You
        # {crop_prediction.upper()}
        Based on your soil nutrients, temperature, humidity, pH and rainfall.
        """)

    with col4:
        st.info(f"""
        ### 🧪 Recommended Fertilizer
        # {fert_prediction}
        Best fertilizer for {crop_prediction} in your soil type.
        """)

    st.markdown("---")
    st.markdown("""
    <div style='background-color: #f1f8e9; padding: 15px; border-radius: 10px;'>
        <p style='color: #555; font-size: 13px; text-align: center;'>
            ⚠️ This is an AI recommendation based on data patterns.
            Always consult a local agricultural expert before making final decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #aaa; font-size: 12px;'>
    Built with ❤️ by Muhammad Arsalan | CS Student, Peshawar 🇵🇰 | AI Learning Journey Week 6-7<br>
    <a href='https://github.com/muhammdarsalan' target='_blank'>GitHub</a> •
    <a href='https://www.linkedin.com/in/muhammad-arsalan-188a03325' target='_blank'>LinkedIn</a>
</p>
""", unsafe_allow_html=True)