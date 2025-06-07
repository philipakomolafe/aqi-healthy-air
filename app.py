# app.py
import streamlit as st
import requests
from datetime import datetime

API_URL = "https://aqi-healthy-air.onrender.com/aqi/online-prediction"

st.set_page_config(page_title="Air Quality Index Dashboard", page_icon="🌫️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .aqi-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        font-size: 22px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌫️ Air Quality Index (AQI) Dashboard")
st.write("Get real-time air quality predictions for Akure, Nigeria.")

refresh = st.button("Refresh Prediction")

def get_aqi_color(color_name):
    # Map color names to hex codes
    colors = {
        "green": "#43a047",
        "yellow": "#fbc02d",
        "orange": "#fb8c00",
        "red": "#e53935",
        "purple": "#8e24aa",
        "maroon": "#6d4c41"
    }
    return colors.get(color_name.lower(), "#607d8b")

def fetch_prediction():
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Error: {e}"

if refresh or "last_data" not in st.session_state:
    data, error = fetch_prediction()
    st.session_state["last_data"] = data
    st.session_state["last_error"] = error
else:
    data = st.session_state["last_data"]
    error = st.session_state["last_error"]

if data:
    aqi_color = get_aqi_color(data.get("color", "green"))
    st.markdown(
        f'<div class="aqi-box" style="background-color:{aqi_color};">'
        f'<span class="big-font">AQI: {data["predicted_aqi"]}</span><br>'
        f'{data["advice"]}'
        f'</div>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    col1.metric("Location", data["location"])
    col2.metric("Status Color", data["color"].capitalize())
    st.caption(f"Last updated: {datetime.fromisoformat(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} UTC")
else:
    st.error(error or "No data available.")

# Optional: Auto-refresh every 60 seconds
if st.button("Auto-refresh (every 60s)"):
    import time
    time.sleep(60)
    st.rerun()