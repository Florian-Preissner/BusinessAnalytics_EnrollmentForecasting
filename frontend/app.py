import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import json
from evidently import Report
from evidently.presets import DataDriftPreset

st.set_page_config(page_title="Dashboard", layout="wide")
models = ["dummy", "dummy2"]
BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# API and data functions
def api_request(url, http_request):
    url = f"{BASE_URL}/{url}"

    try:
        response = http_request(url, timeout=10)

        # Raises an exception for 4xx/5xx responses
        response.raise_for_status()

        # Parse JSON response
        data = response.json()

        print("Success!")
        print(data)

        return True, response.json()

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to the server.")
    except requests.exceptions.Timeout:
        print("The request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return False, {}


@st.cache_data
def load_data():
    success, json_data = api_request("data", requests.get)
    if success:
        hist = json.loads(json_data["historical_data"])
        gen = json.loads(json_data["generated_data"])

        return pd.DataFrame(hist), pd.DataFrame(gen)
    else:
        return pd.DataFrame(),pd.DataFrame()#TODO error handling    

@st.cache_data
def generate_data(days: int):
    success, json_data = api_request(f"data/new/{days}", requests.post)
    if success:
        return pd.DataFrame(json.loads(json_data["generated_data"]))
    else:
        return pd.DataFrame()#TODO error handling

@st.cache_data
def predict_data(model: str, days:int):
    success, json_data = api_request(f"forecast/{model}/{days}", requests.get)
    if success:
        return pd.DataFrame(json.loads(json_data["predicted_data"]))
    else:
        return pd.DataFrame()#TODO error handling


# UI event handler
def onPressed_generateButton():
    number_days = st.session_state["number_days_generate"]
    if number_days:
        st.session_state.data["generated_data"] = generate_data(number_days)

def onPressed_forecastButton():
    number_days = st.session_state["number_days_forecast"]
    if number_days:
        for model in st.session_state.selected_models:
            st.session_state.predicted_data[model]= predict_data(model, number_days)

# init global variables
if "selected_models" not in st.session_state:
    st.session_state.selected_models = models

if "data" not in st.session_state: 
    historical_data, generated_data = load_data()
    st.session_state.data = {
        "historical_data" : historical_data,
        "generated_data" : generated_data
    }

if "predicted_data" not in st.session_state:
    st.session_state.predicted_data = {model: [] for model in models}
if "selected_models" not in st.session_state:
    st.session_state.selected_models = models

# Layout functions

def init_top_layout():
    st.title("Enrollment Forecasting")

    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input("Input number of days for generating data", key="number_days_generate")
    
    with col2:
        st.write("")
        st.button("Generate", on_click=onPressed_generateButton)        

    col3, col4 = st.columns([5, 1])
    with col3:
        st.text_input("Input number of days for forecasting", key="number_days_forecast")
    with col4:
        st.write("")
        st.button("Forecast", on_click=onPressed_forecastButton)

    st.divider()


def init_forecast_plots():

    st.multiselect(
    "Select Models", models,
    key="selected_models" 
    )

    print("historical data")
    print(st.session_state.data["historical_data"])
    print("generated data")
    print(st.session_state.data["generated_data"])
    #TODO
    x = np.arange(0)

    df = pd.DataFrame({
        "x": x,
        "historical_data": [],#st.session_state.data["historical_data"],
        "generated_data": []#st.session_state.data["generated_data"],
    })
    #TDOD stack for each model column to df
    #"dummy": st.session_state.data["historical_data"],
    #    "dummy2": np.sin(x) + np.cos(x)
    #})

        # Placeholder graph
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(df["x"], df["historical_data"], label="Historical Data")
    ax.plot(df["x"], df["generated_data"], label="Generated Data")
    #ax.plot(df["x"], df["dummy"], label="Dummy Prediction")
    #ax.plot(df["x"], df["dummy2"], label="Dummy 2 Model Prediction")

    ax.set_xlabel("Days")
    ax.set_ylabel("Enrolled Students")
    ax.set_title("Enrollment Forecasting for different models")
    ax.legend()

    st.pyplot(fig)

def init_monitoring():
    st.divider()
    st.title("Model Monitoring")
    #TODO set correct data
    reference_df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5]
    })

    current_df = pd.DataFrame({
        "target": [2, 3, 4, 5, 6]
    })

    report = Report(
        metrics=[DataDriftPreset()]
    )
    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    filename="report.html"
    snapshot.save_html(filename)
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    st.iframe(html, height=900)

init_top_layout()
init_forecast_plots()
init_monitoring()