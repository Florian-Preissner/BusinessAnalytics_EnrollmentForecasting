import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Dashboard", layout="wide")
models = ["dummy", "dummy2"]

# API and data functions
def api_request(url, http_request):
    BASE_URL = "http://localhost:8000" #TODO as env variable
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
        return json_data["historical_data"], json_data["generated_data"]
    else:
        return [],[]#TODO error handling    

@st.cache_data
def generate_data(days: int):
    success, json_data = api_request(f"data/new/{days}", requests.post)
    if success:
        return json_data["generated_data"]
    else:
        return []#TODO error handling

@st.cache_data
def predict_data(model: str, days:int):
    success, json_data = api_request(f"forecast/{model}/{days}", requests.get)
    if success:
        return json_data["predicted_data"]
    else:
        return []#TODO error handling


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

    #TODO
    x = np.arange(10)

    df = pd.DataFrame({
        "x": x,
        "Line A": np.sin(x),
        "Line B": np.cos(x),
        "Line C": np.sin(x) + np.cos(x)
    })

        # Placeholder graph
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(df["x"], df["Line A"], label="Line A")
    ax.plot(df["x"], df["Line B"], label="Line B")
    ax.plot(df["x"], df["Line C"], label="Line C")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Sample Line Plot")
    ax.legend()

    st.pyplot(fig)



init_top_layout()
init_forecast_plots()