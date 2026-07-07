from fastapi import FastAPI, HTTPException
from .model.dummymodel import DummyModel
from .data.data_transform import ETL
from .data.data_transform.synthetic_data import load_synthetic_data
import pandas as pd

def init_models(data):
    dummyModel = DummyModel(data)
    return {"dummy" : dummyModel}

def load_data():
    return ETL.load_data().head(5), pd.DataFrame()

app = FastAPI()
historical_data, generated_data = load_data() #historical = model data, generated = new data
models = init_models(historical_data)

@app.get("/")
def root():
    return "Works :)"

@app.get("/data")
def get_data():
    return {"historical_data" : historical_data.to_json(), "generated_data" : generated_data.to_json()}

@app.post("/data/new/{number_days}")
def generate_data(number_days: int):
    global generated_data
    new_data,_ = load_synthetic_data(number_days)
    new_data.head(5)
    generated_data = pd.concat([generated_data, new_data], ignore_index=True)
    return {"generated_data" : generated_data.to_json()}

@app.post("/retrain")
def retrain_models():
    global historical_data, generated_data, models

    historical_data =pd.concat([ historical_data, generated_data], ignore_index=True)
    generated_data = pd.DataFrame()
    models = init_models(historical_data)

@app.get("/forecast/{model_id}/{N}")
def get_NDay_Forecast(model_id: str, N: int):
    global models

    model = models.get(model_id)
    
    if model:
        predictions = model.predict(N)
        return {"predicted_data" : predictions}
    else: 
        raise HTTPException(status_code=404, detail="Model does not exist")