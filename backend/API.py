from fastapi import FastAPI, HTTPException
from .model.dummymodel import DummyModel


def init_models(data):
    dummyModel = DummyModel(data)
    return {"dummy" : dummyModel}

def load_data():
    return [], []

app = FastAPI()
historical_data, generated_data = load_data() #historical = model data, generated = new data
models = init_models(historical_data)

@app.get("/")
def root():
    return "Works :)"

@app.get("/data")
def get_data():
    return {"historical_data" : historical_data, "generated_data" : generated_data}

@app.post("/data/new/{number_days}")
def generate_data(number_days: int):
    #todo
    return {"generated_data" : generated_data}

@app.post("/retrain")
def retrain_models():
    historical_data = historical_data + generated_data
    generated_data = []
    models = init_models(historical_data)

@app.get("/forecast/{model_id}/{N}")
def get_NDay_Forecast(model_id: str, N: int):

    model = models.get(model_id)
    
    if model:
        predictions = model.predict(N)
        return predictions
    else:
        raise HTTPException(status_code=404, detail="Model does not exist")