import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

try:
    from model.EnrollmentForecastModel import EnrollmentForecastModel
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from model.EnrollmentForecastModel import EnrollmentForecastModel


def init_model() -> EnrollmentForecastModel:
    backend_dir = Path(__file__).resolve().parent
    model_path = backend_dir.parent / "lightgbm.pkl"
    test_day_dir = backend_dir / "data" / "feature" / "feature_engineered_test_day_splits"
    return EnrollmentForecastModel(model_path, test_day_dir)

app = FastAPI()
models: Dict[str, EnrollmentForecastModel] = {"enrollment_forecast": init_model()}

@app.get("/")
def root():
    return {"status": "ok", "model": "enrollment_forecast"}

@app.get("/forecast/{dataset}")
def get_forecast(dataset: str):
    model = models.get("enrollment_forecast")
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if dataset == "test":
        payload = model.predict_test().to_dict(orient="records")
    else:
        raise HTTPException(status_code=400, detail="dataset must be 'test'")

    return {"predictions": payload}

@app.get("/forecast/course/{dataset}/{code_module}/{code_presentation}")
def get_forecast_course(dataset: str, code_module: str, code_presentation: str):
    model = models.get("enrollment_forecast")
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        payload = model.predict_course(code_module, code_presentation, dataset=dataset).to_dict(orient="records")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"predictions": payload}

@app.get("/evaluate")
def evaluate_model():
    model = models.get("enrollment_forecast")
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        metrics = model.evaluate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"evaluation": metrics}

@app.get("/courses/{dataset}")
def get_courses(dataset: str):
    model = models.get("enrollment_forecast")
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if dataset != "test":
        raise HTTPException(status_code=400, detail="dataset must be 'test'")

    try:
        courses = model.get_available_courses(dataset=dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"courses": [list(course) for course in courses]}
