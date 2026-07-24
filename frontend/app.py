import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from evidently import Report, Regression, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    from evidently.presets import RegressionPreset
    from evidently.metrics import ValueDrift
except Exception:  # pragma: no cover - optional monitoring dependency
    Report = None
    DataDriftPreset = None

try:
    from backend.model.EnrollmentForecastModel import EnrollmentForecastModel
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(repo_root))
    from backend.model.EnrollmentForecastModel import EnrollmentForecastModel

st.set_page_config(page_title="Enrollment Forecasting", layout="wide")

repo_root = Path(__file__).resolve().parent.parent
model_path = repo_root / "lightgbm.pkl"
test_day_split_dir = repo_root / "backend" / "data" / "feature" / "feature_engineered_test_day_splits"

model = EnrollmentForecastModel(model_path, test_day_split_dir)

@st.cache_data
def get_day_offsets() -> list[int]:
    return sorted(model.test_df["day_offset"].astype(int).unique().tolist())


@st.cache_data
def load_day_dataset(day_offset: int) -> pd.DataFrame:
    file_path = test_day_split_dir / f"feature_engineered_test_day_{day_offset}.csv"
    return pd.read_csv(file_path)


@st.cache_data
def load_day_datasets(max_day_offset: int) -> pd.DataFrame:
    offsets = [offset for offset in get_day_offsets() if offset <= max_day_offset]
    frames = [load_day_dataset(offset) for offset in offsets]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined


@st.cache_data
def predict_day_offset(day_offset: int) -> pd.DataFrame:
    df = load_day_datasets(day_offset)
    if df.empty:
        return pd.DataFrame()

    predictions = model.predict(df)
    result = pd.DataFrame(
        {
            "day_offset": df["day_offset"].values,
            "code_module": df["code_module"].values,
            "code_presentation": df["code_presentation"].values,
            "predicted_enrollment": predictions,
        }
    )
    if "enrollment_num_daily" in df.columns:
        result["actual_enrollment"] = df["enrollment_num_daily"].values
    return result.sort_values(["day_offset", "code_module", "code_presentation"]).reset_index(drop=True)


@st.cache_data
def get_courses(dataset: str) -> list[tuple[str, str]]:
    if dataset != "test":
        raise ValueError("dataset must be 'test'")
    return model.get_available_courses(dataset=dataset)


@st.cache_data
def load_train_data() -> pd.DataFrame:
    train_path = repo_root / "backend" / "data" / "feature" / "feature_engineered_train.csv"
    return pd.read_csv(train_path)


@st.cache_data
def get_train_course_history(course_key: str) -> pd.DataFrame:
    code_module, code_presentation = course_key.split("||")
    df = load_train_data()
    filtered = df[
        (df["code_module"] == code_module) &
        (df["code_presentation"] == code_presentation)
    ].copy()
    if filtered.empty:
        return pd.DataFrame()
    if "enrollment_num_daily" in filtered.columns:
        filtered = filtered.rename(columns={"enrollment_num_daily": "actual_enrollment"})
    return filtered.sort_values("day_offset").reset_index(drop=True)


@st.cache_data
def predict_course_history(day_offset: int, course_key: str) -> pd.DataFrame:
    code_module, code_presentation = course_key.split("||")
    df = load_day_datasets(day_offset)
    if df.empty:
        return pd.DataFrame()

    filtered = df[
        (df["code_module"] == code_module) &
        (df["code_presentation"] == code_presentation)
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    predictions = model.predict(filtered)
    result = pd.DataFrame(
        {
            "day_offset": filtered["day_offset"].values,
            "code_module": filtered["code_module"].values,
            "code_presentation": filtered["code_presentation"].values,
            "predicted_enrollment": predictions,
        }
    )
    if "enrollment_num_daily" in filtered.columns:
        result["actual_enrollment"] = filtered["enrollment_num_daily"].values
    return result.sort_values("day_offset").reset_index(drop=True)


@st.cache_data
def get_forecast_summary(forecast_df: pd.DataFrame) -> dict[str, float]:
    if forecast_df.empty or "predicted_enrollment" not in forecast_df.columns:
        return {}
    return {
        "mean": float(forecast_df["predicted_enrollment"].mean()),
        "max": float(forecast_df["predicted_enrollment"].max()),
        "min": float(forecast_df["predicted_enrollment"].min()),
        "band_width": 10.4,
    }


@st.cache_data
def get_forecast_confidence_bands(forecast_df: pd.DataFrame, band_width: float = 10.4) -> tuple[pd.Series, pd.Series]:
    lower = forecast_df["predicted_enrollment"] - band_width
    upper = forecast_df["predicted_enrollment"] + band_width
    return lower.clip(lower=0), upper


@st.cache_data
def get_model_evaluation() -> dict[str, float]:
    try:
        return model.evaluate()
    except Exception:
        return {}


# UI event handler
def on_pressed_forecast_button():
    day_offset = st.session_state["day_offset"]
    course_key = st.session_state.get("course_selection")
    if course_key:
        st.session_state.predicted_data = predict_course_history(day_offset, course_key)
    else:
        st.session_state.predicted_data = predict_day_offset(day_offset)


def on_pressed_course_button():
    day_offset = st.session_state["day_offset"]
    course_key = st.session_state.get("course_selection")
    if course_key:
        st.session_state.predicted_data = predict_course_history(day_offset, course_key)
    else:
        st.session_state.predicted_data = predict_day_offset(day_offset)


def on_course_selected():
    day_offset = st.session_state["day_offset"]
    course_key = st.session_state["course_selection"]
    st.session_state.predicted_data = predict_course_history(day_offset, course_key)


# init global variables
if "dataset" not in st.session_state:
    st.session_state.dataset = "test"

if "day_offset" not in st.session_state:
    offsets = get_day_offsets()
    st.session_state.day_offset = offsets[0] if offsets else 0

if "course_selection" not in st.session_state:
    courses = get_courses(st.session_state["dataset"])
    course_options = [f"{m}||{p}" for m, p in courses]
    st.session_state.course_selection = course_options[0] if course_options else None

if "predicted_data" not in st.session_state:
    course_key = st.session_state.get("course_selection")
    if course_key:
        st.session_state.predicted_data = predict_course_history(st.session_state["day_offset"], course_key)
    else:
        st.session_state.predicted_data = predict_day_offset(st.session_state["day_offset"])


# Layout functions
def init_top_layout():
    st.title("Enrollment Forecasting")

    offsets = get_day_offsets()
    min_offset, max_offset = (offsets[0], offsets[-1]) if offsets else (0, 0)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Dataset:** test")
        st.slider(
            "Select day offset",
            min_value=min_offset,
            max_value=max_offset,
            value=st.session_state["day_offset"],
            step=1,
            key="day_offset",
            on_change=on_pressed_forecast_button,
        )
    with col2:
        st.button("Forecast day", on_click=on_pressed_forecast_button)

    st.divider()


def init_forecast_plots():
    dataset = st.session_state["dataset"]
    courses = get_courses(dataset)
    course_options = [f"{m}||{p}" for m, p in courses]

    if course_options:
        st.selectbox(
            "Select course",
            course_options,
            key="course_selection",
            on_change=on_course_selected,
        )
        st.button("Forecast course", on_click=on_pressed_course_button)

    course_key = st.session_state.get("course_selection")
    if course_key:
        train_df = get_train_course_history(course_key)
        if not train_df.empty and "actual_enrollment" in train_df.columns:
            st.subheader("Historical Train Enrollment")
            forecast_df = st.session_state.predicted_data
            summary = get_forecast_summary(forecast_df)
            evaluation = get_model_evaluation()

            fig_train, ax_train = plt.subplots(figsize=(10, 3.5))
            if "predicted_enrollment" in train_df.columns:
                ax_train.bar(
                    train_df["day_offset"],
                    train_df["predicted_enrollment"],
                    label="Predicted Enrollment",
                    alpha=0.5,
                    color="tab:blue",
                )
                lower, upper = get_forecast_confidence_bands(train_df)
                ax_train.fill_between(
                    train_df["day_offset"],
                    lower,
                    upper,
                    color="tab:blue",
                    alpha=0.15,
                    label="95% Confidence Band",
                )
            ax_train.plot(
                train_df["day_offset"],
                train_df["actual_enrollment"],
                label="Actual Enrollment",
                color="tab:orange",
                marker="o",
            )
            ax_train.set_xlabel("Day Offset")
            ax_train.set_ylabel("Enrollment")
            ax_train.set_title("Training History by Course")
            ax_train.legend()
            ax_train.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig_train)

            if summary:
                st.subheader("Forecast Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted Mean", f"{summary['mean']:.1f} students/day")
                col2.metric("Forecast Max", f"{summary['max']:.0f} students")
                col3.metric("Forecast Min", f"{summary['min']:.0f} students")
                col4.metric("95% Band Width", f"±{summary['band_width']:.1f} students")

            if evaluation:
                st.subheader("Model Evaluation")
                col5, col6 = st.columns(2)
                col5.metric("MAE", f"{evaluation.get('mae', 0):.2f}")
                col6.metric("RMSE", f"{evaluation.get('rmse', 0):.2f}")

    forecast_df = st.session_state.predicted_data
    if not isinstance(forecast_df, pd.DataFrame):
        forecast_df = pd.DataFrame(forecast_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False

    if not forecast_df.empty and {"day_offset", "predicted_enrollment"}.issubset(forecast_df.columns):
        lower, upper = get_forecast_confidence_bands(forecast_df)
        ax.fill_between(
            forecast_df["day_offset"],
            lower,
            upper,
            color="tab:blue",
            alpha=0.15,
            label="95% Confidence Band",
        )
        ax.bar(forecast_df["day_offset"], forecast_df["predicted_enrollment"], label="Predicted Enrollment", alpha=0.6, color="tab:blue")
        plotted = True

    if not forecast_df.empty and "actual_enrollment" in forecast_df.columns:
        ax.plot(forecast_df["day_offset"], forecast_df["actual_enrollment"], label="Actual Enrollment", color="tab:orange", marker="o")
        plotted = True

    if not plotted:
        st.info("Run a forecast to see predictions.")
    else:
        ax.set_xlabel("Day Offset")
        ax.set_ylabel("Enrollment")
        ax.set_title("Enrollment Forecast")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
        st.dataframe(forecast_df, use_container_width=True)

        evaluation = get_model_evaluation()
        if evaluation:
            st.subheader("Model Evaluation")
            eval_col1, eval_col2 = st.columns(2)
            eval_col1.metric("MAE", f"{evaluation.get('mae', 0):.2f}")
            eval_col2.metric("RMSE", f"{evaluation.get('rmse', 0):.2f}")


def init_monitoring():
    st.divider()
    st.title("Model Monitoring")

    if Report is None or DataDriftPreset is None:
        st.info("Install evidently to enable model monitoring reports.")
        return

    forecast_df = st.session_state.predicted_data
    if not isinstance(forecast_df, pd.DataFrame):
        forecast_df = pd.DataFrame(forecast_df)

    print(forecast_df)
    if not forecast_df.empty:

        # data drift monitoring
        tmp_df = forecast_df
        actual_df = tmp_df.drop(columns=["predicted_enrollment"])
        predicted_df = tmp_df.drop(columns=["actual_enrollment"])
        actual_df = actual_df.rename(columns={"actual_enrollment" : "enrollment"})
        predicted_df = predicted_df.rename(columns={"predicted_enrollment" : "enrollment"})

        reference_dataset = Dataset.from_pandas(
            actual_df,
        )

        current_dataset = Dataset.from_pandas(
            predicted_df,
        )


        report = Report(metrics=[
            ValueDrift(column="enrollment"),
            ])
        snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

        filename = "report.html"
        snapshot.save_html(filename)
        with open(filename, "r", encoding="utf-8") as f:
            html = f.read()
        st.iframe(html, height=900)


        # regression monitoring
        data_definition = DataDefinition(regression=[
            Regression(
                target="actual_enrollment",
                prediction="predicted_enrollment",
            )
        ])
        current_dataset = Dataset.from_pandas(
            forecast_df,
            data_definition=data_definition,
        )
        report = Report(metrics=[
            RegressionPreset()
        ])
        snapshot = report.run(current_data=current_dataset)

        filename = "report_reg.html"
        snapshot.save_html(filename)
        with open(filename, "r", encoding="utf-8") as f:
            html = f.read()
        st.iframe(html, height=900)

    else:
        st.info("Run forecasting to see monitoring")


init_top_layout()
init_forecast_plots()
init_monitoring()