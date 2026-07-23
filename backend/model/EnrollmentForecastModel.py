from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnrollmentForecastModel:
    """Lightweight inference wrapper for a pre-trained enrollment forecast model."""

    def __init__(
        self,
        model_path: Union[str, Path],
        test_feature_day_split_dir: Union[str, Path],
    ) -> None:
        """Load the trained model and engineered test feature files.

        Args:
            model_path: Path to the serialized LightGBM model.
            test_feature_day_split_dir: Path to the directory containing per-day test CSVs.

        Raises:
            TypeError: If any path argument is not a string or Path.
            FileNotFoundError: If the model file or dataset directory cannot be found.
            RuntimeError: If loading fails.
        """
        self.model_path = self._resolve_path(model_path, "model_path")
        self.test_feature_day_split_dir = self._resolve_path(
            test_feature_day_split_dir,
            "test_feature_day_split_dir",
            must_be_file=False,
        )

        self.model = self._load_model(self.model_path)
        self.expected_feature_names = self._extract_expected_feature_names(self.model)
        self.train_df = pd.DataFrame()
        self.test_df = self._load_feature_day_splits(self.test_feature_day_split_dir)

    @staticmethod
    def _resolve_path(path: Union[str, Path], name: str, must_be_file: bool = True) -> Path:
        if not isinstance(path, (str, Path)):
            raise TypeError(f"{name} must be a string or pathlib.Path")
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        if must_be_file and not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        if not must_be_file and not resolved.is_dir():
            raise FileNotFoundError(f"Directory not found: {resolved}")
        return resolved

    @staticmethod
    def _load_model(path: Path) -> Any:
        try:
            return joblib.load(path)
        except Exception as exc:
            logger.exception("Failed to load model from %s", path)
            raise RuntimeError(f"Failed to load model from {path}") from exc

    @staticmethod
    def _load_feature_table(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.exception("Failed to load feature table from %s", path)
            raise RuntimeError(f"Failed to load feature table from {path}") from exc
        if df.empty:
            raise ValueError(f"Loaded feature table is empty: {path}")
        return df

    def _load_feature_day_splits(self, path: Path) -> pd.DataFrame:
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in test feature directory: {path}")

        frames = [self._load_feature_table(csv_path) for csv_path in csv_files]
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            raise ValueError(f"Combined feature table is empty: {path}")
        return combined

    @staticmethod
    def _extract_expected_feature_names(model: Any) -> List[str]:
        if hasattr(model, "feature_name_"):
            feature_names = list(getattr(model, "feature_name_") or [])
        elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
            feature_names = list(model.booster_.feature_name())
        else:
            feature_names = []

        if not feature_names:
            raise RuntimeError("Unable to extract expected feature names from the model")
        return feature_names

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a dataframe for prediction by aligning to the model schema.

        Args:
            df: Dataframe containing pre-engineered features.

        Returns:
            Dataframe reindexed to model feature names.

        Raises:
            TypeError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        features = df.copy()
        if "enrollment_num_daily" in features.columns:
            features = features.drop(columns=["enrollment_num_daily"])
        if "date" in features.columns:
            features = features.drop(columns=["date"])

        categorical_columns = features.select_dtypes(include=["object", "category"]).columns
        if len(categorical_columns) > 0:
            features = pd.get_dummies(features, drop_first=True)

        features = features.reindex(columns=self.expected_feature_names, fill_value=0)
        return features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict enrollment from a prepared or raw engineered dataframe.

        Args:
            df: Dataframe containing engineered features.

        Returns:
            Numpy array of predictions.
        """
        X = self.prepare_features(df)
        try:
            predictions = self.model.predict(X)
        except Exception as exc:
            logger.exception("Model prediction failed")
            raise RuntimeError("Model prediction failed") from exc
        return np.asarray(predictions)

    def _build_prediction_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "day_offset": df["day_offset"].values,
                "code_module": df["code_module"].values,
                "code_presentation": df["code_presentation"].values,
                "predicted_enrollment": self.predict(df),
            }
        )
        if "enrollment_num_daily" in df.columns:
            result["actual_enrollment"] = df["enrollment_num_daily"].values
        return result

    def predict_test(self) -> pd.DataFrame:
        """Predict enrollment for the entire test feature table."""
        return self._build_prediction_frame(self.test_df)

    def predict_train(self) -> pd.DataFrame:
        """Raise if train data is not available in inference-only mode."""
        raise ValueError("Train dataset is not available in inference-only mode")

    def predict_course(
        self,
        code_module: str,
        code_presentation: str,
        dataset: str = "test",
    ) -> pd.DataFrame:
        """Predict enrollment for a single course presentation from the selected dataset."""
        if dataset != "test":
            raise ValueError("dataset must be 'test'")

        source = self.test_df
        required_columns = {"code_module", "code_presentation", "day_offset"}
        missing = required_columns.difference(source.columns)
        if missing:
            raise ValueError(f"Selected dataset missing required columns: {sorted(missing)}")

        filtered = source[
            (source["code_module"] == code_module)
            & (source["code_presentation"] == code_presentation)
        ].copy()
        if filtered.empty:
            raise ValueError(
                f"No rows found for code_module={code_module} and code_presentation={code_presentation} in {dataset} set"
            )

        predictions = self.predict(filtered)
        result = pd.DataFrame(
            {
                "day_offset": filtered["day_offset"].values,
                "predicted_enrollment": predictions,
            }
        )
        if "enrollment_num_daily" in filtered.columns:
            result["actual_enrollment"] = filtered["enrollment_num_daily"].values
        return result.sort_values("day_offset").reset_index(drop=True)

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the test feature table."""
        if "enrollment_num_daily" not in self.test_df.columns:
            raise ValueError("Test feature table must contain enrollment_num_daily to evaluate")

        y_true = self.test_df["enrollment_num_daily"].astype(float).to_numpy()
        y_pred = self.predict(self.test_df)

        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
        except ImportError as exc:
            raise ImportError("scikit-learn is required for evaluation") from exc

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return {"mae": float(mae), "rmse": float(rmse)}

    def get_available_courses(self, dataset: str = "test") -> List[Tuple[str, str]]:
        """Return available course presentation pairs from the selected dataset."""
        if dataset != "test":
            raise ValueError("dataset must be 'test'")

        source = self.test_df
        required_columns = {"code_module", "code_presentation"}
        missing = required_columns.difference(source.columns)
        if missing:
            raise ValueError(f"Selected dataset missing required columns: {sorted(missing)}")

        unique_courses = (
            source[["code_module", "code_presentation"]]
            .drop_duplicates()
            .sort_values(["code_module", "code_presentation"])
        )
        return [tuple(row) for row in unique_courses.to_numpy(dtype=str)]
