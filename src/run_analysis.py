"""Run the Student Performance Capstone analysis from the repository root."""
from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
REPORTS_DIR = ROOT / "reports"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def extract_student_mat(zip_path: Path, csv_path: Path) -> None:
    """Copy student-mat.csv out of the UCI archive.

    The current official download is a zip whose only data member is student.zip.
    The mathematics CSV is inside that inner archive. Older mirrors place the CSV
    at the top level, so both layouts are accepted.
    """
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        direct = next((name for name in names if Path(name).name == "student-mat.csv"), None)
        if direct is not None:
            with archive.open(direct) as source, csv_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            return
        nested_name = next((name for name in names if Path(name).name == "student.zip"), None)
        if nested_name is None:
            raise FileNotFoundError(
                f"student-mat.csv was not found in {zip_path.name}. Archive members: {names}"
            )
        nested_bytes = io.BytesIO(archive.read(nested_name))
    with zipfile.ZipFile(nested_bytes) as nested:
        member = next(name for name in nested.namelist() if Path(name).name == "student-mat.csv")
        with nested.open(member) as source, csv_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def load_math_data() -> pd.DataFrame:
    """Read local UCI math CSV, downloading the official ZIP if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "student-mat.csv"
    if not csv_path.exists():
        zip_path = DATA_DIR / "student-performance.zip"
        try:
            urllib.request.urlretrieve(UCI_ARCHIVE_URL, zip_path)
            extract_student_mat(zip_path, csv_path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the UCI archive. Download student-mat.csv from "
                "https://archive.ics.uci.edu/dataset/320/student+performance and put it in data/."
            ) from exc
    return pd.read_csv(csv_path, sep=";")


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and rows without a supervised target."""
    frame = raw.copy()
    if "G3" not in frame.columns:
        raise ValueError("Expected target column 'G3' was not found in the dataset.")
    frame = frame.drop_duplicates()
    frame = frame.dropna(subset=["G3"])
    frame["G3"] = pd.to_numeric(frame["G3"], errors="coerce")
    frame = frame.dropna(subset=["G3"])
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add compact derived features without using target G3."""
    out = frame.copy()
    parent_cols = [column for column in ["Medu", "Fedu"] if column in out.columns]
    if parent_cols:
        out["parent_education_avg"] = out[parent_cols].mean(axis=1)
    if "absences" in out.columns:
        out["absence_band"] = pd.cut(
            out["absences"],
            bins=[-1, 0, 5, 10, np.inf],
            labels=["0", "1-5", "6-10", ">10"],
        ).astype("object")
    return out


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build train-fold-safe imputing, scaling and one-hot-encoding steps."""
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def make_pipeline_for(features: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("regressor", estimator),
    ])


def fit_compare(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Cross-validate candidate regressors and test both timing scenarios."""
    y = frame["G3"].astype(float)
    all_features = frame.drop(columns=["G3"])
    early_features = all_features.drop(columns=[col for col in ["G1", "G2"] if col in all_features.columns])
    scenarios = {
        "Earlier support (no G1/G2)": early_features,
        "Later in year (with G1/G2)": all_features,
    }
    candidates = {
        "Mean baseline": (DummyRegressor(strategy="mean"), {}),
        "Linear Regression": (LinearRegression(), None),
        "Ridge Regression": (Ridge(), {"regressor__alpha": [0.01, 0.1, 1, 10, 100, 500]}),
        "Lasso Regression": (Lasso(max_iter=30000), {"regressor__alpha": [0.001, 0.01, 0.1, 0.5, 1.0]}),
        "KNN Regression": (KNeighborsRegressor(), {
            "regressor__n_neighbors": [3, 5, 7, 9, 13, 17, 21],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],
        }),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted, splits = [], {}, {}
    for scenario_name, X in scenarios.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        splits[scenario_name] = (X_train, X_test, y_train, y_test)
        for model_name, (estimator, parameter_grid) in candidates.items():
            pipeline = make_pipeline_for(X_train, estimator)
            if parameter_grid:
                search = GridSearchCV(
                    pipeline, parameter_grid, cv=cv,
                    scoring="neg_root_mean_squared_error", n_jobs=-1,
                    refit=True, error_score="raise",
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                cv_rmse = float(-search.best_score_)
                std_rmse = float(search.cv_results_["std_test_score"][search.best_index_])
                params = search.best_params_
            else:
                cv_result = cross_validate(
                    pipeline, X_train, y_train, cv=cv,
                    scoring="neg_root_mean_squared_error", n_jobs=-1,
                )
                cv_rmse_scores = -cv_result["test_score"]
                cv_rmse = float(cv_rmse_scores.mean())
                std_rmse = float(cv_rmse_scores.std(ddof=1))
                best_model = pipeline.fit(X_train, y_train)
                params = {}
            prediction = best_model.predict(X_test)
            row = {
                "Scenario": scenario_name,
                "Model": model_name,
                "CV_RMSE_Mean": cv_rmse,
                "CV_RMSE_SD": std_rmse,
                "Test_RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "Test_MAE": float(mean_absolute_error(y_test, prediction)),
                "Test_R2": float(r2_score(y_test, prediction)),
                "Best_Params": str(params),
            }
            rows.append(row)
            fitted[(scenario_name, model_name)] = best_model
    return pd.DataFrame(rows).sort_values("CV_RMSE_Mean"), fitted, splits


def make_eda_plots(frame: pd.DataFrame) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(frame["G3"], bins=20, color=sns.color_palette("deep")[0], ax=ax)
    ax.set(title="Distribution of Final Mathematics Grades", xlabel="Final grade (G3; 0–20)", ylabel="Number of students")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "grade_distribution.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(frame["age"], discrete=True, color=sns.color_palette("deep")[0], ax=axes[0])
    axes[0].set(title="Student Age Distribution", xlabel="Age (years)", ylabel="Number of students")
    sns.boxplot(data=frame, x="studytime", y="G3", ax=axes[1])
    axes[1].set(title="Final Grade by Weekly Study-Time Category", xlabel="Study time category (1 = lowest, 4 = highest)", ylabel="Final grade (G3)")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "age_and_studytime.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=frame, x="absences", y="G3", alpha=0.65, ax=ax)
    ax.set(title="Absences and Final Grade", xlabel="Recorded absences", ylabel="Final grade (G3)")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "absences_vs_final_grade.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=frame, x="failures", y="G3", ax=ax)
    ax.set(title="Final Grade by Number of Past Class Failures", xlabel="Past class failures", ylabel="Final grade (G3)")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "grade_by_failures.png", dpi=160); plt.close(fig)


def write_summary(readme_results: str, results: pd.DataFrame, importance: pd.DataFrame) -> str:
    top = results.sort_values("CV_RMSE_Mean").iloc[0]
    display_cols = ["Scenario", "Model", "CV_RMSE_Mean", "Test_RMSE", "Test_MAE", "Test_R2"]
    table = results[display_cols].head(8).round(3).to_markdown(index=False)
    top_features = importance.head(5)
    feature_lines = "\n".join(
        f"- `{row.feature}` (permutation importance: {row.importance:.4f})"
        for row in top_features.itertuples(index=False)
    )
    return (
        f"**Best model by 5-fold cross-validation RMSE:** {top['Model']} — {top['Scenario']}.  \n"
        f"Held-out test results: RMSE **{top['Test_RMSE']:.3f}**, MAE **{top['Test_MAE']:.3f}**, R² **{top['Test_R2']:.3f}**.  \n\n"
        "RMSE and MAE are expressed in grade points (lower is better); R² measures the proportion of grade variation explained on the held-out test set. "
        "The model was selected using cross-validation on the training data, not by optimizing against the test set.\n\n"
        "**Top predictive signals by permutation importance on the test set** (these are associations, not causal effects):\n\n"
        f"{feature_lines}\n\n**Leading cross-validation results:**\n\n{table}\n\n"
        "**Practical interpretation:** Use the earlier-support scenario for questions that need an answer before prior-period grades exist. "
        "The later-in-year scenario can be more predictive because it uses G1/G2, but that does not make it suitable for early intervention. "
        "Validate on recent local data and involve educators before any operational use."
    )


def main() -> None:
    for folder in [IMAGES_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    raw = load_math_data()
    frame = engineer_features(clean_data(raw))
    make_eda_plots(frame)
    results, fitted, splits = fit_compare(frame)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best = results.iloc[0]
    scenario = best["Scenario"]
    model_name = best["Model"]
    model = fitted[(scenario, model_name)]
    X_train, X_test, y_train, y_test = splits[scenario]
    permutation = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    importance = pd.DataFrame({"feature": X_test.columns, "importance": permutation.importances_mean})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    y_pred = model.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=axes[0])
    low, high = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([low, high], [low, high], linestyle="--")
    axes[0].set(title="Actual vs Predicted Grades", xlabel="Actual final grade", ylabel="Predicted final grade")
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, ax=axes[1])
    axes[1].axhline(0, linestyle="--")
    axes[1].set(title="Residuals vs Predicted Grade", xlabel="Predicted final grade", ylabel="Actual − predicted")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "prediction_diagnostics.png", dpi=160); plt.close(fig)

    summary = write_summary("", results, importance.rename(columns={"feature":"feature", "importance":"importance"}))
    (REPORTS_DIR / "results_summary.md").write_text(summary, encoding="utf-8")
    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    start, end = "<!-- AUTO_RESULTS_START -->", "<!-- AUTO_RESULTS_END -->"
    if start in text and end in text:
        left, rest = text.split(start, 1)
        _, right = rest.split(end, 1)
        text = left + start + "\n" + summary + "\n" + end + right
        readme_path.write_text(text, encoding="utf-8")
    print("\nModel comparison (sorted by training cross-validation RMSE):")
    print(results[["Scenario", "Model", "CV_RMSE_Mean", "Test_RMSE", "Test_MAE", "Test_R2"]].round(3).to_string(index=False))
    print(f"\nSelected: {model_name} | {scenario}")
    print(f"Saved results to {REPORTS_DIR} and charts to {IMAGES_DIR}.")


if __name__ == "__main__":
    main()
