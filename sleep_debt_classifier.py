from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("bedtime_screentime_sleep_debt.csv")
TARGET = "sleep_debt_category"
ID_COLUMNS = ["user_id"]
RANDOM_STATE = 42


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def audit_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("DATA SHAPE:", df.shape)
    print("\nMISSING VALUES:")
    print(df.isna().sum())
    print("\nDUPLICATE ROWS:", df.duplicated().sum())
    print("DUPLICATE USER IDS:", df["user_id"].duplicated().sum())
    print("\nTARGET DISTRIBUTION:")
    print(df[TARGET].value_counts())

    sanity_checks = {
        "age_outside_18_100": ~df["age"].between(18, 100),
        "phone_minutes_outside_0_300": ~df["bedtime_phone_minutes"].between(0, 300),
        "brightness_outside_0_100": ~df["screen_brightness_pct"].between(0, 100),
        "blue_filter_not_binary": ~df["blue_light_filter_active"].isin([0, 1]),
        "caffeine_negative": df["caffeine_post_5pm_mg"] < 0,
        "activity_negative": df["physical_activity_min"] < 0,
        "sleep_latency_negative": df["sleep_latency_min"] < 0,
        "sleep_hours_outside_0_24": ~df["total_sleep_hours"].between(0, 24),
        "deep_sleep_pct_outside_0_100": ~df["deep_sleep_pct"].between(0, 100),
        "rem_sleep_pct_outside_0_100": ~df["rem_sleep_pct"].between(0, 100),
        "snoozes_negative": df["morning_alarm_snoozes"] < 0,
        "fatigue_score_outside_1_10": ~df["next_day_fatigue_score"].between(1, 10),
    }
    print("\nSANITY CHECK FAIL COUNTS:")
    for name, mask in sanity_checks.items():
        print(f"{name}: {int(mask.sum())}")

    cleaned = df.drop_duplicates().copy()
    numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
    outlier_rows = []

    for col in numeric_cols:
        lower, upper = iqr_bounds(cleaned[col])
        outliers = ~cleaned[col].between(lower, upper)
        outlier_rows.append(
            {
                "column": col,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(outliers.sum()),
            }
        )
        cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

    outlier_report = pd.DataFrame(outlier_rows)
    print("\nOUTLIERS CAPPED WITH IQR BOUNDS:")
    print(outlier_report.to_string(index=False))

    return cleaned, outlier_report


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    cleaned, outlier_report = audit_and_clean(df)

    cleaned.to_csv("bedtime_sleep_debt_cleaned.csv", index=False)
    outlier_report.to_csv("outlier_report.csv", index=False)

    X = cleaned.drop(columns=[TARGET, *ID_COLUMNS])
    y = cleaned[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"model": name, "test_accuracy": accuracy_score(y_test, pred)})

    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    print("\nMODEL COMPARISON:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    if best_model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 8, 14],
            "model__min_samples_leaf": [1, 3],
        }
    elif best_model_name == "Gradient Boosting":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    else:
        param_grid = {
            "model__C": [0.1, 1, 10],
            "model__solver": ["lbfgs"],
        }

    tuned_pipeline = Pipeline(
        steps=[("preprocess", build_preprocessor(X_train)), ("model", models[best_model_name])]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        tuned_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)

    print("\nBEST MODEL:", best_model_name)
    print("BEST CV ACCURACY:", round(grid.best_score_, 4))
    print("BEST PARAMETERS:", grid.best_params_)
    print("TEST ACCURACY:", round(test_accuracy, 4))
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, test_pred))
    print("CONFUSION MATRIX:")
    print(pd.DataFrame(confusion_matrix(y_test, test_pred), index=grid.classes_, columns=grid.classes_))

    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        feature_importance = pd.Series(
            best_pipeline.named_steps["model"].feature_importances_,
            index=best_pipeline.named_steps["preprocess"].get_feature_names_out(),
        ).sort_values(ascending=False)
        feature_importance.to_csv("feature_importance.csv", header=["importance"])
        print("\nTOP FEATURE IMPORTANCES:")
        print(feature_importance.head(15).to_string())

    joblib.dump(best_pipeline, "best_sleep_debt_classifier.joblib")
    results_df.to_csv("model_comparison_results.csv", index=False)


if __name__ == "__main__":
    main()
