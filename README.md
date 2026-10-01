# Sleep Debt Classifier

This project trains a classifier for `sleep_debt_category` from bedtime, screen-time, and sleep-related inputs, then serves the saved model through a Streamlit app.

## Files

- `sleep_debt_classifier.py`: data audit, preprocessing, outlier treatment, model training, evaluation, and model export.
- `app.py`: Streamlit front end for entering user inputs and predicting the sleep debt category.
- `best_sleep_debt_classifier.joblib`: trained model pipeline used by the app.
- `bedtime_screentime_sleep_debt.csv`: original dataset.
- `bedtime_sleep_debt_cleaned.csv`: cleaned dataset after preprocessing checks and IQR outlier capping.
- `outlier_report.csv`: outlier counts and bounds.
- `model_comparison_results.csv`: tested model accuracies.
- `feature_importance.csv`: top model feature importances.

## Run Training

```bash
python sleep_debt_classifier.py
```

## Run App

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Note

The highest-accuracy model uses sleep outcome fields such as `total_sleep_hours` and `next_day_fatigue_score`, which strongly determine the target category. For prediction before sleep outcomes are known, remove those outcome fields and retrain for a more realistic model.
