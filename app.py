from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("best_sleep_debt_classifier.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Sleep Debt Classifier", page_icon="🌙", layout="centered")

st.title("Sleep Debt Classifier")
st.write("Enter bedtime and sleep details to predict the sleep debt category.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python sleep_debt_classifier.py` first.")
    st.stop()

model = load_model()

with st.form("sleep_debt_form"):
    st.subheader("Profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 32)
        gender = st.selectbox("Gender", ["Female", "Male", "Non-Binary"])
        occupation_type = st.selectbox(
            "Occupation type",
            [
                "Corporate 9-to-5",
                "Freelance / Creative",
                "Healthcare / Shift Worker",
                "Remote Tech",
                "Student",
            ],
        )
    with col2:
        chronotype = st.selectbox(
            "Chronotype", ["Intermediate", "Morning Lark", "Night Owl"]
        )
        primary_bedtime_app = st.selectbox(
            "Primary bedtime app",
            [
                "Instagram / Reddit",
                "Messaging / Chat",
                "News / Reading",
                "Streaming (Netflix/Hulu)",
                "TikTok / Reels",
                "YouTube",
            ],
        )
        blue_light_filter_active = st.toggle("Blue light filter active", value=False)

    st.subheader("Bedtime Habits")
    col1, col2 = st.columns(2)
    with col1:
        bedtime_phone_minutes = st.slider("Bedtime phone minutes", 0, 180, 51)
        screen_brightness_pct = st.slider("Screen brightness %", 0, 100, 55)
        caffeine_post_5pm_mg = st.slider("Caffeine after 5 PM, mg", 0, 250, 0)
    with col2:
        physical_activity_min = st.slider("Physical activity minutes", 0, 120, 35)
        sleep_latency_min = st.slider("Sleep latency minutes", 0.0, 125.0, 37.6, 0.1)
        morning_alarm_snoozes = st.slider("Morning alarm snoozes", 0, 7, 3)

    st.subheader("Sleep Outcomes")
    col1, col2 = st.columns(2)
    with col1:
        total_sleep_hours = st.slider("Total sleep hours", 0.0, 12.0, 6.3, 0.1)
        deep_sleep_pct = st.slider("Deep sleep %", 0.0, 40.0, 21.9, 0.1)
    with col2:
        rem_sleep_pct = st.slider("REM sleep %", 0.0, 40.0, 19.1, 0.1)
        next_day_fatigue_score = st.slider("Next-day fatigue score", 1.0, 10.0, 3.2, 0.1)

    submitted = st.form_submit_button("Predict Sleep Debt")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "occupation_type": occupation_type,
                "chronotype": chronotype,
                "bedtime_phone_minutes": bedtime_phone_minutes,
                "primary_bedtime_app": primary_bedtime_app,
                "screen_brightness_pct": screen_brightness_pct,
                "blue_light_filter_active": int(blue_light_filter_active),
                "caffeine_post_5pm_mg": caffeine_post_5pm_mg,
                "physical_activity_min": physical_activity_min,
                "sleep_latency_min": sleep_latency_min,
                "total_sleep_hours": total_sleep_hours,
                "deep_sleep_pct": deep_sleep_pct,
                "rem_sleep_pct": rem_sleep_pct,
                "morning_alarm_snoozes": morning_alarm_snoozes,
                "next_day_fatigue_score": next_day_fatigue_score,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted sleep debt category: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = pd.DataFrame(
            {
                "Category": model.classes_,
                "Probability": model.predict_proba(input_data)[0],
            }
        ).sort_values("Probability", ascending=False)
        st.bar_chart(probabilities.set_index("Category"))
        st.dataframe(
            probabilities.assign(
                Probability=lambda df: (df["Probability"] * 100).round(2).astype(str) + "%"
            ),
            hide_index=True,
            use_container_width=True,
        )
