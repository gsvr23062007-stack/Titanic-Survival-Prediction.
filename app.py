
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Default model features
FEATURES_DEFAULT = ["pclass", "sex", "age", "fare"]


# Load Machine Learning model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


# Application title
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger details to predict the estimated "
    "chance of survival using Machine Learning."
)

# Passenger Information
st.subheader("Passenger Information")

pclass = st.selectbox(
    "Ticket Class",
    [1, 2, 3]
)

sex = st.radio(
    "Gender",
    ["Female", "Male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    min_value=1,
    max_value=80,
    value=30
)

fare = st.slider(
    "Ticket Fare",
    min_value=0,
    max_value=500,
    value=30
)

# Prediction button
if st.button("Predict Survival", type="primary"):

    try:
        # Load model
        model = load_model()

        # Convert gender into numerical value
        sex_value = 1 if sex == "Female" else 0

        # Prepare passenger details
        values = {
            "Pclass": pclass,
            "pclass": pclass,
            "Sex": sex_value,
            "sex": sex_value,
            "Age": age,
            "age": age,
            "Fare": fare,
            "fare": fare
        }

        # Get model feature names
        features = list(
            getattr(
                model,
                "feature_names_in_",
                FEATURES_DEFAULT
            )
        )

        # Check feature names
        if not all(f in values for f in features):
            raise ValueError(
                "Model contains unsupported feature names."
            )

        # Create input DataFrame
        input_data = pd.DataFrame(
            [[values[f] for f in features]],
            columns=features
        )

        # Predict survival
        prediction = model.predict(input_data)[0]

        # Display prediction
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("The model predicts: Survived")
        else:
            st.warning("The model predicts: Not Survived")

        # Survival probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                survival_probability = probabilities[
                    classes.index(1)
                ]

                st.metric(
                    "Estimated Survival Probability",
                    f"{survival_probability:.0%}"
                )

    except FileNotFoundError:
        st.error(
            "Model file is missing. Please save model.joblib first."
        )

    except Exception as e:
        st.error("Prediction could not be completed.")
        st.write(str(e))

st.caption(
    "An educational Machine Learning project using the historical "
    "Titanic dataset. Predictions are estimates, not real-world advice."
)
