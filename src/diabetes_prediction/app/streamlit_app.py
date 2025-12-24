import streamlit as st

from api_client import post_data

@st.cache_data()
def get_prediction(
        gender: str,
        age: int,
        hypertension: str,
        heart_disease: str,
        smoking: str,
        bmi: float,
        mean_glucose: float,
        glucose: float
):
    url = "http://127.0.0.1:8000/predict"
    diabetes = {
        "Gender": gender,
        "Age": age,
        "Hypertension": 1 if hypertension == "yes" else 0,
        "HeartDisease": 1 if heart_disease == "Yes" else 0,
        "SmokingHistory": smoking,
        "BMI": bmi,
        "MeanGlucoseLevel": mean_glucose,
        "GlucoseLevel": glucose
    }
    result = post_data(url, diabetes)
    if not result["data"]:
        return result["message"]

    prediction = result["data"]["prediction"] if result["data"] else None
    return prediction


def main():
    title_div = """
    <div style ="background-color:green;padding:1px">
        <h3 style ="color:black;text-align:center;">
            Diabetes Prediction App
        </h3>
    </div>
    """

    st.markdown(title_div, unsafe_allow_html = True)
    gender = st.selectbox("Gender :", ("Male", "Female", "Other"))
    age = st.number_input("Age :", 1, 80)
    hypertension = st.selectbox("History of hypertension?", ("No", "Yes"))
    heart_disease = st.selectbox("History of heart disease?", ("No", "Yes"))
    smoking = st.selectbox("Smoking status :", (
        "not current", "former", "No Info", "current", "never", "ever"
    ))
    bmi = st.number_input("Body Mass Index (BMI) :", 10.0, 90.0)
    mean_glucose = st.number_input(
        "Average Blood sugar (past 2-3 months) :", 3.0, 9.0
    )
    glucose = st.number_input("Blood sugar :", 80, 300)

    if st.button("Predict"):
        result = get_prediction(
            gender, age, hypertension, heart_disease,
            smoking, bmi, mean_glucose, glucose
        )

        pos_msg = "Congratulations! You do NOT have diabetes."
        neg_msg = "You MAY have diabetes. Please consult your physician."
        if result == "No Diabetes":
            st.success(pos_msg)
        elif result == "Diabetes":
            st.warning(neg_msg)
        else:
            st.error(result)


if __name__=='__main__':
    main()
