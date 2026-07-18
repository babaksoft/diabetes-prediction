import streamlit as st

from diabetes_prediction.app.api_client import post_data
from diabetes_prediction.app.diabetes import DiabetesData


@st.cache_data()
def get_prediction(
    mode: str,
    data: DiabetesData,
) -> str:
    # url = f"http://api:8000/predict?mode={mode}"
    url = f"http://localhost:8000/predict?mode={mode}"  # Uncomment for local inference
    payload = data.__dict__
    payload["hypertension"] = float(payload["hypertension"] == "Yes")
    payload["heart_disease"] = float(payload["heart_disease"] == "Yes")
    result = post_data(url, [payload])
    if not result["data"]:
        return result["message"]

    return str(result["data"]["predictions"][0])


def main():
    title_div = """
    <div style ="background-color:green;padding:1px">
        <h3 style ="color:black;text-align:center;">
            Diabetes Prediction App
        </h3>
    </div>
    """

    st.markdown(title_div, unsafe_allow_html=True)
    st.text(
        "Triage : Catches more true diabetics, Balanced : More accurate diabetic warnings"
    )
    mode = st.selectbox("Model :", ("Triage", "Balanced"))
    gender = st.selectbox("Gender :", ("Male", "Female", "Other"))
    age = st.number_input("Age :", 1, 80)
    hypertension = st.selectbox("History of hypertension?", ("No", "Yes"))
    heart_disease = st.selectbox("History of heart disease?", ("No", "Yes"))
    smoking = st.selectbox(
        "Smoking status :",
        ("not current", "former", "No Info", "current", "never", "ever"),
    )
    bmi = st.number_input("Body Mass Index (BMI) :", 10.0, 90.0)
    mean_glucose = st.number_input("Average Blood sugar (past 2-3 months) :", 3.0, 9.0)
    glucose = st.number_input("Blood sugar :", 80, 300)

    if st.button("Predict"):
        data = DiabetesData(
            gender=gender,
            age=age,
            hypertension=hypertension,
            heart_disease=heart_disease,
            smoking_history=smoking,
            bmi=bmi,
            HbA1c_level=mean_glucose,
            blood_glucose_level=glucose,
        )
        result = get_prediction(str(mode).lower(), data)

        if result == "Negative":
            st.success("Congratulations! You do NOT have diabetes.")
        elif result == "Positive":
            st.warning("You MAY have diabetes. Please consult your physician.")
        else:
            st.error(result)


if __name__ == "__main__":
    main()
