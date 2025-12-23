import pandas as pd
from pydantic import BaseModel


class Diabetes(BaseModel):
    Gender: str
    Age: float
    Hypertension: float
    HeartDisease: float
    SmokingHistory: str
    BMI: float
    MeanGlucoseLevel: float
    GlucoseLevel: float

    def as_dataframe(self):
        data = pd.Series({
            "gender": self.Gender,
            "age": self.Age,
            "hypertension": self.Hypertension,
            "heart_disease": self.HeartDisease,
            "smoking_history": self.SmokingHistory,
            "bmi": self.BMI,
            "HbA1c_level": self.MeanGlucoseLevel,
            "blood_glucose_level": self.GlucoseLevel,
        })
        return pd.DataFrame([data])

    @classmethod
    def from_data(
        cls,
        gender: str,
        age: int,
        hypertension: str,
        heart_disease: str,
        smoking: str,
        bmi: float,
        mean_glucose: float,
        glucose: float
    ):
        hypertension = 1 if hypertension == "Yes" else 0
        heart_disease = 1 if heart_disease == "Yes" else 0
        return Diabetes(
            Gender=gender,
            Age=age,
            Hypertension=hypertension,
            HeartDisease=heart_disease,
            SmokingHistory=smoking,
            BMI=bmi,
            MeanGlucoseLevel=mean_glucose,
            GlucoseLevel=glucose
        )
