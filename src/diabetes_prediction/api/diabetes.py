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

    def as_data_point(self):
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
