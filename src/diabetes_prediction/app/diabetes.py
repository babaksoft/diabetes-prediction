from dataclasses import dataclass


@dataclass
class DiabetesData:
    gender: str
    age: int
    hypertension: str
    heart_disease: str
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
