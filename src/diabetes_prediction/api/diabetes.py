from enum import Enum

from pydantic import BaseModel, Field


class Gender(str, Enum):
    FEMALE = "Female"
    MALE = "Male"
    OTHER = "Other"


class SmokingHistory(str, Enum):
    NOT_CURRENT = "not current"
    FORMER = "former"
    NO_INFO = "No Info"
    CURRENT = "current"
    NEVER = "never"
    EVER = "ever"


class Diabetes(BaseModel):
    gender: Gender = Field(description="Biological sex")
    age: float = Field(description="Patient's age", ge=1.0, le=80.0)
    hypertension: float = Field(
        description="History of hypertension? (0: No, 1: Yes)",
        ge=0.0,
        le=1.0,
    )
    heart_disease: float = Field(
        description="History of heart disease? (0: No, 1: Yes)",
        ge=0.0,
        le=1.0,
    )
    smoking_history: SmokingHistory = Field("Patient's smoking history")
    bmi: float = Field(
        description="Patient's Body Mass Index (BMI)",
        ge=10.0,
        le=90.0,
    )
    HbA1c_level: float = Field(
        description="Average blood sugar level over the past 2-3 months",
        ge=3.0,
        le=9.0,
    )
    blood_glucose_level: float = Field(
        description="Blood sugar level at measurement time",
        ge=80.0,
        le=300.0,
    )
