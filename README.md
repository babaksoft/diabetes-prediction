# Diabetes Prediction

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://github.com/babaksoft/diabetes-prediction/raw/refs/heads/master/pyproject.toml)
![Static Badge](https://img.shields.io/badge/task-classification-orange)
![Static Badge](https://img.shields.io/badge/framework-sklearn-orange)
![GitHub License](https://img.shields.io/github/license/babaksoft/diabetes-prediction)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/babaksoft/diabetes-prediction/build.yml)


### Problem
This is a classification problem where the model predicts whether a person has diabetes or not,
based on given demographic and health information.

### Data
The data corresponds to demographic and health information.

| Variables           | Description                                                     |
|---------------------|-----------------------------------------------------------------|
| gender              | Biological sex (Male/Female/Other)                              |
| age                 | Person's Age                                                    |
| hypertension        | History of hypertension? (Y/N)                                  |
| heart_disease       | History of heart disease? (Y/N)                                 |
| smoking_history     | Smoking history (not current/former/No Info/current/never/ever) |
| bmi                 | Body Mass Index (BMI)                                           |
| HbA1c_level         | Average blood sugar level over the past 2-3 months              |
| blood_glucose_level | Blood sugar level                                               |
| diabetes            | Does this person have diabetes? (0: No, 1: Yes)                 |

Source: Kaggle

## Model requirements

### Performance
Since our business problem is related to healthcare and medical diagnosis,
we would ideally like to miss as few Diabetes (positive) cases as possible.
Therefore, we're aiming for maximum Recall, without sacrificing the overall
model performance.

### Interpretability
Considering the fact that healthcare is a very sensitive area, we'll try to
put special emphasis on model interpretability, to cover Responsible AI
requirements generally associated with Healthcare and Finance domains.

## Model performance

|            | Train  | Test   |
|------------|--------|--------|
| Recall     | 0.9607 | 0.9582 |
| Precision  | 0.1924 | 0.1912 |
| F1 Score   | 0.3205 | 0.3188 |

### Confusion Matrix
![Confusion Matrix](./src/diabetes_prediction/metrics/CM.png)

### ROC Curve (with AUC score)
![ROC-AUC Curve](./src/diabetes_prediction/metrics/ROC.png)

### Precision-Recall (PR) curve
![PR Curve](./src/diabetes_prediction/metrics/PR.png)
