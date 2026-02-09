# Diabetes Prediction

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://github.com/babaksoft/diabetes-prediction/raw/refs/heads/master/pyproject.toml)
![Static Badge](https://img.shields.io/badge/task-classification-orange)
![Static Badge](https://img.shields.io/badge/framework-sklearn-orange)
![GitHub License](https://img.shields.io/github/license/babaksoft/diabetes-prediction)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/babaksoft/diabetes-prediction/build.yml)


## Problem framing
Using historical data on demographic information and health indicators for diabetic and
healthy individuals, we need a predictive model to use in medical triage context.

## Data
Our dataset has 100,000 labelled instances with the following features :

| Feature              | Description                                                     |
|----------------------|-----------------------------------------------------------------|
| gender               | Biological sex (Male/Female/Other)                              |
| age                  | Person's Age                                                    |
| hypertension         | History of hypertension? (0: No, 1: Yes)                        |
| heart_disease        | History of heart disease? (0: No, 1: Yes)                       |
| smoking_history      | Smoking history (not current/former/No Info/current/never/ever) |
| bmi                  | Body Mass Index (BMI)                                           |
| HbA1c_level          | Average blood sugar level over the past 2-3 months              |
| blood_glucose_level  | Blood sugar level                                               |
| diabetes             | Does this person have diabetes? (0: No, 1: Yes)                 |

Source: [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

## Imbalance challenge
With less than 10% positive instances, our dataset suffers severe class imbalance, which is
typical in medical diagnosis datasets. We need to apply a balanced model training approach
to account for very few positive instances our final model should learn from.

## Modeling strategy
In a medical context, we need to put strong emphasis on detecting as much positive cases
(i.e. diabetics) as possible. This calls for maximizing Recall, while keeping overall
performance in an acceptable level.

To strike a good balance between the **clinical cost of a False Negative** and the
**operational cost of a False Positive**, we will aim for the following two models :
- A high recall triage model that can be used during early screening of patients
- A more balanced model that can be used for follow-up tests
