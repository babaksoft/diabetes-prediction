import json
from pathlib import Path

import joblib
import pandas as pd


def model_fn(model_dir):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "thresholds.json", "r") as file:
        thresholds = json.load(file)

    return {
        "model": model,
        "thresholds": thresholds,
    }


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data_dict = json.loads(request_body)
        input_data = pd.DataFrame(data_dict)
        return input_data

    raise ValueError(
        f"Content type '{request_content_type}' is not supported."
        f"Please contact API provider for more information."
    )


def predict_fn(input_data, model_bundle):
    try:
        model = model_bundle["model"]
        thresholds = model_bundle["thresholds"]

        # Last feature must be : 'mode': 'triage' or 'mode': 'balanced'
        mode = input_data.pop("mode").iloc[0]
        threshold = thresholds[mode]

        probabilities = model.predict_proba(input_data)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        rounded_probs = [round(float(prob), 6) for prob in probabilities.ravel()]
        return {
            "mode": mode,
            "predictions": predictions.ravel().tolist(),
            "probabilities": rounded_probs,
            "threshold": threshold,
        }
    except KeyError:
        raise ValueError(
            "Last feature must be : 'mode': 'triage' or 'mode': 'balanced'"
        )


def output_fn(predictions, accept):
    return json.dumps(predictions, sort_keys=False, indent=2), accept


# Must be run as script from aws folder
def smoke_test():
    df = pd.read_csv("../data/prepared/test.csv")
    data = df.drop("diabetes", axis=1).iloc[25:30, :]
    data["mode"] = "triage"
    bundle = {
        "model": joblib.load("../model/hgb_pipeline.joblib"),
        "thresholds": {"triage": 0.3631, "balanced": 0.8871},
    }
    output = predict_fn(data, bundle)
    preds = output["predictions"]
    trues = df["diabetes"].iloc[25:30].values
    result = []
    for pred, true in zip(preds, trues):
        truth = "Diabetes" if true == 1 else "No Diabetes"
        prediction = "Diabetes" if pred == 1 else "No Diabetes"
        verdict = "Correct" if prediction == truth else "Incorrect"
        result.append(f"{prediction}\t({verdict})")
    print("\n".join(result))


if __name__ == "__main__":
    smoke_test()
