import json

import boto3
import pandas as pd

import config

sm_runtime = boto3.client("sagemaker-runtime")


def get_endpoint_name():
    with open("artifacts/deployment.json", "r") as file:
        endpoint_info = json.load(file)
    return endpoint_info["endpoint_name"]


def predict(input_data, mode):
    data = input_data.copy()
    data["mode"] = mode
    items = [json.loads(data.iloc[idx].to_json()) for idx in range(len(data.index))]
    data = json.dumps(items).encode("utf-8")

    endpoint_name = get_endpoint_name()
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name, Body=data, ContentType="application/json"
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def show_predictions(trues, output):
    print("\nGround truth labels :", trues)
    print("\nPredictions :")
    preds = output["predictions"]
    probs = output["probabilities"]
    result = []
    pos_count = 0
    correct_count = 0
    for pred, prob, true in zip(preds, probs, trues):
        truth = "Diabetes" if true == 1 else "No Diabetes"
        prediction = "Diabetes" if pred == 1 else "No Diabetes"
        checked = "✓" if prediction == truth else "✗"
        result.append(f"[{checked}] {prediction}\t(p={round(prob, 4)})")
        if pred == 1:
            pos_count += 1
        if checked == "✓":
            correct_count += 1
    print("\n".join(result))
    print("\nSummary :")
    print(f"Correct : {correct_count}/{len(preds)}")
    print(f"Predicted positives : {pos_count}")
    print("\n")


def demo_predictions(x_batch_, y_batch_, mode):
    threshold = (
        config.TRIAGE_THRESHOLD if mode == "triage" else config.BALANCED_THRESHOLD
    )
    print(f"\nInvoking endpoint (mode={mode}, threshold={threshold})")
    print("-------------------------------------------------------\n")
    print("Input batch :\n", x_batch_)
    output = predict(x_batch_, mode=mode)
    show_predictions(y_batch_, output)


if __name__ == "__main__":
    test_path = "../data/prepared/test.csv"
    df_test = pd.read_csv(test_path)
    x_batch = df_test.drop(config.TARGET, axis=1).iloc[345:350, :]
    y_batch = df_test[config.TARGET].iloc[345:350].values

    demo_predictions(x_batch, y_batch, mode="triage")
    demo_predictions(x_batch, y_batch, mode="balanced")
    print("[INFO] Inference demo successfully completed.")
