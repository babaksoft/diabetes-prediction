import boto3
from sagemaker.sklearn.estimator import SKLearn
from utils import get_session_role

FRAMEWORK_VERSION = "1.4-2"
sm_session, sm_role = get_session_role()


def train(params):
    root_key = (
        f"s3://{sm_session.default_bucket()}/{sm_session.default_bucket_prefix}/data"
    )
    train_path = f"{root_key}/train/train.csv"
    val_path = f"{root_key}/validation/validation.csv"

    sklearn = SKLearn(
        entry_point="train.py",
        role=sm_role,
        instance_count=1,
        instance_type="ml.m6i.large",
        framework_version=FRAMEWORK_VERSION,
        base_job_name="hgb-scikit",
        hyperparameters=params,
    )

    sklearn.fit({"train": train_path, "val": val_path}, wait=True)
    return sklearn


if __name__ == "__main__":
    hyperparameters = {
        "class-weight": "balanced",
        "max-depth": "5",
        "max-features": "0.8",
        "max-iter": 100,
        "validation-fraction": 0.15,
    }
    estimator = train(hyperparameters)
    estimator.latest_training_job.wait(logs="None")

    sm_client = boto3.client("sagemaker")
    artifact = sm_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name
    )["ModelArtifacts"]["S3ModelArtifacts"]
    print("Model artifact :", artifact)
    print("\n[INFO] Model training successfully completed.")
