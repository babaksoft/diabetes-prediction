"""
train.py : Fits best model from MLflow to train+val in a SageMaker training job.
Note : In this module, 'sm' is consistently used as the short form of SageMaker.
"""

import boto3
from sagemaker.sklearn.estimator import SKLearn

from utils import get_session_role

FRAMEWORK_VERSION = "1.4-2"
sm_session, sm_role = get_session_role()


def sm_train(params):
    root_key = (
        f"s3://{sm_session.default_bucket()}/{sm_session.default_bucket_prefix}/data"
    )
    train_path = f"{root_key}/train/train.csv"
    val_path = f"{root_key}/validation/validation.csv"

    sklearn = SKLearn(
        entry_point="sm_train.py",
        role=sm_role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version=FRAMEWORK_VERSION,
        base_job_name="diabetes-scikit",
        hyperparameters=params,
    )

    sklearn.fit({"train": train_path, "val": val_path}, wait=True)
    return sklearn


def train():
    hyperparameters = {
        "class-weight": "balanced",
        "max-depth": "5",
        "max-features": "0.8",
        "max-iter": 100,
        "validation-fraction": 0.15,
    }
    estimator = sm_train(hyperparameters)
    estimator.latest_training_job.wait(logs="None")

    sm_client = boto3.client("sagemaker")
    artifact = sm_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name
    )["ModelArtifacts"]["S3ModelArtifacts"]
    print("Model artifact :", artifact)

    print("\n[INFO] Model training successfully completed.")


if __name__ == "__main__":
    train()
