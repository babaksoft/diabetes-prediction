import json
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from utils import get_session_role

sm_session, _ = get_session_role()


def check_ingest_status() -> bool:
    try:
        s3 = boto3.client("s3")
        key_prefix = f"{sm_session.default_bucket_prefix}/data/train/train.csv"
        s3.head_object(Bucket=sm_session.default_bucket(), Key=key_prefix)
        return True
    except ClientError:
        return False


def get_metadata(data_dir: str | Path) -> dict[str, Any]:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    df_train = pd.read_csv(data_dir / "train.csv")
    df_valid = pd.read_csv(data_dir / "validation.csv")
    df_test = pd.read_csv(data_dir / "test.csv")
    return {
        "dataset_version": "v1",
        "train_size": len(df_train),
        "validation_size": len(df_valid),
        "test_size": len(df_test),
    }


def ingest():
    if check_ingest_status():
        print("[INFO] Data is already ingested.")
        return

    root_dir = Path("../data/prepared")
    root_key = f"{sm_session.default_bucket_prefix}/data"
    train_path = sm_session.upload_data(
        path=root_dir / "train.csv",
        bucket=sm_session.default_bucket(),
        key_prefix=f"{root_key}/train",
    )
    valid_path = sm_session.upload_data(
        path=root_dir / "validation.csv",
        bucket=sm_session.default_bucket(),
        key_prefix=f"{root_key}/validation",
    )
    test_path = sm_session.upload_data(
        path=root_dir / "test.csv",
        bucket=sm_session.default_bucket(),
        key_prefix=f"{root_key}/test",
    )

    metadata = get_metadata(root_dir)
    with open("./artifacts/metadata.json", "wt") as file:
        json.dump(metadata, file)
    meta_path = sm_session.upload_data(
        path="./artifacts/metadata.json",
        bucket=sm_session.default_bucket(),
        key_prefix=root_key,
    )

    print("[INFO] Data successfully ingested.")
    print(f"Train : {train_path}")
    print(f"Validation : {valid_path}")
    print(f"Test : {test_path}")
    print(f"Metadata : {meta_path}")


if __name__ == "__main__":
    ingest()
