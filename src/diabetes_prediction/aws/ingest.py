import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
import boto3
from botocore.exceptions import ClientError
import sagemaker

_ = load_dotenv(find_dotenv())
REGION = os.environ["PROJ_REGION"]
BUCKET = os.environ["PROJ_BUCKET"]
BUCKET_PREFIX = os.environ["PROJ_BUCKET_PREFIX"]


def get_session_role():
    boto3.setup_default_session(region_name=REGION)
    session = sagemaker.Session(default_bucket=BUCKET)
    session.default_bucket_prefix = BUCKET_PREFIX

    role = os.environ["SM_EXEC_ROLE_ARN"]
    return session, role


sm_session, sm_role = get_session_role()


def check_ingest_status() -> bool:
    try:
        s3 = boto3.client("s3")
        key_prefix = f"{sm_session.default_bucket_prefix}/data/train/train.csv"
        s3.head_object(Bucket=sm_session.default_bucket(), Key=key_prefix)
        return True
    except ClientError:
        return False


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

    print("[INFO] Data successfully ingested.")
    print(f"train : {train_path}")
    print(f"validation : {valid_path}")
    print(f"test : {test_path}")


if __name__ == "__main__":
    ingest()
