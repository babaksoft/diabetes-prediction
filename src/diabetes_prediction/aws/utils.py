import os

import boto3
import sagemaker
from dotenv import find_dotenv, load_dotenv

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
