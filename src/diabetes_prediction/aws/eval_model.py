import json

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.s3 import S3Downloader
from sagemaker.sklearn.processing import SKLearnProcessor
from utils import get_session_role

FRAMEWORK_VERSION = "1.4-2"


def show_metrics(job_desc):
    s3_uri = ""
    for output in job_desc["ProcessingOutputConfig"]["Outputs"]:
        if output["OutputName"] == "evaluation":
            s3_uri = output["S3Output"]["S3Uri"] + "/evaluation.json"
            break

    eval_output = S3Downloader.read_file(s3_uri)
    eval_output_dict = json.loads(eval_output)
    print("\nEvaluation metrics :")
    print(json.dumps(eval_output_dict, sort_keys=False, indent=4))


def evaluate():
    session, role = get_session_role()
    root_uri = f"{session.default_bucket()}/{session.default_bucket_prefix}"
    data_uri = f"s3://{root_uri}/data/test/test.csv"
    model_uri = f"s3://{root_uri}/models/model.tar.gz"

    processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        base_job_name="scikit-eval",
        sagemaker_session=session,
    )
    processor.run(
        code="evaluate.py",
        inputs=[
            ProcessingInput(source=model_uri, destination="/opt/ml/processing/model"),
            ProcessingInput(source=data_uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            )
        ],
    )
    job_description = processor.jobs[-1].describe()
    show_metrics(job_description)


if __name__ == "__main__":
    evaluate()
    print("[INFO] Model evaluation successfully completed.")
