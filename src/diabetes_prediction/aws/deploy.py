from datetime import datetime
import json

from sagemaker.sklearn.model import SKLearnModel

from utils import get_session_role

FRAMEWORK_VERSION = "1.4-2"


def deploy():
    session, role = get_session_role()
    model_data = (
        f"s3://{session.default_bucket()}/{session.default_bucket_prefix}"
        f"/models/model.tar.gz"
    )
    model = SKLearnModel(
        entry_point="inference.py",
        model_data=model_data,
        role=role,
        framework_version=FRAMEWORK_VERSION,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    endpoint_name = f"diabetes-predictor-{timestamp}"
    predictor = model.deploy(
        endpoint_name=endpoint_name,
        instance_type="ml.t2.large",
        initial_instance_count=1,
    )

    endpoint_info = {
        "endpoint_name": predictor.endpoint,
        "region": session.boto_session.region_name,
        "deployed_at": datetime.now().isoformat(),
        "model_version": "v1",
    }

    artifact_path = "artifacts/deployment.json"
    with open(artifact_path, "w") as file:
        json.dump(endpoint_info, file)
    print("Deployment info saved to :", artifact_path)


if __name__ == "__main__":
    deploy()
    print("[INFO] Model deployment successfully completed.")
