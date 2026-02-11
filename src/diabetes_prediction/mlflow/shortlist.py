import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from ..config import config
from ..utils import evaluate_model


def train_candidate_models():
    rs = config.RANDOM_STATE
    agg_metrics = []
    model_names = ["LR", "Nyst-SVC", "RF", "HGB", "KNN"]
    models = [
        LogisticRegression(class_weight="balanced", random_state=rs),
        make_pipeline(
            Nystroem(random_state=rs),
            LinearSVC(class_weight="balanced", random_state=rs)),
        RandomForestClassifier(
            class_weight="balanced", random_state=rs),
        HistGradientBoostingClassifier(
            class_weight="balanced", random_state=rs),
        KNeighborsClassifier()
    ]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment("Model Shortlisting")
    for name, model in tqdm(zip(model_names, models)):
        print(f"\nEvaluating '{name}' model...")
        model_name = "Nystroem+LinearSVC" if name == "Nyst-SVC" else None
        metrics = evaluate_model(model, name, model_name)
        agg_metrics.append(pd.Series(metrics, name=name))

    path = config.METRICS_PATH / "sl_metrics.csv"
    df_metrics = pd.DataFrame(agg_metrics, index=model_names)
    df_metrics = df_metrics.reset_index().rename(columns={"index": "model"})
    df_metrics.to_csv(path, index=False, header=True)

    with mlflow.start_run(
        run_name="Performance",
        experiment_id=experiment.experiment_id) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.log_artifact(path)
        mlflow.end_run()


if __name__ == "__main__":
    train_candidate_models()
