import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from ..config import config
from ..pipeline import pipeline
from ..utils import feature_target_split, mlflow_register
from ..utils import plot_metrics, train_and_validate


def get_svc_model():
    rs = config.RANDOM_STATE
    svc = make_pipeline(
        Nystroem(gamma=0.2, random_state=rs),
        LinearSVC(dual=False, random_state=rs),
    )
    return svc


# Train and validate several models on given train data
def train_candidate_models(x_train, y_train):
    # Pick a random state for repeatability
    rs = config.RANDOM_STATE

    agg_metrics = []
    model_names = [
        "LR", "Tree", "RF", "Nyst-SVC", "Hist-GB", "KNN", "MLP"
    ]
    models = [
        LogisticRegression(max_iter=1000, random_state=rs),
        DecisionTreeClassifier(random_state=rs),
        RandomForestClassifier(random_state=rs),
        get_svc_model(),
        HistGradientBoostingClassifier(random_state=rs),
        KNeighborsClassifier(),
        MLPClassifier(random_state=rs)
    ]

    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.set_experiment("model-shortlisting")
    for name, model in tqdm(zip(model_names, models)):
        print(f"\nTraining '{name}' model...")
        metrics = train_and_validate(model, x_train, y_train)
        agg_metrics.append(metrics)
        mlflow_register(model, name, x_train, name, dict(list(metrics.items())))

    print("Plotting metrics...")
    metric_names = ["Recall", "Precision", "FScore"]
    df_metrics = pd.DataFrame(agg_metrics, index=model_names)

    with mlflow.start_run(
        run_name="Performance",
        experiment_id=experiment.experiment_id
    ) as run:
        for metric_name in metric_names:
            plot_metrics(
                df_metrics[f"Train {metric_name}"],
                df_metrics[f"Val {metric_name}"],
                metric_name,
                model_names
            )
            path = config.METRICS_PATH / f"Train_Val_{metric_name}.png"
            mlflow.log_artifact(path)

        # NOTE: sle stands for ShortListing Experiment
        path = config.METRICS_PATH / "sle_metrics.csv"
        df_metrics.to_csv(path, index=True, header=True)
        mlflow.log_artifact(path)
        mlflow.set_tag("project", config.PROJECT_NAME)
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.end_run()


def main():
    data_path = config.DATA_PATH / "prepared" / config.TRAIN_FILE
    x_raw, y_train = feature_target_split(data_path)
    x_train = pipeline.fit_transform(x_raw)
    x_train = pd.DataFrame(x_train, columns=pipeline.get_feature_names_out())
    train_candidate_models(x_train, y_train)


if __name__ == "__main__":
    main()
