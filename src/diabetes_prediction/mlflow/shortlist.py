import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from config import config
from pipeline import pipeline
from utils import feature_target_split, mlflow_register
from utils import plot_metrics, train_and_validate


# Train and validate several models on given train data
def train_candidate_models(x_train, y_train):
    # Pick a random state for repeatability
    rs = 147

    train_metrics = []
    val_metrics = []
    model_names = [
        "Tree", "RF", "Lin-SVC", "Hist-GB", "KNN", "GaussianNB", "MLP"
    ]
    models = [
        DecisionTreeClassifier(random_state=rs),
        RandomForestClassifier(random_state=rs),
        LinearSVC(random_state=rs),
        HistGradientBoostingClassifier(random_state=rs),
        KNeighborsClassifier(),
        GaussianNB(),
        MLPClassifier(random_state=rs),
    ]

    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.set_experiment("model-shortlisting")
    for name, model in tqdm(zip(model_names, models)):
        print(f"\nTraining '{name}' model...")
        train_recall, val_recall = train_and_validate(model, x_train, y_train)
        train_metrics.append(train_recall)
        val_metrics.append(val_recall)
        metrics = {
            "Training Recall": train_recall,
            "Validation Recall": val_recall,
        }
        mlflow_register(model, name, name, metrics)

    print("Plotting metrics...")
    plot_metrics(train_metrics, val_metrics, model_names)

    with mlflow.start_run(
        run_name="Performance",
        experiment_id=experiment.experiment_id
    ):
        path = config.METRICS_PATH / "Train_Val_Recall.png"
        mlflow.log_artifact(path)
        mlflow.end_run()


def main():
    data_path = config.DATA_PATH / "prepared" / config.TRAIN_FILE
    x_raw, y_train = feature_target_split(data_path)
    x_train = pipeline.fit_transform(x_raw)
    x_train = pd.DataFrame(x_train, columns=pipeline.get_feature_names_out())
    train_candidate_models(x_train, y_train)


if __name__ == "__main__":
    main()
