from datetime import datetime

import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from diabetes_prediction.config import settings
from diabetes_prediction.pipeline import build_pipeline
from diabetes_prediction.utils import get_data


def tune_model(name, model, x, y, param_grid):
    rs = settings.RANDOM_STATE
    scoring = "average_precision"
    params = {
        "model_type": type(model).__name__,
        "class_weight": model.class_weight,
        "random_state": model.random_state,
        "param_grid": param_grid,
        "scoring": scoring,
    }

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        transform = build_pipeline()
        pipeline = Pipeline([("transformer", transform), ("estimator", model)])
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)
        params["cv_splits"] = cv.n_splits
        params["cv_shuffle"] = cv.shuffle
        params["cv_random_state"] = cv.random_state

        start = datetime.now()
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(x, y)
        end = datetime.now()

        params["best_params"] = search.best_params_
        params["best_pr_auc"] = round(search.best_score_, 4)
        params["tuning_duration"] = str(end - start)
        mlflow.log_params(params)
        mlflow.end_run()

    print(type(model).__name__)
    print("-" * (len(type(model).__name__) + 2))
    print(f"Best params :\n{search.best_params_}\n")
    print(f"PR-AUC : {round(search.best_score_, 4)}")
    print("Tuning time :", str(end - start))


def tune_models():
    rs = settings.RANDOM_STATE

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="Hyperparameter Tuning")
    x_train, y_train = get_data()

    ## Tune first model shortlisted for tuning (Hist-GB)
    hgb_clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=rs)
    param_grid = {
        "estimator__max_depth": [5, 8, 10, 12],
        "estimator__max_iter": [70, 85, 100],
        "estimator__max_features": [0.8, 0.85, 0.9, 0.95],
        "estimator__validation_fraction": [0.1, 0.15, 0.2],
    }
    tune_model("Hist-GB", hgb_clf, x_train, y_train, param_grid)

    ## Tune second model shortlisted for tuning (LR)
    lr_clf = LogisticRegression(class_weight="balanced", random_state=rs)
    param_grid = [
        {
            "estimator__l1_ratio": [0.0],
            "estimator__solver": ["lbfgs", "newton-cg", "newton-cholesky"],
            "estimator__C": [0.5, 0.7, 1.0, 2.0, 5.0],
            "estimator__max_iter": [500, 1000],
        },
        {
            "estimator__solver": ["saga"],
            "estimator__l1_ratio": [0.2, 0.3, 0.5],
            "estimator__C": [0.5, 0.7, 1.0, 2.0, 5.0],
            "estimator__max_iter": [500, 1000],
        },
    ]
    tune_model("LR", lr_clf, x_train, y_train, param_grid)


if __name__ == "__main__":
    tune_models()
