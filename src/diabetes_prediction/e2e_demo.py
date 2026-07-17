from diabetes_prediction.config.logging import configure_logging
from diabetes_prediction.evaluate import evaluate
from diabetes_prediction.ingest import ingest
from diabetes_prediction.predict import predict
from diabetes_prediction.train import train


def show_banner():
    print("\n==========================================")
    print("Diabetes Prediction")
    print("End-to-End ML Pipeline Demo")
    print("\nPipeline:")
    print("✓ Data ingestion")
    print("✓ Model training")
    print("✓ Model evaluation")
    print("✓ Batch inference")
    print("\n(c) 2026, babaksoft")
    print("==========================================")

    _ = input("\nPress ENTER to continue...")


def demo():
    show_banner()
    print("\n====== Data Ingestion ======")
    ingest()

    print("\n====== Training on train+validation set ======")
    train()

    print("\n====== Evaluating on test set ======")
    evaluate()

    print("\n====== Batch inference on test set ======")
    predict()

    print("End-to-end demo completed.")


if __name__ == "__main__":
    configure_logging()
    demo()
