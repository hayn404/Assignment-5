import argparse
import sys
import os
import mlflow
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum required validation accuracy (default: 0.85)",
    )
    parser.add_argument(
        "--model-info",
        type=str,
        default="model_info.txt",
        help="Path to the file containing the MLflow Run ID",
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    # --- Read Run ID ---
    if not os.path.exists(args.model_info):
        print(f"[ERROR] {args.model_info} not found.")
        sys.exit(1)
 
    with open(args.model_info, "r") as f:
        run_id = f.read().strip()
 
    if not run_id:
        print("[ERROR] model_info.txt is empty.")
        sys.exit(1)
 
    print(f"[INFO] Checking Run ID: {run_id}")
 
    # --- Fetch metric from MLflow ---
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
 
    client = mlflow.MlflowClient()
 
    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"[ERROR] Could not retrieve run from MLflow: {e}")
        sys.exit(1)
 
    metrics = run.data.metrics
    accuracy = metrics.get("best_val_accuracy")
 
    if accuracy is None:
        print("[ERROR] 'best_val_accuracy' metric not found in MLflow run.")
        sys.exit(1)
 
    print(f"[INFO] best_val_accuracy = {accuracy:.4f}  |  threshold = {args.threshold}")
 
    if accuracy < args.threshold:
        print(
            f"[FAIL] Accuracy {accuracy:.4f} is below the required threshold "
            f"{args.threshold}. Deployment blocked."
        )
        sys.exit(1)
 
    print(
        f"[PASS] Accuracy {accuracy:.4f} meets the threshold {args.threshold}. "
        "Proceeding to deployment."
    )
 
 
if __name__ == "__main__":
    main()