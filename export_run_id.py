import argparse
import os
import sys
import mlflow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Assignment5_CI",
        help="Name of the MLflow experiment to query",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="model_info.txt",
        help="File to write the Run ID into",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        print(f"[ERROR] Experiment '{args.experiment_name}' not found in MLflow.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        print("[ERROR] No MLflow runs found after training.")
        sys.exit(1)

    run_id = runs[0].info.run_id
    print(f"[INFO] Captured Run ID: {run_id}")

    with open(args.output_file, "w") as f:
        f.write(run_id)

    print(f"[INFO] Run ID written to {args.output_file}")


if __name__ == "__main__":
    main()