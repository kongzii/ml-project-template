import os
import getpass
import logging
import argparse
import contextlib
import mlflow
import mlflow.tensorflow

logging.basicConfig(level=logging.INFO)


@contextlib.contextmanager
def init_mlflow(
    run_name: str = "default",
    args: argparse.Namespace = None,
    enable_tensorflow_autolog: bool = False
):
    user_id = os.environ.get("USER_ID", getpass.getuser())
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "unknown")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{user_id} - {experiment_name}")

    if enable_tensorflow_autolog:
        mlflow.tensorflow.autolog(every_n_iter=1)

    with mlflow.start_run(run_name=run_name):
        if args is not None:
            mlflow.log_params(vars(args))

        yield
    
