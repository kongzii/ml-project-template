import mlflow
import argparse
import logging
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from utilities import SEED_VALUE, timing
from utilities.tracking import init_mlflow
from utilities.files import download, extract
from models.keras_model import KerasModel

COMPONENT_NAME = "test"


@timing
def main(args: argparse.Namespace):
    keras_model = KerasModel(args.model_dir)

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        f"{args.dataset_dir}/test",
        label_mode="int",
        batch_size=args.batch_size,
        class_names=keras_model.class_names,
        seed=SEED_VALUE,
    )

    y_true, y_pred = [], []

    for x, y in raw_test_ds:
        predictions = keras_model.predict(x, return_id=True)
        predictions = [max(p.keys(), key=lambda k: p[k]) for p in predictions]

        y_true.extend(y.numpy().tolist())
        y_pred.extend(predictions)
        assert len(y_true) == len(y_pred)

        if not len(y_true) % 1_000:
            logging.info(f"Tested {len(y_true)} samples.")

    scores = {
        "accuracy_score": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision_score": precision_score(y_true, y_pred),
        "recall_score": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    logging.info(scores)
    mlflow.set_tags(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/datas/model")
    parser.add_argument("--dataset-dir", default="/datas/aclImdb")
    parser.add_argument("--batch-size", default=32)

    args = parser.parse_args()

    with init_mlflow(run_name=COMPONENT_NAME, args=args):
        main(args)
