import re
import json
import string
import mlflow
import argparse
import tensorflow as tf

from pathlib import Path
from utilities import SEED_VALUE
from utilities.tracking import init_mlflow

COMPONENT_NAME = "train"


def create_model(
    dataset_to_adapt,
    embedding_dim: int,
    max_tokens: int,
) -> tf.keras.Model:
    vectorize = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=max_tokens,
    )
    vectorize.adapt(dataset_to_adapt)

    text_inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="input_raw_text")

    x = vectorize(text_inputs)
    x = tf.keras.layers.Embedding(max_tokens + 1, embedding_dim)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64))(x)

    outputs = tf.keras.layers.Dense(2, activation="softmax", name="output")(x)

    return vectorize, tf.keras.Model(inputs=text_inputs, outputs=outputs)


def main(args: argparse.Namespace):
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        f"{args.dataset_dir}/train",
        label_mode="categorical",
        batch_size=args.batch_size,
        validation_split=0.2,
        subset="training",
        seed=SEED_VALUE,
        class_names=args.class_names,
    )
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        f"{args.dataset_dir}/train",
        label_mode="categorical",
        batch_size=args.batch_size,
        validation_split=0.2,
        subset="validation",
        seed=SEED_VALUE,
        class_names=args.class_names,
    )

    vectorize_layer, model = create_model(
        dataset_to_adapt=raw_train_ds.map(lambda x, y: x),
        embedding_dim=args.embedding_dim,
        max_tokens=args.max_tokens,
    )

    model_dir = Path(f"/{args.datas_dir}/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(f"{model_dir}/vocabulary.txt", "w") as f:
        f.write(json.dumps(vectorize_layer.get_vocabulary()))
    
    with open(f"{model_dir}/class_names.txt", "w") as f:
        f.write(json.dumps(args.class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"{args.tensorboard_dir}/{COMPONENT_NAME}/", 
    )
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=2,
        restore_best_weights=True,
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{args.datas_dir}/model",
        monitor="val_categorical_accuracy",
        save_best_only=True,
    )

    model.fit(
        raw_train_ds, 
        validation_data=raw_val_ds, 
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[
            tensorboard_callback,
            earlystopping_callback,
            checkpoint_callback,
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard-dir", default="/tensorboard")
    parser.add_argument("--datas-dir", default="/datas")
    parser.add_argument("--dataset-dir", default="/datas/aclImdb")
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--steps-per-epoch", default=None)
    parser.add_argument("--batch-size", default=32)
    parser.add_argument("--embedding-dim", default=128)
    parser.add_argument("--max-tokens", default=50_000)
    parser.add_argument("--class-names", default=["neg", "pos"])

    args = parser.parse_args()

    with init_mlflow(run_name=COMPONENT_NAME, args=args, enable_tensorflow_autolog=True):
        main(args)
