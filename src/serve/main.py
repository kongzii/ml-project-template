import os
import fastapi
import logging
import numpy as np
import tensorflow as tf

from pydantic import BaseModel
from models.model import Model
from models.keras_model import KerasModel

logging.basicConfig(level=logging.WARNING)

MODEL_PATH = os.environ.get("MODEL_PATH", "/datas/model")


if not os.path.exists(MODEL_PATH):
    raise ValueError(
        f"Directory with saved model does not exist: {MODEL_PATH}"
    )


class Item(BaseModel):
    text: str


app = fastapi.FastAPI()
keras_model = KerasModel(MODEL_PATH)


def predict(model: Model, item: Item):
    prediction = model.predict([item.text])[0]

    return {
        "text": item.text,
        "prediction": prediction,
    }


@app.post("/keras_model")
def keras_model_predict(item: Item):
    return predict(keras_model, item)
