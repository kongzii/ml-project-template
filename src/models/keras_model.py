import json
import tensorflow as tf

from models.model import Model, ModelInput, ModelOutput
from typing import Union, Dict


class KerasModel(Model):
    def load(self):
        self.model = tf.keras.models.load_model(self.model_path)

        with open(self.model_path + "/class_names.txt", "r") as f:
            self.class_names = json.load(f)

    def predict(
        self, inputs: ModelInput, return_id: bool = False
    ) -> ModelOutput:
        return [
            {
                (self.id_to_class_name(i) if not return_id else i): p
                for i, p in enumerate(row)
            }
            for row in self.model.predict(inputs).tolist()
        ]

    def class_name_to_id(self, name: str) -> int:
        return self.class_names.index(name)

    def id_to_class_name(self, id_: int) -> str:
        return self.class_names[id_]
