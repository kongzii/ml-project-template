import abc

# TODO: Meaningful typings
ModelInput = object
ModelOutput = object


class Model:
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path

        self.load()

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def predict(self, inputs: ModelInput, *args, **kwargs) -> ModelOutput:
        pass
