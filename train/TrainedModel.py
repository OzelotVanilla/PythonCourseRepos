from keras.engine.training import Model as KerasModel


class TrainedModel:
    def __init__(self, model: KerasModel, dataset_path: str, result_column_name: str):
        self.model = model
        self.dataset_path = dataset_path
        self.result_column_name = result_column_name
