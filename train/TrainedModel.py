from keras.engine.training import Model as KerasModel


class TrainedModel:
    def __init__(self, model: KerasModel, result_column, data_column):
        self.model = model
        self.result_column = result_column
        self.data_column = data_column
