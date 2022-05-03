from pandas import read_csv as readCSV
from keras.engine.sequential import Sequential as KerasSeqModel
from keras.layers import Dense as KerasDenseLayer
from keras.callbacks import EarlyStopping as KerasEarlyStop

from util.console import console


# This file contains training according to single file


def getModel(dataset_path: str, result_column_name: str):
    console.clear()
    console.info("Prepare to traine model from file \"" + dataset_path + "\".")
    dataset_frame = readCSV(dataset_path)

    # Separate whole dataframe to data column and result column
    result_column = dataset_frame[result_column_name]
    data_column = dataset_frame.drop(columns=[result_column_name])

    # Summon the model
    model = KerasSeqModel([
        KerasDenseLayer(10, activation="relu"),
        KerasDenseLayer(10, activation="relu"),
        KerasDenseLayer(10, activation="relu")
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(data_column, result_column, validation_split=0.2, callbacks=[KerasEarlyStop(patience=10)])

    # Return the model
    console.info(
        "Successfully trained model from file \"" + dataset_path +
        "\" predicting \"" + result_column_name + "\"."
    )
    return model
