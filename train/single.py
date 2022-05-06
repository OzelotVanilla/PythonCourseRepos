from util.console import console
from util.helper import getBestGPUTensorFlow

import tensorflow
from pandas import DataFrame, Series, read_csv as readCSV
from keras.engine.sequential import Sequential as KerasSeqModel
from keras.layers import Dense as KerasDenseLayer
from keras.callbacks import EarlyStopping as KerasEarlyStop
from keras.engine.training import Model as KerasModel


# This file contains training according to single file


def getModel(dataset_path: str, result_column_name: str, /, use_CPU: bool = False) -> KerasModel:
    console.clear()
    console.info("Prepare to traine model from file \"" + dataset_path + "\".")

    # Read and separate whole dataframe to data column and result column
    result_column, data_column = __splitOneColumn(readCSV(dataset_path), result_column_name)

    # Summon the model
    model = KerasSeqModel([
        KerasDenseLayer(10, activation="relu"),
        KerasDenseLayer(10, activation="relu"),
        KerasDenseLayer(10, activation="relu")
    ])

    # Choose use CPU or GPU
    with tensorflow.device("/cpu:0" if use_CPU else getBestGPUTensorFlow()):
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(data_column, result_column, validation_split=0.2, callbacks=[KerasEarlyStop(patience=10)])

    # Return the model
    console.info(
        "Successfully trained model from file \"" + dataset_path +
        "\" predicting \"" + result_column_name + "\"."
    )
    return model


# Return at least the loss of a model in list
def testModel(model: KerasModel, dataset_path: str, result_column_name: str, use_CPU: bool = False) -> dict[str, object]:
    console.info("Testing model...")
    result_column, data_column = __splitOneColumn(readCSV(dataset_path), result_column_name)

    # Choose use CPU or GPU
    with tensorflow.device("/cpu:0" if use_CPU else getBestGPUTensorFlow()):
        eval_result = model.evaluate(data_column, result_column)

    # Check if returned value is not list, change to list for zip function
    if type(eval_result) != type([]):
        eval_result = [eval_result]
    console.info("Model trained, these information available:")
    result = dict()
    for (key, value) in zip(model.metrics_names, eval_result):
        print("\t" + str(key) + ":", value)
        result[key] = value
    return result


def __splitOneColumn(dataframe: DataFrame, split_column_name: str) -> tuple[Series, DataFrame]:
    return (dataframe[split_column_name], dataframe.drop(columns=[split_column_name]))
