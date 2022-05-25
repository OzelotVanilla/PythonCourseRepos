from train.tool import splitOneColumn, readCSV
from train.TrainedModel import TrainedModel
from util.console import console
from util.helper import getBestGPUTensorFlow

import tensorflow
from keras.engine.base_layer import Layer as KerasLayer
from keras.engine.sequential import Sequential as KerasSeqModel
from keras.layers import Dense as KerasDenseLayer
from keras.callbacks import EarlyStopping as KerasEarlyStop
import pandas as pd
from pandas import DataFrame


# This file contains training according to single file


def getModel(dataset_path_or_dataframe, target_column_name: str,
             /, use_CPU: bool = False, descr_convert: dict[str, dict[str, int]] = None,
             layers: list[KerasLayer] = [KerasDenseLayer(10, activation="relu")] * 3,
             compile_optimizer: str = "adam", compile_loss_function="mean_squared_error",
             compile_metrics=["accuracy"],
             fit_callbacks=[KerasEarlyStop(patience=3)], fit_epoch: int = 1, validation_split=0.2) -> TrainedModel:
    console.clear()
    path_type = type(dataset_path_or_dataframe)
    if path_type == str:
        console.info("Prepare to read from file \"" + dataset_path_or_dataframe + "\".")
        result_column, data_column = splitOneColumn(
            readCSV(dataset_path_or_dataframe, descr_convert=descr_convert),
            target_column_name
        )
    elif path_type == DataFrame:
        console.info("Prepare to read using pandas' dataframe:\n")
        print(dataset_path_or_dataframe.head(), end="\n\n")
        result_column, data_column = splitOneColumn(
            dataset_path_or_dataframe,
            target_column_name
        )
    else:
        console.err(
            "Wrong parameter for dataset_path_or_dataframe: " +
            str(dataset_path_or_dataframe) + ", check again."
        )
        raise TypeError()

    # Read and separate whole dataframe to data column and result column

    # Summon the model
    console.info(
        "Ready to train model predicting \"" + target_column_name + "\"."
    )

    model = summonModel(
        result_column, data_column, layers, use_CPU=use_CPU, compile_optimizer=compile_optimizer,
        compile_loss_function=compile_loss_function, compile_metrics=compile_metrics, fit_callbacks=fit_callbacks,
        fit_epoch=fit_epoch, validation_split=validation_split
    )

    console.info(
        "Successfully trained model predicting \"" + target_column_name + "\"."
    )

    return model


# Return at least the loss of a model in list
def testModel(model: TrainedModel, test_result_column=None, test_data_column=None, /, use_CPU: bool = False) -> dict[str, object]:
    console.info("Testing model...")
    # No test reference input -> use the train set to test
    if (test_result_column is None):
        test_result_column = model.result_column
    if (test_data_column is None):
        test_data_column = model.data_column

    # Choose use CPU or GPU
    model = model.model
    with tensorflow.device("/cpu:0" if use_CPU else getBestGPUTensorFlow()):
        # Evaluate the model
        eval_result = model.evaluate(test_data_column, test_result_column)

    # Convert output to dict (for plotting's convenience)
    # Check if returned value is not list, change to list for zip function
    if type(eval_result) != type([]):
        eval_result = [eval_result]
    console.info("Model tested, these information available:")
    # Save output to dict
    result = dict()
    for (key, value) in zip(model.metrics_names, eval_result):
        # Display evaluation output
        print("\t" + str(key) + ":", value)
        result[key] = value

    console.wait(2)
    return result


def getModelByXYColumn(result_column, data_column,
                       /, use_CPU: bool = False, descr_convert: dict[str, dict[str, int]] = None,
                       layers: list[KerasLayer] = [KerasDenseLayer(10, activation="relu")] * 3,
                       compile_optimizer: str = "adam", compile_loss_function="mean_squared_error",
                       compile_metrics=["accuracy"],
                       fit_callbacks=[KerasEarlyStop(patience=3)], fit_epoch: int = 1, validation_split=0.2) -> TrainedModel:
    console.clear()
    console.info(
        "Start model training."
    )

    model = summonModel(
        result_column, data_column, layers, use_CPU=use_CPU, compile_optimizer=compile_optimizer,
        compile_loss_function=compile_loss_function, compile_metrics=compile_metrics, fit_callbacks=fit_callbacks,
        fit_epoch=fit_epoch, validation_split=validation_split
    )

    console.info(
        "Successfully trained model"
    )
    console.wait(4)

    return model


def summonModel(result_column, data_column, layers: list[KerasLayer] = [KerasDenseLayer(10, activation="relu")] * 3,
                /, use_CPU: bool = False,
                compile_optimizer: str = "adam", compile_loss_function="mean_squared_error",
                compile_metrics=["accuracy"],
                fit_callbacks=[KerasEarlyStop(patience=3)], fit_epoch: int = 1, validation_split=0.2) -> TrainedModel:

    # Summon the model
    model = KerasSeqModel(layers)

    # Choose use CPU or GPU
    with tensorflow.device("/cpu:0" if use_CPU else getBestGPUTensorFlow()):
        model.compile(optimizer=compile_optimizer, loss=compile_loss_function, metrics=compile_metrics)
        model.fit(data_column, result_column, validation_split=validation_split,
                  callbacks=fit_callbacks, epochs=fit_epoch)

    # Return the model
    return TrainedModel(model, result_column, data_column)
