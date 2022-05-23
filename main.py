from util.PyplotDiagram import PyplotDiagram
from util.console import console
from util.helper import prepareEnv, cleanEnv
from train.single import getModel, testModel
from train.pre_processing import classToDigitReplace, getClassToDigitDict, unifyColNames

from keras.layers import Dense as KerasDenseLayer
from pandas import read_csv as readCSV

# Use VSCode to open the entire folder, then run this script
# Otherwise, the import may not be solved


def main():
    # Do preparation jobs, like install tools, download datasets to specified path (datasets directory)
    # Do not worry, almost all config would be clean-up if you want
    prepareEnv()

    # Train the model according to 2015 data
    model_2015 = getModel(
        "./datasets/data_2015.csv", "HeartDiseaseorAttack",
        use_CPU=True, layers=[KerasDenseLayer(10, activation="relu")],
        fit_epoch=10
    )
    model_2015_result = testModel(model_2015, use_CPU=True)

    # Train the model according to 2020 data
    df_2020 = readCSV("./datasets/data_2020.csv")
    unifyColNames(df_2020)
    classToDigitReplace(df_2020, getClassToDigitDict())
    model_2020 = getModel(
        df_2020, "HeartDisease",
        use_CPU=True, descr_convert=getClassToDigitDict(),
        layers=[KerasDenseLayer(10, activation="relu")],
        fit_epoch=10
    )
    model_2020_result = testModel(model_2020, use_CPU=True)

    # Draw the plot of loss and accuracy
    diagram = PyplotDiagram()
    diagram.addAsSeries(
        {"Original 2015 Model": model_2015_result, "Original 2020 Model": model_2020_result}
    ).setTitle("Trained Result")
    PyplotDiagram.showAllPlot()

    # This function do the clean-up and finishing job
    cleanEnv()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stopped the program manually.")
        cleanEnv()
