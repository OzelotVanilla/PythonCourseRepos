from util.console import console
from util.helper import prepareEnv, cleanEnv
from train.single import getModel, testModel
from train.pre_processing import getCSV2020ReplaceDict

from keras.layers import Dense as KerasDenseLayer

# Use VSCode to open the entire folder, then run this script
# Otherwise, the import may not be solved


def main():
    # Do preparation jobs, like install tools, download datasets to specified path
    # Do not worry, almost all config would be clean-up if you want
    prepareEnv()

    # Train the model according to 2015 data
    single_2015_model = getModel(
        "./datasets/data_2015.csv", "HeartDiseaseorAttack",
        use_CPU=True, layers=[KerasDenseLayer(10, activation="relu")],
        fit_epoch=10
    )
    testModel(single_2015_model, use_CPU=True)

    # Train the model according to 2020 data
    single_2020_model = getModel(
        "./datasets/data_2020.csv", "HeartDisease",
        use_CPU=True, descr_convert=getCSV2020ReplaceDict(),
        layers=[KerasDenseLayer(10, activation="relu")],
        fit_epoch=10
    )
    testModel(single_2020_model, use_CPU=True)

    # This function do the clean-up and finishing job
    cleanEnv()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stopped the program manually.")
        cleanEnv()
