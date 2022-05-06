from util.console import console
from util.helper import executePrelude, executeFinale
from train.single import getModel, testModel

# Use VSCode to open the entire folder, then run this script
# Otherwise, the import may not be solved


def main():
    # Do preparation jobs, like install tools, download datasets to specified path
    # Do not worry, almost all config would be clean-up if you want
    executePrelude()

    # Train the model according to 2015 data
    single_2015_model = getModel("./datasets/data_2015.csv", "HeartDiseaseorAttack", use_CPU=True)
    test_result_single_2015_model = testModel(
        single_2015_model,
        "./datasets/data_2015.csv", "HeartDiseaseorAttack",
        use_CPU=True
    )

    # This function do the clean-up and finishing job
    executeFinale()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stopped the program manually.")
        executeFinale()
