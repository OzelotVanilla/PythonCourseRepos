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


def getCSV2020ReplaceDict() -> dict[str, dict[str, int]]:
    # This function just return the dict to replace the discriptive data
    # To make main concise, this function is used
    return {"Smoking": {"Yes": 1, "No": 0},
            "AlcoholDrinking": {"Yes": 1, "No": 0},
            "Stroke": {"Yes": 1, "No": 0},
            "DiffWalking": {"Yes": 1, "No": 0},
            "Sex": {"Other": 0, "Female": 1, "Male": 2},
            "AgeCategory": {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                            "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12,
                            "80 or older": 13},
            "Race": {"Other": 0, "White": 1, "Black": 2, "Asian": 3,
                     "American Indian/Alaskan Native": 4, "Hispanic": 5},
            "Diabetic": {"No": 0, "No, borderline diabetes": 1, "Yes (during pregnancy)": 2, "Yes": 3},
            "PhysicalActivity": {"Yes": 1, "No": 0},
            "GenHealth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
            "Asthma": {"Yes": 1, "No": 0},
            "KidneyDisease": {"Yes": 1, "No": 0},
            "SkinCancer": {"Yes": 1, "No": 0}}


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stopped the program manually.")
        executeFinale()
