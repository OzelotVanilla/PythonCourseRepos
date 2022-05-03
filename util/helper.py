import os
from os import system as run
from util.console import console


def executePrelude():
    console.clear()
    installKaggleCommandLineTool()
    console.info("The python \"kaggle\" module should be correctly installed now")
    console.clear()
    downloadKaggleDatasets([
        ("johnsmith88/heart-disease-dataset", "heart", "2015"),
        ("kamilpytlak/personal-key-indicators-of-heart-disease", "heart_2020_cleaned", "2020")
    ])


def installKaggleCommandLineTool() -> None:
    run("pip install kaggle")


def downloadKaggleDatasets(dataset_info_list: list[tuple[str, str]]):
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    for dataset_info in dataset_info_list:
        # Inside the tuple, first is the dataset name, next is its download name
        # third one is name to rename
        run(
            "kaggle datasets download --force --unzip -d \"" + dataset_info[0] + "\" -p \"datasets\""
        )
        new_file_path = "./datasets/data_" + dataset_info[2] + ".csv"
        if os.path.isfile(new_file_path) and os.path.exists(new_file_path):
            os.remove(new_file_path)
        os.rename("./datasets/" + dataset_info[1] + ".csv", new_file_path)
