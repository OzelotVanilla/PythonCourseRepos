import os
from os import system as run
from util.console import console


def executePrelude() -> None:
    # Install and configure Kaggle command line tool
    installKaggleCommandLineTool()
    prepareKaggleRunningEnv()

    # Use that tool to download datasets
    console.info("Downloading datasets from Kaggle...")
    downloadKaggleDatasets([
        ("johnsmith88/heart-disease-dataset", "heart", "2015"),
        ("kamilpytlak/personal-key-indicators-of-heart-disease", "heart_2020_cleaned", "2020")
    ])


def executeFinale() -> None:
    console.clear()

    # Ask if remove kaggle command line tool
    if input("Do you want to remove kaggle command line tool (type \"yes\" to uninstall)? ") == "yes":
        run("pip uninstall kaggle")
        if input("Also remove config file (type \"yes\" to delete)? ") == "yes":
            deleteKaggleConfig()
        return None

    # Restore backup file if exist
    restoreKaggleConfig()


def installKaggleCommandLineTool() -> None:
    console.clear()
    console.info("Downloading Kaggle tool for downloading datasets.")
    status = run("pip install kaggle")
    if status != 0:
        console.err("Kaggle cannot be installed with Python's pip.")
        print("\tSolution possible: See error above, solve it and re-run.")
        exit()


def prepareKaggleRunningEnv() -> None:
    # Show install success (it should) info
    console.info("Kaggle is installed, preparing Kaggle running environment")

    # Get current user's home directory
    user_home_path = os.path.expanduser("~")

    # Check if there are ".kaggle" folder under current user dir
    if not os.path.exists(user_home_path + "/.kaggle"):
        os.mkdir(user_home_path + "/.kaggle/")
        console.info("Created \"~/.kaggle/\" folder.")

    # Check if "kaggle.json" exist, if exist, backup it
    if (os.path.isfile(user_home_path + "/.kaggle/kaggle.json")
            and os.path.exists(user_home_path + "/.kaggle/kaggle.json")):
        # If the backup also exists (happens when you have run this script for multiple times)
        if os.path.exists(user_home_path + "/.kaggle/kaggle.json.course_projcet_backup"):
            os.remove(user_home_path + "/.kaggle/kaggle.json.course_projcet_backup")
        os.rename(
            user_home_path + "/.kaggle/kaggle.json",
            user_home_path + "/.kaggle/kaggle.json.course_projcet_backup"
        )
        console.info("Found existing config, backing it up...")

    with open(user_home_path + "/.kaggle/kaggle.json", "w") as kaggle_config:
        kaggle_config.write("{\"username\":\"ozelotvanilla\",\"key\":\"0691674bcb905405fa7ccb4738f34ea2\"}")
        console.info("Kaggle config file written.")

    console.info("The python \"kaggle\" module should be correctly installed and configured now")
    print("Continue in 3 seconds")
    console.wait(3)
    console.clear()


def downloadKaggleDatasets(dataset_info_list: list[tuple[str, str]]):
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    for dataset_info in dataset_info_list:
        # Inside the tuple, first is the dataset name, next is its download name
        # third one is name to rename
        status = run(
            "kaggle datasets download --force --unzip -d \"" + dataset_info[0] + "\" -p \"datasets\""
        )
        if status != 0:
            console.err("Kaggle failed doing its job. See error message above and re-run.")
            exit()
        new_file_path = "./datasets/data_" + dataset_info[2] + ".csv"
        if os.path.isfile(new_file_path) and os.path.exists(new_file_path):
            os.remove(new_file_path)
        os.rename("./datasets/" + dataset_info[1] + ".csv", new_file_path)


def deleteKaggleConfig():
    user_home_path = os.path.expanduser("~")
    if os.path.exists(user_home_path + "/.kaggle/kaggle.json"):
        os.remove(user_home_path + "/.kaggle/kaggle.json")
    os.remove(user_home_path + "./.kaggle/")


def restoreKaggleConfig():
    user_home_path = os.path.expanduser("~")
    if os.path.exists(user_home_path + "/.kaggle/kaggle.json.course_projcet_backup"):
        if os.path.exists(user_home_path + "/.kaggle/kaggle.json"):
            os.remove(user_home_path + "/.kaggle/kaggle.json")
        os.rename(
            user_home_path + "/.kaggle/kaggle.json.course_projcet_backup",
            user_home_path + "/.kaggle/kaggle.json"
        )
