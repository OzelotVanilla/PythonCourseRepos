from util.console import console

import os
from os import system as run
import tensorflow


def executePrelude() -> None:
    # Install and configure Kaggle command line tool
    __installKaggleCommandLineTool()
    __prepareKaggleRunningEnv()

    # Use that tool to download datasets
    console.info("Downloading datasets from Kaggle...")
    __downloadKaggleDatasets([
        ("alexteboul/heart-disease-health-indicators-dataset", "heart_disease_health_indicators_BRFSS2015", "2015"),
        ("kamilpytlak/personal-key-indicators-of-heart-disease", "heart_2020_cleaned", "2020")
    ])


def executeFinale() -> None:
    # Prompt that the whole program is end
    print("\n\n")
    console.info("The whole program is end. Press enter to continue.")
    input()
    console.clear()

    # Ask if remove kaggle command line tool
    if input("Do you want to remove kaggle command line tool (type \"yes\" to uninstall)? ") == "yes":
        console.warn("Uninstalling Kaggle command line tool...")
        run("pip uninstall kaggle")
        if input("Also remove config file (type \"yes\" to delete)? ") == "yes":
            __deleteKaggleConfig()
        console.info("Kaggle command line tool uninstalled.\n")
        return None

    # Restore backup file if exist
    __restoreKaggleConfig()
    console.clear()


def getBestGPUTensorFlow() -> str:
    available_GPU_list = tensorflow.config.list_physical_devices('GPU')
    if type(available_GPU_list) != type([]) or len(available_GPU_list) <= 0:
        console.warn("Assigned to use GPU, but no GPU found by TensorFlow")
        return "/cpu:0"
    else:
        return available_GPU_list[0].name


def __installKaggleCommandLineTool() -> None:
    console.clear()
    console.info("Downloading Kaggle tool for downloading datasets.")

    # Use pip to download Kaggle command line tool
    status = run("pip install kaggle")

    # If not success, this program cannot run anymore
    # So it force exit
    if status != 0:
        console.err("Kaggle cannot be installed with Python's pip.")
        print("\tSolution possible: See error above, solve it and re-run.", end="\n\n")
        exit()


def __prepareKaggleRunningEnv() -> None:
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

    # Write Ozelot's valid API key for kaggle to download
    # Please note that this key may become INVALID at any time after this semester
    with open(user_home_path + "/.kaggle/kaggle.json", "w") as kaggle_config:
        kaggle_config.write("{\"username\":\"ozelotvanilla\",\"key\":\"0691674bcb905405fa7ccb4738f34ea2\"}")
        console.info("Kaggle config file written.")

    console.info("The python \"kaggle\" module should be correctly installed and configured now")
    print("Continue in 4 seconds...")
    console.wait(4)
    console.clear()


def __downloadKaggleDatasets(dataset_info_list: list[tuple[str, str]]):
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    for dataset_info in dataset_info_list:
        # Inside the tuple, first is the dataset name, next is its download name
        # third one is name to rename downloaded file
        status = run(
            "kaggle datasets download --force --unzip -d \"" + dataset_info[0] + "\" -p \"datasets\""
        )

        # If not success, this program cannot run anymore
        if status != 0:
            console.err("Kaggle failed doing its job. See error message above and re-run.")
            exit()

        # Renaming datasets with name given
        new_file_path = "./datasets/data_" + dataset_info[2] + ".csv"
        if os.path.isfile(new_file_path) and os.path.exists(new_file_path):
            try:
                os.remove(new_file_path)
            except PermissionError:
                console.err(
                    "There is no permission writing file. Did you open that?"
                )
                for info_tuple in dataset_info_list:
                    os.remove("./datasets/" + info_tuple[1] + ".csv")
                exit()
        os.rename("./datasets/" + dataset_info[1] + ".csv", new_file_path)


def __deleteKaggleConfig():
    # If user want to uninstall, delete kaggle config folder under current user's folder
    user_home_path = os.path.expanduser("~")
    if os.path.exists(user_home_path + "/.kaggle/kaggle.json"):
        os.remove(user_home_path + "/.kaggle/kaggle.json")
    os.remove(user_home_path + "./.kaggle/")


def __restoreKaggleConfig():
    # If there is backuped settings, restore them
    user_home_path = os.path.expanduser("~")
    if os.path.exists(user_home_path + "/.kaggle/kaggle.json.course_projcet_backup"):
        if os.path.exists(user_home_path + "/.kaggle/kaggle.json"):
            os.remove(user_home_path + "/.kaggle/kaggle.json")
        os.rename(
            user_home_path + "/.kaggle/kaggle.json.course_projcet_backup",
            user_home_path + "/.kaggle/kaggle.json"
        )
