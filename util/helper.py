from importlib.util import find_spec as hasModule
from util.console import console

import os
from os import system as run
import requests
import tensorflow


__module_to_check = ["tensorflow", "keras", "numpy", "pandas", "sklearn"]


def prepareEnv() -> None:
    # Check if required package is installed
    __checkIfRequiredModuleInstalled()

    # Download required util from GitHub
    # __downloadFromWebRaw([
    #     {
    #         "url": "https://raw.githubusercontent.com/SantiagoEG/FCBF_module/master/FCBF_module.py",
    #         "path": "lib/",
    #         "name": "FCBF_module.py"
    #     }
    # ])

    # Install and configure Kaggle command line tool
    __installKaggleCommandLineTool()
    __prepareKaggleRunningEnv()

    # Use that tool to download datasets
    console.info("Downloading datasets from Kaggle...")
    __downloadKaggleDatasets([
        ("alexteboul/heart-disease-health-indicators-dataset", "heart_disease_health_indicators_BRFSS2015", "2015"),
        ("kamilpytlak/personal-key-indicators-of-heart-disease", "heart_2020_cleaned", "2020")
    ])


def cleanEnv() -> None:
    # Prompt that the whole program is end
    print("\n\n")
    console.info("The whole program is end. Press enter to continue.")
    input()
    console.clear()

    # Ask if remove kaggle command line tool
    if input("Do you want to remove kaggle command line tool (type \"yes\" to uninstall, enter=no)? ") == "yes":
        __uninstallKaggle()

    # Restore backup file if not uninstall and it exist
    else:
        __restoreKaggleConfig()
        console.clear()


def getBestGPUTensorFlow() -> str:
    available_GPU_list = tensorflow.config.list_physical_devices('GPU')
    if type(available_GPU_list) != type([]) or len(available_GPU_list) <= 0:
        console.warn("Assigned to use GPU, but no GPU found by TensorFlow")
        return "/cpu:0"
    else:
        return available_GPU_list[0].name


def __checkIfRequiredModuleInstalled() -> None:
    console.info("Checking if required modules are installed...")
    should_exit = False

    for mod_name in __module_to_check:
        if not hasModule(mod_name):
            console.err("No module named \"" + mod_name + "\". Try to install using pip3, and re-run.")
            should_exit = True
        elif not should_exit:
            print("\tSatisfied:", mod_name)
    exit() if should_exit else None
    console.info("All required modules found.")
    console.wait(2)


def __downloadFromWebRaw(req_list: list[dict[str, ]]) -> None:
    console.info("Downloading raw content from web...")

    # Using dict to send request info in a list
    for req_info in req_list:
        # First, get dir, create if not exist
        path_dir = req_info["path"]
        if not (os.path.exists(path_dir) and os.path.isdir(path_dir)):
            os.mkdir(path_dir)
        with open(os.path.join(path_dir, req_info["name"]), "wb") as output_file:
            content_url = req_info["url"]
            print("\tDownloading from \"" + str(content_url) + "\"...", end="")
            output_file.write(requests.get(content_url).content)
            print("\tDone.")
    console.info("All required things downloaded.")
    console.wait(2)


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
    if (os.path.exists(user_home_path + "/.kaggle/kaggle.json" and os.path.isfile(user_home_path + "/.kaggle/kaggle.json"))):
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
        if os.path.exists(new_file_path) and os.path.isfile(new_file_path):
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


def __uninstallKaggle() -> None:
    console.warn("Uninstalling Kaggle command line tool...")
    run("pip uninstall kaggle")
    if input("Also remove config file (type \"yes\" to delete)? ") == "yes":
        __deleteKaggleConfig()
    console.info("Kaggle command line tool uninstalled.\n")


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
