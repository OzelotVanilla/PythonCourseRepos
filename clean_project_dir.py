import os
from util.console import console
# This Python Script is aimed to restore the project directory to its initial structure

# Recursively remove all files and directory

def removeDirTree(dir_path, ignoring: list[str] = None):
    """
    Parameter \"ignoring: list<string>\" is case sensitive.
    """

    if type(ignoring) == list and len(ignoring) == 0:
        console.warn("Parameter \"ignoring\" was set, but empty.")
    elif ignoring != None and type(ignoring) != list:
        console.warn("Parameter \"ignoring\" was set to invalid value.")

    files_or_dirs = os.listdir(dir_path)
    # Remove contents
    for file_or_dir in files_or_dirs:
        path = os.path.join(dir_path, file_or_dir)
        if type(ignoring) == list and file_or_dir in ignoring:
            console.info(f"File or directory \"{file_or_dir}\" ignored.")
        elif os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            removeDirTree(path)

    # Remove the directory
    if ignoring == None or len(ignoring) == 0:
        os.rmdir(dir_path)
    else:
        console.info("Some files was held, and the directory remained.")
        print("\tHeld files:", str(ignoring))


# It will clean the datasets and all intermediate products of the program (models, maked_up datasets, feature selected datasets, etc.)
# You can generate those files by running main.py (Project_Interactive_Demo.ipynb cannot automaticly download datasets)

def main():
    # Get User Response (Avoid accidental operations)
    console.warn("This python script is going to remove the intermediate product of the program")
    print("You can generate those files again by running main.py")
    console.warn("The following directory will be deleted: ")
    dir_to_del = ['datasets', 'models', 'lib']
    for dir in dir_to_del:
        if os.path.exists(dir):
            if os.path.isdir(dir):
                print('\t{} - directory'.format(dir))
            if os.path.isfile(dir):
                console.warn('Incorrect File Type: Found files instead of directory')
                console.warn('Please check the file: {}'.format(dir))
                console.warn('\nProgram Exited')
                exit(1)
    input_str = input('\n[INPUT] Are you sure to continue (yes/[no]): ')
    # Yes -> Remove the directories
    if input_str.lower() == 'yes':
        console.info('Start Removing')
        for dir in dir_to_del:
            if os.path.exists(dir):
                console.info('Removing \'{}\''.format(dir))
                removeDirTree(dir, ["README.md"])
        console.info('Project Directory Cleaning Complete')
    # No -> Cancel
    else:
        console.info('Project Directory Cleaning Cancelled')


if __name__ == '__main__':
    main()
