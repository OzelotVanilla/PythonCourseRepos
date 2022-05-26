import os
from util.console import console
from util.helper import removeDirTree
# This Python Script is aimed to restore the project directory to its initial structure as cloned from GitHub

# It will clean the datasets and all intermediate products of the program (models, maked_up datasets, feature selected datasets, etc.)
# You can generate those files by running main.py (Project_Interactive_Demo.ipynb cannot automaticly download datasets)

# Recursively remove all files and directory


def main():
    # Get User Response (Avoid accidental operations)
    console.warn("This python script is going to remove the intermediate product of the program")
    print("You can generate those files again by running main.py")
    console.warn("The following directory will be delected: ")
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
    input_str = input('Are you sure to continue (yes/[no]): ')
    # Yes -> Remove the directories
    if input_str.lower() == 'yes':
        console.info('Start Removing')
        for dir in dir_to_del:
            if os.path.exists(dir):
                console.info('Removing \'{}\''.format(dir))
                removeDirTree(dir)
        console.info('Project Directory Cleaning Complete')
    # No -> Cancel
    else:
        console.info('Project Directory Cleaning Canceled')


if __name__ == '__main__':
    main()
