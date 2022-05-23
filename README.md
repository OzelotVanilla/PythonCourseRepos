# Welcome to this PythonCourseRepos

Only for class use.

## Before running

1. Download the entire project to a folder
2. **IMPORTANT** use VSCode to **open that folder**,
   ensure that the `main.py` is in the root directory after
   opening the folder.
3. Check if the required package is installed.
   If not, please see error message, and install them
   (There is an installation check, if required library not installed, there will be an error).
   * tensorflow
   * numpy
   * keras
   * pandas
   * scikit-learn
   * matplotlib
4. Also, make sure your Python version is greater or equal to 3.10.
   Because Python is interpret code files, some notations will be considered as error,
   but actually they are not error.
5. Make sure the VSCode terminal can access to the internet.

## Program Interaction Range

This program will:

* Use internet connection.
* Call system terminal to run command which will download files.
* Create temp file and delete them.
* Create data inside this project's folder.
* Clean created/modified file outside of project folder after running this program.

This program may:

* Install new module like `kaggle` of Python.


This program **will NOT**:

* Delete existing file without permission.

## Introduction

The Repository for CPS Python 1's assignment and project.
After checking all contents in 
**Before Running** and **Program Interation Range** section,
click `main.py`, then run this file.

The aim of the project is: train a model with dataset `A`,
and let this model be able to predict dataset `B`, which is similar to `A`.

First, the dataset `B` will be proceed to match the column in `A`.
For missing data in `B`, we will fill it.

The main value of filling data will be:
**average**, **mode** (number happens most), **abnormal value** (like `-1`),
**outlier** (too small or too big value), **NaN** (in different libraies).


## Content

`datasets` folder: save training datasets. Will be created during running the program.

`lib` folder: useful tools **from others**.

`train` folder: for training models.

`util` folder: tools that help programming.

`main.py` file: the entry for the project.

## Other Information

For the format, we use autopep8 to manage it.

