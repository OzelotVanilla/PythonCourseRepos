# Cross-Dataset Model Adaptation Feasibility Study

A CPS 3320 Python Programming Course Project - Wenzhou-Kean University

## Before running

1. Download the entire project to a folder (Recommended: Clone the project from GitHub)
2. **IMPORTANT** use VSCode to **open that folder**,
   ensure that the `main.py` and `Project_Interactive_Demo.ipynb` is in the root directory after
   opening the folder.
3. Check if the required package is installed.
   If not, please see error message, and install them
   (There is an installation check in `main.py`, if required library not installed, there will be an error).
   * tensorflow
   * numpy
   * keras
   * pandas
   * scikit-learn
   * matplotlib
4. Also, make sure your Python version is greater or equal to 3.9.
   Because Python is interpret code files, some notations will be considered as error,
   but actually they are not error.
5. Make sure the VSCode terminal can access to the internet. (For automatic dataset downloads)

## Program Interaction Range

This program will:

* Use Internet connection.
* Call system terminal to run command which will download files.
* Create temp file and delete them.
* Create data inside this project's folder.
* Clean created/modified file outside of project folder after running this program.

This program may:

* Install new python packages like `kaggle`.


This program **will NOT**:

* Delete existing file without permission.

## Project Introduction

### Motivation
The project is aimed to verify the feasibility of using an already **trained model** to do prediction on **a newer dataset** about the same topic with slightly different feature settings, which is supposed to save the **time cost** of training a new model for each dataset when the population is large and make it easier to use existing trained models on the Internet.
The Repository is for CPS 3320 Python Programming's course project.  

<!-- 
### Main Idea
The aim of the project is: train a model with dataset `A`,
and let this model be able to predict dataset `B`, which is similar to `A`.

First, the dataset `B` will be proceed to match the column in `A`.
For missing data in `B`, we will fill it.

The main value of filling data will be:
**average**, **mode** (number happens most), **abnormal value** (like `-1`),
**outlier** (too small or too big value), **NaN** (in different libraies). -->

### Datasets
The project uses two heart disease datasets from Kaggle  
The datasets are excerpts of the annual CDC survey data. And each of them has more than `250k` of records
- [Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) shows the data of year `2015`
    
    253680 records   22 columns

- [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) shows the data of year `2020`
    
    319795 records   18 columns
    
The two datasets has a **difference** in their **data and available features** that can cause problem to the universal use of models  
Using the normal entrence of the program `main.py` to run the project will automatically download the datasets from Kaggle

### Major Steps
1. Train a model for the `2015 dataet` for cross-dataset adaptation experiment  
   
2. Make up the missing values reqired by the trained model that the `2020 dataset` does not have.  
   
   1. Default Value Filling  
   
   2. Average Value Filling  
   
   3. ML Model Prediction  
   
3. Test out the performance of the `"maked up"` data of 2020 in predicting the heart disease  

4. Train a new model for the '2020 dataset` as a comparison  

5. Analyze the result  

## Content

`datasets` directory: save training datasets. Will be created during running the program.

`models` directory: save trained models. Will be created during running the program.

`images` directory: project output graph example.

~~`lib` directory: useful tools **from others**.~~

`train` directory: self-defined python modules for training models.

`util` directory: self-defined tools that help programming.

`clean_project_dir.py` file: remove all datasets and intermediate product of the project.

`main.py` file: the entry for the project.

`Project_Interactive_Demo.ipynb` file: the interactive version of the project with code explanation

## Usage
Make sure you have checked all contents in **Before Running** and **Program Interation Range** section.  
### main.py & Project_Iteractive_Demo.ipynb
Both `main.py` and `Project_Interactive_Demo.ipynb` contains the main logic of the program.  
The following listed some differences between them:  
* `main.py` offers automatic dataset download and runtime environment check. `Project_Interactive_Demo.ipynb` does not.
* `main.py` is a automate running program that provides no pauses in the middle. `Project_Interactive_Demo.ipynb` is an interactive demo of the program, which allows users to run the program step by step and check the intermediate product. In addition, utilizing the itermediate product saved in files, `Project_Interactive_Demo.ipynb` offers Check Points for the users to do quick evalutions to certain parts of the program.
* Functional code in `main.py` is extensively encapsulated in self-defined methods for programming convenience. Code in `Project_Interactive_Demo.ipynb` is relatively easy to understand as it uses more standard APIs. 
* `main.py` is a regular python file that can be directly run in command line or VSCode. `Project_Interactive_Demo.ipynb` is a jupyter notebook file which needes `jupyter` package to open. It is recommended to install the `Jupyter` extension and open the notebook in VSCode, which will guide you to configure the jupyter environment.
### Which one to choose?  
Both `main.py` and `Project_Interactive_Demo.ipynb` are available entrance of the project program. However, directly running the `Project_Interactive_Demo.ipynb` requires the user to manually download the datasets (if not included).
### Recommended Order
It is recommended to **firstly** run `main.py` to download the datasets and generate the intermediate products, which will be saved in files, **then** employ the `Project_Iteractive_Demo.ipynb` to evalute the program using check points to qickly resume the progress from the intermediate files.

## References
[Intrusion_Detection_Using_CICIDS2017](https://github.com/arif6008/Intrusion_Detection_Using_CICIDS2017)  
[Deep learning under the keras framework (2) two-class and multi-class problems](https://blog.51cto.com/u_15060545/4350777)

## Other Information

autopep8 was used as the format manager of the python files.

