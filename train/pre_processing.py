from util.console import console

from math import nan
from sys import api_version
import pandas as pd
import pandas.api as pdapi
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
# from FCBF_module import FCBFK


default_value_to_fill = (-1, 0, -100000, 100000, nan, np.NaN)


def getClassToDigitDict() -> dict[str, dict[str, int]]:
    # This function just return the dict to replace the discriptive data
    # According to the standard of https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
    # To make main concise, this function is used
    return {"HeartDisease": {"Yes": 1, "No": 0},
            "Smoking": {"Yes": 1, "No": 0},
            "AlcoholDrinking": {"Yes": 1, "No": 0},
            "Stroke": {"Yes": 1, "No": 0},
            "DiffWalking": {"Yes": 1, "No": 0},
            "Sex": {"Other": 0, "Female": 1, "Male": 2},
            "Age": {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                    "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12,
                    "80 or older": 13},
            "Race": {"Other": 0, "White": 1, "Black": 2, "Asian": 3,
                     "American Indian/Alaskan Native": 4, "Hispanic": 5},
            "Diabetes": {"No": 0, "No, borderline diabetes": 1, "Yes (during pregnancy)": 2, "Yes": 3},
            "PhysActivity": {"Yes": 1, "No": 0},
            "GenHlth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
            "Asthma": {"Yes": 1, "No": 0},
            "KidneyDisease": {"Yes": 1, "No": 0},
            "SkinCancer": {"Yes": 1, "No": 0}}


def classToDigitReplace(dataframe: pd.DataFrame, replace_dict: dict[str, dict[str, int]]):
    console.info("Converting descriptive data to numbers:")
    for key in replace_dict:
        print("\t", key + ":", replace_dict[key])
    dataframe.replace(replace_dict, inplace=True)
    console.info("Replaced required data.")


# Pre-defined column names for replacement
# For method unifyColNames


def getNameDict():
    return {'HeartDiseaseorAttack': 'HeartDisease',
            'AgeCategory': 'Age',
            'PhysicalHealth': 'PhysHlth',
            'GenHealth': 'GenHlth',
            'HvyAlcoholConsump': 'AlcoholDrinking',
            'Diabetic': 'Diabetes',
            'Smoker': 'Smoking',
            'PhysicalActivity': 'PhysActivity',
            'MentHlth': 'MentalHealth',
            'DiffWalk': 'DiffWalking'}

# Modify the column names of the input data frame (data_2020)
# According to the pre-defined name dictionary


def unifyColNames(*dfs: pd.DataFrame, name_dict: dict[str, str] = getNameDict()) -> None:
    # For each input dataset
    for df in dfs:
        # Get column names
        cols = df.columns.to_list()
        # Repace specific column names according to the name dict
        for i in range(len(cols)):
            if cols[i] in name_dict:
                cols[i] = name_dict[cols[i]]
        df.columns = cols

# Modify the column order of the input DataFrames
# According to the pre-defined order list
# By default use the column order of the first data frame


def unifyColOrder(*dfs: pd.DataFrame, order_list: list[str] = None) -> None:
    if (len(dfs) < 2):
        return None
    # Use default order_list ?
    start = 0
    if order_list is None:
        order_list = dfs[0].columns.to_list()
        start = 1

    # Get columns in order
    for i in range(start, len(dfs)):
        count = 0
        for col_name in order_list:
            # Has column -> exchange two columns
            if col_name in dfs[i].columns:
                dfs[i].insert(loc=count, column=col_name,
                              value=dfs[i].pop(col_name))
                count += 1

# Get the shared features among the input dataframes
# Used in mlPredictValueMakeUp()

def getSharedFeatures(*dfs: pd.DataFrame) -> list:
    if len(dfs) == 0: return None
    if len(dfs) == 1: dfs[0].columns.to_list()
    shared_features = []
    # For every feature in the first data frame
    for feature in dfs[0].columns.to_list():
        # Not in every input dataset -> not a shared feature -> ignore
        for df in dfs[1:]:
            if not (feature in df.columns.to_list()): continue
        # Is a shared feature -> accept
        shared_features.append(feature)
    return shared_features

# Feature Selection Method
# Select influential features from the columns of the dataframe


def selectFeatures(df: pd.DataFrame, train_size=0.8, threshold=0.9, labelColName='HeartDisease'):
    # Split train and test set
    x = df.drop([labelColName], axis=1).values
    y = df.iloc[:, df.columns.to_list().index(
        labelColName)].values.reshape(-1, 1)
    y = np.ravel(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_size, test_size=1 - train_size, random_state=0, stratify=y
    )

    # Conduct feature selection using sklearn
    importances = mutual_info_classif(x_train, y_train)

    # Get all features
    features = df.dtypes[df.dtypes != 'object'].drop(labelColName).index

    # Create pairs of importances and features
    f_list = sorted(
        zip(map(lambda x: round(x, 4), importances), features), reverse=True)

    # Calculate the sum of importance scores
    sum_of_importance = 0
    for i in range(len(f_list)):
        sum_of_importance += f_list[i][0]

    # Select the important features from top to bottom until the accumulated importance reaches threshold
    temp_sum = 0
    selected_features = []
    f_list = sorted(zip(map(lambda x: round(x, 4), importances /
                    sum_of_importance), features), reverse=True)
    for i in range(len(f_list)):
        temp_sum += f_list[i][0]
        selected_features.append(f_list[i][1])
        if temp_sum >= threshold:
            break  # Stop adding features when reaching threshold

    # Return result
    return selected_features


# Make up for missing features (***MakeUp)

# Use Default Value

def defaultValueMakeUp(df_src: pd.DataFrame, df_dist: pd.DataFrame, col_name, default_val=-1, insert_loc=-1):
    # Column exists -> do nothing
    if col_name in df_dist.columns.to_list():
        return None
    # Create a new column and fill default value
    df_dist.insert(loc=insert_loc, column=col_name, value=default_val)

# Use Average Value


def averageValueMakeUp(df_src: pd.DataFrame, df_dist: pd.DataFrame, col_name, default_val=-1, insert_loc=-1):
    # Because all features to make up for are digits that represent certain classifications
    # The average value is the most frequent appearing digit
    average = df_src[col_name].value_counts().index[0]
    # Fill the missing feature with average value
    defaultValueMakeUp(df_src, df_dist, col_name,
                       default_val=average, insert_loc=insert_loc)

# Use ML model to predict the missing values
# according to the shared features between the two datasets


def mlPredictValueMakeUp(df_src: pd.DataFrame, df_dist: pd.DataFrame, col_name, default_val=-1, insert_loc=-1):
    if col_name in df_dist.columns.to_list(): return None
    # Get the shared features between the two datasets
    shared_features = getSharedFeatures(df_src, df_dist)
    # number of features > 10 -> feature selection
    if len(shared_features) > 10:
        df_src_shared = df_src[shared_features.append(col_name)]
        shared_features = selectFeatures(df_src_shared, labelColName=col_name)
    # Train prediction model to generate values
    x = df_src.loc[shared_features].values
    y = df_src.loc[col_name].values
    ###### Default Values ########
    train_size = 0.8
    test_size = 0.2
    ##############################
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_size, test_size=test_size, random_state=0, stratify=y
    )
    pass

# Feature Making up function
# use the specific makeUpFunc to create values


def makeUpAllMissingValue(df_src: pd.DataFrame, df_dist: pd.DataFrame, makeUpFunc=defaultValueMakeUp, default_val=-1):
    # Unify Column Order
    unifyColOrder(df_src, df_dist)
    # Get column names from df_src
    col_names = df_src.columns.to_list()
    # Find missing columns -> defaultValueMakeUp
    for i in range(len(col_names)):
        if not (col_names[i] in df_dist.columns):
            makeUpFunc(df_src=df_src, df_dist=df_dist,
                       col_name=col_names[i], default_val=default_val, insert_loc=i)
            # df_dist.insert(loc=i, column=col_names[i], value=default_val)


# Test code


def main():
    # Read dataset
    # Unify Column Names
    df_2015 = pd.read_csv('datasets/data_2015.csv')
    df_2020 = pd.read_csv('datasets/data_2020.csv')
    unifyColNames(df_2015, df_2020)
    print("Columns after unification:")
    print(df_2015.columns)
    print(df_2020.columns)
    # Feature Selection
    featureSelected = selectFeatures(df_2015, labelColName="HeartDisease")
    df_2015_fs = df_2015[featureSelected]
    # Missing Value Make Up
    makeUpAllMissingValue(df_2015_fs, df_2020, averageValueMakeUp)
    print(df_2015_fs.columns)
    print(df_2020.columns)
    df_2020_fs = df_2020[featureSelected]
    print(df_2020_fs)


if __name__ == '__main__':
    main()
