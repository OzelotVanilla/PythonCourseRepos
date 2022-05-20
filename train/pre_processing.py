from math import nan
from sys import api_version
import pandas as pd
import pandas.api as pdapi
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
# from FCBF_module import FCBFK


default_value_to_fill = (-1, 0, -100000, 100000, nan, np.NaN)


def getCSV2020ReplaceDict() -> dict[str, dict[str, int]]:
    # This function just return the dict to replace the discriptive data
    # According to the standard of https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
    # To make main concise, this function is used
    return {"HeartDisease": {"Yes": 1, "No": 0},
            "Smoking": {"Yes": 1, "No": 0},
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
    average = df_src[col_name].mean()
    if pdapi.types.is_integer_dtype(df_src[col_name].dtype): average = round(average)
    # average = round(average) 
    defaultValueMakeUp(df_src, df_dist, col_name,
                       default_val=average, insert_loc=insert_loc)

# Use Default Value (make up for all missing value)


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
    # # Unify Column Order
    # unifyColOrder(df_2015, df_2020)
    # print(df_2015.columns)
    # print(df_2020.columns)
    # Feature Selection
    featureSelected = selectFeatures(df_2015, labelColName="HeartDisease")
    df_2015_fs = df_2015[featureSelected]
    # Missing Value Make Up
    makeUpAllMissingValue(df_2015_fs, df_2020, averageValueMakeUp)
    print(df_2015_fs.columns)
    print(df_2020.columns)
    df_2020_fs = df_2020[featureSelected]
    print(df_2020_fs)
    # # Feature Selection (sklearn.feature_selection.mutual_info_classif)
    # selected_features = selectFeatures(df, threshold=0.6)
    # print("Features selected:")
    # print(selected_features)
    # # # Feature Selection (FCBF)
    # # x_selected = df[selectedFeatures].values
    # # # print(x_selected.shape)
    # # y = df.iloc[:, df.columns.to_list().index('HeartDisease')].values
    # # # print(y)
    # # fcbf = FCBFK(k = 6)
    # # x_fcbf_selected = fcbf.fit_transform(x_selected, y)
    # # print(x_fcbf_selected)


if __name__ == '__main__':
    main()
