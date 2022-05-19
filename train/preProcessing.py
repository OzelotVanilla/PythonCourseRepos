import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import numpy as np

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
            if cols[i] in name_dict: cols[i] = name_dict[cols[i]]
        df.columns = cols

# Feature Selection Method
# Select influential features from the columns of the dataframe
def selectFeatures(df: pd.DataFrame, train_size=0.8, threshold=0.9, labelColName='HeartDisease'):
    # Split train and test set
    x = df.drop([labelColName],axis=1).values
    y = df.iloc[:, df.columns.to_list().index(labelColName)].values.reshape(-1,1)
    y = np.ravel(y)
    x_train, x_test, y_train, y_test = train_test_split(x , y, train_size = train_size, test_size = 1-train_size, random_state = 0, stratify = y)
    # Conduct feature selection using sklearn
    importances = mutual_info_classif(x_train, y_train)
    # Get all features
    features = df.dtypes[df.dtypes != 'object'].drop(labelColName).index
    # Create pairs of importances and features
    f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
    # Calculate the sum of importance scores
    Sum = 0
    for i in range(len(f_list)):
        Sum += f_list[i][0]
    # Select the important features from top to bottom until the accumulated importance reaches threshold
    tempSum = 0
    selectedFeatures = []
    f_list = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
    for i in range(len(f_list)):
        tempSum += f_list[i][0]
        selectedFeatures.append(f_list[i][1])
        if tempSum >= threshold: break  # Stop adding features when reaching threshold
    # Return result
    return selectedFeatures

# Test code
def main():
    # Read dataset
    df = pd.read_csv('datasets/data_2015.csv')
    # Unify Column Names
    unifyColNames(df)
    print("Columns after unification:")
    print(df.columns)
    print()
    # Feature Selection
    selectedFeatures = selectFeatures(df, threshold=0.6)
    print("Features selected:")
    print(selectedFeatures)

if __name__ == '__main__':
    main()