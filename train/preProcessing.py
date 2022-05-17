from typing import Any
import pandas as pd

# Pre-defined column names for replacement
# For method unifyColNames
def get_name_dict():
    return {'HeartDiseaseorAttack': 'HeartDisease',
        'AgeCategory': 'Age',
    'PhysicalHealth': 'PhysHlth',
    'GenHealth': 'GenHlth',
    'HvyAlcoholConsump': 'AlcoholDrinking',
    'Diabetic': 'Diabetes',
    'Smoker': 'Smoking',
    'PhysicalActivity': 'PhysActivity',
    'MentHlth': 'MentalHealth',
    'DiffWalk': 'DiffWalking',
    
    }

# Modify the column names of the input data frame (data_2020)
# According to the pre-defined name dictionary
def unifyColNames(*dfs: pd.DataFrame, name_dict = get_name_dict()) -> None:
    # For each input dataset
    for df in dfs:
        # Get column names
        cols = df.columns.to_list()
        # Repace specific column names according to the name dict
        for i in range(len(cols)):
            if cols[i] in name_dict: cols[i] = name_dict[cols[i]]
        df.columns = cols