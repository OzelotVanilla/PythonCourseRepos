from pandas import DataFrame, Series


def splitOneColumn(dataframe: DataFrame, split_column_name: str) -> tuple[Series, DataFrame]:
    # Split the result column and other column
    # Return as (result, others)
    return (dataframe[split_column_name], dataframe.drop(columns=[split_column_name]))
