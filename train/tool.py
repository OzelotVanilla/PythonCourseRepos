from util.console import console

from pandas import DataFrame, Series
from pandas import read_csv


def readCSV(path: str, /, column_name_unify=None, descr_convert: dict[str, dict[str, int]] = None) -> DataFrame:
    dataframe = read_csv(path)

    # If need to change descriptive data to numbers
    if descr_convert != None:
        console.info("Converting descriptive data to numbers:")
        for key in descr_convert:
            print("\t", key + ":", descr_convert[key])
        dataframe.replace(descr_convert, inplace=True)
        console.info("Replaced required data.")

    return dataframe


def splitOneColumn(dataframe: DataFrame, split_column_name: str) -> tuple[Series, DataFrame]:
    # Split the result column and other column
    # Return as (result, others)
    return (dataframe[split_column_name], dataframe.drop(columns=[split_column_name]))
