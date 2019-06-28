import os
import pandas as pd
import numpy as np


def all_paths_in_dir(path, file_type=".txt"):
    files = []
    for r, d, f in os.walk(path):
        for file_path in f:
            if file_type in file_path:
                files.append(os.path.join(r, file_path))
    if not files:
        print("No {} files in {}, is it correct path?".format(file_type, path))
    return files


def load_excel_files_into_df(file_path, file_type):
    parking_data = all_paths_in_dir(file_path, file_type)
    df_list = []
    for elem in parking_data:
        df_list.append(pd.read_excel(elem))
    print("All files in " + file_path + " loaded!")
    return pd.concat(df_list)


def split_dataframe(df, split_elements_list):
    """
    Splits a dataframe and returns two new dfs with the split_elements in the new df
    :param df: dataframe to be splitted
    :param split_elements_list: array of elements to remove from old df (x) and place into new (y)
    :return: df and y (new df)
    """
    y = df.filter(split_elements_list)
    x = df.drop(split_elements_list, axis=1)

    return x, y
