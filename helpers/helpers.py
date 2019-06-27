import os
import pandas as pd


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
