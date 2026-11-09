
import pandas as pd

def read_excel_file(file_path):

    excel_file = pd.ExcelFile(file_path)
    dataframes = {sheet_name: excel_file.parse(sheet_name) for sheet_name in excel_file.sheet_names}


    dataframes = {sheet_name: df for sheet_name, df in dataframes.items() if len(df) == 870}


    for key in dataframes.keys():
        df = dataframes[key]
        df.set_index(df.iloc[:, 0], inplace=True)
        df.index.name = None
        df = df.iloc[:, 1:]
        dataframes[key] = df
    return dataframes

def restructure_dataframes(dataframes):

    new_dataframes = {}
    for attr_name, df in dataframes.items():
        for sample_name in df.columns:
            if sample_name not in new_dataframes:
                new_dataframes[sample_name] = pd.DataFrame(index=df.index)
            new_dataframes[sample_name][attr_name] = df[sample_name]

    return new_dataframes

def remove_keys(dataframes, keys):


    if isinstance(keys, str):
        keys = [keys]


    for key in keys:
        try:
            del dataframes[key]
        except KeyError:
            pass

def main(file_path, keys_to_remove):

    dataframes = read_excel_file(file_path)
    print(f"Initial keys: {list(dataframes.keys())}")

    new_dataframes = restructure_dataframes(dataframes)

    key_list = list(new_dataframes.keys())


    print(f"Last key before deletion: {key_list[-1]}")

    remove_keys(new_dataframes, keys_to_remove)

    key_list = list(new_dataframes.keys())


    print(f"Last key after deletion: {key_list[-1]}" if key_list else "No keys left.")

    return new_dataframes


if __name__ == "__main__":
    excel_file_path = './neuronID/summarize_181013_modified.xlsx'


    single_key_to_remove = '141212_White_Straightened_Worm_Neuron_Locations_12-12-2014.txt'
    neuronID_dfs = main(excel_file_path, single_key_to_remove)
