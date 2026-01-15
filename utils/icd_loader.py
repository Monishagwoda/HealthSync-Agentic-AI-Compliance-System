import os
import glob
import pandas as pd

def load_icd_data(folder_path):
    icd_dfs = []
    for file in glob.glob(os.path.join(folder_path, "*.pkl")):
        try:
            obj = pd.read_pickle(file)
            if isinstance(obj, pd.DataFrame):
                icd_dfs.append(obj)
                print(f"Loaded {file} with shape {obj.shape}")
            else:
                print(f"Skipped {file}, not a DataFrame")
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    if icd_dfs:
        return pd.concat(icd_dfs, ignore_index=True)
    else:
        print("No valid ICD data found!")
        return pd.DataFrame(columns=["Code", "Description"])
