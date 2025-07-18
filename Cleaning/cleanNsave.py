from clean import cleanse_df
import os
import pandas as pd

UNCLEANED_DIR = "datasets"
CLEAN_DIR = "cleaned_dataset"

for filename in os.listdir(UNCLEANED_DIR):
    if filename.endswith(".csv"):
        uncleaned_path = os.path.join(UNCLEANED_DIR, filename)
        df = pd.read_csv(uncleaned_path)

        try:
            cleaned_df = cleanse_df(df)
            cleaned_path = os.path.join(CLEAN_DIR, filename)
            cleaned_df.to_csv(cleaned_path)
            print(f"Cleaned and saved: {filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")