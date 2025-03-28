# script written to move currently existing files to new format
# data/output/{var_group}__{var}__{yyyymmdd}.parquet --> 
# data/output/{var_group}/{var}/{var}__yyyymmdd.parquet

import os
import shutil
import glob

# Define the base directory
base_dir = "data/output/"

# Get all matching parquet files
files = glob.glob(os.path.join(base_dir, "*__*__*.parquet"))

for file in files:
    filename = os.path.basename(file)
    
    # Parse the filename structure
    parts = filename.split("__")
    if len(parts) != 3:
        continue  # Skip if the filename doesn't match the expected pattern

    var_group, var, date_ext = parts
    yyyymmdd = date_ext.replace(".parquet", "")

    # Define the new directory and file path
    new_dir = os.path.join(base_dir, var_group, var)
    new_filename = f"{var}__{yyyymmdd}.parquet"
    new_path = os.path.join(new_dir, new_filename)

    # Create the directory if it doesn't exist
    os.makedirs(new_dir, exist_ok=True)

    # Move the file
    shutil.move(file, new_path)
    print(f"Moved {file} → {new_path}")