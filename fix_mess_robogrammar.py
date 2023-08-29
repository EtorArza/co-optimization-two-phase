import glob
import re
import shutil

file_pattern = "results/robogrammar/data/reevaleachvsend_*_FlatTerrainTask_1.0_1.0_*.txt"
matching_files = glob.glob(file_pattern)

for file in matching_files:
    match = re.search(r'_([0-9]+)\.txt$', file)
    if match:
        random_seed = int(match.group(1))
        new_filename = f"results/robogrammar/data/proposedmethod_{540 + random_seed - 1}_FlatTerrainTask_standard_{random_seed}.txt"
        shutil.copy(file, new_filename)
        print(f"Copied '{file}' to '{new_filename}'")
    else:
        raise ValueError(f"Regex did not match for file: {file}")
