
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

data_dir = Path("data/raw/grazpedwri-dx")

# 1. Inspect YOLO class configuration
with ZipFile(data_dir / "folder_structure.zip") as archive:
    yaml_files = [
        name for name in archive.namelist()
        if name.endswith(".yaml")
    ]

    for name in yaml_files:
        print("YAML:", name)
        print(archive.read(name).decode("utf-8"))

# 2. Inspect CSV metadata
df = pd.read_csv(data_dir / "dataset.csv")

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())
