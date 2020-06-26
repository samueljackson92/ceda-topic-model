import pandas as pd
import xarray as xr
import spacy
from tqdm import tqdm
from pathlib import Path
import json

CRU_PATH = Path('data/raw/cru/')
cru_files = list(CRU_PATH.glob('**/*.nc.gz'))

corpi = []
for file_name in tqdm(cru_files):
    data = {}
    ds = xr.open_dataset(file_name)
    data.update(ds.attrs)
    data['file_name'] = file_name.stem
    data['file_path'] = str(file_name)

    corpi.append(data)
    
with Path('data/processed/cru/cru_corpi.json').open('w') as handle:
    json.dump(corpi, handle)