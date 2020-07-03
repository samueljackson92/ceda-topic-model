import click
import pandas as pd
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import json
from pandarallel import pandarallel

def extract_meta_data(file_name, output_folder):
    data = {}
    # Must use decode times = False as some timestamps cannot be
    # correct decoded, causing opening to fail.
    ds = xr.open_dataset(file_name, decode_times=False)
    data.update(ds.attrs)

    data = {k: str(v) for k, v in data.items()}

    data['file_name'] = Path(file_name).stem
    data['file_path'] = str(file_name)

    name = file_name.replace('/', '.')
    name = name if name[0] != '.' else name[1:]
    name = Path(name).with_suffix('.json')

    with (output_folder / name).open('w') as handle:
        json.dump(data, handle)

    return data

@click.command()
@click.argument('input-file')
@click.argument('output-folder')
def main(input_file, output_folder):
    pandarallel.initialize(progress_bar=True)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    def extract(file_name):
        return extract_meta_data(file_name, output_folder)

    file_names = pd.read_csv(input_file, header=0, names=['path'], index_col=None)
    file_names.path.parallel_map(extract)

if __name__ == "__main__":
    main()
