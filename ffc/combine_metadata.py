import click
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_meta_data(path: Path) -> dict: 
    with path.open('r') as handle:
        doc = json.load(handle)
    return doc

@click.command()
@click.argument('input-folder')
@click.argument('output-file')
def main(input_folder, output_file):
    input_folder = Path(input_folder)
    meta_data_files = list(input_folder.glob('*.json'))
    
    docs = [load_meta_data(path) for path in tqdm(meta_data_files)]
    doc_df = pd.DataFrame(docs)
    doc_df = doc_df.fillna('')
    
    doc_df.to_hdf(output_file, key='data')

if __name__ == "__main__":
    main()