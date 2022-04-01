import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image


def walk(path) -> Path:
    """Generator that walks down a directory and yields Path objects pointing to .TIF files.""" 
    for p in Path(path).iterdir(): 
        if p.is_dir(): 
            yield from walk(p)
            continue

        if p.suffix != '.TIF':
            continue

        yield p

def parse_image_filename(filepath: Union[Path, str]) -> dict:
    """Parse the column, row, field, and channel from a filepath

    Args:
        filepath (Union[Path, str]): filepath to a single image file.

    Raises:
        ValueError: if the file is not in the expected format

    Returns:
        dict: dict containing the values parsed from the filename.
    """

    if isinstance(filepath, str):
        filepath = Path(filepath)

    sample_str = filepath.stem.split('_')[-1]
    sample_pattern = '(?P<column>[A-Z])(?P<row>\d\d)f(?P<field>\d\d)d(?P<channel>\d)'
    match = re.match(sample_pattern, sample_str)
    if match is None:
        raise ValueError(f'filepath does not match expected pattern: {sample_str}')
    
    vals = match.groupdict()
    vals['row'] = int(vals['row'])
    vals['field'] = int(vals['field'])
    vals['channel'] = int(vals['channel'])
    return vals

def file_to_nparray(filepath: Union[Path, str]) -> np.array:  
    """Accepts filepath of a TIF image and returns an image of the array."""
    img = Image.open(filepath)
    img_array = np.asarray(img)
    return img_array

def make_images_df(data_dir='MFGTMP_220317120003', output_file='images_df.csv') -> None:
    """Finds all image files in the data directory, parses their filenames, and compiles them into a dataframe.

    Args:
        data_dir (str, optional): filepath of directory holding image files. Defaults to 'MFGTMP_220317120003'.
        output_file (str, optional): file to write dataframe to. will be written as a csv. Defaults to 'images_df.csv'.
    """
    rows = []
    for image_file in walk(data_dir):
        row = parse_image_filename(image_file)
        row['rel_fp'] = str(image_file)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

def load_images_df(csv_file='images_df.csv'):
    return pd.read_csv(csv_file)


if __name__ == "__main__":
    test_str = 'MFGTMP_220317120003/MFGTMP_220317120003_A01f00d0.TIF'
    test_path = Path(test_str)
