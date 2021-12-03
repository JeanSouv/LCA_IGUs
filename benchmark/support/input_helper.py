import os
import pandas as pd

def get_data(filename, directory=ROOT_DIR, force_download=False):
    """Download Excel dataset

    Parameters
    ----------
    filename: string (optional)
        name of an Excel dataset
    dir: string (optional)
        Root_dir, location of the data
    force_download: bool (optional)
        if True, force redownload of data

    Returns
    -------
    data: pandas.DataFrame
    """
    if force_download or not os.path.exists(filename):
        data = pd.ExcelFile(os.path.join(directory, filename))
    return data
