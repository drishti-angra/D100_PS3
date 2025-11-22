import pandas as pd
import hashlib

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.

def deterministic_split(value, training_frac=0.8):
    """Hash-based split (used when ID column is not numeric)."""
    hashed_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
    return "train" if hashed_value % 100 < training_frac * 100 else "test"


def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (Full dataset available before splitting)
    id_column : str
        Name of ID column. The train/test split is done based on this id_column 
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    if np.issubdtype(df[id_column].dtype, np.integer):
        df['sample']=np.where(df[id_column] % 100 < training_frac*100, "train", "test")
    
    else:
        df['sample']=df[id_column].apply(lambda x: deterministic_split(x, training_frac))


    return df
