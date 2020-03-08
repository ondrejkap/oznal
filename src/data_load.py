import os
import sys
import pandas as pd


def get_data_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def load_data():
    return pd.read_csv('../data/raw/winemag-data-130k-v2.csv', index_col=0)

