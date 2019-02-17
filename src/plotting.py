import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import numpy as np
np.warnings.filterwarnings('ignore')

import scipy
from scipy.stats import kendalltau

import pandas as pd

import matplotlib.pylab as plt
plt.switch_backend('TkAgg')

import seaborn as sns
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})


def readCSV(fname, header=None, drop=False, usecols=None):
    """
    Reads a csv file into a Pandas Data Frame.

    :param fname: Filename (including its path).
    :return: Pandas DF
    """

    filename = fname
    df = pd.read_csv(filename, sep=',', header=header, usecols=usecols)

    if drop:
        df.dropna(how="all", inplace=True)
    return df
