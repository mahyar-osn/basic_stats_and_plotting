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


def plotSimpleScatter(x, y, data, xlim=None, ylim=None, save=False, path=None):
    """

    :param x: x-variable
    :param y: y-variable
    :param data: Data as Pandas DF
    :param xlim: X-axis range
    :param ylim: Y-axis range
    :param save: Boolean for saving the plot
    :param path: Path to save the plot
    :return:
    """

    sns.set(color_codes=True)
    sns.set(font_scale=2.5)
    plt.figure(figsize=(11, 10))

    if xlim is not None:
        plt.xlim(xmin=xlim[0])
        plt.xlim(xmax=xlim[1])
    if ylim is not None:
        plt.xlim(xmin=ylim[0])
        plt.xlim(xmax=ylim[1])

    sns.regplot(x=x, y=y, data=data, scatter_kws={"s": 100})
    plt.show(0)

    if save:
        plt.savefig(path, format='png')
