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


