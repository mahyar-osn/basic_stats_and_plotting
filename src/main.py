import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from analysis import CSV, Plotting, Stats


""" A dict for some parameter configuration """
config = dict()
config['root'] = '../data/'
config['file_name'] = 'HLA_data.csv'
config['variables'] = ['age', 'Height', 'Weight', 'BMI', 'fvc_pred']

""" File name """
filename = os.path.join(config['root'], config['file_name'])

csv = CSV(filename)
plot = Plotting()
stats = Stats()

""" Read the data file into data frame """
df = csv.readCSV(header=0, drop=True, usecols=config['variables'])

print('done')
