import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

np.warnings.filterwarnings('ignore')

import scipy
from scipy.stats import kendalltau

import pandas as pd

import matplotlib.pylab as plt

plt.switch_backend('TkAgg')

import seaborn as sns

sns.set()
sns.set_style("whitegrid", {'axes.grid': False})


class CSV:

    def __init__(self, fname):
        self._filename = fname

    def readCSV(self, header=None, drop=False, usecols=None):
        """
        Reads a csv file into a Pandas Data Frame.

        :param fname: Filename (including its path).
        :return: Pandas DF
        """

        filename = self._filename
        df = pd.read_csv(filename, sep=',', header=header, usecols=usecols)

        if drop:
            df.dropna(how="all", inplace=True)
        return df


class Plotting:

    @staticmethod
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

    @staticmethod
    def plotScatterWithColorBar(x, y, data, color, color_title, xlim=None, ylim=None, save=False, path=None):
        """

        :param x: x-variable
        :param y: y-variable
        :param data: Data as Pandas DF
        :param color: Data column to be used for color plotting
        :param color_title: Title of the color bar
        :param xlim: X-axis range
        :param ylin: Y-axis range
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

        points = plt.scatter(data[x], data[y], c=data[color], s=250, cmap="Blues")

        cbar_title = color_title
        cb = plt.colorbar(points)
        cb.set_label(cbar_title)
        g = sns.regplot(x=x, y=y, data=data, scatter=False)

        g.set(xlim=(xlim[0], xlim[1]))
        plt.show()

        if save:
            plt.savefig(path, format='png')

    @staticmethod
    def plotScatterMatrix(data, columns=[None, None], hue=None):
        """

        :param data:
        :param columns:
        :param hue:
        :return:
        """

        sns.set(style="ticks")
        start = columns[0]
        end = columns[1]
        sns.pairplot(data.ix[start:end], hue=hue)

    @staticmethod
    def plotScatterWithVaryingSizeAndHue(x, y, data, hue=None, size=None, sizes=(40, 400), alpha=.5, palette="muted",
                                         height=6, xlim=None, ylim=None, save=False, path=None):
        """

        :param x:
        :param y:
        :param data:
        :param hue:
        :param size:
        :param sizes:
        :param alpha:
        :param palette:
        :param height:
        :return:
        """
        sns.set(color_codes=True)
        sns.set(font_scale=2.5)

        if xlim is not None:
            plt.xlim(xmin=xlim[0])
            plt.xlim(xmax=xlim[1])
        if ylim is not None:
            plt.xlim(xmin=ylim[0])
            plt.xlim(xmax=ylim[1])

        sns.relplot(x=x, y=y, hue=hue, size=size, sizes=sizes, alpha=alpha, palette=palette, height=height, data=data)

        if save:
            plt.savefig(path, format='png')

    @staticmethod
    def plotLine(data):
        sns.set(style="whitegrid")
        sns.lineplot(data=data, palette="tab10", linewidth=2.5)

    @staticmethod
    def plotHist(x, save=False, path=None):
        """

        :param x:
        :param save:
        :param path:
        :return:
        """

        sns.distplot(x)
        if save:
            plt.savefig(path, format='png')

    @staticmethod
    def plotCorrelationMatrix(data):
        """

        :param data:
        :return:
        """

        corr = data.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # @staticmethod
    # def plotKernelDensityEstimate():
    #     """
    #
    #     :return:
    #     """
    #
    #     sns.set()
    #     sns.jointplot(x1, x2, kind="kde", height=7, space=0)

    @staticmethod
    def plotHeatMap(x, y, z, data, save=False, path=None):
        """

        :param x:
        :param y:
        :param z:
        :param data:
        :param save:
        :param path:
        :return:
        """

        sns.set()
        df = data.pivot(x, y, z)
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(df, annot=True, linewidths=.5, ax=ax)

        if save:
            plt.savefig(path, format='png')


class Stats:

    @staticmethod
    def linearRegression(x, y, data, save=False, path=None, useSklearn=False):
        """
        Creates a linear regression model given x and y variables.

        :param x:
        :param y:
        :param data:
        :param save:
        :param path:
        :return:
        """

        X, Y = np.asarray(data[x]), np.asarray(data[y])
        X = X.reshape(-1, 1)

        #  remove row if NaN
        X, Y = X[~np.isnan(X).reshape(-1, 1).any(axis=1)], Y[~np.isnan(Y).reshape(-1, 1).any(axis=1)]

        if useSklearn:
            from sklearn.linear_model import LinearRegression

            fullmodel = LinearRegression()
            fullmodel.fit(X, Y)
            yhat = fullmodel.predict(X)
            SS_Residual = sum((Y - yhat) ** 2)
            SS_Total = sum((Y - np.mean(Y)) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1 - (1 - r_squared) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)

        _, _, r_value, p_value, _ = scipy.stats.linregress(X.reshape(1, -1)[0], Y)

        if save:
            output_file = path
            with open(output_file, 'w') as opfile:
                opfile.write("\n\t==============================================\n")
                opfile.write("\n\tStatistics of %s vs %s  \n\n" % (x, y))
                opfile.write("\n\t==============================================\n")
                opfile.write("\tR Value: \n")
                opfile.write("\t\t\t\t\t\t\t\t\t%0.9f\n" % (r_value))
                opfile.write("\tP Value: \n")
                opfile.write("\t\t\t\t\t\t\t\t\t%0.9f\n" % (p_value))
                opfile.write("\tR Value (M): \n")
                opfile.write("\n\t==============================================\n")
                opfile.write("\n")

        return [r_value, p_value]

    @staticmethod
    def multivariateLinearRegression(X, y, data, save=False, path=None, useSklearn=False):
        """

        :param X:
        :param y:
        :param data:
        :param save:
        :param path:
        :param useSklearn:
        :return:
        """
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        X, Y = np.asarray(data[X]), np.asarray(data[y])
        X = X.reshape(-1, 1)

        #  remove row if NaN
        X, Y = X[~np.isnan(X).reshape(-1, 1).any(axis=1)], Y[~np.isnan(Y).reshape(-1, 1).any(axis=1)]

        X = sm.add_constant(X)
        est = sm.OLS(y, X).fit()
        print(est.summary())
