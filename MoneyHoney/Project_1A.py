import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm as cm


#######################################################################################################################
# load the data  and correlation matrix to find the correlation between the co efficients
#######################################################################################################################


def Data():
    url = "data_banknote_authentication.txt"
    names = ["variance of Wavelet Transformed image", "skewness of Wavelet Transformed image",
             "curtosis of Wavelet Transformed image", "entropy of image", "class"]
    data = pd.read_csv(url, names=names)
    dataframe = DataFrame(data)

    data.head()
    print(data.groupby('class').size())
    print(data.describe())

    # to store the data into features and the targets
    features = dataframe.iloc[:, 0:3]
    Targets = dataframe.iloc[:, -1]

    return features, Targets, dataframe


def analyze_data(df, top_number):
    corr_mat = df.corr()
    corr_mat *= np.tri(*corr_mat.values.shape, k=-1).T
    corr_mat = corr_mat.stack()
    corr_mat = corr_mat.reindex(corr_mat.abs().sort_values(ascending=False).index).reset_index()
    corr_mat.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    covar_mat = df.cov()
    covar_mat *= np.tri(*covar_mat.values.shape, k=-1).T
    covar_mat = covar_mat.stack()
    covar_mat = covar_mat.reindex(covar_mat.abs().sort_values(ascending=False).index).reset_index()
    covar_mat.columns = ["FirstVariable", "SecondVariable", "Covariance"]
    return corr_mat.head(top_number), covar_mat.head(top_number)


def pair_plot(df):
    sns.set(style='whitegrid', context='notebook')  # set the apearance
    sns.pairplot(df, height=2.5)  # create the pair plots
    plt.show()


def correl_matrix(X):
    # create a figure that's 7x7 (inches?) with 100 dots per inch
    fig = plt.figure(figsize=(7, 7), dpi=100)

    # add a subplot that has 1 row, 1 column, and is the first subplot
    ax1 = fig.add_subplot(111)

    # get the 'jet' color map
    cmap = cm.get_cmap('jet', 30)

    # Perform the correlation and take the absolute value of it. Then map
    # the values to the color map using the "nearest" value
    cax = ax1.imshow(np.abs(X.corr()), interpolation='nearest', cmap=cmap)

    # now set up the axes
    major_ticks = np.arange(0, len(X.columns), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True, which='both', axis='both')
    plt.title('Correlation Matrix')
    ax1.set_xticklabels(X.columns, fontsize=9)
    ax1.set_yticklabels(X.columns, fontsize=12)

    # add the legend and show the plot
    fig.colorbar(cax, ticks=[-0.4, -0.25, -.1, 0, 0.1, .25, .5, .75, 1])
    plt.show()


def cov_matrix(X):
    # create a figure that's 7x7 (inches?) with 100 dots per inch
    fig = plt.figure(figsize=(7, 7), dpi=100)

    # add a subplot that has 1 row, 1 column, and is the first subplot
    ax1 = fig.add_subplot(111)

    # get the 'jet' color map
    cmap = cm.get_cmap('cubehelix', 30)

    # Perform the covariance and take the absolute value of it. Then map
    # the values to the color map using the "nearest" value
    cax = ax1.imshow(np.abs(X.cov()), interpolation='nearest', cmap=cmap)

    # now set up the axes
    major_ticks = np.arange(0, len(X.columns), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True, which='both', axis='both')
    plt.title('Covariance Matrix')
    ax1.set_xticklabels(X.columns, fontsize=9)
    ax1.set_yticklabels(X.columns, fontsize=12)

    # add the legend and show the plot
    fig.colorbar(cax, ticks=[-0.1, 0, 350, 750, 1000, 1500, 2000, 2500])
    plt.show()


#######################################################################################################################
# Main Program Starts Here
#######################################################################################################################

features, Target, df = Data()
corr_matrix, covar_matrix = analyze_data(df, 60)
pair_plot(df)
correl_matrix(df)
cov_matrix(df)
