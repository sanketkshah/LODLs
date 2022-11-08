import argparse
from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import seaborn as sns
sns.set_theme(style="darkgrid")
 
if __name__ == '__main__':
    # Get hyperparams from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--title', type=str, default="")
    args = parser.parse_args()

    # Read data from file
    df = pd.read_csv(args.file, header=None, names=["m", "c"])

    # Plot
    plt.xlabel("Feature", fontsize=25)
    plt.ylabel("Prediction", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim(-1, 1)
    plt.ylim(-8, 8)
    plt.locator_params(nbins=4)
    #   Get x values
    axes = plt.gca()
    x_min, x_max = axes.get_xlim()
    x_vals = np.linspace(x_min, x_max, 1000)
    #   Plot true function
    y_true = 10 * np.power(x_vals, 3) - 6.5 * x_vals
    plt.plot(x_vals, y_true)
    #   Plot preidcted function
    for _, row in df.iterrows():
        slope, intercept = row["m"], row["c"]
        y_pred = intercept + slope * x_vals
        plt.plot(x_vals, y_pred, alpha=0.5, color=sns.color_palette()[1])
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.18, left=0.15)
    plt.show()