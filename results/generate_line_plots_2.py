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
    parser.add_argument('--slope', type=float)
    parser.add_argument('--intercept', type=float)
    parser.add_argument('--title', type=str, default="")
    args = parser.parse_args()

    # Plot
    plt.xlabel("MAE in Empirical Neighbourhood", fontsize=25)
    plt.ylabel("Normalised Test DQ", fontsize=25)
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title(args.title if args.title != "" else args.file, fontsize=25)
    plt.xlim(-0.02, 0.1)
    plt.ylim(0, 1.2)
    #   Get x values
    axes = plt.gca()
    x_min, x_max = axes.get_xlim()
    x_vals = np.linspace(x_min, x_max, 1000)
    #   Plot data points
    df = pd.read_csv(args.file, header=None, names=["x", "y"])
    sns.scatterplot(x="x",y="y",data=df, s=200)
    #   Plot line
    y_true = args.intercept + args.slope * x_vals
    plt.plot(x_vals, y_true, color=sns.color_palette()[0])
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.18, left=0.16)
    plt.show()