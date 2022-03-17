import os
from glob import glob
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def fig_ax(figsize=(15, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.margins(x=0)
    return fig, ax


def load_data(folder="data"):
    """Returns X_train, y_train, X_test as a list of DataFrame, Series, DataFrame.
    """
    filespecs = ["input_training_*.csv", "output_training_*.csv", "input_testing.csv"]
    out = []
    for filespec in filespecs:
        filename = glob(os.path.join(folder, filespec))[0]
        out.append(pd.read_csv(
            filename,
            index_col=0
        ).squeeze().astype("float64"))
    return out


def save_results(y_pred: np.ndarray, test_index: pd.Index, out_path="submission.csv"):
    """Saves the predictions for submission."""
    out_df = pd.DataFrame(y_pred, index=test_index)
    out_df.to_csv(out_path)
