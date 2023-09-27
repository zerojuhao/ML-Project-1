import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import table
import os
# plot table

def plot_table(data_tab):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    tbl = table(ax, data_tab, loc='center', cellLoc='center', colWidths=[0.2]*len(data_tab.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.show()