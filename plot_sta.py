import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import table
# plot table

def plot_table(data_tab):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    tbl = table(ax, data_tab, loc='center', cellLoc='center', colWidths=[0.2]*len(data_tab.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.show()
# save image as output.png
# plt.savefig('output.png', bbox_inches='tight', dpi=300, transparent=True)