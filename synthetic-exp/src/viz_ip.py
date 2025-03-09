import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
import matplotlib as mpl
import matplotlib.patheffects as PathEffects

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath} \usepackage{amssymb} \usepackage{mathptmx}",
    "font.weight": "bold",
    "font.size": 16,
    "axes.titleweight": "bold",
    "axes.labelweight": 1000,  
    "axes.linewidth": 2.5,
    "axes.grid": False,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 500,
})

colors = [
    "#1F77B4",  # Blue (similar to GBT in reference)
    "#D62728",  # Red (similar to NN in reference)
    "#2CA02C",  # Green (keeping as third color)
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#7F7F7F",  # Gray
    "#BCBD22",  # Olive
    "#17BECF",  # Cyan
    "#FF7F0E"   # Orange
]

markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', 'p']

csv_file = './innerprod.csv'
df = pd.read_csv(csv_file)

if 'Step' in df.columns:
    epochs = df['Step'].tolist()
    data_columns = df.columns.drop('Step')
else:
    epochs = range(1, len(df) + 1)
    data_columns = df.columns.tolist()

run_ids = ['sumdot', 'polynomial']

for run_id in run_ids:
    fig, ax = plt.subplots(figsize=(12, 8)) 

    color_idx = 0
    
    for column in data_columns:
        if run_id in column:
            x_data = epochs[:401][::40]
            y_data = df[column][:401][::40]
            
            if "cft" in column and 'neg' not in column:
                ax.plot(x_data, y_data, 
                       label='CFT', 
                       linewidth=4, 
                       marker='*', 
                       markersize=32,
                       markeredgewidth=1.5,
                       markeredgecolor='white',
                       color=colors[0],
                       alpha=1.0)
            elif 'uft' in column:
                ax.plot(x_data, y_data, 
                       label='CPT', 
                       linewidth=4, 
                       marker=markers[1], 
                       markersize=16,
                       markeredgewidth=1.5,
                       markeredgecolor='white',
                       color=colors[1],
                       alpha=1.0)
            else:
                ax.plot(x_data, y_data, 
                       label='NEG-CFT', 
                       linewidth=4, 
                       marker=markers[2], 
                       markersize=16,
                       markeredgewidth=1.5,
                       markeredgecolor='white',
                       color=colors[2],
                       alpha=1.0)
            
            color_idx += 1
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    
    ax.tick_params(axis='both', which='major', labelsize=32, width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2, length=5)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(1000) 
        label.set_color('black')
        label.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=0.5, foreground='white')])
    
    plt.locator_params(axis='x', nbins=5)
    
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.2, color='#666666')
    ax.set_xlabel(r'\textbf{Step}', fontsize=32)
    ax.set_ylabel(r'\textbf{Inner product}', fontsize=32)
    
    legend = ax.legend(
        fontsize=32,
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fancybox=False,
        loc='best',
        handlelength=4,
        handletextpad=1.0,
    )
    legend.get_frame().set_linewidth(1.5)
    
    for text in legend.get_texts():
        text.set_fontweight(800) 
        text.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=0.5, foreground='white')])
    legend.get_frame().set_linewidth(0.5)
    
    plt.tight_layout()
    
    plt.savefig(f'innerprod_{run_id}.png', dpi=500, bbox_inches='tight')
    plt.clf()