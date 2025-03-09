import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("deep")
csv_file = './loss.csv'
df = pd.read_csv(csv_file)

if 'Step' in df.columns:
    epochs = df['Step'].tolist()
    data_columns = df.columns.drop('Step')
else:
    epochs = range(1, len(df) + 1)
    data_columns = df.columns.tolist()

run_ids = ['sumdot', 'polynomial']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd', 'P', '8']
custom_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

for run_id in run_ids:
    plt.figure(figsize=(12, 8))

    for column in data_columns:
        in_prod = df[column].tolist()
        if run_id in column:
            if "cft" in column and 'neg' not in column:
                plt.plot(epochs[:401][::40], df[column][:401][::40], label='CFT', linewidth=6, marker='*', markersize=30, color='royalblue')
            elif 'uft' in column:
                plt.plot(epochs[:401][::40], df[column][:401][::40], label='CPT', linewidth=4, marker='o', markersize=22, color='#ff7f0e')
            else:
                plt.plot(epochs[:401][::40], df[column][:401][::40], label='NEG-CFT', linewidth=4, marker='o', markersize=22, color='#2ca02c')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.tick_params(axis='both', which='major', labelsize=12, width=2)

    plt.locator_params(axis='x', nbins=3)
    plt.xlabel('Step', fontsize=32)
    plt.ylabel('Loss', fontsize=32)

    legend = plt.legend(fontsize=30, title_fontsize=16)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)

    legend.get_frame().set_alpha(0)
    legend.get_frame().set_facecolor((0, 0, 0, 0))

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'loss_{run_id}.png')
    plt.clf()
