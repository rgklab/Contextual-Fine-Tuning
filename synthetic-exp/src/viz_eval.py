from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
from schema import schema
from quinine import QuinineArgumentParser

def valid_row(r):
    return r.task == task and r.run_id == run_id


sns.set_palette("deep") 
run_dir = '/voyager/projects/younwoo/in-context-learning/models'
df = read_run_dir(run_dir)

# task = "sumdot_2nn_regression"
task = "polynomial_2nn_regression"

run_ids = ['cft', 'uft', 'neg-cft']

steps = [-1]
for step in steps:
    metrics_collect = {}
    for run_id in run_ids:
        run_path = os.path.join(run_dir, task, run_id)
        parser = QuinineArgumentParser(schema=schema)
        args = parser.parse_quinfig()

        tmp_args = args.training
        tmp_args.method = run_id

        args.training = tmp_args
        
        recompute_metrics = True
        if recompute_metrics:
            get_run_metrics(run_path, args, step=step,  skip_baselines=True)

        metrics = collect_results(run_dir, df, valid_row=valid_row, args=args, step=step) 
        
        
        metrics_collect[run_id] = metrics['standard']
        models = relevant_model_names[task]

    models = run_ids
    plt.figure(figsize=(12, 8)) 
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bolden the remaining spines (left and bottom)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Bolden the tick lines as well
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)
    
    if models is not None:
        metrics = {k: metrics_collect[k]['Transformer'] for k in models}

    for name, vs in metrics.items():
        low = vs["bootstrap_low"][::2]
        high = vs["bootstrap_high"][::2]
        if name == 'cft':
            plt.plot(list(range(41))[::2], vs["mean"][::2], label='CFT', color='royalblue', lw=4, marker='*', markersize=28, zorder=3)
            plt.fill_between(range(41)[::2], low, high, alpha=0.3, color='dodgerblue')
        elif name =='uft':
            plt.plot(list(range(41))[::2], vs["mean"][::2], label='CPT', color='#ff7f0e', lw=4, marker='o', markersize=16, zorder=2)
            plt.fill_between(range(41)[::2], low, high, alpha=0.3, color='gold')
        else:
            plt.plot(list(range(41))[::2], vs["mean"][::2], label='NEG-CFT', color='#2ca02c', lw=4, marker='o', markersize=16, zorder=1)
            plt.fill_between(range(41)[::2], low, high, alpha=0.3, color='limegreen')

    plt.locator_params(axis='x', nbins=3)
    plt.xlabel("In-context examples", fontsize=32)
    plt.ylabel("Squared error", fontsize=32)

    legend = plt.legend(fontsize=30, title_fontsize=16)  # Legend with larger fonts
    legend.get_frame().set_alpha(0)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    plt.xticks(fontsize=34)  # X-axis ticks
    plt.yticks(fontsize=34)  # Y-axis ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'eval.png')