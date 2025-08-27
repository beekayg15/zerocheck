import os
import re
from collections import OrderedDict
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def parse_log_file(filename, leading_dots=4):
    """Parse a log file, extract lines with specified leading dots (·) before 'End:', and average repeated tasks."""
    task_times = OrderedDict()
    task_order = []

    # Use · (U+00B7) for leading dots
    pattern = re.compile(
    r'^' + r'·{' + str(leading_dots) + r'}End:\s+(.*?)\s+[·.]{2,}\s*([\d.]+)(µs|ms|s)$')


    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                task_name, value, unit = match.groups()
                value = float(value)
                if unit == 's':
                    value *= 1_000_000
                elif unit == 'ms':
                    value *= 1_000
                if task_name not in task_times:
                    task_times[task_name] = []
                    task_order.append(task_name)
                task_times[task_name].append(value)

    averaged_times = OrderedDict()
    for task in task_order:
        averaged_times[task] = sum(task_times[task])/len(task_times[task])/1000  # Convert to milliseconds

    return averaged_times


plt.rcParams.update({'font.size': 14})

def plot_stacked_bars_from_folder(root_dir):
    testcases = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    testcases.sort()

    ntt_dicts = []
    sumcheck_dicts = []

    for case in testcases:
        ntt_file = os.path.join(root_dir, case, 'univar_opt_bench_24_24_run_12_ligero_open_60.log')
        sumcheck_file = os.path.join(root_dir, case, 'mullin_opt_bench_24_24_run_12_ligero_open_48.log')

        ntt_dict = parse_log_file(ntt_file, 8) if os.path.exists(ntt_file) else {}
        sumcheck_dict = parse_log_file(sumcheck_file, 4) if os.path.exists(sumcheck_file) else {}

        ntt_dicts.append(ntt_dict)
        sumcheck_dicts.append(sumcheck_dict)

    # --- Build consistent color maps
    ntt_tasks = sorted({task for d in ntt_dicts for task in d})
    sumcheck_tasks = sorted({task for d in sumcheck_dicts for task in d})

    ntt_colors = {task: cm.tab20(i / len(ntt_tasks)) for i, task in enumerate(ntt_tasks)}
    sumcheck_colors = {task: cm.Paired(i / len(sumcheck_tasks)) for i, task in enumerate(sumcheck_tasks)}

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2
    group_spacing=0.5
    ind = np.arange(len(testcases)) * group_spacing

    # Keep handles only once for legend
    ntt_handles = {}
    sumcheck_handles = {}

    for i, (ntt_data, sumcheck_data) in enumerate(zip(ntt_dicts, sumcheck_dicts)):
        # --- NTT
        bottom = 0
        for task, value in ntt_data.items():
            color = ntt_colors[task]
            bar = ax.bar(ind[i] - width/2, value, width, bottom=bottom, color=color, label=f"NTT {task}")
            bottom += value
            if task not in ntt_handles:
                ntt_handles[task] = bar

        # --- SumCheck
        bottom = 0
        for task, value in sumcheck_data.items():
            color = sumcheck_colors[task]
            bar = ax.bar(ind[i] + width/2, value, width, bottom=bottom, color=color, label=f"SumCheck {task}")
            bottom += value
            if task not in sumcheck_handles:
                sumcheck_handles[task] = bar

    ax.set_xticks(ind)
    ax.set_xticklabels(testcases, rotation=45, ha="right", fontsize=14)
    ax.set_ylabel("Time (µs)", fontsize=14)
    ax.set_title("Stacked Bar Chart of NTT and SumCheck by Testcase", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    # Legends with fixed mapping
    first_legend = ax.legend(ntt_handles.values(), ntt_handles.keys(), title="NTT Tasks", loc='upper left',bbox_to_anchor=(1.02, 1.1), fontsize=14, title_fontsize=14, frameon = False)
    ax.add_artist(first_legend)
    ax.legend(sumcheck_handles.values(), sumcheck_handles.keys(), title="SumCheck Tasks", loc='upper left',bbox_to_anchor=(1.02, 0.4), fontsize=14, title_fontsize=14, frameon = False)

    plt.tight_layout()
    plt.savefig("file.pdf")

def plot_horizontal_stacked_bars(ntt_dict, sumcheck_dict, filename="horizontal_stacked_bar.pdf"):
    """
    Plot horizontal stacked bar charts for NTT and SumCheck, showing percentage breakdown of each task.
    """

    ntt_total = sum(ntt_dict.values()) if ntt_dict else 1
    sumcheck_total = sum(sumcheck_dict.values()) if sumcheck_dict else 1

    ntt_total_s = ntt_total / 1000
    sumcheck_total_s = sumcheck_total / 1000

    ntt_tasks = list(ntt_dict.keys())
    sumcheck_tasks = list(sumcheck_dict.keys())

    ntt_percents = [100 * v / ntt_total for v in ntt_dict.values()]
    sumcheck_percents = [100 * v / sumcheck_total for v in sumcheck_dict.values()]

    ntt_colors = [cm.tab20(i / max(1, len(ntt_tasks))) for i in range(len(ntt_tasks))]
    sumcheck_colors = [cm.Paired(i / max(1, len(sumcheck_tasks))) for i in range(len(sumcheck_tasks))]

    fig, ax = plt.subplots(figsize=(8, 1.5))  # Smaller height, tighter width

    # Bar positions and height
    bar_height = 0.6
    y_ntt = 1
    y_sumcheck = 0

    # Plot NTT bar
    left = 0
    for percent, task, color in zip(ntt_percents, ntt_tasks, ntt_colors):
        ax.barh(y_ntt, percent, left=left, color=color, edgecolor='k', height=bar_height, label=f"NTT: {task}")
        left += percent

    # Plot SumCheck bar
    left = 0
    for percent, task, color in zip(sumcheck_percents, sumcheck_tasks, sumcheck_colors):
        ax.barh(y_sumcheck, percent, left=left, color=color, edgecolor='k', height=bar_height, label=f"SumCheck: {task}")
        left += percent

    # Remove y-ticks and add vertical text labels for bars
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Time (%)", fontsize=26)
    ax.tick_params(axis='x', labelsize=20)


    # Add horizontal bar names (no rotation)
    ax.text(-13, y_sumcheck, "SumCheck", va='center', ha='center', fontsize=26, rotation=0, clip_on=False)
    ax.text(-13, y_ntt, "NTT", va='center', ha='center', fontsize=26, rotation=0, clip_on=False)

    # Add total time (in seconds) at the end of each bar
    ax.text(102, y_ntt, f"{ntt_total_s:.2f}s", va='center', ha='left', fontsize=22, color='black', clip_on=False)
    ax.text(102, y_sumcheck, f"{sumcheck_total_s:.2f}s", va='center', ha='left', fontsize=22, color='black', clip_on=False)  # fontweight='bold', 

    # Build legends for each bar
    ntt_handles = [plt.Rectangle((0,0),1,1, color=color) for color in ntt_colors]
    sumcheck_handles = [plt.Rectangle((0,0),1,1, color=color) for color in sumcheck_colors]
    ntt_labels = [f"{task}" for task in ntt_tasks]
    sumcheck_labels = [f"{task}" for task in sumcheck_tasks]

    # Place legends outside the plot area on the right
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height * 0.95])  # Make plot area tighter

    first_legend = ax.legend(ntt_handles, ntt_labels, loc='upper left', bbox_to_anchor=(1.13, 1.2), fontsize=26, title_fontsize=13, frameon=False, framealpha=0.5)  # title="NTT Tasks",
    ax.add_artist(first_legend)
    ax.legend(sumcheck_handles, sumcheck_labels, loc='upper left', bbox_to_anchor=(1.6, 1.2), fontsize=26, title_fontsize=13, frameon=False, framealpha=0.5)  # title="SumCheck Tasks",

    # Remove whitespace around the figure
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.15)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.03)
    print(f"Saved to {filename}")
    plt.show()


# Reads all folders in the specified directory and plots the stacked bar chart
# The folders names would be the x axis label. Each folder should contain a NTT log and a SumCheck log.
if __name__ == "__main__":
    root_directory = "output_log"
    # plot_stacked_bars_from_folder(root_directory)

    ntt_file = os.path.join(root_directory, 'univar_opt_bench_20_20_run_64_ligero_open_64.log')
    sumcheck_file = os.path.join(root_directory, 'mullin_opt_bench_20_20_run_64_ligero_open_64.log')
    ntt_dict = parse_log_file(ntt_file, 8) if os.path.exists(ntt_file) else {}
    sumcheck_dict = parse_log_file(sumcheck_file, 4) if os.path.exists(sumcheck_file) else {}

    ntt_dict_need_key = [
        "IFFT for g,h,s,o from evaluations to coefficients",
        "FFT Compute g,h,s,o,z,q evaluations over coset domain",
        "IFFT for q from evaluations to coefficients",
        "Compute coset domain",
    ]
    sumcheck_dict_need_key = [
        "computing inital challenge using which f_hat is computed",
        "Build MLE: computing f_hat(X) = sum_{B^m} f(X).eq(X, r)",
        "running sumcheck proving algorithm for X rounds",
        "computing evaluations of input MLEs at challenges",
    ]
    ntt_dict = {k: v for k, v in ntt_dict.items() if k in ntt_dict_need_key}
    sumcheck_dict = {k: v for k, v in sumcheck_dict.items() if k in sumcheck_dict_need_key}
    ntt_dict_key_rename = {
        "IFFT for g,h,s,o from evaluations to coefficients": "Inp iNTT",
        "FFT Compute g,h,s,o,z,q evaluations over coset domain": "Inp NTT & q",
        "IFFT for q from evaluations to coefficients": "q iNTT",
        "Compute coset domain": "Others",
    }
    # Rename keys in ntt_dict using ntt_dict_key_rename
    ntt_dict = {ntt_dict_key_rename.get(k, k): v for k, v in ntt_dict.items()}
    # Order ntt_dict by the specified order
    ntt_order = ["Inp iNTT", "Inp NTT & q", "q iNTT", "Others"]
    ntt_dict = {k: ntt_dict[k] for k in ntt_order if k in ntt_dict}
    
    # Merge two sumcheck items into one "Build MLE"
    build_mle_keys = [
        "computing inital challenge using which f_hat is computed",
        "Build MLE: computing f_hat(X) = sum_{B^m} f(X).eq(X, r)",
    ]
    build_mle_sum = sum(sumcheck_dict.get(k, 0) for k in build_mle_keys)
    # Remove the two keys
    for k in build_mle_keys:
        sumcheck_dict.pop(k, None)
    # Add merged key
    sumcheck_dict["Build MLE"] = build_mle_sum
    
    sumcheck_dict_key_rename = {
        # "computing inital challenge using which f_hat is computed": "SumCheck: computing inital challenge using which f_hat is computed",
        # "Build MLE: computing f_hat(X) = sum_{B^m} f(X).eq(X, r)": "SumCheck: Build MLE: computing f_hat(X) = sum_{B^m} f(X).eq(X, r)",
        "running sumcheck proving algorithm for X rounds": "SumCheck Rds",
        "computing evaluations of input MLEs at challenges": "Others",
    }
    sumcheck_dict = {sumcheck_dict_key_rename.get(k, k): v for k, v in sumcheck_dict.items()}
    sumcheck_order = ["Build MLE", "SumCheck Rds", "Others"]
    sumcheck_dict = {k: sumcheck_dict[k] for k in sumcheck_order if k in sumcheck_dict}


    plot_horizontal_stacked_bars(ntt_dict, sumcheck_dict, filename=f"{root_directory}/horizontal_stacked_bar.pdf")

