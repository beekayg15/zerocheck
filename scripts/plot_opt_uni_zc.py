import os
import re
from collections import OrderedDict
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

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


# Reads all folders in the specified directory and plots the stacked bar chart
# The folders names would be the x axis label. Each folder should contain a NTT log and a SumCheck log.
if __name__ == "__main__":
    root_directory = "output_log"
    plot_stacked_bars_from_folder(root_directory)
