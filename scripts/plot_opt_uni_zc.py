import os
import re
from collections import OrderedDict
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
        averaged_times[task] = sum(task_times[task])/len(task_times[task])

    return averaged_times


def plot_stacked_bars_from_folder(root_dir):
    testcases = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    testcases.sort()

    ntt_data = []
    sumcheck_data = []
    ntt_labels = []
    sumcheck_labels = []

    for case in testcases:
        # ntt_file = os.path.join(root_dir, case, 'ntt.log')
        # sumcheck_file = os.path.join(root_dir, case, 'sumcheck.log')
        print(case)
        ntt_file = os.path.join(root_dir, case, 'univar_opt_bench_24_24_run_12_ligero_open_60.log')
        sumcheck_file = os.path.join(root_dir,case, 'mullin_opt_bench_24_24_run_12_ligero_open_48.log')

        if os.path.exists(ntt_file):
            print("Parsing NTT file:", ntt_file)
            ntt_dict = parse_log_file(ntt_file, 8)
            ntt_data.append(ntt_dict)
            ntt_labels.append(case + " NTT")
            print(len(ntt_dict))
        if os.path.exists(sumcheck_file):
            print("Parsing SumCheck file:", sumcheck_file)
            sumcheck_dict = parse_log_file(sumcheck_file, 4)
            sumcheck_data.append(sumcheck_dict)
            sumcheck_labels.append(case + " SumCheck")

        
    fig, ax = plt.subplots(figsize=(12, 6))
    ind = np.arange(len(testcases) * 2)
    width = 0.35

    # Colors cycle automatically, but we'll separate legends
    ntt_handles = []
    sumcheck_handles = []

    for i, data in enumerate(ntt_data):
        bottom = 0
        for task, value in data.items():
            bar = ax.bar(i*2, value, width, bottom=bottom, label=f"NTT {task}")
            bottom += value
            if i == 0:
                ntt_handles.append(bar)

    for i, data in enumerate(sumcheck_data):
        bottom = 0
        for task, value in data.items():
            bar = ax.bar(i*2+1, value, width, bottom=bottom, label=f"SumCheck {task}")
            bottom += value
            if i == 0:
                sumcheck_handles.append(bar)

    ax.set_xticks(ind)
    ax.set_xticklabels(ntt_labels + sumcheck_labels, rotation=45, ha="right")
    ax.set_ylabel("Time (µs)")
    ax.set_title("Stacked Bar Chart of NTT and SumCheck by Testcase")

    # Two separate legends
    first_legend = ax.legend(ntt_handles, [h.get_label() for h in ntt_handles], title="NTT Tasks", loc='upper left')
    ax.add_artist(first_legend)
    ax.legend(sumcheck_handles, [h.get_label() for h in sumcheck_handles], title="SumCheck Tasks", loc='upper right')

    plt.tight_layout()
    plt.savefig("file")


# Example usage
if __name__ == "__main__":
    root_directory = "output_log"  # replace with your directory
    plot_stacked_bars_from_folder(root_directory)
