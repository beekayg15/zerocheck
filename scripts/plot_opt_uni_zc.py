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


def _draw_3d_pie(ax, values, colors, depth=0.18, yscale=0.55, startangle=-20, label_min_pct=0.05):
    """Draw one pie as a tilted disc: elliptical top wedges plus a shaded side wall, with outside percentage labels."""
    from matplotlib.patches import Polygon
    from matplotlib.colors import to_rgb

    total = sum(values)
    fracs = [v / total for v in values]
    bounds = np.cumsum([0] + fracs) * 360 + startangle

    def ellipse(theta_deg, dy=0.0):
        t = np.radians(theta_deg)
        return np.cos(t), yscale * np.sin(t) - dy

    # Side wall: only the front half of the rim (sin < 0, i.e., angles in [180, 360) mod 360) is visible.
    for t1, t2, color in zip(bounds[:-1], bounds[1:], colors):
        dark = tuple(0.7 * c for c in to_rgb(color))
        for k in range(-1, 3):  # wedge may wrap around; check each visible window
            lo, hi = max(t1, 180 + 360 * k), min(t2, 360 + 360 * k)
            if hi <= lo:
                continue
            ts = np.linspace(lo, hi, 60)
            top = [ellipse(t) for t in ts]
            bottom = [ellipse(t, depth) for t in ts[::-1]]
            ax.add_patch(Polygon(top + bottom, closed=True, facecolor=dark, edgecolor=dark, linewidth=0.5, zorder=1))

    # Top face
    for t1, t2, color in zip(bounds[:-1], bounds[1:], colors):
        ts = np.linspace(t1, t2, max(2, int((t2 - t1) * 2)))
        pts = [(0.0, 0.0)] + [ellipse(t) for t in ts]
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor='white', linewidth=0.8, zorder=2))

    # Outside labels with leader lines; nudge labels on the same side apart vertically
    placed = {1: [], -1: []}
    for t1, t2, frac in zip(bounds[:-1], bounds[1:], fracs):
        pct = 100 * frac
        if pct < label_min_pct:
            continue
        mid = (t1 + t2) / 2
        x0, y0 = ellipse(mid)
        if np.sin(np.radians(mid)) < 0:
            y0 -= depth / 2
        side = 1 if np.cos(np.radians(mid)) >= 0 else -1
        lx, ly = 1.28 * np.cos(np.radians(mid)), 1.45 * yscale * np.sin(np.radians(mid))
        if np.sin(np.radians(mid)) < 0:
            ly -= depth
        for other in placed[side]:
            if abs(ly - other) < 0.42:
                ly = other - 0.42 if ly <= other else other + 0.42
        placed[side].append(ly)
        ax.plot([x0, lx], [y0, ly], color='0.3', linewidth=0.8, zorder=3)
        ax.text(lx + 0.04 * side, ly, f"{pct:.1f}%", ha='left' if side > 0 else 'right', va='center', fontsize=18, zorder=4)

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-yscale - depth - 0.35, yscale + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_3d_pies(ntt_dict, sumcheck_dict, filename="pie_breakdown.pdf"):
    """
    Two tilted pie charts (NTT, SumCheck) showing the percentage breakdown of each task, with total runtime in the titles.
    """
    datasets = [
        ("NTT", ntt_dict, [cm.tab20(i / max(1, len(ntt_dict))) for i in range(len(ntt_dict))]),
        ("SumCheck", sumcheck_dict, [cm.Paired(i / max(1, len(sumcheck_dict))) for i in range(len(sumcheck_dict))]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
    # fix the axes geometry first, so the legend positions computed below are final
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.2, wspace=0.18)
    legends = []
    for ax, (name, data, colors) in zip(axes, datasets):
        # Drop negligible tasks (< 0.05% of the total, e.g., NTT's coset-domain setup): no visible slice, no legend entry
        total = sum(data.values())
        keep = [i for i, v in enumerate(data.values()) if 100 * v / total >= 0.05]
        data = {k: v for i, (k, v) in enumerate(data.items()) if i in keep}
        colors = [colors[i] for i in keep]
        _draw_3d_pie(ax, list(data.values()), colors)
        total_s = total / 1000
        ax.set_title(f"{name}: {total_s:.2f} s", fontsize=19, pad=0)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
        # Two legend rows, laid out the same way for both pies: two shorter labels side by side on the first row, the
        # longest label ("Inp NTT & q", "Sample Chal. & Build MLE") starting the second row, left-aligned with the first,
        # followed by any remaining label (NTT's "Others")
        labels = list(data.keys())
        long_i = max(range(len(labels)), key=lambda i: len(labels[i]))
        short = [i for i in range(len(labels)) if i != long_i]
        first, second = short[:2], [long_i] + short[2:]
        kw = dict(fontsize=17, frameon=False, handlelength=0.9, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
        # Measure the first row once (centered under the pie), then anchor BOTH rows by their upper-left corner at the same
        # x: the colour boxes then share one left edge in any renderer. A centered anchor would not work, because the PDF
        # backend measures text slightly differently from the on-screen one and would shift the rows by different amounts.
        probe = ax.legend([handles[i] for i in first], [labels[i] for i in first], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, **kw)
        fig.canvas.draw()
        bb = probe.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        probe.remove()
        leg = ax.legend([handles[i] for i in first], [labels[i] for i in first], loc='upper left', bbox_to_anchor=(bb.x0, -0.07), ncol=2, **kw)
        ax.add_artist(leg)
        leg2 = ax.legend([handles[i] for i in second], [labels[i] for i in second], loc='upper left', bbox_to_anchor=(bb.x0, bb.y0), ncol=len(second), **kw)
        legends += [leg, leg2]
    # bbox_extra_artists replaces the default list, so keep the default artists (outside labels) and add the legends
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.03, bbox_extra_artists=fig.get_default_bbox_extra_artists() + legends)
    print(f"Saved to {filename}")


# Reads all folders in the specified directory and plots the stacked bar chart
# The folders names would be the x axis label. Each folder should contain a NTT log and a SumCheck log.
if __name__ == "__main__":
    root_directory = "output_log"
    # plot_stacked_bars_from_folder(root_directory)

    # NTT: 64-thread run with the parallel quotient fix (Nov 2026), rep2 of 3; SumCheck: 64-thread rerun (Nov 2026),
    # rep1 is the median-total repetition
    ntt_file = os.path.join(root_directory, 'epyc_64thr', 'univar_opt_bench_20_20_run_64_ligero_open_64_after_qfix_rep2.log')
    sumcheck_file = os.path.join(root_directory, 'epyc_64thr', 'mullin_opt_bench_20_20_run_64_ligero_open_64_rep1.log')
    ntt_dict = parse_log_file(ntt_file, 8) if os.path.exists(ntt_file) else {}
    sumcheck_dict = parse_log_file(sumcheck_file, 4) if os.path.exists(sumcheck_file) else {}

    ntt_dict_need_key = [
        "IFFT for g,h,s,o from evaluations to coefficients",
        "FFT Compute g,h,s,o,z,q evaluations over coset domain",
        "IFFT for q from evaluations to coefficients",
        "Compute coset domain",
        "computing evaluations at challenge, get challenge",
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
        # challenge sampling and evaluation at r, the counterpart of SumCheck's "Others" and challenge sampling
        "computing evaluations at challenge, get challenge": "Others",
    }
    # Rename keys in ntt_dict using ntt_dict_key_rename; keys mapped to the same name are summed
    renamed = {}
    for k, v in ntt_dict.items():
        renamed[ntt_dict_key_rename.get(k, k)] = renamed.get(ntt_dict_key_rename.get(k, k), 0) + v
    ntt_dict = renamed
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
    sumcheck_dict["Sample Chal. & Build MLE"] = build_mle_sum
    
    sumcheck_dict_key_rename = {
        # "computing inital challenge using which f_hat is computed": "SumCheck: computing inital challenge using which f_hat is computed",
        # "Build MLE: computing f_hat(X) = sum_{B^m} f(X).eq(X, r)": "SumCheck: Build MLE: computing f_hat(X) = sum_{B^m} f(X).eq(X, r)",
        "running sumcheck proving algorithm for X rounds": "SumCheck Rds",
        "computing evaluations of input MLEs at challenges": "Others",
    }
    sumcheck_dict = {sumcheck_dict_key_rename.get(k, k): v for k, v in sumcheck_dict.items()}
    sumcheck_order = ["Sample Chal. & Build MLE", "SumCheck Rds", "Others"]
    sumcheck_dict = {k: sumcheck_dict[k] for k in sumcheck_order if k in sumcheck_dict}


    plot_horizontal_stacked_bars(ntt_dict, sumcheck_dict, filename=f"{root_directory}/horizontal_stacked_bar.pdf")
    plot_3d_pies(ntt_dict, sumcheck_dict, filename=f"{root_directory}/pie_breakdown.pdf")

