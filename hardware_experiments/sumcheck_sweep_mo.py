from helper_funcs import sumcheck_only_sweep
from itertools import product
import params
from poly_list import *
import pandas as pd
import openpyxl
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from util import is_pareto_efficient


def sweep_sumcheck_configs():
    """
    Sweeps through all combinations of hardware configs and available bandwidths,
    runs sumcheck_only_sweep, and records all results.

    Returns:
        results_dict: dict keyed by (available_bw, num_pes, num_eval_engines, num_product_lanes, onchip_mle_size)
    """
    results_dict = {}

    # constant params
    mle_update_latency = 10
    extensions_latency = 20
    modmul_latency = 10
    modadd_latency = 1
    latencies = mle_update_latency, extensions_latency, modmul_latency, modadd_latency
    bits_per_element = 256
    freq = 1e9
    modmul_area = params.modmul_area
    modadd_area = params.modadd_area
    reg_area = params.reg_area
    rr_ctrl_area = params.rr_ctrl_area
    per_pe_delay_buffer_count = params.per_pe_delay_buffer_count  # support degree up to 31 now.

    # sweeping params
    sumcheck_polynomials = [
        [["q1", "q2", "fz"]],  # a gate of degree 2
        # [["q1", "q2", "q3", "fz"]],
        # [["q1", "q2", "q3", "q4", "fz"]],
        [["q1", "q2", "q3", "q4", "q5", "fz"]],  # a gate of degree 5
        # [["q1", "q2", "q3", "q4", "q5", "q6", "fz"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "fz"]],
        [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fz"]],  # a gate of degree 8
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "fz"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "fz"]],
    ]

    sweep_num_vars = [20]
    sweep_sumcheck_pes_range = [2, 4, 8, 16, 32]
    sweep_eval_engines_range = range(2, 15, 4)
    sweep_product_lanes_range = range(3, 15, 4)
    sweep_onchip_mle_sizes_range = [128, 1024, 16384]  # in number of field elements
    sweep_available_bw_list = [128, 256, 512, 1024]  # in GB/s

    # testing all combinations
    for (available_bw, num_vars, num_pes, num_eval_engines, num_product_lanes, onchip_mle_size, sumcheck_gate) in product(
        sweep_available_bw_list,
        sweep_num_vars,
        sweep_sumcheck_pes_range,
        sweep_eval_engines_range,
        sweep_product_lanes_range,
        sweep_onchip_mle_sizes_range,
        sumcheck_polynomials
    ):
        ##################################################################
        # 1. #num_mle*2(double bf) buffers: for buffering input MLEs.
        #     a. Each size: onchip_mle_size (words)
        # 2. One buffer for Tmp MLE
        #     a. its size: (highest_degree_of_f + 1)*onchip_mle_size/2 (words)
        ##################################################################
        gate_degree = max(len(term) for term in sumcheck_gate)
        num_accumulate_regs = gate_degree + 1
        num_unique_mle_in_gate = len(set(sum(sumcheck_gate, [])))
        num_sumcheck_sram_buffers = num_unique_mle_in_gate * 2  # double buffering
        tmp_mle_sram_scale_factor = (gate_degree + 1) / 2
        constants = (
            bits_per_element,
            freq,
            modmul_area,
            modadd_area,
            reg_area,
            num_accumulate_regs,
            rr_ctrl_area,
            per_pe_delay_buffer_count,
            num_sumcheck_sram_buffers,
            tmp_mle_sram_scale_factor
        )

        sweep_params = (
            num_vars,
            [num_pes],
            [num_eval_engines],
            [num_product_lanes],
            [onchip_mle_size]
        )
        # print(f"Running sweep for available_bw={available_bw} GB/s, num_vars={num_vars}, num_pes={num_pes}, "
        #       f"num_eval_engines={num_eval_engines}, num_product_lanes={num_product_lanes}, "
        #       f"onchip_mle_size={onchip_mle_size}")
        stats_dict = sumcheck_only_sweep(
            sweep_params,
            [sumcheck_gate],
            latencies,
            constants,
            available_bw
        )

        # Update the key to match all sweeping parameters
        results_dict[
            (
                available_bw,
                num_vars,
                num_pes,
                num_eval_engines,
                num_product_lanes,
                onchip_mle_size,
                gate_to_string(sumcheck_gate)
            )
        ] = {
            "result": stats_dict,
            "params": {
                "available_bw": available_bw,
                "num_vars": num_vars,
                "num_pes": num_pes,
                "num_eval_engines": num_eval_engines,
                "num_product_lanes": num_product_lanes,
                "onchip_mle_size": onchip_mle_size,
                "sumcheck_gate": gate_to_string(sumcheck_gate),
            }
        }

    return results_dict


def plot_area_latency_one(df, filename):
    """
    Draw a scatter plot: x="total_latency", y="area", color by "available
    _bw", marker by "sumcheck_gate".
    """
    plt.figure(figsize=(10, 7))
    # Define marker styles for sumcheck_gate
    marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
    sumcheck_gates = sorted(df["sumcheck_gate"].unique())
    marker_dict = {gate: marker_styles[i % len(marker_styles)] for i, gate in enumerate(sumcheck_gates)}
    # Use seaborn color palette for available_bw
    available_bw_list = sorted(df["available_bw"].unique())
    palette = sns.color_palette("tab10", n_colors=len(available_bw_list))
    color_dict = {bw: palette[i % len(palette)] for i, bw in enumerate(available_bw_list)}
    # Plot each combination
    for gate in sumcheck_gates:
        for bw in available_bw_list:
            sub_df = df[(df["sumcheck_gate"] == gate) & (df["available_bw"] == bw)]
            plt.scatter(
                sub_df["total_latency"],
                sub_df["area"],
                label=f"{gate}, {bw}GB/s",
                color=color_dict[bw],
                marker=marker_dict[gate],
                s=30,
                # edgecolor="k",
                alpha=0.8
            )
    plt.ylabel("Area")
    plt.title("Sumcheck Sweep: Area vs Total Latency")
    plt.xlim(left=0)  # Ensure x-axis starts at 0
    plt.legend(title="Gate (marker) / BW (color)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', direction='in', length=4)
    # Set x-axis ticks in units of 1e6
    plt.xlabel("Total Latency (x10^6)")
    locs, labels = plt.xticks()
    plt.xticks(locs, [f"{x/1e6}" for x in locs if x >= 0])  # Filter out negative ticks
    plt.tight_layout()
    plt.savefig(filename + "_area_latency_one.png", bbox_inches='tight')
    plt.close()


def plot_gate_acrx_bw(df, filename):
    """
    Draw multiple subplots: each subplot corresponds to one available_bw.
    Within each subplot, use different marker styles to distinguish sumcheck_gate types.
    """
    available_bw_list = sorted(df["available_bw"].unique())
    sumcheck_gates = sorted(df["sumcheck_gate"].unique())

    # Define marker styles for sumcheck_gate
    marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
    marker_dict = {gate: marker_styles[i % len(marker_styles)] for i, gate in enumerate(sumcheck_gates)}

    # Create subplots: 3 rows (area-latency, memory-latency, modmul_count-latency), columns = available_bw
    num_subplots = len(available_bw_list)
    fig, axes = plt.subplots(3, num_subplots, figsize=(6 * num_subplots, 18), sharey='row')

    if num_subplots == 1:
        axes = axes.reshape(3, 1)

    for col, bw in enumerate(available_bw_list):
        sub_df = df[df["available_bw"] == bw]
        # First row: area vs latency
        ax_area = axes[0, col]
        for gate in sumcheck_gates:
            gate_df = sub_df[sub_df["sumcheck_gate"] == gate]
            if not gate_df.empty:
                # Pareto filter: minimize both area and latency
                costs = gate_df[["area", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_df = gate_df[pareto_mask]
                ax_area.scatter(
                    pareto_gate_df["total_latency"],
                    pareto_gate_df["area"],
                    label=gate,
                    marker=marker_dict[gate],
                    color='C0',
                    s=30,
                    edgecolor="k",
                    alpha=0.8
                )
        ax_area.set_title(f"Available BW: {bw} GB/s")
        ax_area.set_xlim(left=0)
        ax_area.set_xlabel("Total Latency (x10^6)")
        locs = ax_area.get_xticks()
        locs = [x for x in locs if x >= 0]
        ax_area.set_xticks(locs)
        ax_area.set_xticklabels([f"{x/1e6:g}" for x in locs])
        ax_area.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        if col == 0:
            ax_area.set_ylabel("Area")
        # Custom legend: only show unique marker combos (only for first subplot)
        if col == 0:
            handles = []
            for gate in sumcheck_gates:
                handles.append(Line2D([0], [0], marker=marker_dict[gate], color='w', label=gate,
                                       markerfacecolor='C0', markeredgecolor='k', markersize=10, linestyle='None'))
            ax_area.legend(handles=handles, title="Gate (marker)", loc='best', fontsize='small')

        # Save x-tick locations and limits for use in other rows
        xlim = ax_area.get_xlim()
        xticks = locs
        xticklabels = [f"{x/1e6:g}" for x in xticks]

        # Second row: total_onchip_memory_MB vs latency
        ax_mem = axes[1, col]
        for gate in sumcheck_gates:
            gate_df = sub_df[sub_df["sumcheck_gate"] == gate]
            if not gate_df.empty:
                # Pareto filter: minimize both total_onchip_memory_MB and latency
                costs = gate_df[["total_onchip_memory_MB", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                # pareto_gate_df = gate_df[pareto_mask]
                pareto_gate_df = gate_df  # no pareto filter
                ax_mem.scatter(
                    pareto_gate_df["total_latency"],
                    pareto_gate_df["total_onchip_memory_MB"],
                    label=gate,
                    marker=marker_dict[gate],
                    color='C1',
                    s=30,
                    edgecolor="k",
                    alpha=0.8
                )
        ax_mem.set_xlim(xlim)
        ax_mem.set_xlabel("Total Latency (x10^6)")
        ax_mem.set_xticks(xticks)
        ax_mem.set_xticklabels(xticklabels)
        ax_mem.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        if col == 0:
            ax_mem.set_ylabel("Total Onchip Memory (MB)")

        # Third row: modmul_count vs latency (Pareto-efficient only)
        ax_modmul = axes[2, col]
        for gate in sumcheck_gates:
            gate_df = sub_df[sub_df["sumcheck_gate"] == gate]
            if not gate_df.empty:
                # Pareto filter: minimize both modmul_count and latency
                costs = gate_df[["modmul_count", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_df = gate_df[pareto_mask]
                # pareto_gate_df = gate_df  # no pareto filter
                ax_modmul.scatter(
                    pareto_gate_df["total_latency"],
                    pareto_gate_df["modmul_count"],
                    label=gate,
                    marker=marker_dict[gate],
                    color='C2',
                    s=30,
                    edgecolor="k",
                    alpha=0.8
                )
        ax_modmul.set_xlim(xlim)
        ax_modmul.set_xlabel("Total Latency (x10^6)")
        ax_modmul.set_xticks(xticks)
        ax_modmul.set_xticklabels(xticklabels)
        ax_modmul.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        if col == 0:
            ax_modmul.set_ylabel("Modmul Count")

    plt.tight_layout()
    plt.savefig(filename + "_gate_acrx_bw.png", bbox_inches='tight')
    plt.close()


def save_results(results, filename, save_excel=False, draw_plots_type=0):
    """
    Save the sweep results to an Excel file.
    Each row contains the sweep parameters (from 'params') and the stats_dict items as columns.
    Optionally, draw a scatter plot: x="total_latency", y="area", color by "available_bw", marker by "sumcheck_gate".
    """
    rows = []
    for value in results.values():
        params = value["params"]
        stats_dict = value["result"]
        # Flatten stats_dict (which may be nested)
        for idx, config_stats in stats_dict.items():
            for config, stat_items in config_stats.items():
                row = dict(params)  # copy params
                row["poly_idx"] = idx
                row["hardware_config"] = str(config)
                # Add all stat_items as columns
                for k, v in stat_items.items():
                    row[k] = v
                rows.append(row)
    df = pd.DataFrame(rows)
    if save_excel:
        df.to_excel(filename + ".xlsx", index=False)
    if draw_plots_type:
        if draw_plots_type == 1:
            plot_area_latency_one(df, filename)
        elif draw_plots_type == 2:
            plot_gate_acrx_bw(df, filename)


if __name__ == "__main__":
    results = sweep_sumcheck_configs()
    save_results(results, "sumcheck_sweep_results_mo", save_excel=True, draw_plots_type=2)
    
    print("End...")

