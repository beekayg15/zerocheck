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
from test_ntt_func_sim import run_fit_onchip
from tqdm import tqdm
import math


def analyze_polynomial_gate(gate):
    """
    Analyze a single gate (list of terms).
    Returns a dict with:
      - num_terms: number of terms (sublists) in the gate
      - num_unique_items: number of unique strings in the gate
      - degree: size of the longest term (max sublist length) e.g.: 2, 3, etc.
    """
    num_terms = len(gate)
    unique_items = set()
    max_degree = 0
    for term in gate:
        unique_items.update(term)
        if len(term) > max_degree:
            max_degree = len(term)
    return {
        "num_terms": num_terms,
        "num_unique_mle": len(unique_items),
        "degree": max_degree
    }


def sweep_NTT_configs(n_size_values: list, bw_values: list, polynomial_list: list):
    """
    Sweep all combinations of n and bw, calling run_fit_onchip for each.
    Returns a dictionary keyed by (n, bw) with the results.

    :param n_size_values: List of NTT sizes to sweep. Should be the exp `μ`. E.g., [16, 17, 18, ...]
    :param bw_values: List of available bandwidths to sweep (GB/s).
    :param polynomial_list: List of polynomials to sweep. Each polynomial is a list of terms, where each term is a list of strings.
    :return: Dictionary of results keyed by (n, bw)
    """

    all_rows = []
    for gate in tqdm(polynomial_list, desc="NTT Sweep for gate"):
        gate_name = gate_to_string(gate)
        gate_stats = analyze_polynomial_gate(gate)
        gate_num_terms = gate_stats["num_terms"]
        gate_num_unique_mle = gate_stats["num_unique_mle"]
        gate_degree = gate_stats["degree"]
        gate_degree_n = int(math.log2(gate_degree - 1))  # TODO: degree is not always a power of 2

        for n in tqdm(n_size_values, desc=f"NTT Sweep for n"):
            for bw in tqdm(bw_values, desc=f"NTT sweep bw"):
                print(f"Running NTT sweep for n={gate_degree - 1}x2^{n}, bw={bw}...")
                res = run_fit_onchip(target_n=n+gate_degree_n, target_bw=bw, save_pkl=False)
                # res is a dict: key=(n_pow, available_bw, unroll_factor, pe_amt), value=dict
                for key, value in res.items():
                    n_pow, available_bw, unroll_factor, pe_amt = key
                    row = {
                        "gate_name": gate_name,
                        "gate_num_terms": gate_num_terms,
                        "gate_num_unique_mle": gate_num_unique_mle,
                        "gate_degree": gate_degree,
                        "n": n,
                        "target_n": n + gate_degree_n,
                        "n_pow": n_pow,
                        "available_bw": available_bw,
                        "unroll_factor": unroll_factor,
                        "pe_amt": pe_amt,
                    }
                    
                    value = value.copy()
                    if "total_cycles" in value:
                        # Repeat NTT for each MLE in series
                        value["total_latency"] = value["total_cycles"] * gate_num_unique_mle

                        # area cost
                        value["total_comp_area"] = value["total_modmuls"] * params.modmul_area + value["total_modadds"] * params.modadd_area
                        value["total_onchip_memory_MB"] = value["total_num_words"] * params.bits_per_scalar / 8 / (1 << 20)
                        value["total_mem_area_mm2"] = value["total_onchip_memory_MB"] * params.MB_CONVERSION_FACTOR
                        value["total_area"] = value["total_comp_area"] + value["total_mem_area_mm2"]
                    
                    row.update(value)
                    all_rows.append(row)

    singleNTT = pd.DataFrame(all_rows)

    return singleNTT


def sweep_sumcheck_configs(num_var_list: list, available_bw_list: list, polynomial_list: list):
    """
    Sweeps through all combinations of hardware configs and available bandwidths,
    runs sumcheck_only_sweep, and records all results.

    Args:
        num_var_list: list of num_vars to sweep (e.g., [20])
        available_bw_list: list of available bandwidths to sweep (e.g., [128, 256, 512, 1024])
        polynomial_list: list of sumcheck polynomials to sweep (e.g., [ [["q1", "q2"], ["q3", "q4"]], gate2, ...])
    Returns:
        results_dict: dict keyed by (available_bw, num_pes, num_eval_engines, num_product_lanes, onchip_mle_size)
    """
    results_dict = {}

    # constant params
    mle_update_latency = params.mle_update_latency
    extensions_latency = params.extensions_latency
    modmul_latency = params.modmul_latency
    modadd_latency = params.modadd_latency
    latencies = mle_update_latency, extensions_latency, modmul_latency, modadd_latency
    bits_per_element = params.bits_per_scalar
    freq = params.freq
    modmul_area = params.modmul_area
    modadd_area = params.modadd_area
    reg_area = params.reg_area
    rr_ctrl_area = params.rr_ctrl_area
    per_pe_delay_buffer_count = params.per_pe_delay_buffer_count  # support degree up to 31 now.

    # sweeping params. 
    # Use polynomial_list, append 'fz' to the end of each term of each gate
    sumcheck_polynomials = [
        [[*term, "fz"] for term in gate]
        for gate in polynomial_list
    ]

    sweep_sumcheck_pes_range = [2, 4, 8, 16, 32]
    sweep_eval_engines_range = range(2, 15, 4)
    sweep_product_lanes_range = range(3, 15, 4)
    sweep_onchip_mle_sizes_range = [128, 1024, 16384]  # in number of field elements

    # testing all combinations
    loop_iter = product(
        available_bw_list,
        num_var_list,
        sweep_sumcheck_pes_range,
        sweep_eval_engines_range,
        sweep_product_lanes_range,
        sweep_onchip_mle_sizes_range,
        sumcheck_polynomials
    )
    for (available_bw, num_vars, num_pes, num_eval_engines, num_product_lanes, onchip_mle_size, sumcheck_gate) in tqdm(list(loop_iter), desc="Sumcheck sweep"):
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

    rows = []
    for value in results_dict.values():
        vparams = value["params"]
        stats_dict = value["result"]
        # Flatten stats_dict (which may be nested)
        for idx, config_stats in stats_dict.items():
            for config, stat_items in config_stats.items():
                row = dict(vparams)  # copy params
                row["poly_idx"] = idx
                row["hardware_config"] = str(config)
                # Add all stat_items as columns
                for k, v in stat_items.items():
                    row[k] = v
                rows.append(row)
    df = pd.DataFrame(rows)

    return df


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


def plot_gate_acrx_bw(sc_df: pd.DataFrame, ntt_df: pd.DataFrame, filename):
    """
    Draw multiple subplots: each subplot corresponds to one available_bw.
    Within each subplot, use different marker styles to distinguish sumcheck_gate types.
    """
    available_bw_list = sorted(sc_df["available_bw"].unique())
    sumcheck_gates = sorted(sc_df["sumcheck_gate"].unique())

    # Define marker styles for sumcheck_gate
    marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
    marker_dict = {gate: marker_styles[i % len(marker_styles)] for i, gate in enumerate(sumcheck_gates)}

    # Create subplots: 3 rows (area-latency, memory-latency, modmul_count-latency), columns = available_bw
    num_subplots = len(available_bw_list)
    fig, axes = plt.subplots(3, num_subplots, figsize=(6 * num_subplots, 18), sharey='row')

    if num_subplots == 1:
        axes = axes.reshape(3, 1)


    for col, bw in enumerate(available_bw_list):
        sub_sc_df = sc_df[sc_df["available_bw"] == bw]
        sub_ntt_df = ntt_df[ntt_df["available_bw"] == bw]

        # First row: area vs latency
        ax_area = axes[0, col]
        # Find common gates between sumcheck and NTT for this bw
        common_gates = set(sub_sc_df["sumcheck_gate"].unique())
        min_latency = None
        max_latency = None
        for gate in common_gates:
            gate_ntt = gate.replace(" fz", "")  # Remove 'fz' suffix for NTT
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["gate_name"] == gate_ntt]
            # Plot sumcheck area-latency (color C0)
            if not gate_sc_df.empty:
                costs = gate_sc_df[["area", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_sc_df = gate_sc_df[pareto_mask]
                ax_area.scatter(
                    pareto_gate_sc_df["total_latency"],
                    pareto_gate_sc_df["area"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C0',
                    s=30,
                    edgecolor="k",
                    alpha=0.8
                )
                min_sc = pareto_gate_sc_df["total_latency"].min()
                max_sc = pareto_gate_sc_df["total_latency"].max()
                min_latency = min(min_latency, min_sc) if min_latency is not None else min_sc
                max_latency = max(max_latency, max_sc) if max_latency is not None else max_sc
            # Plot NTT area-latency (color C3)
            if not gate_ntt_df.empty:
                costs_ntt = gate_ntt_df[["total_area", "total_latency"]].values
                pareto_mask_ntt = is_pareto_efficient(costs_ntt)
                pareto_gate_ntt_df = gate_ntt_df[pareto_mask_ntt]
                ax_area.scatter(
                    pareto_gate_ntt_df["total_latency"],
                    pareto_gate_ntt_df["total_area"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C3',
                    s=20,
                    edgecolor="k",
                    alpha=0.8
                )
                min_ntt = pareto_gate_ntt_df["total_latency"].min()
                max_ntt = pareto_gate_ntt_df["total_latency"].max()
                min_latency = min(min_latency, min_ntt) if min_latency is not None else min_ntt
                max_latency = max(max_latency, max_ntt) if max_latency is not None else max_ntt
        # Set x range based on min/max of both SumCheck and NTT dots
        if min_latency is not None and max_latency is not None:
            xlim_min = min_latency * 0.8
            xlim_max = max_sc * 3.5  # max_latency * 1.2
        else:
            xlim_min = 0
            xlim_max = None
        ax_area.set_title(f"Available BW: {bw} GB/s")
        ax_area.set_xlim(left=xlim_min, right=xlim_max)
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
            for gate in common_gates:
                gate_ntt = gate.replace(" fz", "")  # Remove 'fz' suffix for NTT
                handles.append(Line2D([0], [0], marker=marker_dict[gate], color='w', label=gate_ntt,
                                       markerfacecolor='C0', markeredgecolor='k', markersize=10, linestyle='None'))
            # Add color legend for Sumcheck/NTT
            handles.append(Line2D([0], [0], marker='o', color='w', label='Sumcheck', markerfacecolor='C0', markeredgecolor='k', markersize=10, linestyle='None'))
            handles.append(Line2D([0], [0], marker='o', color='w', label='NTT', markerfacecolor='C3', markeredgecolor='k', markersize=10, linestyle='None'))
            ax_area.legend(handles=handles, title="Gate (marker), Color (type)", loc='best', fontsize='small')

        # Save x-tick locations and limits for use in other rows
        xlim = ax_area.get_xlim()
        xticks = locs
        xticklabels = [f"{x/1e6:g}" for x in xticks]

        # Second row: total_onchip_memory_MB vs latency
        ax_mem = axes[1, col]
        for gate in sumcheck_gates:
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            if not gate_sc_df.empty:
                # Pareto filter: minimize both total_onchip_memory_MB and latency
                costs = gate_sc_df[["total_onchip_memory_MB", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                # pareto_gate_sc_df = gate_sc_df[pareto_mask]
                pareto_gate_sc_df = gate_sc_df  # no pareto filter
                ax_mem.scatter(
                    pareto_gate_sc_df["total_latency"],
                    pareto_gate_sc_df["total_onchip_memory_MB"],
                    label=f"Sumcheck: {gate}",
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
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            if not gate_sc_df.empty:
                # Pareto filter: minimize both modmul_count and latency
                costs = gate_sc_df[["modmul_count", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_sc_df = gate_sc_df[pareto_mask]
                ax_modmul.scatter(
                    pareto_gate_sc_df["total_latency"],
                    pareto_gate_sc_df["modmul_count"],
                    label=f"Sumcheck: {gate}",
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
    print(f"Saved plot to {filename}_gate_acrx_bw.png")
    plt.close()


def save_results(sumcheck_result: pd.DataFrame, ntt_result: pd.DataFrame, filename, save_excel=False, draw_plots_type=0):
    """
    Save the sweep results to an Excel file.
    Each row contains the sweep parameters (from 'params') and the stats_dict items as columns.
    Optionally, draw a scatter plot: x="total_latency", y="area", color by "available_bw", marker by "sumcheck_gate".
    """
    
    if save_excel:
        sumcheck_result.to_excel(f"{filename}_sc.xlsx", index=False)
        ntt_result.to_excel(f"{filename}_ntt.xlsx", index=False)
    if draw_plots_type:
        if draw_plots_type == 1:
            plot_area_latency_one(sumcheck_result, filename)
        elif draw_plots_type == 2:
            plot_gate_acrx_bw(sumcheck_result, ntt_result, filename)


if __name__ == "__main__":
    n_values = [20]
    bw_values = [128, 256, 512, 1024]  # in GB/s
    polynomial_list = [
        # [["q1", "q2"]],  # a gate of degree 2
        [["q1", "q2", "q3"]],
        # [["q1", "q2", "q3", "q4"]],
        [["q1", "q2", "q3", "q4", "q5"]],  # a gate of degree 5
        # [["q1", "q2", "q3", "q4", "q5", "q6"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]],  # a gate of degree 8
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]],
    ]

    # NTT
    ntt_result_df = sweep_NTT_configs(
        n_size_values=n_values, 
        bw_values=bw_values,
        polynomial_list=polynomial_list,
    )

    # SumCheck
    sc_results_df = sweep_sumcheck_configs(
        num_var_list=n_values, 
        available_bw_list=bw_values,
        polynomial_list=polynomial_list,
    )

    save_results(sc_results_df, ntt_result_df, "sumcheck_sweep_results_mo", save_excel=True, draw_plots_type=2)

    print("End...")

