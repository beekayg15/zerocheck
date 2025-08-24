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
from test_ntt_func_sim import run_fit_onchip, run_miniNTT_fit_onchip, characterize_poly
from tqdm import tqdm
import math
import os
from pathlib import Path


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


def sweep_miniNTT_all_onchip_configs(n_size_values: list, polynomial_list: list, consider_sparsity=True):
    """
    Sweep all combinations of n and polynomial, calling run_miniNTT_fit_onchip for each.
    Returns a DataFrame with the results.

    :param n_size_values: List of NTT sizes to sweep. Should be the exp `μ`. E.g., [16, 17, 18, ...]
    :param polynomial_list: List of polynomials to sweep. Each polynomial is a list of terms, where each term is a list of strings.
    :return: DataFrame of results
    """
    all_rows = []
    for gate in tqdm(polynomial_list, desc="miniNTT Sweep for gate"):
        poly_features = characterize_poly(gate)
        num_gate_unique_mles, num_gate_reused_mles, num_gate_adds, num_gate_products = poly_features

        gate_name = gate_to_string(gate)
        for n in tqdm(n_size_values, desc=f"miniNTT Sweep for n"):
            max_degree = max(len(term) for term in gate)
            N = 2 ** n
            ntt_len = (max_degree - 1) * N

            # iNTT, NTT, iNTT:
            res_dict = run_miniNTT_fit_onchip(target_n=n, polynomial=gate, modadd_latency=params.modadd_latency,
                                              modmul_latency=params.modmul_latency, consider_sparsity=consider_sparsity)
            # res_dict: key=(num_butterflies), value=dict of cost for that config
            for num_butterflies, value in res_dict.items():
                row = {
                    "sumcheck_gate": gate_name,
                    "n": n,
                    "num_butterflies": num_butterflies,
                }
                value = value.copy()
                # Area calculations (example, adjust as needed for your params)
                value["design_modmul_area"] = value["total_modmuls"] * params.modmul_area  # 22nm, mm^2
                value["total_comp_area_22"] = value["design_modmul_area"] + value["total_modadds"] * params.modadd_area
                value["total_onchip_memory_MB"] = value["total_num_words"] * params.bits_per_scalar / 8 / (1 << 20)
                value["total_mem_area_22"] = value["total_onchip_memory_MB"] * params.MB_CONVERSION_FACTOR
                value["total_area_22"] = value["total_comp_area_22"] + value["total_mem_area_22"]
                value["total_area"] = value["total_area_22"] / params.scale_factor_22_to_7nm
                value["total_latency"] = value["total_cycles"] + num_gate_adds * (ntt_len // (2*num_butterflies)) \
                    + 1 * ntt_len // (num_butterflies)  # term-add term-mul element-wise
                row.update(value)
                all_rows.append(row)
    miniNTT_df = pd.DataFrame(all_rows)
    return miniNTT_df


def sweep_onchip_sumcheck_configs(num_var_list: list, available_bw_list: list, polynomial_list: list):
    """
    Sweeps through all combinations of hardware configs and available bandwidths,
    runs sumcheck_only_sweep, and records all results.

    Args:
        num_var_list: list of num_vars to sweep (e.g., [20])
        available_bw_list: list of available bandwidths to sweep (e.g., [128, 256, 512, 1024])
        polynomial_list: list of sumcheck polynomials to sweep (e.g., [ [["g1", "g2"], ["g3", "g4"]], gate2, ...])
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

    max_num_vars = max(num_var_list)
    # Sweep PE count from 2 to 256 with stride 4, then powers of 2 up to 2**(max_num_vars-1)
    sweep_sumcheck_pes_range = list(range(2, 513, 32))
    # Add higher powers of 2 if needed (above 256)
    sweep_sumcheck_pes_range += [2 ** i for i in range(9, max_num_vars) if 2 ** i > 256]
    sweep_eval_engines_range = range(2, 9, 1)
    sweep_product_lanes_range = range(3, 9, 1)
    # sweep_onchip_mle_sizes_range = [128, 1024, 16384]  # in number of field elements

    # testing all combinations
    loop_iter = product(
        available_bw_list,
        num_var_list,
        sweep_sumcheck_pes_range,
        sweep_eval_engines_range,
        sweep_product_lanes_range,
        # sweep_onchip_mle_sizes_range,
        sumcheck_polynomials
    )
    for (available_bw, num_vars, num_pes, num_eval_engines, num_product_lanes, sumcheck_gate) in tqdm(list(loop_iter), desc="Sumcheck sweep"):
        ##################################################################
        # 1. #num_mle*2(double bf) buffers: for buffering input MLEs.
        #     a. Each size: onchip_mle_size (words)
        # 2. One buffer for Tmp MLE
        #     a. its size: (highest_degree_of_f + 1)*onchip_mle_size/2 (words)
        ##################################################################
        gate_degree = max(len(term) for term in sumcheck_gate)
        num_accumulate_regs = gate_degree + 1
        num_unique_mle_in_gate = len(set(sum(sumcheck_gate, [])))
        num_sumcheck_sram_buffers = num_unique_mle_in_gate  # * 1.5  # no double buffering, but MLE update storage
        tmp_mle_sram_scale_factor = 0  # (gate_degree + 1) / 2
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

        # make the same onchip mem size for sumcheck and NTT
        assert gate_degree >= 2, "Gate degree must be at least 2"
        ntt_length = (gate_degree - 2) * (2**num_vars)
        num_word_in_ntt = (num_unique_mle_in_gate - 1) * ntt_length * 2 + ntt_length / 2
        onchip_mle_size = num_word_in_ntt // (num_sumcheck_sram_buffers + tmp_mle_sram_scale_factor)
        # onchip_mle_size = max(onchip_mle_size, 2**num_vars)  # try to match NTT and SumCheck SRAM size
        onchip_mle_size = 2**num_vars  # always fit

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


def plot_gate_acrx_bw(sc_df: pd.DataFrame, ntt_df: pd.DataFrame, filename, xranges=None, yranges=None):
    """
    Draw multiple subplots: each subplot corresponds to one available_bw.
    Within each subplot, use different marker styles to distinguish sumcheck_gate types.
    """
    n_list = sorted(sc_df["num_vars"].unique())
    sumcheck_gates = sorted(sc_df["sumcheck_gate"].unique())
    marker_styles = ['o', 'X', '^', 's', 'D', 'P', '*', 'v', '<', '>']
    marker_dict = {gate: marker_styles[i % len(marker_styles)] for i, gate in enumerate(sumcheck_gates)}

    # --- First row: area vs latency ---
    num_subplots = len(n_list)
    fig_area, axes_area = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6), sharey=True)
    if num_subplots == 1:
        axes_area = [axes_area]
    if xranges is not None:
        if len(xranges) != num_subplots:
            raise ValueError(f"xranges {xranges} must have length {num_subplots}")
    if yranges is not None:
        if len(yranges) != 3:
            raise ValueError(f"yranges {yranges} must have length {3}")
    saved_xlims = []
    for col, n in enumerate(n_list):
        sub_sc_df = sc_df[sc_df["num_vars"] == n]
        sub_ntt_df = ntt_df[ntt_df["n"] == n]
        ax_area = axes_area[col]
        common_gates = set(sub_sc_df["sumcheck_gate"].unique())
        all_pareto_latencies = []
        for gate in common_gates:
            ntt_gate = gate.replace(" fz", "") if isinstance(gate, str) else gate
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["sumcheck_gate"] == ntt_gate]
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
                all_pareto_latencies.extend(pareto_gate_sc_df["total_latency"].values)
            if not gate_ntt_df.empty:
                costs_ntt = gate_ntt_df[["total_area", "total_latency"]].values if "total_area" in gate_ntt_df.columns else gate_ntt_df[["area", "total_latency"]].values
                pareto_mask_ntt = is_pareto_efficient(costs_ntt)
                pareto_gate_ntt_df = gate_ntt_df[pareto_mask_ntt]
                ax_area.scatter(
                    pareto_gate_ntt_df["total_latency"],
                    pareto_gate_ntt_df["total_area"] if "total_area" in pareto_gate_ntt_df.columns else pareto_gate_ntt_df["area"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C3',
                    s=35,
                    edgecolor="k",
                    alpha=0.8
                )
        if xranges is not None:
            ax_area.set_xlim(*xranges[col])
            saved_xlims.append(tuple(xranges[col]))
        else:
            if all_pareto_latencies:
                min_x = min(all_pareto_latencies)
                max_x = max(all_pareto_latencies)
                ax_area.set_xlim(0.8 * min_x, 1.2 * max_x)
            saved_xlims.append(ax_area.get_xlim())
        ax_area.set_title(f"n = {n}")
        ax_area.set_xlabel("Total Latency (x10^3)")
        xticks = ax_area.get_xticks()
        ax_area.set_xticklabels([f"{x/1e3:g}" for x in xticks])
        ax_area.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_area.minorticks_on()
        if yranges is not None:
            ax_area.set_ylim(*yranges[0])
        if col == 0:
            ax_area.set_ylabel("Area")
        if col == 0:
            handles = []
            for gate in common_gates:
                ntt_gate = gate.replace(" fz", "") if isinstance(gate, str) else gate
                handles.append(Line2D([0], [0], marker=marker_dict[gate], color='w', label=ntt_gate,
                                       markerfacecolor='C0', markeredgecolor='k', markersize=10, linestyle='None'))
            handles.append(Line2D([0], [0], marker='o', color='w', label='Sumcheck', markerfacecolor='C0', markeredgecolor='k', markersize=10, linestyle='None'))
            handles.append(Line2D([0], [0], marker='o', color='w', label='NTT', markerfacecolor='C3', markeredgecolor='k', markersize=10, linestyle='None'))
            ax_area.legend(handles=handles, title="Gate (marker), Color (type)", loc='best', fontsize='small')

    fig_area.tight_layout()
    fig_area.savefig(f"{filename}_gate_acrx_bw_area.png", bbox_inches='tight')
    print(f"Saved plot to {filename}_gate_acrx_bw_area.png")
    plt.close(fig_area)

    # --- Second row: total_onchip_memory_MB vs latency ---
    fig_mem, axes_mem = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6), sharey=True)
    if num_subplots == 1:
        axes_mem = [axes_mem]
    for col, n in enumerate(n_list):
        sub_sc_df = sc_df[sc_df["num_vars"] == n]
        sub_ntt_df = ntt_df[ntt_df["n"] == n]
        ax_mem = axes_mem[col]
        common_gates = set(sub_sc_df["sumcheck_gate"].unique())
        for gate in common_gates:
            ntt_gate = gate.replace(" fz", "") if isinstance(gate, str) else gate
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["sumcheck_gate"] == ntt_gate]
            if not gate_sc_df.empty:
                costs = gate_sc_df[["total_onchip_memory_MB", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_sc_df = gate_sc_df  # Plot all points, not just Pareto-efficient ones
                ax_mem.scatter(
                    pareto_gate_sc_df["total_latency"],
                    pareto_gate_sc_df["total_onchip_memory_MB"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C0',
                    s=30,
                    edgecolor="k",
                    alpha=0.8
                )
            if not gate_ntt_df.empty and "total_onchip_memory_MB" in gate_ntt_df:
                costs_ntt = gate_ntt_df[["total_onchip_memory_MB", "total_latency"]].values
                pareto_mask_ntt = is_pareto_efficient(costs_ntt)
                pareto_gate_ntt_df = gate_ntt_df[pareto_mask_ntt]
                ax_mem.scatter(
                    pareto_gate_ntt_df["total_latency"],
                    pareto_gate_ntt_df["total_onchip_memory_MB"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C3',
                    s=35,
                    edgecolor="k",
                    alpha=0.8
                )
        ax_mem.set_xlim(*saved_xlims[col])
        xticks = axes_area[col].get_xticks()
        ax_mem.set_xticks(xticks)
        ax_mem.set_xticklabels([f"{x/1e3:g}" for x in xticks])
        ax_mem.set_xlabel("Total Latency (x10^3)")
        ax_mem.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_mem.minorticks_on()
        if yranges is not None:
            ax_mem.set_ylim(*yranges[1])
        if col == 0:
            ax_mem.set_ylabel("Total Onchip Memory (MB)")
        ax_mem.set_title(f"n = {n}")
    fig_mem.tight_layout()
    fig_mem.savefig(f"{filename}_gate_acrx_bw_mem.png", bbox_inches='tight')
    print(f"Saved plot to {filename}_gate_acrx_bw_mem.png")
    plt.close(fig_mem)

    # --- Third row: modmul_count vs latency ---
    fig_modmul, axes_modmul = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6), sharey=True)
    if num_subplots == 1:
        axes_modmul = [axes_modmul]
    for col, n in enumerate(n_list):
        sub_sc_df = sc_df[sc_df["num_vars"] == n]
        sub_ntt_df = ntt_df[ntt_df["n"] == n]
        ax_modmul = axes_modmul[col]
        common_gates = set(sub_sc_df["sumcheck_gate"].unique())
        for gate in common_gates:
            ntt_gate = gate.replace(" fz", "") if isinstance(gate, str) else gate
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["sumcheck_gate"] == ntt_gate]
            if not gate_sc_df.empty:
                costs = gate_sc_df[["modmul_count", "total_latency"]].values if "modmul_count" in gate_sc_df.columns else gate_sc_df[["total_modmuls", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_sc_df = gate_sc_df[pareto_mask]
                ax_modmul.scatter(
                    pareto_gate_sc_df["total_latency"],
                    pareto_gate_sc_df["modmul_count"] if "modmul_count" in pareto_gate_sc_df.columns else pareto_gate_sc_df["total_modmuls"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C0',
                    s=30,
                    edgecolor="k",
                    alpha=0.8
                )
            if not gate_ntt_df.empty:
                costs_ntt = gate_ntt_df[["total_modmuls", "total_latency"]].values
                pareto_mask_ntt = is_pareto_efficient(costs_ntt)
                pareto_gate_ntt_df = gate_ntt_df[pareto_mask_ntt]
                ax_modmul.scatter(
                    pareto_gate_ntt_df["total_latency"],
                    pareto_gate_ntt_df["total_modmuls"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C3',
                    s=35,
                    edgecolor="k",
                    alpha=0.8
                )
        ax_modmul.set_xlim(*saved_xlims[col])
        xticks = axes_area[col].get_xticks()
        ax_modmul.set_xticks(xticks)
        ax_modmul.set_xticklabels([f"{x/1e3:g}" for x in xticks])
        ax_modmul.set_xlabel("Total Latency (x10^3)")
        ax_modmul.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_modmul.minorticks_on()
        if yranges is not None:
            ax_modmul.set_ylim(*yranges[2])
        if col == 0:
            ax_modmul.set_ylabel("Modmul Count")
        ax_modmul.set_title(f"n = {n}")
    fig_modmul.tight_layout()
    fig_modmul.savefig(f"{filename}_gate_acrx_bw_modmul.png", bbox_inches='tight')
    print(f"Saved plot to {filename}_gate_acrx_bw_modmul.png")
    plt.close(fig_modmul)


def save_results(sumcheck_result: pd.DataFrame, ntt_result: pd.DataFrame, filename, save_excel=False):
    """
    Save the sweep results to Excel files.
    Each row contains the sweep parameters (from 'params') and the stats_dict items as columns.
    """
    if save_excel:
        sumcheck_result.to_excel(f"{filename}_sc.xlsx", index=False)
        ntt_result.to_excel(f"{filename}_ntt.xlsx", index=False)
        print(f"Saved results to {filename}_sc.xlsx and {filename}_ntt.xlsx")


def load_results(filename):
    """
    Load the sweep results from Excel files into DataFrames.
    Returns (sumcheck_result_df, ntt_result_df).
    """
    print(f"Loading results from {filename}_sc.xlsx and {filename}_ntt.xlsx")
    sumcheck_path = f"{filename}_sc.xlsx"
    ntt_path = f"{filename}_ntt.xlsx"
    sumcheck_result_df = pd.read_excel(sumcheck_path)
    ntt_result_df = pd.read_excel(ntt_path)
    return sumcheck_result_df, ntt_result_df


def plot_gate_acrx_groups(sc_df, ntt_df, filename, poly_groups):
    """
    Draw a 1x3 grid of subplots: each column is a group of polynomials (gates).
    Bandwidth is not considered.
    Args:
        sc_df: DataFrame for sumcheck results
        ntt_df: DataFrame for NTT results
        filename: output file prefix (no extension)
        poly_groups: list of 3 lists, each is a group of gates (gate as list of lists)
    """
    assert len(poly_groups) == 3, "Need 3 poly groups"
    marker_styles = ['o', 's', '^', 'X', 'D', 'P', '*', 'v', '<', '>']
    # Flatten all groups to get unique gates
    all_gates = [gate for group in poly_groups for gate in group]
    unique_gate_names = sorted(set(gate_to_string(gate) for gate in all_gates))
    marker_dict = {gate_name: marker_styles[i % len(marker_styles)] for i, gate_name in enumerate(unique_gate_names)}

    fig = plt.figure(figsize=(24, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.8], 
                         )  # Reduced spacing between all subplots
    # Create first 3 subplots normally
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # Add the 4th subplot separately since it will be 3D
    axes.append(fig.add_subplot(gs[0, 3], projection='3d'))

    # 1-3: original scatter plots
    for col, group in enumerate(poly_groups):
        ax = axes[col]
        group_gate_names = [gate_to_string(gate) for gate in group]
        # For SumCheck, add another 'fz' to each term for indexing
        group_sc_gate_names = [gate_to_string([[*term, "fz"] for term in gate]) for gate in group]
        sub_sc_df = sc_df[sc_df["sumcheck_gate"].isin(group_sc_gate_names)]
        sub_ntt_df = ntt_df[ntt_df["sumcheck_gate"].isin(group_gate_names)]
        for gate in group:
            gate_name = gate_to_string(gate)
            sc_gate_name = gate_to_string([[*term, "fz"] for term in gate])
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == sc_gate_name]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["sumcheck_gate"] == gate_name]
            # Pareto filter for Sumcheck
            if not gate_sc_df.empty:
                costs = gate_sc_df[["area", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_sc_df = gate_sc_df[pareto_mask]
                # Convert total_latency from ns to ms
                ax.scatter(
                    pareto_gate_sc_df["total_latency"] / 1e3,
                    pareto_gate_sc_df["area"],
                    marker=marker_dict[gate_name],
                    color='C0',
                    s=30,
                    edgecolor="k",
                    alpha=0.8,
                    label=f"{gate_name} (Sumcheck)"
                )
            # Pareto filter for NTT
            if not gate_ntt_df.empty:
                area_col = "total_area" if "total_area" in gate_ntt_df.columns else "area"
                costs_ntt = gate_ntt_df[[area_col, "total_latency"]].values
                pareto_mask_ntt = is_pareto_efficient(costs_ntt)
                pareto_gate_ntt_df = gate_ntt_df[pareto_mask_ntt]
                # Convert total_latency from ns to ms
                ax.scatter(
                    pareto_gate_ntt_df["total_latency"] / 1e3,
                    pareto_gate_ntt_df[area_col],
                    marker=marker_dict[gate_name],
                    color='C3',
                    s=35,
                    edgecolor="k",
                    alpha=0.8,
                    label=f"{gate_name} (NTT)"
                )
        # Change x-axis to ms units
        xticks = ax.get_xticks()
        ax.set_xlabel("Runtime (μs)", fontsize=14)
        ax.set_ylim(0, 650)  # Limit y-axis between 0 and 650
        ax.set_xlim(1, 30)  # Convert limits from ns to ms
        ax.set_xticklabels([f"{x:.0f}" for x in ax.get_xticks()], fontsize=14)
        ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.minorticks_on()
        ax.tick_params(axis='y', labelsize=14)
        if col == 0:
            ax.set_ylabel("Area (mm²)", fontsize=14)
        # Custom legend for each subplot (group)
        handles = []
        for gate in group:
            gate_name = gate_to_string(gate)
            handles.append(Line2D([0], [0], marker=marker_dict[gate_name], color='w', label=f"{gate_name} (SumCheck)",
                      markerfacecolor='C0', markeredgecolor='k', markersize=10, linestyle='None'))
            handles.append(Line2D([0], [0], marker=marker_dict[gate_name], color='w', label=f"{gate_name} (NTT)",
                      markerfacecolor='C3', markeredgecolor='k', markersize=10, linestyle='None'))
        ax.legend(handles=handles, loc='best', fontsize=12)


    # 4th subplot: 3D surface from plot_n17_ntt_sumcheck_allonchip
    from sram_budget_plot import plot_n17_ntt_sumcheck_allonchip
    plot_n17_ntt_sumcheck_allonchip(max_MB=200, ax=axes[3])
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    axes[3].spines['bottom'].set_visible(False)
    axes[3].spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{filename}_gate_acrx_groups.pdf", bbox_inches='tight')
    plt.savefig(f"{filename}_gate_acrx_groups.png", bbox_inches='tight')
    print(f"Saved plot to {filename}_gate_acrx_groups.pdf")
    plt.close(fig)


if __name__ == "__main__":
    n_values = 17
    # bw_values = [128, 256, 1024, 2048]  # in GB/s

    # ################################################
    # poly_style_name = "deg_inc_mle_inc_term_fixed"
    # polynomial_list = [
    #     [["g1", "g2"]],
    #     [["g1", "g2", "g3"]],  # a gate of degree 3
    #     [["g1", "g2", "g3", "g4"]],
    #     # [["g1", "g2", "g3", "g4", "g5"]],  # a gate of degree 5
    #     # [["g1", "g2", "g3", "g4", "g5", "g6"]],
    #     # [["g1", "g2", "g3", "g4", "g5", "g6", "g7"]],
    #     # [["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]],
    #     # [["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"]],  # a gate of degree 9
    #     # [["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10"]],
    # ]

    # output_dir = Path(f"outplot_mo_onchip/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # ntt_result_df = sweep_miniNTT_all_onchip_configs(
    #     n_size_values=[n_values],
    #     polynomial_list=polynomial_list,
    #     consider_sparsity=True
    # )
    # sc_results_df = sweep_onchip_sumcheck_configs(
    #     num_var_list=[n_values],
    #     available_bw_list=[1e9],
    #     polynomial_list=polynomial_list,
    # )
    # save_results(
    #     sc_results_df,
    #     ntt_result_df,
    #     output_dir.joinpath(f"{poly_style_name}"),
    #     save_excel=True
    # )
    # # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    # xlim_area = None # [(0e3, 40e3)]
    # ylim_area = None # [(0, 400), (8, 60), (0, 2500)]
    # plot_gate_acrx_bw(sc_df=sc_results_df, 
    #                   ntt_df=ntt_result_df,
    #                   filename=output_dir.joinpath(f"{poly_style_name}"),
    #                   xranges=xlim_area,
    #                   yranges=ylim_area)

    # # ################################################
    # # poly_style_name = "deg_inc_mle_fixed_term_fixed"
    # # polynomial_list = [
    # #     [["g1", "g2"]],
    # #     [["g1", "g2", "g2"]],
    # #     [["g1", "g2", "g2", "g2"]],
    # #     [["g1", "g2", "g2", "g2", "g2"]],
    # # ]

    # # output_dir = Path(f"outplot_mo_onchip/n_{n_values}")
    # # if not os.path.exists(output_dir):
    # #     os.makedirs(output_dir, exist_ok=True)
    # # ntt_result_df = sweep_miniNTT_all_onchip_configs(
    # #     n_size_values=[n_values],
    # #     polynomial_list=polynomial_list,
    # #     consider_sparsity=True
    # # )
    # # sc_results_df = sweep_onchip_sumcheck_configs(
    # #     num_var_list=[n_values],
    # #     available_bw_list=[1e9],
    # #     polynomial_list=polynomial_list,
    # # )
    # # save_results(
    # #     sc_results_df,
    # #     ntt_result_df,
    # #     output_dir.joinpath(f"{poly_style_name}"),
    # #     save_excel=True
    # # )
    # # # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    # # xlim_area = None # [(1.5e6, 9.5e6), (1.5e6, 9e6), (0, 3e6), (0, 1.5e6)]
    # # ylim_area = None # [(0, 350), (0, 7), (0, 4500)]
    # # plot_gate_acrx_bw(sc_df=sc_results_df, 
    # #                   ntt_df=ntt_result_df,
    # #                   filename=output_dir.joinpath(f"{poly_style_name}"),
    # #                   xranges=xlim_area,
    # #                   yranges=ylim_area)
    
    # # ################################################
    # poly_style_name = "deg_fixed_mle_fixed_term_inc"
    # polynomial_list = [
    #     [["g1", "g2"], ["g3"]],
    #     [["g1", "g2"], ["g1"], ["g3"]],
    #     [["g1", "g2"], ["g1"], ["g2"], ["g3"]],
    # ]

    # output_dir = Path(f"outplot_mo_onchip/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # ntt_result_df = sweep_miniNTT_all_onchip_configs(
    #     n_size_values=[n_values],
    #     polynomial_list=polynomial_list,
    #     consider_sparsity=True
    # )
    # sc_results_df = sweep_onchip_sumcheck_configs(
    #     num_var_list=[n_values],
    #     available_bw_list=[1e9],
    #     polynomial_list=polynomial_list,
    # )
    # save_results(
    #     sc_results_df,
    #     ntt_result_df,
    #     output_dir.joinpath(f"{poly_style_name}"),
    #     save_excel=True
    # )
    # # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    # xlim_area = None # [(2e6, 7e6), (1e6, 6e6), (0, 2e6), (0, 2e6)]
    # ylim_area = None # [(0, 180), (0, 6.4), (0, 2200)]
    # plot_gate_acrx_bw(sc_df=sc_results_df, 
    #                   ntt_df=ntt_result_df,
    #                   filename=output_dir.joinpath(f"{poly_style_name}"),
    #                   xranges=xlim_area,
    #                   yranges=ylim_area)
    
    # ################################################
    # poly_style_name = "deg_fixed_mle_inc_term_inc"
    # polynomial_list = [
    #     [["g1", "g2"]],
    #     [["g1", "g2"], ["g3"]],
    #     [["g1", "g2"], ["g3"], ["g4"]],
    #     # [["g1", "g2"], ["g1", "g3"]],
    #     # [["g1", "g2"], ["g1", "g3"], ["g1", "g4"]],
    #     # [["g1", "g2"], ["g1", "g3"], ["g1", "g4"], ["g1", "g5"]],
    # ]

    # output_dir = Path(f"outplot_mo_onchip/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # ntt_result_df = sweep_miniNTT_all_onchip_configs(
    #     n_size_values=[n_values],
    #     polynomial_list=polynomial_list,
    #     consider_sparsity=True
    # )
    # sc_results_df = sweep_onchip_sumcheck_configs(
    #     num_var_list=[n_values],
    #     available_bw_list=[1e9],
    #     polynomial_list=polynomial_list,
    # )
    # save_results(
    #     sc_results_df,
    #     ntt_result_df,
    #     output_dir.joinpath(f"{poly_style_name}"),
    #     save_excel=True
    # )
    # # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    # xlim_area = None # [(1.5e6, 12e6), (0e6, 8e6), (0, 2.5e6), (0, 2e6)]
    # ylim_area = None # [(0, 160), (0, 11.5), (0, 2200)]
    # plot_gate_acrx_bw(sc_df=sc_results_df, 
    #                   ntt_df=ntt_result_df,
    #                   filename=output_dir.joinpath(f"{poly_style_name}"),
    #                   xranges=xlim_area,
    #                   yranges=ylim_area)
    
    # ################################################
    ################################################
    poly_style_name = "general_runs"
    polynomial_list = [
        [["g1", "g2"]],
        [["g1", "g2", "g3"]],  # a gate of degree 3
        [["g1", "g2", "g3", "g4"]],

        [["g1", "g2"], ["g3"]],
        [["g1", "g2"], ["g1"], ["g3"]],
        [["g1", "g2"], ["g1"], ["g2"], ["g3"]],

        [["g1", "g2"], ["g3"]],
        [["g1", "g2"], ["g3"], ["g4"]],
    ]

    output_dir = Path(f"outplot_mo_onchip/n_{n_values}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    ntt_result_df = sweep_miniNTT_all_onchip_configs(
        n_size_values=[n_values],
        polynomial_list=polynomial_list,
        consider_sparsity=True
    )
    sc_results_df = sweep_onchip_sumcheck_configs(
        num_var_list=[n_values],
        available_bw_list=[1e9],
        polynomial_list=polynomial_list,
    )
    save_results(
        sc_results_df,
        ntt_result_df,
        output_dir.joinpath(f"{poly_style_name}"),
        save_excel=True
    )
    # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    polynomial_list = [
        [
            [["g1", "g2"]],
            [["g1", "g2"], ["g3"]],
            [["g1", "g2"], ["g3"], ["g4"]],
        ],
        [
            [["g1", "g2"], ["g3"]],
            [["g1", "g2"], ["g1"], ["g3"]],
            [["g1", "g2"], ["g1"], ["g2"], ["g3"]],
        ],
        [
            [["g1", "g2"]],
            [["g1", "g2", "g3"]],  # a gate of degree 3
            [["g1", "g2", "g3", "g4"]],
        ],
    ]
    plot_gate_acrx_groups(
        sc_df=sc_results_df,
        ntt_df=ntt_result_df,
        filename=output_dir.joinpath(f"{poly_style_name}_n{n_values}"),
        poly_groups=polynomial_list,
    )
    
    
    ################################################
    print("End...")

