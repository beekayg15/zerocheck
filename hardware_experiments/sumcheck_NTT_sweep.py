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


def get_step_radix_gate_degree(gate_degree):
    """
    Get the step radix of breaking a larger (deg-1) to two 2's power sum.
    Then we can run NTT on each size.
    """
    assert gate_degree >= 1, "Gate degree must be at least 1"
    degree_minus1 = gate_degree - 1

    if gate_degree == 1:
        return [0]
    elif degree_minus1 & (degree_minus1 - 1) == 0:
        # degree_minus1 is a power of 2: do degree_minus1*2^n size NTT
        return [degree_minus1]
    elif degree_minus1 != 3 and (degree_minus1 + 1) & degree_minus1 == 0:
        # degree_minus1 == 2^k - 1, so use 2^k * lengthN
        msb = degree_minus1.bit_length()
        return [2 ** msb]
    else:
        # Not a power of 2 or 2^k-1: break into at most two 2's powers (a+b >= degree_minus1)
        msb = degree_minus1.bit_length() - 1
        a = 2 ** msb
        # Find next set bit below MSB
        b = 0
        for i in range(msb - 1, -1, -1):
            if (degree_minus1 >> i) & 1:
                b = 2 ** i
                break
        if b == 0:
            # Only one set bit, should not happen here
            b = 0
        # If a+b < degree_minus1, bump b to next lower power of 2 (covering the case where degree_minus1 is not sum of two 2's powers)
        if a + b < degree_minus1:
            # Use the next lower power of 2
            b = 2 ** (msb - 1)

        return [a, b]


def run_step_radix_ntt(gate_degree, n, bw, polynomial=None, **kwargs):
    """
    Simulate NTT runtime for non-2's power size using step-radix decomposition.
    The NTT size is (gate_degree-1) * 2^n, which may not be a power of 2.
    This function uses get_step_radix_gate_degree to break (gate_degree-1) into its 2's power chunks,
    runs run_fit_onchip for the corresponding 2's power NTTs, then sums the results.

    Args:
        gate_degree: Degree of the gate (int)
        n: log2(N), where N is the base NTT size (int)
        bw: Bandwidth in GB/s (int)
        polynomial: The polynomial to analyze (optional, passed to run_fit_onchip)
        **kwargs: Additional arguments for run_fit_onchip

    Returns:
        res: dict with summed total_cycles, total_modmuls, total_modadds, total_num_words, etc.
    """
    lengthN = 2 ** n
    assert gate_degree >= 1, "Gate degree must be at least 1"
    degree_minus1 = gate_degree - 1

    # Use get_step_radix_gate_degree to get the list of 2's power chunks
    chunk_sizes = get_step_radix_gate_degree(gate_degree)
    if chunk_sizes == [0]:
        # degree 1: no NTT needed, set all cycles to 0
        res = run_fit_onchip(target_n=n, target_bw=bw, polynomial=polynomial, **kwargs)
        for v in res.values():
            for k in ["total_cycles", "single_ntt_cycles", "all_ntt_cycles", "elementwise_cycles"]:
                if k in v:
                    v[k] = 0
        return res

    # For each chunk, run NTT of size chunk_size * lengthN
    if len(chunk_sizes) == 1:
        n_chunk = int(math.log2(chunk_sizes[0]) + n)  # n_chunk = 2+20
        return run_fit_onchip(target_n=n_chunk, target_bw=bw, polynomial=polynomial, **kwargs)
    else:
        # Combine results: use the first chunk's res_chunk as the base, and only add 'total_cycles' from the second chunk
        a, b = math.log2(chunk_sizes[0]), math.log2(chunk_sizes[1])
        res_a = run_fit_onchip(target_n=int(a + n), target_bw=bw, polynomial=polynomial, **kwargs)
        res_b = run_fit_onchip(target_n=int(b + n), target_bw=bw, polynomial=polynomial, unroll_factors_pow=math.ceil((a + n)/2), **kwargs)
        # Build combined result dict: keys from res_a, sum total_cycles with matching key in res_b
        res = {}
        for key_a, val_a in res_a.items():
            # key_a: (a, available_bw, unroll_factor, pe_amt)
            # Find corresponding key in res_b: replace a with b
            key_b = (b, key_a[1], key_a[2], key_a[3])  # key=(n_pow, available_bw, unroll_factor, pe_amt)
            val_b = res_b.get(key_b)
            if val_b is not None:
                combined = val_a.copy()
                # Sum total_cycles and other relevant fields if needed
                if "total_cycles" in combined and "total_cycles" in val_b:
                    combined["total_cycles"] += val_b["total_cycles"]
                # if "all_ntt_cycles" in combined and "all_ntt_cycles" in val_b:
                #     combined["all_ntt_cycles"] += val_b["all_ntt_cycles"]
                # if "single_ntt_cycles" in combined and "single_ntt_cycles" in val_b:
                #     combined["single_ntt_cycles"] += val_b["single_ntt_cycles"]
                # if "elementwise_cycles" in combined and "elementwise_cycles" in val_b:
                #     combined["elementwise_cycles"] += val_b["elementwise_cycles"]
                res[(math.log2(degree_minus1) + n, key_a[1], key_a[2], key_a[3])] = combined
            else:
                raise ValueError(f"Key {key_b} not found in res_b. This should not happen for valid NTT sizes.")
        return res


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
        gate_degree_n = int(math.log2(gate_degree - 1))

        for n in tqdm(n_size_values, desc=f"NTT Sweep for n"):
            for bw in tqdm(bw_values, desc=f"NTT sweep bw"):
                print(f"Running NTT sweep for n={gate_degree - 1}x2^{n}, bw={bw}...")
                step_sizes = get_step_radix_gate_degree(gate_degree)
                res_input_iNTT = run_fit_onchip(target_n=n, target_bw=bw, save_pkl=False, unroll_factors_pow=math.ceil((n+math.log2(step_sizes[0]))/2))
                # res = run_fit_onchip(target_n=n+gate_degree_n, target_bw=bw, save_pkl=False)
                res = run_step_radix_ntt(gate_degree=gate_degree, n=n, bw=bw, polynomial=None, save_pkl=False)
                # res is a dict: key=(n_pow, available_bw, unroll_factor, pe_amt), value=dict
                for key, value in res.items():
                    n_pow, available_bw, unroll_factor, pe_amt = key
                    row = {
                        "gate_name": gate_name,
                        "gate_num_terms": gate_num_terms,
                        "gate_num_unique_mle": gate_num_unique_mle,
                        "gate_degree": gate_degree,
                        "n": n,
                        # "target_n": n + gate_degree_n,
                        "n_pow": n_pow,  # target_n
                        "available_bw": available_bw,
                        "unroll_factor": unroll_factor,
                        "pe_amt": pe_amt,
                    }
                    
                    value = value.copy()
                    value_input_iNTT = res_input_iNTT.get((n, available_bw, unroll_factor, pe_amt), {})
                    if "total_cycles" in value:
                        # Repeat NTT for each MLE in series
                        value["total_latency"] = value_input_iNTT["total_cycles"] * gate_num_unique_mle + value["total_cycles"] * (gate_num_unique_mle + 1)  # +q

                        # area cost
                        value["design_modmul_area"] = value["total_modmuls"] * params.modmul_area  # 22nm, mm^2
                        value["total_comp_area_22"] = value["design_modmul_area"] + value["total_modadds"] * params.modadd_area
                        value["total_onchip_memory_MB"] = value["total_num_words"] * params.bits_per_scalar / 8 / (1 << 20)
                        value["total_mem_area_22"] = value["total_onchip_memory_MB"] * params.MB_CONVERSION_FACTOR
                        value["total_area_22"] = value["total_comp_area_22"] + value["total_mem_area_22"]
                        value["total_area"] = value["total_area_22"] / params.scale_factor_22_to_7nm
                    
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


def plot_gate_acrx_bw(sc_df: pd.DataFrame, ntt_df: pd.DataFrame, filename, xranges=None, yranges=None):
    """
    Draw multiple subplots: each subplot corresponds to one available_bw.
    Within each subplot, use different marker styles to distinguish sumcheck_gate types.
    """
    available_bw_list = sorted(sc_df["available_bw"].unique())
    sumcheck_gates = sorted(sc_df["sumcheck_gate"].unique())

    # Define marker styles for sumcheck_gate
    marker_styles = ['o', 'X', '^', 's', 'D', 'P', '*', 'v', '<', '>']
    marker_dict = {gate: marker_styles[i % len(marker_styles)] for i, gate in enumerate(sumcheck_gates)}

    # --- First row: area vs latency ---
    num_subplots = len(available_bw_list)
    fig_area, axes_area = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6), sharey=True)
    if num_subplots == 1:
        axes_area = [axes_area]

    if xranges is not None:
        if len(xranges) != num_subplots:
            raise ValueError(f"xranges {xranges} must have length {num_subplots}")
    if yranges is not None:
        if len(yranges) != 3:  # area, Mem MB, modmul num
            raise ValueError(f"yranges {yranges} must have length {3}")

    # Save xlims from area plot for each subplot
    saved_xlims = []
    for col, bw in enumerate(available_bw_list):
        sub_sc_df = sc_df[sc_df["available_bw"] == bw]
        sub_ntt_df = ntt_df[ntt_df["available_bw"] == bw]
        ax_area = axes_area[col]
        common_gates = set(sub_sc_df["sumcheck_gate"].unique())
        all_pareto_latencies = []
        for gate in common_gates:
            gate_ntt = gate.replace(" fz", "")
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["gate_name"] == gate_ntt]
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
                costs_ntt = gate_ntt_df[["total_area", "total_latency"]].values
                pareto_mask_ntt = is_pareto_efficient(costs_ntt)
                pareto_gate_ntt_df = gate_ntt_df[pareto_mask_ntt]
                ax_area.scatter(
                    pareto_gate_ntt_df["total_latency"],
                    pareto_gate_ntt_df["total_area"],
                    label=None,
                    marker=marker_dict[gate],
                    color='C3',
                    s=35,
                    edgecolor="k",
                    alpha=0.8
                )
        # Set xlim: use xranges if provided, else use (0.8*min, 1.2*max) of Pareto-efficient sumcheck points
        if xranges is not None:
            ax_area.set_xlim(*xranges[col])
            saved_xlims.append(tuple(xranges[col]))
        else:
            if all_pareto_latencies:
                min_x = min(all_pareto_latencies)
                max_x = max(all_pareto_latencies)
                ax_area.set_xlim(0.8 * min_x, 1.2 * max_x)
            saved_xlims.append(ax_area.get_xlim())
        ax_area.set_title(f"Available BW: {bw} GB/s")
        ax_area.set_xlabel("Total Latency (x10^6)")
        # Format x-tick labels to show values divided by 1e6
        xticks = ax_area.get_xticks()
        ax_area.set_xticklabels([f"{x/1e6:g}" for x in xticks])
        ax_area.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_area.minorticks_on()
        if yranges is not None:
            ax_area.set_ylim(*yranges[0])
        if col == 0:
            ax_area.set_ylabel("Area")
        if col == 0:
            handles = []
            for gate in common_gates:
                gate_ntt = gate.replace(" fz", "")
                handles.append(Line2D([0], [0], marker=marker_dict[gate], color='w', label=gate_ntt,
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
    for col, bw in enumerate(available_bw_list):
        sub_sc_df = sc_df[sc_df["available_bw"] == bw]
        sub_ntt_df = ntt_df[ntt_df["available_bw"] == bw]
        ax_mem = axes_mem[col]
        common_gates = set(sub_sc_df["sumcheck_gate"].unique())
        for gate in common_gates:
            gate_ntt = gate.replace(" fz", "")
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["gate_name"] == gate_ntt]
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
        # Use the saved xlim from area plot
        ax_mem.set_xlim(*saved_xlims[col])
        # Set x-ticks and labels to match the first plot
        xticks = axes_area[col].get_xticks()
        ax_mem.set_xticks(xticks)
        ax_mem.set_xticklabels([f"{x/1e6:g}" for x in xticks])
        ax_mem.set_xlabel("Total Latency (x10^6)")
        ax_mem.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_mem.minorticks_on()
        if yranges is not None:
            ax_mem.set_ylim(*yranges[1])
        if col == 0:
            ax_mem.set_ylabel("Total Onchip Memory (MB)")
        ax_mem.set_title(f"Available BW: {bw} GB/s")
    fig_mem.tight_layout()
    fig_mem.savefig(f"{filename}_gate_acrx_bw_mem.png", bbox_inches='tight')
    print(f"Saved plot to {filename}_gate_acrx_bw_mem.png")
    plt.close(fig_mem)

    # --- Third row: modmul_count vs latency ---
    fig_modmul, axes_modmul = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6), sharey=True)
    if num_subplots == 1:
        axes_modmul = [axes_modmul]
    for col, bw in enumerate(available_bw_list):
        sub_sc_df = sc_df[sc_df["available_bw"] == bw]
        sub_ntt_df = ntt_df[ntt_df["available_bw"] == bw]
        ax_modmul = axes_modmul[col]
        common_gates = set(sub_sc_df["sumcheck_gate"].unique())
        for gate in common_gates:
            gate_ntt = gate.replace(" fz", "")
            gate_sc_df = sub_sc_df[sub_sc_df["sumcheck_gate"] == gate]
            gate_ntt_df = sub_ntt_df[sub_ntt_df["gate_name"] == gate_ntt]
            if not gate_sc_df.empty:
                costs = gate_sc_df[["modmul_count", "total_latency"]].values
                pareto_mask = is_pareto_efficient(costs)
                pareto_gate_sc_df = gate_sc_df[pareto_mask]
                ax_modmul.scatter(
                    pareto_gate_sc_df["total_latency"],
                    pareto_gate_sc_df["modmul_count"],
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
        # Use the saved xlim from area plot
        ax_modmul.set_xlim(*saved_xlims[col])
        # Set x-ticks and labels to match the first plot
        xticks = axes_area[col].get_xticks()
        ax_modmul.set_xticks(xticks)
        ax_modmul.set_xticklabels([f"{x/1e6:g}" for x in xticks])
        ax_modmul.set_xlabel("Total Latency (x10^6)")
        ax_modmul.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_modmul.minorticks_on()
        if yranges is not None:
            ax_modmul.set_ylim(*yranges[2])
        if col == 0:
            ax_modmul.set_ylabel("Modmul Count")
        ax_modmul.set_title(f"Available BW: {bw} GB/s")  # Add title to each subplot
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


if __name__ == "__main__":
    n_values = 20
    bw_values = [128, 256, 1024, 2048]  # in GB/s

    ################################################
    poly_style_name = "deg_inc_mle_inc_term_fixed"
    polynomial_list = [
        [["q1", "q2"]],
        [["q1", "q2", "q3"]],  # a gate of degree 3
        [["q1", "q2", "q3", "q4"]],
        [["q1", "q2", "q3", "q4", "q5"]],  # a gate of degree 5
        # [["q1", "q2", "q3", "q4", "q5", "q6"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]],
        # a gate of degree 9
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]],
        # [["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]],
    ]

    output_dir = Path(f"outplot_mo/n_{n_values}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    ntt_result_df = sweep_NTT_configs(
        n_size_values=[n_values],
        bw_values=bw_values,
        polynomial_list=polynomial_list,
    )
    sc_results_df = sweep_sumcheck_configs(
        num_var_list=[n_values],
        available_bw_list=bw_values,
        polynomial_list=polynomial_list,
    )
    save_results(
        sc_results_df,
        ntt_result_df,
        output_dir.joinpath(f"{poly_style_name}"),
        save_excel=True
    )
    # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    xlim_area = None # [(1.8e6, 14.5e6), (0.5e6, 8e6), (0, 6e6), (0, 4e6)]
    ylim_area = None # [(0, 350), (0, 13), (0, 4500)]
    plot_gate_acrx_bw(sc_df=sc_results_df, 
                      ntt_df=ntt_result_df,
                      filename=output_dir.joinpath(f"{poly_style_name}"),
                      xranges=xlim_area,
                      yranges=ylim_area)

    # ################################################
    # poly_style_name = "deg_inc_mle_fixed_term_fixed"
    # polynomial_list = [
    #     [["q1", "q2", "q3"]],
    #     [["q1", "q2", "q3", "q3", "q3"]],
    #     [["q1", "q2", "q3", "q3", "q3", "q3", "q3", "q3", "q3"]],
    # ]

    # output_dir = Path(f"outplot_mo/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # # ntt_result_df = sweep_NTT_configs(
    # #     n_size_values=[n_values],
    # #     bw_values=bw_values,
    # #     polynomial_list=polynomial_list,
    # # )
    # # sc_results_df = sweep_sumcheck_configs(
    # #     num_var_list=[n_values],
    # #     available_bw_list=bw_values,
    # #     polynomial_list=polynomial_list,
    # # )        
    # # save_results(
    # #     sc_results_df,
    # #     ntt_result_df,
    # #     output_dir.joinpath(f"{poly_style_name}"),
    # #     save_excel=True
    # # )
    # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    # xlim_area = [(1.5e6, 9.5e6), (1.5e6, 9e6), (0, 3e6), (0, 1.5e6)]
    # ylim_area = [(0, 350), (0, 7), (0, 4500)]
    # plot_gate_acrx_bw(sc_df=sc_results_df, 
    #                   ntt_df=ntt_result_df,
    #                   filename=output_dir.joinpath(f"{poly_style_name}"),
    #                   xranges=xlim_area,
    #                   yranges=ylim_area)
    
    # ################################################
    # poly_style_name = "deg_fixed_mle_fixed_term_inc"
    # polynomial_list = [
    #     [["q1", "q2", "q3"], ["q1", "q2", "q4"]],
    #     [["q1", "q2", "q3"], ["q1", "q2", "q4"], ["q1", "q3", "q4"]],
    #     [["q1", "q2", "q3"], ["q1", "q2", "q4"], [
    #         "q1", "q3", "q4"], ["q2", "q3", "q4"]],
    # ]

    # output_dir = Path(f"outplot_mo/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # # ntt_result_df = sweep_NTT_configs(
    # #     n_size_values=[n_values],
    # #     bw_values=bw_values,
    # #     polynomial_list=polynomial_list,
    # # )
    # # sc_results_df = sweep_sumcheck_configs(
    # #     num_var_list=[n_values],
    # #     available_bw_list=bw_values,
    # #     polynomial_list=polynomial_list,
    # # )    
    # # save_results(
    # #     sc_results_df,
    # #     ntt_result_df,
    # #     output_dir.joinpath(f"{poly_style_name}"),
    # #     save_excel=True
    # # )
    # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    # xlim_area = [(2e6, 7e6), (1e6, 6e6), (0, 2e6), (0, 2e6)]
    # ylim_area = [(0, 180), (0, 6.4), (0, 2200)]
    # plot_gate_acrx_bw(sc_df=sc_results_df, 
    #                   ntt_df=ntt_result_df,
    #                   filename=output_dir.joinpath(f"{poly_style_name}"),
    #                   xranges=xlim_area,
    #                   yranges=ylim_area)
    
    # ################################################
    # poly_style_name = "deg_fixed_mle_inc_term_inc"
    # polynomial_list = [
    #     [["q1", "q2"], ["q1", "q2", "q3"]],
    #     [["q1", "q2", "q3"], ["q1", "q2", "q4"], ["q1", "q3", "q4"]],
    #     [["q1", "q2", "q3"], ["q1", "q3", "q4"], [
    #         "q2", "q3", "q4"], ["q1", "q2", "q5"]],
    #     [["q1", "q2", "q3"], ["q1", "q2", "q4"], ["q1", "q2", "q5"], ["q1", "q2", "q6"], [
    #         "q1", "q2", "q7"], ["q1", "q2", "q8"], ["q1", "q2", "q9"]],
    # ]

    # output_dir = Path(f"outplot_mo/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # # ntt_result_df = sweep_NTT_configs(
    # #     n_size_values=[n_values],
    # #     bw_values=bw_values,
    # #     polynomial_list=polynomial_list,
    # # )
    # # sc_results_df = sweep_sumcheck_configs(
    # #     num_var_list=[n_values],
    # #     available_bw_list=bw_values,
    # #     polynomial_list=polynomial_list,
    # # )
    # # save_results(
    # #     sc_results_df,
    # #     ntt_result_df,
    # #     output_dir.joinpath(f"{poly_style_name}"),
    # #     save_excel=True
    # # )
    # sc_results_df, ntt_result_df = load_results(output_dir.joinpath(f"{poly_style_name}"))

    # xlim_area = [(1.5e6, 12e6), (0e6, 8e6), (0, 2.5e6), (0, 2e6)]
    # ylim_area = [(0, 160), (0, 11.5), (0, 2200)]
    # plot_gate_acrx_bw(sc_df=sc_results_df, 
    #                   ntt_df=ntt_result_df,
    #                   filename=output_dir.joinpath(f"{poly_style_name}"),
    #                   xranges=xlim_area,
    #                   yranges=ylim_area)
    
    # ################################################
    print("End...")

