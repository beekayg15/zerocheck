from helper_funcs import sumcheck_only_sweep
from itertools import product
import params
from poly_list import *
import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt


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
    t_gate = [["f"]]
    sumcheck_polynomials = [
        t_gate,
    ]

    sweep_num_vars = [20]
    sweep_sumcheck_pes_range = range(16, 19)
    sweep_eval_engines_range = range(2, 3)
    sweep_product_lanes_range = range(3, 4)
    sweep_onchip_mle_sizes_range = [16384]  # in number of field elements
    sweep_available_bw_list = [256, 512, 1024]  # in GB/s

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
            sumcheck_polynomials,
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
                "sumcheck_gate": sumcheck_gate,
            }
        }

    return results_dict


def save_results(results, filename, save_excel=False, draw_plots=False):
    """
    Save the sweep results to an Excel file.
    Each row contains the sweep parameters (from 'params') and the stats_dict items as columns.
    Optionally, draw a dots plot: x="total_latency", y="area", color by "available_bw".
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
    if draw_plots:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="total_latency",
            y="area",
            hue="available_bw",
            palette="tab10",
            s=60,
            edgecolor="k"
        )
        plt.xlabel("Total Latency")
        plt.ylabel("Area")
        plt.title("Sumcheck Sweep: Area vs Total Latency")
        plt.legend(title="Available BW")
        plt.tight_layout()
        plt.savefig(filename + ".png")
        plt.close()


if __name__ == "__main__":
    results = sweep_sumcheck_configs()
    save_results(results, "sumcheck_sweep_results_mo", save_excel=True, draw_plots=True)
    
    print("End...")

