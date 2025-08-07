from params import BITS_PER_MB, MB_CONVERSION_FACTOR, sha_area
import os
from poly_list import *
import argparse
import pickle
import numpy as np
from util import is_pareto_efficient
from sumcheck_only_experiments import get_phy_cost, num_modmul_ops_in_polynomial, calc_utilization, find_robust_design_with_objective
import math
from sumcheck_models import * 
import matplotlib.pyplot as plt

def get_area_cost(hw_config, latencies, constants, available_bw, scale_factor_22_to_7nm=None):
    num_pes, num_eval_engines, num_product_lanes, onchip_mle_size = hw_config
    _, _, modmul_latency, modadd_latency = latencies
    bits_per_scalar, freq, modmul_area, modadd_area, reg_area, num_accumulate_regs, rr_ctrl_area, per_pe_delay_buffer_count, num_sumcheck_sram_buffers, multifunction_tree_sram_scale_factor = constants

    # mulitifunction tree area
    multifunction_tree_regs = 0
    multifunction_tree_modmuls = num_pes*(num_eval_engines - 1)*num_product_lanes # minimum number of modmuls for full pipelining
    multifunction_tree_modadds = math.ceil(1.5*multifunction_tree_modmuls) # estimate

    multifunction_tree_regs_mm2 = multifunction_tree_regs*reg_area
    multifunction_tree_modmuls_mm2 = multifunction_tree_modmuls*modmul_area
    multifunction_tree_modadds_mm2 = multifunction_tree_modadds*modadd_area

    multifunction_tree_compute_area_mm2 = multifunction_tree_regs_mm2 + multifunction_tree_modmuls_mm2 + multifunction_tree_modadds_mm2

    # eval engine area
    eval_engine_modmuls = 2*num_eval_engines # to process MLE Updates

    eval_engine_area_mm2 = (eval_engine_modmuls*modmul_area)*num_pes
    total_rr_ctrl_area_mm2 = (rr_ctrl_area)*num_pes
    delay_buffer_area_mm2 = (per_pe_delay_buffer_count*reg_area)*num_pes

    eval_engine_area_mm2 += total_rr_ctrl_area_mm2 + delay_buffer_area_mm2

    # accumulation registers area
    accumulation_reg_area_mm2 = num_accumulate_regs*reg_area

    # SRAM buffers area
    sumcheck_buffer_area_mb = num_sumcheck_sram_buffers*onchip_mle_size*bits_per_scalar/BITS_PER_MB
    sumcheck_buffer_area_mm2 = sumcheck_buffer_area_mb*MB_CONVERSION_FACTOR

    multifunction_tree_temp_buffer_area_mb = multifunction_tree_sram_scale_factor*onchip_mle_size*bits_per_scalar/BITS_PER_MB
    multifunction_tree_temp_buffer_area_mm2 = multifunction_tree_temp_buffer_area_mb*MB_CONVERSION_FACTOR

    hbm_phy_area_mm2 = get_phy_cost(available_bw)

    total_area_mm2 = multifunction_tree_compute_area_mm2 + eval_engine_area_mm2 + accumulation_reg_area_mm2 + sumcheck_buffer_area_mm2 + multifunction_tree_temp_buffer_area_mm2 + sha_area
    if scale_factor_22_to_7nm is not None:
        total_area_mm2 /= scale_factor_22_to_7nm # convert to 7nm area

    total_modmuls = multifunction_tree_modmuls + eval_engine_modmuls * num_pes
    design_modmul_area = total_modmuls * modmul_area

    # Calculate area breakdown
    nonmodmul_logic_area = \
        multifunction_tree_regs_mm2 + \
        multifunction_tree_modadds_mm2 + \
        total_rr_ctrl_area_mm2 + \
        delay_buffer_area_mm2 + \
        accumulation_reg_area_mm2 + \
        sha_area


    memory_area = sumcheck_buffer_area_mm2 + multifunction_tree_temp_buffer_area_mm2
    
    area_breakdown = {
        'design_modmul_area': design_modmul_area / (scale_factor_22_to_7nm if scale_factor_22_to_7nm else 1),
        'nonmodmul_logic_area': nonmodmul_logic_area / (scale_factor_22_to_7nm if scale_factor_22_to_7nm else 1),
        'memory_area': memory_area / (scale_factor_22_to_7nm if scale_factor_22_to_7nm else 1),
    }

    return round(total_area_mm2, 3), hbm_phy_area_mm2, total_modmuls, design_modmul_area, area_breakdown

def sumcheck_only_sweep(sweep_params, sumcheck_polynomials, latencies, constants, available_bw, scale_factor_22_to_7nm=None):

    num_vars, sumcheck_pes_range, eval_engines_range, product_lanes_range, onchip_mle_sizes_range = sweep_params
    mle_update_latency, extensions_latency, modmul_latency, modadd_latency = latencies

    bits_per_scalar, freq, *_ = constants

    sumcheck_core_stats = dict()
    for idx, sumcheck_polynomial in enumerate(sumcheck_polynomials):
        
        modmul_ops, per_round_ops = num_modmul_ops_in_polynomial(num_vars, sumcheck_polynomial, debug=False)
        
        sumcheck_core_stats[idx] = dict()
        for num_pes in sumcheck_pes_range:
            for num_eval_engines in eval_engines_range:

                assert num_eval_engines > 1

                for num_product_lanes in product_lanes_range:
                    assert num_product_lanes > 2

                    for onchip_mle_size in onchip_mle_sizes_range:
                        sumcheck_hardware_params = num_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, onchip_mle_size

                        sumcheck_hardware_config = num_pes, num_eval_engines, num_product_lanes, onchip_mle_size
                        sumcheck_core_stats[idx][sumcheck_hardware_config] = dict()
                        s_dict = sumcheck_core_stats[idx][sumcheck_hardware_config]

                        total_area_mm2, hbm_phy_area_mm2, total_modmuls, design_modmul_area, area_breakdown = get_area_cost(sumcheck_hardware_config, latencies, constants, available_bw, scale_factor_22_to_7nm)

                        supplemental_data = bits_per_scalar, available_bw, freq
                        num_build_mle = len({elem for sublist in sumcheck_polynomial for elem in sublist if isinstance(elem, str) and elem.startswith("fz")})
                        round_latencies, *_ = create_sumcheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, num_build_mle, supplemental_data, debug=False, debug_just_start=False, use_max_extensions=True)
                        # print(round_latencies)

                        # Calculate per-round utilization
                        per_round_utilization = []
                        for ops, lat in zip(per_round_ops, round_latencies):
                            util = calc_utilization(ops, total_modmuls, lat)
                            per_round_utilization.append(util)


                        total_latency = sum(round_latencies)

                        utilization = calc_utilization(modmul_ops, total_modmuls, total_latency)

                        s_dict['total_latency'] = total_latency
                        s_dict['area'] = total_area_mm2
                        s_dict['area_with_hbm'] = total_area_mm2 + hbm_phy_area_mm2
                        s_dict['modmul_count'] = total_modmuls
                        s_dict['design_modmul_area'] = design_modmul_area
                        s_dict['round_latencies'] = round_latencies
                        s_dict['utilization'] = utilization
                        s_dict['per_round_utilization'] = per_round_utilization
                        s_dict['area_breakdown'] = area_breakdown

    return sumcheck_core_stats

def get_pareto_curve(sumcheck_core_stats, print_only_best=True, area_max=None):
    if area_max is None:
        exit("Error: area_max must be specified for comparisons")

    pareto_curves = dict()
    # sumcheck_core_stats: {poly_idx: {hw_config: stats}}
    for poly_idx, hw_configs in sumcheck_core_stats.items():
        points = []
        configs = []
        utils = []
        per_round_utils = []
        for hw_config, stats in hw_configs.items():
            latency = stats['total_latency']
            area = stats['area']
            util = stats['utilizations']
            per_round_util = stats['per_round_utilizations']
            if area > area_max:
                continue  # Skip points exceeding area_max
            points.append((latency, area))
            configs.append(hw_config)
            utils.append(util)
            per_round_utils.append(per_round_util)
        points = np.array(points)
        if len(points) == 0:
            exit("No design points found for polynomial index {}".format(poly_idx))
        mask = is_pareto_efficient(points)
        pareto_points = points[mask]
        pareto_configs = [configs[i] for i, m in enumerate(mask) if m]
        pareto_utils = [utils[i] for i, m in enumerate(mask) if m]
        pareto_per_round_utils = [per_round_utils[i] for i, m in enumerate(mask) if m]
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_curves[poly_idx] = [
            {
                "hw_config": pareto_configs[i],
                "latency": pareto_points[i, 0],
                "area": pareto_points[i, 1],
                "utilization": pareto_utils[i],
                "per_round_utilization": pareto_per_round_utils[i]
            }
            for i in sorted_indices
        ]
    
    if print_only_best:
        print("Fastest Pareto Optimal Design:")
    else:
        print("Pareto Design Points:")
    for poly_idx, points in pareto_curves.items():
        print(f"  Polynomial Index: {poly_idx}")
        for point in points:
            hw_config = point["hw_config"]
            latency = point["latency"]
            area = point["area"]
            util = point["utilization"]
            per_round_util = point["per_round_utilization"]
            output = f"    Config: {hw_config}, Latency: {int(latency)}, Area: {area:.6f}"
            if util is not None:
                output += f", Utilization: {util}"
            print(output)
            # if per_round_util is not None:
            #     # per_round_util_str = [f'{u:.3f}' for u in per_round_util]
            #     print(f"    Per-round utilization: {per_round_util}")
            #     print()
            if print_only_best:
                break  # Only print the first point for each polynomial
    fastest_point = pareto_curves[0][0]
    fastest_latency = fastest_point['latency']
    fastest_config = fastest_point['hw_config']
    fastest_area = fastest_point['area']
    return pareto_curves, fastest_latency, fastest_config, fastest_area

def plot_runtime_bars(fastest_latencies, labels, freq, target_runtimes_ms, design="zkspeed", folder="."):
    runtimes_ms = []   
    for latency in fastest_latencies:
        runtime_ms = (latency / freq) * 1000
        runtimes_ms.append(runtime_ms)

    # print(fastest_latencies)
    # print(target_runtimes_ms)
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(5, 3))
    bars_target = plt.bar(x, target_runtimes_ms, width, color='salmon', label=design)
    bars_fastest = plt.bar([i + width for i in x], runtimes_ms, width, color='skyblue', label="Ours")

    plt.xticks([i + width / 2 for i in x], labels)
    plt.ylabel('Runtime (ms)')
    plt.title('First Pareto Point Runtime per Polynomial')
    plt.legend()
    plt.tight_layout()

    # Annotate speedup on fastest latency bars
    for i, (target, fastest) in enumerate(zip(target_runtimes_ms, runtimes_ms)):
        if fastest > 0:
            speedup = target / fastest
            plt.text(i + width, fastest + 0.01 * max(runtimes_ms + target_runtimes_ms), f"{speedup:.2f}$\\times$", ha='center', va='bottom', fontsize=8, color='black')

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{design}_runtime_comp.png"))
    plt.close()

def plot_zkspeed_only_runtime_comparison(all_results, freq, jellyfish_scaledowns, folder="."):
    """
    Plot runtime comparison focusing only on zkSpeed designs (no NoCap section):
    - zkSpeed vs zkSpeed+ vs zkPHIRE Vanilla vs zkPHIRE Jellyfish (scaledown 2, 4, 8)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    
    # Extract zkSpeed data only
    zkspeed_data = all_results['zkspeed_arb_prime']
    jellyfish_data = all_results['jellyfish']
    jellyfish_all_data = all_results.get('jellyfish_all', {4: jellyfish_data})
    
    # Convert all to ms
    zkspeed_target_ms = zkspeed_data['target_runtimes_ms']
    zkspeed_pp_target_ms = jellyfish_data['zkspeed_pp_target_runtimes_ms']
    vanilla_our_ms = [(lat / freq) * 1000 for lat in zkspeed_data['fastest_latencies']]
    
    # Get jellyfish data for all scaledown factors
    jellyfish_all_ms = {}
    for scaledown in jellyfish_scaledowns:
        if scaledown in jellyfish_all_data:
            jellyfish_all_ms[scaledown] = [(lat / freq) * 1000 for lat in jellyfish_all_data[scaledown]['fastest_latencies']]
        else:
            jellyfish_all_ms[scaledown] = [0] * len(zkspeed_data['labels'])
    
    # Use zkSpeed labels plus total
    all_labels = zkspeed_data['labels'] + ['Total']
    num_zkspeed = len(zkspeed_data['labels'])
    
    # Set up positions
    x = np.arange(len(all_labels))
    
    # Calculate bar positioning
    num_jellyfish = len(jellyfish_scaledowns)
    total_bars = 3 + num_jellyfish  # zkSpeed, zkSpeed+, zkPHIRE, + jellyfish variants
    width = 0.8 / total_bars
    
    # Center positions around x-tick
    positions = []
    for bar_idx in range(total_bars):
        pos_offset = (bar_idx - (total_bars - 1) / 2) * width
        positions.append(pos_offset)
    
    # Colors
    zkspeed_colors = ['#6399E8', '#EE8183', '#6CCBAB']  # zkSpeed, zkSpeed+, zkPHIRE
    jellyfish_colors = ['#B19EE1', '#F4B07F', '#C9A96E', '#FF8C69', '#FF6347', '#FF4500', '#DC143C', '#B22222']
    
    # 1. zkSpeed bars (exclude Total Time)
    bars_zkspeed = []
    for i in range(len(all_labels) - 1):
        pos = x[i] + positions[0]
        val = zkspeed_target_ms[i]
        label = 'zkSpeed' if i == 0 else None
        bar = ax.bar(pos, val, width, color=zkspeed_colors[0], alpha=0.8, label=label)
        bars_zkspeed.append(bar)
    
    # 2. zkSpeed+ bars (exclude Total Time)
    bars_zkspeed_pp = []
    for i in range(len(all_labels) - 1):
        pos = x[i] + positions[1]
        val = zkspeed_pp_target_ms[i]
        label = 'zkSpeed+' if i == 0 else None
        bar = ax.bar(pos, val, width, color=zkspeed_colors[1], alpha=0.8, label=label)
        bars_zkspeed_pp.append(bar)
    
    # 3. zkPHIRE bars (exclude Total Time)
    bars_zkphire = []
    for i in range(len(all_labels) - 1):
        pos = x[i] + positions[2]
        val = vanilla_our_ms[i]
        label = 'zkPHIRE' if i == 0 else None
        bar = ax.bar(pos, val, width, color=zkspeed_colors[2], alpha=0.8, label=label)
        bars_zkphire.append(bar)
    
    # 4. zkPHIRE Jellyfish bars (exclude Total Time)
    bars_jellyfish_all = {}
    for idx, scaledown in enumerate(jellyfish_scaledowns):
        bars_jellyfish_all[scaledown] = []
        for i in range(len(all_labels) - 1):
            pos = x[i] + positions[3 + idx]
            val = jellyfish_all_ms[scaledown][i]
            label = f'zkPHIRE (Jellyfish {scaledown}×)' if i == 0 else None
            bar = ax.bar(pos, val, width, color=jellyfish_colors[idx], alpha=0.8, label=label)
            bars_jellyfish_all[scaledown].append(bar)
    
    # 5. Total Time bars (new cluster)
    total_idx = len(all_labels) - 1
    
    # Calculate total times
    zkspeed_total = sum(zkspeed_target_ms)
    zkspeed_pp_total = sum(zkspeed_pp_target_ms)  
    vanilla_total = sum(vanilla_our_ms)
    jellyfish_totals = {}
    for scaledown in jellyfish_scaledowns:
        jellyfish_totals[scaledown] = sum(jellyfish_all_ms[scaledown])
    
    # Total bars with same positioning
    ax.bar(x[total_idx] + positions[0], zkspeed_total, width, color=zkspeed_colors[0], alpha=0.8)
    ax.bar(x[total_idx] + positions[1], zkspeed_pp_total, width, color=zkspeed_colors[1], alpha=0.8)
    ax.bar(x[total_idx] + positions[2], vanilla_total, width, color=zkspeed_colors[2], alpha=0.8)
    
    for idx, scaledown in enumerate(jellyfish_scaledowns):
        pos = x[total_idx] + positions[3 + idx]
        ax.bar(pos, jellyfish_totals[scaledown], width, color=jellyfish_colors[idx], alpha=0.8)
    
    # Calculate max height for annotations
    all_jellyfish_values = []
    for scaledown in jellyfish_scaledowns:
        all_jellyfish_values.extend(jellyfish_all_ms[scaledown])
    
    all_values = (zkspeed_target_ms + zkspeed_pp_target_ms + vanilla_our_ms + 
                 all_jellyfish_values + [zkspeed_total, zkspeed_pp_total, vanilla_total] + 
                 list(jellyfish_totals.values()))
    max_val = max(all_values)
    
    # Annotate speedups for individual polynomials
    # zkPHIRE speedups vs zkSpeed and zkSpeed+
    for i in range(num_zkspeed):
        zkspeed_target = zkspeed_target_ms[i]
        zkspeed_pp_target = zkspeed_pp_target_ms[i]
        ours = vanilla_our_ms[i]
        speedup_vs_zkspeed = zkspeed_target / ours
        speedup_vs_zkspeed_pp = zkspeed_pp_target / ours
        text_str = f"{speedup_vs_zkspeed:.2f}×/{speedup_vs_zkspeed_pp:.2f}×"
        
        ax.text(x[i] + positions[2], ours + 0.01 * max_val, 
                text_str, ha='center', va='bottom', 
                fontsize=12, rotation=90, fontweight='bold', color='black')
    
    # zkPHIRE Jellyfish speedups vs zkSpeed and zkSpeed+
    for idx_scale, scaledown in enumerate(jellyfish_scaledowns):
        for i in range(num_zkspeed):
            zkspeed_target = zkspeed_target_ms[i]
            zkspeed_pp_target = zkspeed_pp_target_ms[i]
            ours = jellyfish_all_ms[scaledown][i]
            speedup_vs_zkspeed = zkspeed_target / ours
            speedup_vs_zkspeed_pp = zkspeed_pp_target / ours
            text_str = f"{speedup_vs_zkspeed:.2f}×/{speedup_vs_zkspeed_pp:.2f}×"
            
            ax.text(x[i] + positions[3 + idx_scale], ours + 0.01 * max_val,
                    text_str, ha='center', va='bottom', rotation=90,
                    fontsize=12, fontweight='bold', color='black')
    
    # Total time speedup annotations
    total_idx = len(all_labels) - 1
    
    # zkPHIRE total speedups
    speedup_vs_zkspeed_total = zkspeed_total / vanilla_total
    speedup_vs_zkspeed_pp_total = zkspeed_pp_total / vanilla_total
    text_str = f"{speedup_vs_zkspeed_total:.2f}×/{speedup_vs_zkspeed_pp_total:.2f}×"
    ax.text(x[total_idx] + positions[2], vanilla_total + 0.01 * max_val,
            text_str, ha='center', va='bottom', rotation=90,
            fontsize=12, fontweight='bold', color='black')
    
    # zkPHIRE Jellyfish total speedups
    for idx_scale, scaledown in enumerate(jellyfish_scaledowns):
        jellyfish_total = jellyfish_totals[scaledown]
        speedup_vs_zkspeed_jelly_total = zkspeed_total / jellyfish_total
        speedup_vs_zkspeed_pp_jelly_total = zkspeed_pp_total / jellyfish_total
        text_str = f"{speedup_vs_zkspeed_jelly_total:.2f}×/{speedup_vs_zkspeed_pp_jelly_total:.2f}×"
        
        ax.text(x[total_idx] + positions[3 + idx_scale], jellyfish_total + 0.01 * max_val,
                text_str, ha='center', va='bottom', rotation=90,
                fontsize=12, fontweight='bold', color='black')
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=16)
    ax.set_ylabel('Runtime (ms)', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)


    # ax.set_title('zkSpeed Hardware Comparison')
    
    # Create legend
    handles = [bars_zkspeed[0], bars_zkspeed_pp[0], bars_zkphire[0]]
    labels = ['zkSpeed    (Vanilla)', 'zkSpeed+ (Vanilla)', 'zkPHIRE    (Vanilla)']
    
    # Add jellyfish handles and labels
    for idx, scaledown in enumerate(jellyfish_scaledowns):
        handles.append(bars_jellyfish_all[scaledown][0])
        labels.append(f'zkPHIRE (Jellyfish {scaledown}×)')
    
    ax.legend(handles, labels, loc='upper left', ncol=2,
              frameon=True, fancybox=True, shadow=False, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max_val + 25)
    
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, "zkspeed_only_runtime_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(folder, "zkspeed_only_runtime_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("ZKSPEED-ONLY RUNTIME COMPARISON SUMMARY")
    print("="*60)
    for i, label in enumerate(zkspeed_data['labels']):
        zkspeed_target = zkspeed_target_ms[i]
        zkspeed_pp_target = zkspeed_pp_target_ms[i]
        vanilla_ours = vanilla_our_ms[i]
        
        vanilla_speedup = zkspeed_target / vanilla_ours if vanilla_ours > 0 else 0
        vanilla_speedup_pp = zkspeed_pp_target / vanilla_ours if vanilla_ours > 0 else 0
        
        print(f"  {label}:")
        print(f"    zkSpeed: {zkspeed_target:.3f}ms")
        print(f"    zkSpeed+: {zkspeed_pp_target:.3f}ms") 
        print(f"    zkPHIRE (Vanilla): {vanilla_ours:.3f}ms = {vanilla_speedup:.2f}× vs zkSpeed, {vanilla_speedup_pp:.2f}× vs zkSpeed+")
        
        # Print results for all jellyfish scaledown factors
        for scaledown in jellyfish_scaledowns:
            if scaledown in jellyfish_all_ms and i < len(jellyfish_all_ms[scaledown]):
                jellyfish_ours = jellyfish_all_ms[scaledown][i]
                jellyfish_vs_zkspeed = zkspeed_target / jellyfish_ours if jellyfish_ours > 0 else 0
                jellyfish_vs_zkspeed_pp = zkspeed_pp_target / jellyfish_ours if jellyfish_ours > 0 else 0
                print(f"    zkPHIRE (Jellyfish {scaledown}×): {jellyfish_ours:.3f}ms = {jellyfish_vs_zkspeed:.2f}× vs zkSpeed, {jellyfish_vs_zkspeed_pp:.2f}× vs zkSpeed+")
    
    # Print totals
    print(f"\n  Combined HyperPlonk:")
    print(f"    zkSpeed Total: {zkspeed_total:.3f}ms")
    print(f"    zkSpeed+ Total: {zkspeed_pp_total:.3f}ms")
    print(f"    zkPHIRE (Vanilla) Total: {vanilla_total:.3f}ms = {zkspeed_total/vanilla_total:.2f}× vs zkSpeed, {zkspeed_pp_total/vanilla_total:.2f}× vs zkSpeed+")
    
    for scaledown in jellyfish_scaledowns:
        jelly_total = jellyfish_totals[scaledown]
        print(f"    zkPHIRE (Jellyfish {scaledown}×) Total: {jelly_total:.3f}ms = {zkspeed_total/jelly_total:.2f}× vs zkSpeed, {zkspeed_pp_total/jelly_total:.2f}× vs zkSpeed+")
    
    print("="*60)

def plot_comprehensive_runtime_comparison(all_results, freq, jellyfish_scaledowns, folder="."):
    """
    Plot comprehensive runtime comparison in a single plot:
    - NoCap vs zkPHIRE Goldilocks
    - zkSpeed vs zkSpeed+ vs zkPHIRE Vanilla vs zkPHIRE Jellyfish (scaledown 2, 4, 8)
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))  # Made wider to accommodate more bars
    
    # Extract all data
    nocap_data = all_results['nocap']
    zkspeed_data = all_results['zkspeed_arb_prime']
    jellyfish_data = all_results['jellyfish']
    jellyfish_all_data = all_results.get('jellyfish_all', {4: jellyfish_data})  # Fallback for compatibility
    
    # Convert all to ms
    nocap_target_ms = nocap_data['target_runtimes_ms']
    nocap_our_ms = [(lat / freq) * 1000 for lat in nocap_data['fastest_latencies']]
    
    zkspeed_target_ms = zkspeed_data['target_runtimes_ms']
    zkspeed_pp_target_ms = jellyfish_data['zkspeed_pp_target_runtimes_ms']
    vanilla_our_ms = [(lat / freq) * 1000 for lat in zkspeed_data['fastest_latencies']]
    
    # Get jellyfish data for all scaledown factors
    jellyfish_all_ms = {}
    for scaledown in jellyfish_scaledowns:
        if scaledown in jellyfish_all_data:
            jellyfish_all_ms[scaledown] = [(lat / freq) * 1000 for lat in jellyfish_all_data[scaledown]['fastest_latencies']]
        else:
            jellyfish_all_ms[scaledown] = [0] * len(zkspeed_data['labels'])  # Fallback
    
    # Combine all labels and data
    all_labels = nocap_data['labels'] + zkspeed_data['labels'] + ['Combined HyperPlonk']
    num_nocap = len(nocap_data['labels'])
    num_zkspeed = len(zkspeed_data['labels'])
    
    # Set up positions
    x = np.arange(len(all_labels))
    
    # Calculate dynamic positioning based on number of jellyfish scaledowns
    num_jellyfish = len(jellyfish_scaledowns)
    # For zkSpeed section: 3 + num_jellyfish bars per group (zkSpeed, zkSpeed+, zkPHIRE, + jellyfish variants)
    # For NoCap section: 2 bars per group
    
    # Calculate bar width dynamically to fit all bars
    max_bars_per_group = max(2, 3 + num_jellyfish)  # NoCap has 2, zkSpeed section has 3 + num_jellyfish
    width = 0.8 / max_bars_per_group  # Use 80% of space, leave 20% for gaps between groups
    
    # Generate jellyfish colors dynamically
    jellyfish_colors = []
    base_colors = [ '#B19EE1', '#F4B07F', '#C9A96E', '#FF8C69', '#FF6347', '#FF4500', '#DC143C', '#B22222']  # Orange to red gradient
    # base_colors = ["#6CCBAB",
    #                "#B19EE1",
    #                "#F4B07F",
    #                "#6399E8",
    #                "#778898",
    #                "#EE8183",]
    for i in range(num_jellyfish):
        jellyfish_colors.append(base_colors[i % len(base_colors)])
    
    # For zkSpeed section: calculate positions to center all bars around x-tick
    # Total bars: zkSpeed, zkSpeed+, zkPHIRE, + jellyfish variants
    zkspeed_section_bars = 3 + num_jellyfish
    # Center positions around x[i]: from -(zkspeed_section_bars-1)/2 * width to +(zkspeed_section_bars-1)/2 * width
    zkspeed_positions = []
    for bar_idx in range(zkspeed_section_bars):
        pos_offset = (bar_idx - (zkspeed_section_bars - 1) / 2) * width
        zkspeed_positions.append(pos_offset)
    
    # Create all bars in legend order: NoCap, zkPHIRE (Goldilocks), zkSpeed, zkSpeed+, zkPHIRE, zkPHIRE Jellyfish (dynamic)
    sumflex_colors = ['lightgreen', '#6CCBAB', '#F4B07F']  # Same as combined plot
    zkspeed_colors = ['#6399E8', '#EE8183', '#6CCBAB', '#F4B07F']  # zkSpeed, zkSpeed+, zkPHIRE

    # 1. NoCap bars (first in legend) - 2 bars centered around each x-tick, touching each other
    bars_nocap = []
    for i in range(num_nocap):
        pos = x[i] - width/2  # Left bar positioned at left edge
        val = nocap_target_ms[i]
        label = 'NoCap' if i == 0 else None
        bar = ax.bar(pos, val, width, color='#6399E8', alpha=0.8, label=label) # gray
        bars_nocap.append(bar)
    
    # 2. zkPHIRE Goldilocks (second in legend) - only for NoCap section
    bars_goldilocks = []
    for i in range(num_nocap):
        pos = x[i] + width/2  # Right bar positioned immediately adjacent (no gap)
        val = nocap_our_ms[i]
        label = 'zkPHIRE (Goldilocks)' if i == 0 else None
        bar = ax.bar(pos, val, width, color='#EE8183', alpha=0.8, label=label) # blue
        bars_goldilocks.append(bar)
    
    # 3. zkSpeed bars (third in legend) - only for zkSpeed section (exclude Total Time)
    bars_zkspeed = []
    for i in range(num_nocap, len(all_labels) - 1):  # Exclude 'Total Time' label
        pos = x[i] + zkspeed_positions[0]  # First position
        val = zkspeed_target_ms[i - num_nocap]
        label = 'zkSpeed' if i == num_nocap else None
        bar = ax.bar(pos, val, width, color=zkspeed_colors[0], alpha=0.8, label=label)
        bars_zkspeed.append(bar)
    
    # 4. zkSpeed+ bars (fourth in legend) - only for zkSpeed section (exclude Total Time)
    bars_zkspeed_pp = []
    for i in range(num_nocap, len(all_labels) - 1):  # Exclude 'Total Time' label
        pos = x[i] + zkspeed_positions[1]  # Second position
        val = zkspeed_pp_target_ms[i - num_nocap]
        label = 'zkSpeed+' if i == num_nocap else None
        bar = ax.bar(pos, val, width, color=zkspeed_colors[1], alpha=0.8, label=label)
        bars_zkspeed_pp.append(bar)
    
    # 5. zkPHIRE (fifth in legend) - only for zkSpeed section (exclude Total Time)
    bars_zkphire = []
    for i in range(num_nocap, len(all_labels) - 1):  # Exclude 'Total Time' label
        pos = x[i] + zkspeed_positions[2]  # Third position
        val = vanilla_our_ms[i - num_nocap]
        label = 'zkPHIRE' if i == num_nocap else None
        bar = ax.bar(pos, val, width, color=sumflex_colors[1], alpha=0.8, label=label)
        bars_zkphire.append(bar)
    
    # 6. zkPHIRE Jellyfish bars (dynamic number of scaledowns) - only for zkSpeed section
    bars_jellyfish_all = {}
    for idx, scaledown in enumerate(jellyfish_scaledowns):
        bars_jellyfish_all[scaledown] = []
        for i in range(num_nocap, len(all_labels) - 1):  # Exclude 'Total Time' label
            pos = x[i] + zkspeed_positions[3 + idx]  # Positions starting from 4th slot
            val = jellyfish_all_ms[scaledown][i - num_nocap]
            label = f'zkPHIRE (Jelly {scaledown}×)' if i == num_nocap else None
            bar = ax.bar(pos, val, width, color=jellyfish_colors[idx], alpha=0.8, label=label)
            bars_jellyfish_all[scaledown].append(bar)
    
    # 7. Total Time bars (new cluster) - show sum of all zkSpeed polynomials
    total_idx = len(all_labels) - 1  # Last position is 'Total Time'
    
    # Calculate total times for each design
    zkspeed_total = sum(zkspeed_target_ms)
    zkspeed_pp_total = sum(zkspeed_pp_target_ms)  
    vanilla_total = sum(vanilla_our_ms)
    jellyfish_totals = {}
    for scaledown in jellyfish_scaledowns:
        jellyfish_totals[scaledown] = sum(jellyfish_all_ms[scaledown])
    
    # Use same dynamic positioning for total time bars
    # zkSpeed total
    bar = ax.bar(x[total_idx] + zkspeed_positions[0], zkspeed_total, width, color='#6399E8', alpha=0.8)
    
    # zkSpeed+ total  
    bar = ax.bar(x[total_idx] + zkspeed_positions[1], zkspeed_pp_total, width, color='#FF9999', alpha=0.8)
    
    # zkPHIRE total
    bar = ax.bar(x[total_idx] + zkspeed_positions[2], vanilla_total, width, color=sumflex_colors[1], alpha=0.8)
    
    # zkPHIRE Jellyfish totals for all scaledowns
    for idx, scaledown in enumerate(jellyfish_scaledowns):
        pos = x[total_idx] + zkspeed_positions[3 + idx]
        bar = ax.bar(pos, jellyfish_totals[scaledown], width, color=jellyfish_colors[idx], alpha=0.8)
    
    # Calculate max height for annotations (including total times)
    all_jellyfish_values = []
    for scaledown in jellyfish_scaledowns:
        all_jellyfish_values.extend(jellyfish_all_ms[scaledown])
        
    all_values = (nocap_target_ms + nocap_our_ms + zkspeed_target_ms + 
                 zkspeed_pp_target_ms + vanilla_our_ms + all_jellyfish_values +
                 [zkspeed_total, zkspeed_pp_total, vanilla_total] + list(jellyfish_totals.values()))
    max_val = max(all_values)
    
    # Annotate speedups
    # NoCap speedups (zkPHIRE Goldilocks vs NoCap)
    for i in range(num_nocap):
        target = nocap_target_ms[i]
        ours = nocap_our_ms[i]
        speedup = target / ours
        ax.text(x[i] + width/2, ours + 0.01 * max_val, 
                f"{speedup:.1f}×", ha='center', va='bottom', 
                fontsize=8, rotation=90, fontweight='bold', color='black')
    
    # zkPHIRE speedups vs zkSpeed and zkSpeed+
    for i in range(num_zkspeed):
        idx = i + num_nocap
        zkspeed_target = zkspeed_target_ms[i]
        zkspeed_pp_target = zkspeed_pp_target_ms[i]
        ours = vanilla_our_ms[i]
        # Speedup vs zkSpeed
        speedup_vs_zkspeed = zkspeed_target / ours
        speedup_vs_zkspeed_pp = zkspeed_pp_target / ours

        text_str = f"{speedup_vs_zkspeed:.2f}×/{speedup_vs_zkspeed_pp:.2f}×"

        ax.text(x[idx] + zkspeed_positions[2], ours + 0.01 * max_val, 
                text_str, ha='center', va='bottom', 
                fontsize=8, rotation=90, fontweight='bold', color='black')
        

    # zkPHIRE Jellyfish speedups vs zkSpeed and zkSpeed+ (for all scaledowns)
    for idx_scale, scaledown in enumerate(jellyfish_scaledowns):
        for i in range(num_zkspeed):
            idx = i + num_nocap
            zkspeed_target = zkspeed_target_ms[i]
            zkspeed_pp_target = zkspeed_pp_target_ms[i]
            ours = jellyfish_all_ms[scaledown][i]
            # Speedup vs zkSpeed
            speedup_vs_zkspeed = zkspeed_target / ours
            speedup_vs_zkspeed_pp = zkspeed_pp_target / ours
            text_str = f"{speedup_vs_zkspeed:.2f}×/{speedup_vs_zkspeed_pp:.2f}×"

            ax.text(x[idx] + zkspeed_positions[3 + idx_scale], ours + 0.01 * max_val,
                    text_str, ha='center', va='bottom', rotation=90,
                    fontsize=8, fontweight='bold', color='black')
    
    # Total time speedup annotations
    total_idx = len(all_labels) - 1
    
    # zkPHIRE total speedups vs zkSpeed and zkSpeed+
    # vs zkSpeed
    speedup_vs_zkspeed_total = zkspeed_total / vanilla_total
    speedup_vs_zkspeed_pp_total = zkspeed_pp_total / vanilla_total
    text_str = f"{speedup_vs_zkspeed_total:.2f}×/{speedup_vs_zkspeed_pp_total:.2f}×"
    ax.text(x[total_idx] + zkspeed_positions[2], vanilla_total + 0.01 * max_val,
            text_str, ha='center', va='bottom', rotation=90,
            fontsize=8, fontweight='bold', color='black')
    
    # zkPHIRE Jellyfish total speedups vs zkSpeed and zkSpeed+ (for all scaledowns)
    for idx_scale, scaledown in enumerate(jellyfish_scaledowns):
        jellyfish_total = jellyfish_totals[scaledown]
        # vs zkSpeed
        speedup_vs_zkspeed_jelly_total = zkspeed_total / jellyfish_total
        speedup_vs_zkspeed_pp_jelly_total = zkspeed_pp_total / jellyfish_total
        text_str = f"{speedup_vs_zkspeed_jelly_total:.2f}×/{speedup_vs_zkspeed_pp_jelly_total:.2f}×"
        if idx_scale == 0:
            text_str = f"{speedup_vs_zkspeed_jelly_total:.2f}×\n/{speedup_vs_zkspeed_pp_jelly_total:.2f}×"
            
        # if idx_scale == 0:
        #     ax.text(x[total_idx] + zkspeed_positions[3 + idx_scale] + 0.14, jellyfish_total - 0.2 * max_val,
        #             text_str, ha='center', va='bottom', rotation=90,
        #             fontsize=8, fontweight='bold', color='black')
        # else:
        ax.text(x[total_idx] + zkspeed_positions[3 + idx_scale], jellyfish_total + 0.01 * max_val,
                text_str, ha='center', va='bottom', rotation=90,
                fontsize=8, fontweight='bold', color='black')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel('Runtime (ms)')
    # ax.set_title('Comprehensive Speedup Comparison with Prior ASICs')
    
    # Create two separate legends
    # Legend 1: NoCap section (upper left)
    nocap_handles = [bars_nocap[0], bars_goldilocks[0]]
    nocap_labels = ['NoCap', 'zkPHIRE']
    legend1 = ax.legend(nocap_handles, nocap_labels, loc='upper left', 
                       frameon=True, fancybox=True, shadow=False)
    
    # Legend 2: zkSpeed section (upper right)
    zkspeed_handles = [bars_zkspeed[0], bars_zkspeed_pp[0], bars_zkphire[0]]
    zkspeed_labels = ['zkSpeed', 'zkSpeed+', 'zkPHIRE']
    
    # Add jellyfish handles and labels dynamically
    for idx, scaledown in enumerate(jellyfish_scaledowns):
        zkspeed_handles.append(bars_jellyfish_all[scaledown][0])
        zkspeed_labels.append(f'zkPHIRE (Jelly {scaledown}×)')
    
    legend2 = ax.legend(zkspeed_handles, zkspeed_labels, loc='upper center', ncol = 3,
                       frameon=True, fancybox=True, shadow=False)
    
    # Add the first legend back to the plot (matplotlib removes it when creating the second)
    ax.add_artist(legend1)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max_val + 9)
    vertical_frac = 0.9
    # Add vertical lines to separate NOCAP, ZKSPEED, and Total
    ax.axvline(x=num_nocap - 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(num_nocap/2 - 0.5, max_val * vertical_frac, 'NoCap', 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#6399E8", alpha=0.7))
    ax.text(num_nocap + num_zkspeed/2 - 0.2, max_val * vertical_frac, 'zkSpeed', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, "comprehensive_runtime_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(folder, "comprehensive_runtime_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE RUNTIME COMPARISON SUMMARY")
    print("="*80)
    print("NOCAP Design:")
    for label, target, ours in zip(nocap_data['labels'], nocap_target_ms, nocap_our_ms):
        speedup = target / ours if ours > 0 else 0
        print(f"  {label}: {target:.3f}ms (NoCap) vs {ours:.3f}ms (Ours Goldilocks) = {speedup:.2f}× speedup")
    
    print("\nZKSPEED Design:")
    for i, label in enumerate(zkspeed_data['labels']):
        zkspeed_target = zkspeed_target_ms[i]
        zkspeed_pp_target = zkspeed_pp_target_ms[i]
        vanilla_ours = vanilla_our_ms[i]
        
        vanilla_speedup = zkspeed_target / vanilla_ours if vanilla_ours > 0 else 0
        
        print(f"  {label}:")
        print(f"    zkSpeed: {zkspeed_target:.3f}ms")
        print(f"    zkSpeed+: {zkspeed_pp_target:.3f}ms") 
        print(f"    zkPHIRE (Vanilla): {vanilla_ours:.3f}ms = {vanilla_speedup:.2f}× vs zkSpeed")
        
        # Print results for all jellyfish scaledown factors
        for scaledown in jellyfish_scaledowns:
            if scaledown in jellyfish_all_ms and i < len(jellyfish_all_ms[scaledown]):
                jellyfish_ours = jellyfish_all_ms[scaledown][i]
                jellyfish_vs_zkspeed = zkspeed_target / jellyfish_ours if jellyfish_ours > 0 else 0
                jellyfish_vs_zkspeed_pp = zkspeed_pp_target / jellyfish_ours if jellyfish_ours > 0 else 0
                print(f"    zkPHIRE (Jelly {scaledown}×): {jellyfish_ours:.3f}ms = {jellyfish_vs_zkspeed:.2f}× vs zkSpeed, {jellyfish_vs_zkspeed_pp:.2f}× vs zkSpeed+")
    print("="*80)

def plot_combined_runtime_comparison_simple(all_results, freq, folder="."):
    """
    Plot combined runtime comparison across nocap and zkspeed designs only
    all_results: dict with design names as keys and results as values
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Extract data
    nocap_data = all_results['nocap']
    zkspeed_arb_data = all_results['zkspeed_arb_prime']
    
    # Convert to ms
    nocap_target_ms = nocap_data['target_runtimes_ms']
    nocap_our_ms = [(lat / freq) * 1000 for lat in nocap_data['fastest_latencies']]
    
    zkspeed_target_ms = zkspeed_arb_data['target_runtimes_ms']
    zkspeed_arb_our_ms = [(lat / freq) * 1000 for lat in zkspeed_arb_data['fastest_latencies']]
    
    # Combine all labels and data
    all_labels = nocap_data['labels'] + zkspeed_arb_data['labels']
    all_target_ms = nocap_target_ms + zkspeed_target_ms
    all_sumflex_ms = nocap_our_ms + zkspeed_arb_our_ms  # All zkPHIRE results (Goldilocks + Arb Prime)
    
    # Set up positions
    x = range(len(all_labels))
    width = 0.35
    
    # Create bars with custom colors for each target type
    target_colors = ['#6399E8'] + ['#EE8183'] * 3  # Blue for NOCAP, salmon for ZKSPEED
    bars_target = []
    for i, (pos, val, color) in enumerate(zip([i - width/2 for i in x], all_target_ms, target_colors)):
        if i == 0:
            bar = ax.bar(pos, val, width, color=color, alpha=0.8, label='NoCap')
        elif i == 1:
            bar = ax.bar(pos, val, width, color=color, alpha=0.8, label='zkSpeed')
        else:
            bar = ax.bar(pos, val, width, color=color, alpha=0.8)
        bars_target.append(bar)
    
    # zkPHIRE bars with different colors
    sumflex_colors = ['lightgreen', '#B19EE1']  # Two distinct colors for zkPHIRE variants
    bars_sumflex = []
    for i, (pos, val) in enumerate(zip([i + width/2 for i in x], all_sumflex_ms)):
        if i == 0:
            color = sumflex_colors[0]
            label = 'Us (64b Fixed Prime)'
        else:
            color = sumflex_colors[1]
            if i == 1:  # Only add label once for ZKSPEED arb prime
                label = 'Us (255b Arb Prime)'
            else:
                label = None
        bar = ax.bar(pos, val, width, color=color, alpha=0.8, label=label)
        bars_sumflex.append(bar)
    
    # Annotate speedups
    for i, (target, sumflex) in enumerate(zip(all_target_ms, all_sumflex_ms)):
        if sumflex > 0:
            speedup = target / sumflex
            ax.text(i + width/2, sumflex + 0.01 * max(all_target_ms + all_sumflex_ms), 
                   f"{speedup:.2f}$\\times$", ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel('Runtime (ms)')
    # ax.set_title('Speedup Comparison with Prior ASICs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add vertical line to separate NOCAP and ZKSPEED
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(0.2, max(all_target_ms + all_sumflex_ms) * 0.9, 'NoCap', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(1.9, max(all_target_ms + all_sumflex_ms) * 0.9, 'zkSpeed', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, "combined_runtime_comparison_simple.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("COMBINED RUNTIME COMPARISON SUMMARY (SIMPLE)")
    print("="*60)
    print("NOCAP Design:")
    for label, target, goldilocks in zip(nocap_data['labels'], nocap_target_ms, nocap_our_ms):
        speedup = target / goldilocks if goldilocks > 0 else 0
        print(f"  {label}: {target:.3f}ms (target) vs {goldilocks:.3f}ms (zkPHIRE Goldilocks) = {speedup:.2f}x speedup")
    
    print("\nZKSPEED Design:")
    for label, target, arb_prime in zip(zkspeed_arb_data['labels'], zkspeed_target_ms, zkspeed_arb_our_ms):
        speedup_arb = target / arb_prime if arb_prime > 0 else 0
        print(f"  {label}: {target:.3f}ms (target) vs {arb_prime:.3f}ms (zkPHIRE Vanilla) = {speedup_arb:.2f}x speedup")
    print("="*60)

def plot_combined_runtime_comparison(all_results, freq, folder="."):
    """
    Plot combined runtime comparison across all designs on a single plot
    all_results: dict with design names as keys and results as values
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Extract data
    nocap_data = all_results['nocap']
    zkspeed_arb_data = all_results['zkspeed_arb_prime']
    zkspeed_fixed_data = all_results['zkspeed_fixed_prime']
    
    # Convert to ms
    nocap_target_ms = nocap_data['target_runtimes_ms']
    nocap_our_ms = [(lat / freq) * 1000 for lat in nocap_data['fastest_latencies']]
    
    zkspeed_target_ms = zkspeed_arb_data['target_runtimes_ms']
    zkspeed_arb_our_ms = [(lat / freq) * 1000 for lat in zkspeed_arb_data['fastest_latencies']]
    zkspeed_fixed_our_ms = [(lat / freq) * 1000 for lat in zkspeed_fixed_data['fastest_latencies']]
    
    # Combine all labels and data
    all_labels = nocap_data['labels'] + zkspeed_arb_data['labels']
    all_target_ms = nocap_target_ms + zkspeed_target_ms
    all_sumflex_ms = nocap_our_ms + zkspeed_arb_our_ms  # All zkPHIRE results (Goldilocks + Arb Prime)
    all_fixed_prime_ms = [None] * len(nocap_our_ms) + zkspeed_fixed_our_ms  # Only zkSpeed has fixed prime variant
    
    # Set up positions
    x = range(len(all_labels))
    width = 0.25
    
    # Create bars with custom colors for each target type
    target_colors = ['#6399E8'] + ['#EE8183'] * 3  # Blue for NOCAP, salmon for ZKSPEED
    bars_target = []
    for i, (pos, val, color) in enumerate(zip([i - width for i in x], all_target_ms, target_colors)):
        if i == 0:
            bar = ax.bar(pos, val, width, color=color, alpha=0.8, label='NoCap')
        elif i == 1:
            bar = ax.bar(pos, val, width, color=color, alpha=0.8, label='zkSpeed')
        else:
            bar = ax.bar(pos, val, width, color=color, alpha=0.8)
        bars_target.append(bar)
    
    # zkPHIRE bars with different colors
    sumflex_colors = ['lightgreen', '#B19EE1', '#F4B07F']  # Three distinct colors for zkPHIRE variants
    bars_sumflex = []
    for i, (pos, val) in enumerate(zip(x, all_sumflex_ms)):
        if i == 0:
            color = sumflex_colors[0]
            label = 'zkPHIRE (Goldilocks)'
        else:
            color = sumflex_colors[1]
            if i == 1:  # Only add label once for ZKSPEED arb prime
                label = 'zkPHIRE (Arb Prime)'
            else:
                label = None
        bar = ax.bar(pos, val, width, color=color, alpha=0.8, label=label)
        bars_sumflex.append(bar)
    
    # Only add fixed prime bars for ZKSPEED (indices 1, 2, 3)
    fixed_x = [i + width for i in x[1:]]  # Skip NOCAP (index 0)
    fixed_vals = [val for val in all_fixed_prime_ms if val is not None]
    bars_fixed_prime = []
    for i, (pos, val) in enumerate(zip(fixed_x, fixed_vals)):
        if i == 0:  # Only add label once
            bar = ax.bar(pos, val, width, color=sumflex_colors[2], alpha=0.8, label='zkPHIRE (Fixed Prime)')
        else:
            bar = ax.bar(pos, val, width, color=sumflex_colors[2], alpha=0.8)
        bars_fixed_prime.append(bar)
    
    # Annotate speedups with colors matching the bars
    annotation_colors = ['black', 'black', 'black']  # Match zkPHIRE bar colors
    for i, (target, sumflex) in enumerate(zip(all_target_ms, all_sumflex_ms)):
        if sumflex > 0:
            speedup = target / sumflex
            color = annotation_colors[0] if i == 0 else annotation_colors[1]
            # Move Goldilocks speedup label slightly to the right
            x_offset = i + (0.02 if i == 0 else 0)
            ax.text(x_offset, sumflex + 0.01 * max(all_target_ms + all_sumflex_ms + fixed_vals), 
                   f"{speedup:.2f}$\\times$", ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
    
    # Annotate speedups for fixed prime (ZKSPEED only)
    for i, (target, fixed_prime) in enumerate(zip(zkspeed_target_ms, zkspeed_fixed_our_ms)):
        if fixed_prime > 0:
            speedup = target / fixed_prime
            ax.text(i + 1 + width, fixed_prime + 0.01 * max(all_target_ms + all_sumflex_ms + fixed_vals), 
                   f"{speedup:.2f}$\\times$", ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    
    # Formatting
    ax.set_xticks(x)
    # ax.set_xticklabels(all_labels, rotation=15, ha='right')
    ax.set_xticklabels(all_labels)
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Speedup Comparison with Prior ASICs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add vertical line to separate NOCAP and ZKSPEED
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(0.2, max(all_target_ms + all_sumflex_ms + fixed_vals) * 0.9, 'NoCap', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(1.9, max(all_target_ms + all_sumflex_ms + fixed_vals) * 0.9, 'zkSpeed', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, "combined_runtime_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("COMBINED RUNTIME COMPARISON SUMMARY")
    print("="*60)
    print("NOCAP Design:")
    for label, target, goldilocks in zip(nocap_data['labels'], nocap_target_ms, nocap_our_ms):
        speedup = target / goldilocks if goldilocks > 0 else 0
        print(f"  {label}: {target:.3f}ms (target) vs {goldilocks:.3f}ms (zkPHIRE Goldilocks) = {speedup:.2f}x speedup")
    
    print("\nZKSPEED Design:")
    for label, target, arb_prime, fixed_prime in zip(zkspeed_arb_data['labels'], zkspeed_target_ms, zkspeed_arb_our_ms, zkspeed_fixed_our_ms):
        speedup_arb = target / arb_prime if arb_prime > 0 else 0
        speedup_fixed = target / fixed_prime if fixed_prime > 0 else 0
        print(f"  {label}:")
        print(f"    Target: {target:.3f}ms")
        print(f"    zkPHIRE (Arb Prime): {arb_prime:.3f}ms = {speedup_arb:.2f}x speedup")
        print(f"    zkPHIRE (Fixed Prime): {fixed_prime:.3f}ms = {speedup_fixed:.2f}x speedup")
    print("="*60)


def combine_sumcheck_stats(sumcheck_core_stats):
    # Find all unique hardware configs across all polynomials
    all_hw_configs = set()
    for hw_configs in sumcheck_core_stats.values():
        all_hw_configs.update(hw_configs.keys())

    combined_stats = {0: {}}
    for hw_config in all_hw_configs:
        combined_entry = {
            'round_latencies': [],
            'utilizations': [],
            'per_round_utilizations': [],
            'total_latency': 0,
            'area': None,
            'area_with_hbm': None,
            'modmul_count': None,
            'design_modmul_area': None
        }
        for poly_idx, hw_configs in sumcheck_core_stats.items():
            if hw_config in hw_configs:
                stats = hw_configs[hw_config]
                combined_entry['round_latencies'].append(stats['round_latencies'])
                combined_entry['utilizations'].append(stats['utilization'])
                combined_entry['per_round_utilizations'].append(stats['per_round_utilization'])
                combined_entry['total_latency'] += stats['total_latency']
                # Set area and other static values from the first occurrence
                if combined_entry['area'] is None:
                    combined_entry['area'] = stats['area']
                    combined_entry['area_with_hbm'] = stats['area_with_hbm']
                    combined_entry['modmul_count'] = stats['modmul_count']
                    combined_entry['design_modmul_area'] = stats['design_modmul_area']
        combined_stats[0][hw_config] = combined_entry
    return combined_stats

def get_specific_design_data(num_vars, sumcheck_polynomial, sumcheck_hardware_params, latencies, constants, available_bw, scale_factor_22_to_7nm=None):
    bits_per_scalar, freq, *_ = constants

    num_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, onchip_mle_size = sumcheck_hardware_params

    sumcheck_hardware_config = num_pes, num_eval_engines, num_product_lanes, onchip_mle_size

    total_area_mm2, hbm_phy_area_mm2, total_modmuls, design_modmul_area, area_breakdown = get_area_cost(sumcheck_hardware_config, latencies, constants, available_bw, scale_factor_22_to_7nm)

    supplemental_data = bits_per_scalar, available_bw, freq
    num_build_mle = len({elem for sublist in sumcheck_polynomial for elem in sublist if isinstance(elem, str) and elem.startswith("fz")})
    round_latencies, *_ = create_sumcheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, num_build_mle, supplemental_data, debug=False, debug_just_start=False, use_max_extensions=True)
    
    return sum(round_latencies), total_area_mm2, hbm_phy_area_mm2

if __name__ == "__main__":
    freq = 1e9
    num_vars = 24
    
    parser = argparse.ArgumentParser(description="Sumcheck hardware design space exploration")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data files")

    args = parser.parse_args()
    overwrite_data = args.overwrite

    # Dictionary to store all results for combined plotting
    all_results = {}

    # Run nocap design first
    print("=" * 50)
    print("RUNNING ISO-NOCAP DESIGN")
    print("=" * 50)
    
    design = "nocap"
    bits_per_element = 64
    mle_update_latency = 2
    extensions_latency = 10
    modmul_latency = 2
    modadd_latency = 1
    bandwidth = 1024
    comp_area = 8.14
    modmul_area = 11682e-6
    poly_list = [spartan_1, spartan_2]
    scale_factor_22_to_7nm = None
    target_runtimes_ms = [36.816]

    print("comparing with design:", design)
    data_folder = f"sumcheck_dse/asic_comp/"
    os.makedirs(data_folder, exist_ok=True)

    latencies = (
        mle_update_latency,
        extensions_latency,
        modmul_latency,
        modadd_latency
    )

    sumcheck_pes_range = [(1 << i) for i in range(6)]
    eval_engines_range = range(2, 8)
    product_lanes_range = range(3, 9)
    onchip_mle_sizes_range = [(1 << i) for i in range(10, 16)]

    modadd_area = 0.002*11682e-6/0.264       # mm^2 per modular adder
    reg_area = 580e-6/256*64  # mm^2 per register
    num_accumulate_regs = 32
    rr_ctrl_area = 49040e-6   # mm^2 per round-robin controller
    
    per_pe_delay_buffer_count = 32
    multifunction_tree_sram_scale_factor = 16
    num_sumcheck_sram_buffers = 24

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
        multifunction_tree_sram_scale_factor
    )

    sweep_params = (
        num_vars,
        sumcheck_pes_range,
        eval_engines_range,
        product_lanes_range,
        onchip_mle_sizes_range
    )

    out_path = os.path.join(data_folder, f"sumcheck_stats_{design}_comp.pkl")
    if not os.path.exists(out_path) or overwrite_data:
        sumcheck_comp_stats = sumcheck_only_sweep(sweep_params, poly_list, latencies, constants, bandwidth, None)
        with open(out_path, "wb") as f:
            pickle.dump(sumcheck_comp_stats, f)
    else:
        with open(out_path, "rb") as f:
            sumcheck_comp_stats = pickle.load(f)
    
    # Process nocap results with pareto search
    sumcheck_comp_stats = combine_sumcheck_stats(sumcheck_comp_stats)

    print("Pareto Design Points for ISO-NOCAP:")
    _, fastest_latency, fastest_config, fastest_area = get_pareto_curve(sumcheck_comp_stats, print_only_best=False, area_max=comp_area)
    print("-------------------------------------")
    labels = ["Combined Spartan"]
    plot_runtime_bars([fastest_latency], labels, freq, target_runtimes_ms=target_runtimes_ms, folder=data_folder, design=design)

    # Store nocap results
    all_results['nocap'] = {
        'fastest_latencies': [fastest_latency],
        'labels': labels,
        'target_runtimes_ms': target_runtimes_ms
    }

    print(f"ISO-NOCAP fastest latency: {fastest_latency}")
    print(f"ISO-NOCAP fastest config: {fastest_config}")
    print(f"ISO-NOCAP fastest area: {fastest_area}")
    print("=" * 50)
    print("ISO-NOCAP DESIGN COMPLETE")
    print("=" * 50)
    

    # iso-zkspeed fixed params
    zerocheck_sc_latency = 8390552
    zerocheck_mu_latency = 6566059
    permcheck_sc_latency = 8391296
    permcheck_mu_latency = 8025180
    opencheck_sc_latency = 8389280
    opencheck_mu_latency = 8025180

    # Create vector of runtimes by adding like numbers together and converting to ms
    zkspeed_target_runtimes = [
        zerocheck_sc_latency + zerocheck_mu_latency,
        permcheck_sc_latency + permcheck_mu_latency,
        opencheck_sc_latency + opencheck_mu_latency
    ]
    zkspeed_target_runtimes_ms = [lat / freq * 1000 for lat in zkspeed_target_runtimes]
    labels = ["ZeroCheck", "PermCheck", "OpenCheck"]

    # Create vector of runtimes by adding like numbers together and converting to ms
    zkspeed_pp_target_runtimes = [
        8572762,
        9545774,
        9545246
    ]
    zkspeed_ops = [
        1392509388, # ZeroCheck
        1694499808, # PermCheck
        704643018   # OpenCheck
    ]
    # zkspeed_ops = [
    #     855638508, # ZeroCheck only product and mle update
    #     1023411208, # PermCheck only product and mle update
    #     503316438   # OpenCheck only product and mle update
    # ]
    zkspeed_pp_target_runtimes_ms = [lat / freq * 1000 for lat in zkspeed_pp_target_runtimes]



    # Now run zkspeed design
    print("\n" + "=" * 50)
    print("RUNNING zkPHIRE (Iso-zkSpeed) DESIGN")
    print("=" * 50)
    
    design = "zkspeed_arb_prime"
    bandwidth = 2048
    # poly_list = [vanilla_gate_zkspeed, vanilla_perm_zkspeed, opencheck]
    # training set
    poly_list = [
        verifiable_asics,
        spartan_1,
        spartan_2,
        witness_non_id_point,
        witness_id_point_1,
        witness_id_point_2,
        incomplete_addition_1,
        incomplete_addition_2,
        complete_addition_1,
        complete_addition_2,
        complete_addition_3,
        complete_addition_4,
        complete_addition_5,
        complete_addition_6,
        complete_addition_7,
        complete_addition_8,
        complete_addition_9,
        complete_addition_10,
        complete_addition_11,
        complete_addition_12
    ]

    bits_per_element = 255
    mle_update_latency = 10
    extensions_latency = 20
    modmul_latency = 10
    modadd_latency = 1
    modmul_area = 0.478
    scale_factor_22_to_7nm = 3.6
    modadd_area = 0.006 # mm^2 per modular adder
    reg_area = 580e-6  # mm^2 per register

    print("comparing with design:", design)
    data_folder = f"sumcheck_dse/asic_comp/"
    os.makedirs(data_folder, exist_ok=True)

    latencies = (
        mle_update_latency,
        extensions_latency,
        modmul_latency,
        modadd_latency
    )

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
        multifunction_tree_sram_scale_factor
    )


    out_path = os.path.join(data_folder, f"sumcheck_stats_{design}_comp.pkl")
    if not os.path.exists(out_path) or overwrite_data:
        sumcheck_comp_stats = sumcheck_only_sweep(sweep_params, poly_list, latencies, constants, bandwidth, scale_factor_22_to_7nm)
        with open(out_path, "wb") as f:
            pickle.dump(sumcheck_comp_stats, f)
    else:
        with open(out_path, "rb") as f:
            sumcheck_comp_stats = pickle.load(f)
    
    comp_area = 37 # in 7nm
    
    # Process zkspeed results with robust objective search
    lambda_weight = 0.5  # tradeoff weight for slowdown/util in robust design search
    slowdown_objective="geomean"
    util_objective="mean"  # objective for utilization in robust design search
    obj_set = None

    # lambda_weight = 0.5 # tradeoff weight for
    # obj_set = "latency"

    print("Running robust design search for zkPHIRE (Iso-zkSpeed) ARBITRARY PRIME...")

    # find the best design on the training set
    robust_design = find_robust_design_with_objective(sumcheck_comp_stats, slowdown_objective=slowdown_objective, util_objective=util_objective, lambda_weight=lambda_weight, area_max=comp_area, include_hbm_area=False, objective_setting=obj_set)
    design_config = robust_design['best_design']
    latencies_best = robust_design['latencies_per_poly']
    print("Best design configuration found:", design_config)
    print("Best design latencies per polynomial:", latencies_best)
    # Print area breakdown for the robust design
    print("\n" + "=" * 50)
    print("AREA BREAKDOWN FOR ROBUST DESIGN")
    print("=" * 50)
    
    # Get area breakdown from the robust design configuration
    num_pes_rb, num_eval_engines_rb, num_product_lanes_rb, onchip_mle_size_rb = design_config
    sumcheck_hardware_config = num_pes_rb, num_eval_engines_rb, num_product_lanes_rb, onchip_mle_size_rb
    _, _, _, _, area_breakdown = get_area_cost(sumcheck_hardware_config, latencies, constants, bandwidth, scale_factor_22_to_7nm)
    
    print(f"Hardware Configuration: {design_config}")
    print(f"Total Area: {sum(area_breakdown.values()):.3f} mm²")
    print("\nArea Breakdown:")
    print(f"  Design ModMul Area:    {area_breakdown['design_modmul_area']:.3f} mm² ({area_breakdown['design_modmul_area']/sum(area_breakdown.values())*100:.1f}%)")
    print(f"  Non-ModMul Logic Area: {area_breakdown['nonmodmul_logic_area']:.3f} mm² ({area_breakdown['nonmodmul_logic_area']/sum(area_breakdown.values())*100:.1f}%)")
    print(f"  Memory Area:           {area_breakdown['memory_area']:.3f} mm² ({area_breakdown['memory_area']/sum(area_breakdown.values())*100:.1f}%)")
    
    print("\nDetailed Breakdown:")
    print(f"  ModMul Count: {num_pes_rb*(num_eval_engines_rb - 1)*num_product_lanes_rb + 2*num_eval_engines_rb*num_pes_rb}")
    print(f"  PE Count: {num_pes_rb}")
    print(f"  Eval Engines per PE: {num_eval_engines_rb}")
    print(f"  Product Lanes per PE: {num_product_lanes_rb}")
    print(f"  On-chip MLE Size: {onchip_mle_size_rb}")
    print("=" * 50)


    num_pes_rb, num_eval_engines_rb, num_product_lanes_rb, onchip_mle_size_rb = design_config
    shp = (num_pes_rb, num_eval_engines_rb, num_product_lanes_rb, mle_update_latency, extensions_latency, modmul_latency, onchip_mle_size_rb)
    
    # Get total_modmuls for utilization calculation
    sumcheck_hardware_config = num_pes_rb, num_eval_engines_rb, num_product_lanes_rb, onchip_mle_size_rb
    _, _, total_modmuls, _, _ = get_area_cost(sumcheck_hardware_config, latencies, constants, bandwidth, scale_factor_22_to_7nm)
    
    test_set_polys = [vanilla_gate_zkspeed, vanilla_perm_zkspeed, opencheck]
    latencies_arb = []
    modmul_ops_arb = []
    prod_ops_arb = []
    mu_ops_arb = []
    utilizations_arb = []
    for poly in test_set_polys:
        total_latency, _, _ = get_specific_design_data(
            num_vars,
            poly,
            shp,
            latencies,
            constants,
            bandwidth,
            scale_factor_22_to_7nm
        )
        latencies_arb.append(total_latency)
        
        # Calculate modmul ops for this polynomial
        modmul_ops, prod_ops, mu_ops = num_modmul_ops_in_polynomial(num_vars, poly, debug=False, return_only_core_ops=True)
        modmul_ops_arb.append(modmul_ops)
        prod_ops_arb.append(prod_ops)
        mu_ops_arb.append(mu_ops)
        
        # Calculate utilization
        utilization = calc_utilization(modmul_ops, total_modmuls, total_latency)
        utilizations_arb.append(utilization)

    plot_runtime_bars(latencies_arb, labels, freq, target_runtimes_ms=zkspeed_target_runtimes_ms, folder=data_folder, design=design)
    
    # Store zkspeed arb prime results
    all_results['zkspeed_arb_prime'] = {
        'fastest_latencies': latencies_arb,
        'labels': labels,
        'target_runtimes_ms': zkspeed_target_runtimes_ms,   
    }
    
    print()
    print("---------- zkPHIRE (Iso-zkSpeed) DESIGN RESULTS ----------")
    print()
    for idx, (label, latency, modmul_ops, prod_ops, mu_ops, utilization) in enumerate(zip(labels, latencies_arb, modmul_ops_arb, prod_ops_arb, mu_ops_arb, utilizations_arb)):
        print(f"{label}: {latency} cycles, {modmul_ops} modmul ops, ops/cycle = {modmul_ops/latency:.2f}, utilization = {utilization:.2f}")
        print(f"  Prod Ops: {prod_ops}, Mu Ops: {mu_ops}")
    
    print()
    print("---------- zkPHIRE (Iso-zkSpeed) DESIGN COMPLETE ----------")
    print()
    print("=" * 50)
    

    print("\nzkSpeed+ runtimes and ops:")
    vanilla_polys = [vanilla_gate_zkspeed, vanilla_perm_zkspeed, opencheck]
    # for label, poly, target_cycles in zip(labels, vanilla_polys, zkspeed_pp_target_runtimes):
    #     modmul_ops, _ = num_modmul_ops_in_polynomial(num_vars, poly, debug=False)
    #     print(f"  {label}: {target_cycles} cycles, {modmul_ops} modmul ops, ops/cycle = {modmul_ops/target_cycles}")
    # print()
    for label, cycle_count, modops in zip(labels, zkspeed_pp_target_runtimes, zkspeed_ops):
        print(f"  {label}: {cycle_count} cycles, {modops} modmul ops, ops/cycle = {modops/cycle_count}")
    print("=" * 50)

    # Get results for jellyfish polynomials using the same robust design configuration
    print("\n" + "=" * 50)
    print("GETTING JELLYFISH POLYNOMIAL RESULTS WITH SAME CONFIG")
    print("=" * 50)

    jellyfish_polys = [jellyfish_gate_hyperplonk, jellyfish_perm_hyperplonk, opencheck]
    jellyfish_labels = ["Jellyfish Gate", "Jellyfish Perm", "OpenCheck"]
    
    # Run 3 experiments with different scaledown factors
    scaledown_factors = [2, 4, 8]
    # scaledown_factors = [4, 8]
    jellyfish_results = {}
    
    for gate_scaledown in scaledown_factors:
        print(f"\n--- Running Jellyfish experiment with scaledown factor {gate_scaledown} ---")
        
        latencies_jellyfish = []
        modmul_ops_jellyfish = []
        prod_ops_jellyfish = []
        mu_ops_jellyfish = []
        utilizations_jellyfish = []
        
        num_vars_jellyfish = num_vars - int(math.log2(gate_scaledown))
        print(f"Using robust design configuration: {design_config}")
        print(f"num_vars_jellyfish = {num_vars_jellyfish} (original {num_vars} - log2({gate_scaledown}))")
        
        for poly, label in zip(jellyfish_polys, jellyfish_labels):
            total_latency, _, _ = get_specific_design_data(
                num_vars_jellyfish,
                poly,
                shp,
                latencies,
                constants,
                bandwidth,
                scale_factor_22_to_7nm
            )
            latencies_jellyfish.append(total_latency)
            
            # Calculate modmul ops for this polynomial
            modmul_ops, product_ops, mu_ops = num_modmul_ops_in_polynomial(num_vars_jellyfish, poly, debug=False, return_only_core_ops=True)
            modmul_ops_jellyfish.append(modmul_ops)
            prod_ops_jellyfish.append(product_ops)
            mu_ops_jellyfish.append(mu_ops)
            
            # Calculate utilization (using same total_modmuls as it's the same hardware config)
            utilization = calc_utilization(modmul_ops, total_modmuls, total_latency)
            utilizations_jellyfish.append(utilization)

            print(f"{label}: {total_latency} cycles, {modmul_ops} modmul ops, ops/cycle = {modmul_ops/total_latency:.2f}, utilization = {utilization:.2f}")
            print(f"  Prod Ops: {product_ops}, Mu Ops: {mu_ops}")

        # Store results for this scaledown factor
        jellyfish_results[gate_scaledown] = {
            'fastest_latencies': latencies_jellyfish,
            'labels': jellyfish_labels,
            'zkspeed_target_runtimes_ms': zkspeed_target_runtimes_ms,
            'zkspeed_pp_target_runtimes_ms': zkspeed_pp_target_runtimes_ms,
        }
        
        # Convert to ms and calculate speedups against zkspeed targets
        jellyfish_runtimes_ms = [(lat / freq) * 1000 for lat in latencies_jellyfish]

        print(f"\nJellyfish polynomial speedup vs zkSpeed (scaledown {gate_scaledown}):")
        for label, target_ms, our_ms in zip(jellyfish_labels, zkspeed_target_runtimes_ms, jellyfish_runtimes_ms):
            speedup = target_ms / our_ms if our_ms > 0 else 0
            print(f"  {label}: {target_ms:.3f}ms (zkSpeed) vs {our_ms:.3f}ms (Ours) = {speedup:.2f}x speedup")

        print(f"\nJellyfish polynomial speedup vs zkSpeed+ (scaledown {gate_scaledown}):")
        for label, target_ms, our_ms in zip(jellyfish_labels, zkspeed_pp_target_runtimes_ms, jellyfish_runtimes_ms):
            speedup = target_ms / our_ms if our_ms > 0 else 0
            print(f"  {label}: {target_ms:.3f}ms (zkSpeed+) vs {our_ms:.3f}ms (Ours) = {speedup:.2f}x speedup")

    # Store all jellyfish results (use scaledown=4 as default for backward compatibility)
    all_results['jellyfish'] = jellyfish_results[4]
    all_results['jellyfish_all'] = jellyfish_results
    
    print()
    print("=" * 50)
    print("JELLYFISH POLYNOMIAL RESULTS COMPLETE")
    print("=" * 50)
    
   
    skip_fixed_prime = True

    if not skip_fixed_prime:
    # Now run zkspeed design
        print("\n" + "=" * 50)
        print("RUNNING zkPHIRE (Iso-zkSpeed) FIXED PRIME DESIGN")
        print("=" * 50)
        
        design = "zkspeed_fixed_prime"
        bandwidth = 2048
        comp_area = 24.96 + 5.84 # in 7nm
        poly_list = [vanilla_gate_zkspeed, vanilla_perm_zkspeed, opencheck]
        bits_per_element = 256
        mle_update_latency = 20
        extensions_latency = 20
        modmul_latency = 20
        modadd_latency = 1
        modmul_area = 0.264
        scale_factor_22_to_7nm = 3.6
        modadd_area = 0.002 # mm^2 per modular adder
        reg_area = 580e-6  # mm^2 per register

        print("comparing with design:", design)
        data_folder = f"sumcheck_dse/asic_comp/"
        os.makedirs(data_folder, exist_ok=True)

        latencies = (
            mle_update_latency,
            extensions_latency,
            modmul_latency,
            modadd_latency
        )

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
            multifunction_tree_sram_scale_factor
        )

        out_path = os.path.join(data_folder, f"sumcheck_stats_{design}_comp.pkl")
        if not os.path.exists(out_path) or overwrite_data:
            sumcheck_comp_stats = sumcheck_only_sweep(sweep_params, poly_list, latencies, constants, bandwidth, scale_factor_22_to_7nm)
            with open(out_path, "wb") as f:
                pickle.dump(sumcheck_comp_stats, f)
        else:
            with open(out_path, "rb") as f:
                sumcheck_comp_stats = pickle.load(f)
        

        
        print("Running robust design search for zkPHIRE (Iso-zkSpeed)...")
        robust_design = find_robust_design_with_objective(sumcheck_comp_stats, slowdown_objective=slowdown_objective, util_objective=util_objective, lambda_weight=lambda_weight, area_max=comp_area, include_hbm_area=False)
        latencies_fixed = robust_design['latencies_per_poly']

        utilizations = robust_design['utilizations_per_poly']
        print("Utilizations per polynomial:", utilizations)
        
        plot_runtime_bars(latencies_fixed, labels, freq, target_runtimes_ms=zkspeed_target_runtimes_ms, folder=data_folder, design=design)
        
        # Store zkspeed fixed prime results
        all_results['zkspeed_fixed_prime'] = {
            'fastest_latencies': latencies_fixed,
            'labels': labels,
            'target_runtimes_ms': zkspeed_target_runtimes_ms
        }
        
        print("---------- zkPHIRE (Iso-zkSpeed) DESIGN RESULTS ----------")
        for label, latency in zip(labels, latencies_fixed):
            print(f"{label}: {latency} cycles")
        print("---------- zkPHIRE (Iso-zkSpeed) DESIGN COMPLETE ----------")
        print("=" * 50)
    
    # Create comprehensive comparison plot
    print("\n" + "=" * 50)
    print("CREATING COMPREHENSIVE COMPARISON PLOT")
    print("=" * 50)
    plot_comprehensive_runtime_comparison(all_results, freq, scaledown_factors, data_folder)
    print("Comprehensive comparison plot saved!")
    print("=" * 50)
    
    # Create zkSpeed-only comparison plot
    print("\n" + "=" * 50)
    print("CREATING ZKSPEED-ONLY COMPARISON PLOT")
    print("=" * 50)
    plot_zkspeed_only_runtime_comparison(all_results, freq, scaledown_factors, data_folder)
    print("zkSpeed-only comparison plot saved!")
    print("=" * 50)