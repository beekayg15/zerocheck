import math
from params import *
from sumcheck_models import create_sumcheck_schedule
import numpy as np

def get_phy_cost(available_bw):
    if available_bw <= 512:
        # bad estimate for now
        hbm_area = (available_bw/512)*14.9
    
    else:
        hbm_area = (available_bw/1024)*29.6

    return hbm_area

def get_area_cost(hw_config, latencies, constants, available_bw):
    num_pes, num_eval_engines, num_product_lanes, onchip_mle_size = hw_config
    _, _, modmul_latency, modadd_latency = latencies
    bits_per_scalar, freq, modmul_area, modadd_area, reg_area, num_accumulate_regs, rr_ctrl_area, per_pe_delay_buffer_count, num_sumcheck_sram_buffers, tmp_mle_sram_scale_factor = constants

    # tree_module = shared_tree_cost(setup_config_dict = {
    #     "basic": {
    #         "mod_mul_latency": modmul_latency,
    #         "mod_add_latency": modadd_latency,
    #     },
    #     "multiply_lane_tree": {
    #         "num_input_entries_per_cycle": num_eval_engines,  # should be at least num_eval_engines
    #         "num_lanes_per_sc_pe": num_product_lanes,
    #         "total_sc_pe": num_pes
    #     },
    # })
    # hw_cost = tree_module.get_hardware_cost_cost()

    # mulitifunction tree area
    # multifunction_tree_regs = hw_cost['req_mem_reg_num']
    multifunction_tree_regs = 0
    multifunction_tree_modmuls = num_pes*(num_eval_engines - 1)*num_product_lanes # minimum number of modmuls for full pipelining
    # multifunction_tree_modadds = hw_cost['req_mod_add_num']
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

    tmp_mle_buffer_area_mb = tmp_mle_sram_scale_factor*onchip_mle_size*bits_per_scalar/BITS_PER_MB
    tmp_mle_buffer_area_mm2 = tmp_mle_buffer_area_mb*MB_CONVERSION_FACTOR

    hbm_phy_area_mm2 = get_phy_cost(available_bw)

    total_area_mm2 = multifunction_tree_compute_area_mm2 + eval_engine_area_mm2 + accumulation_reg_area_mm2 + sumcheck_buffer_area_mm2 + tmp_mle_buffer_area_mm2
    total_area_mm2 /= scale_factor_22_to_7nm  # convert to 7nm area

    total_onchip_memory_MB = (
        (num_sumcheck_sram_buffers + tmp_mle_sram_scale_factor) * onchip_mle_size * bits_per_scalar
        + num_accumulate_regs * bits_per_scalar
    ) / BITS_PER_MB

    total_modmuls = multifunction_tree_modmuls + eval_engine_modmuls * num_pes
    design_modmul_area = total_modmuls * modmul_area

    return round(total_area_mm2, 3), hbm_phy_area_mm2, total_modmuls, design_modmul_area, total_onchip_memory_MB

def num_modmul_ops_in_polynomial(num_vars, sumcheck_polynomial, debug=True, return_only_core_ops=False):
    degrees = [len(term) for term in sumcheck_polynomial]
    num_build_mle = len({elem for sublist in sumcheck_polynomial for elem in sublist if isinstance(elem, str) and elem.startswith("fz")})
    per_round_ops = np.zeros(num_vars)

    per_round_ops[0] += num_build_mle*(1<<num_vars)

    max_degree = max(degrees)
    modmul_ops_per_term = [(d - 1)*(max_degree + 1) for d in degrees]
    product_modmul_ops = sum(modmul_ops_per_term)
    total_product_modmul_ops = 0
    
    if debug:
        print()
        print(f"num_vars: {num_vars}, num_terms: {len(sumcheck_polynomial)}, max_degree: {max_degree}")
        print(f"  num_build_mle: {num_build_mle}")
        print(f"  degrees: {degrees}, max_degree: {max_degree}")
        print(f"  modmul_ops_per_term: {modmul_ops_per_term}")
        print(f"  product_modmul_ops: {product_modmul_ops}")

    idx = 0
    for i in range(num_vars, 0, -1):
        num_pairs = (1 << (i - 1))
        per_round_ops[idx] += product_modmul_ops * num_pairs
        total_product_modmul_ops += product_modmul_ops * num_pairs
        idx += 1

    if debug:
        print(f"num_modmul_ops_in_polynomial: {num_vars} vars, {len(sumcheck_polynomial)} terms, {max_degree} max degree")
        print(f"  product_modmul_ops: {product_modmul_ops}, total_product_modmul_ops: {total_product_modmul_ops}")
    
    unique_entries = set()
    for term in sumcheck_polynomial:
        unique_entries.update(term)
    num_unique_mles = len(unique_entries)
    
    total_mle_update_modmul_ops = 0

    idx = 1
    for i in range(num_vars, 1, -1):
        num_pairs = (1 << (i - 1))
        per_round_ops[idx] += num_unique_mles * num_pairs
        total_mle_update_modmul_ops += num_unique_mles * num_pairs
        idx += 1

    per_round_ops = [int(x) for x in per_round_ops]

    if debug:
        print(f"  num_unique_mles: {num_unique_mles}, total_mle_update_modmul_ops: {total_mle_update_modmul_ops}")
        print(f"  num_build_mle_modops: {num_build_mle*(1<<num_vars)}")
        print(f"  per_round_ops: {per_round_ops}")
        print(f"  total_modmul_ops: {total_product_modmul_ops + total_mle_update_modmul_ops + num_build_mle*(1<<num_vars)}")
        print()
    
    assert sum(per_round_ops) == total_product_modmul_ops + total_mle_update_modmul_ops + num_build_mle*(1<<num_vars)
    if return_only_core_ops:
        return total_product_modmul_ops + total_mle_update_modmul_ops, total_product_modmul_ops, total_mle_update_modmul_ops
    else:
        return total_product_modmul_ops + total_mle_update_modmul_ops + num_build_mle*(1<<num_vars), per_round_ops

def calc_utilization(modmul_ops, num_modmuls, actual_cycle_count):
    min_cycles = modmul_ops / num_modmuls
    if actual_cycle_count == 0:
        exit("why")
    utilization = min_cycles / actual_cycle_count
    return utilization

def sumcheck_only_sweep(sweep_params, sumcheck_polynomials, latencies, constants, available_bw):

    num_vars, sumcheck_pes_range, eval_engines_range, product_lanes_range, onchip_mle_sizes_range = sweep_params
    mle_update_latency, extensions_latency, modmul_latency, modadd_latency = latencies

    bits_per_scalar, freq, *_ = constants

    sumcheck_core_stats = dict()
    for idx, sumcheck_polynomial in enumerate(sumcheck_polynomials):
        
        modmul_ops, per_round_ops = num_modmul_ops_in_polynomial(num_vars, sumcheck_polynomial, debug=True)
        
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

                        total_area_mm2, hbm_phy_area_mm2, total_modmuls, design_modmul_area, total_onchip_memory_MB = get_area_cost(sumcheck_hardware_config, latencies, constants, available_bw)

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

                        s_dict['total_latency'] = total_latency + (2 * num_pes * num_vars)  # + PE accumulation per round
                        s_dict['area'] = total_area_mm2
                        s_dict['area_with_hbm'] = total_area_mm2 + hbm_phy_area_mm2
                        s_dict['modmul_count'] = total_modmuls
                        s_dict['design_modmul_area'] = design_modmul_area
                        s_dict['total_onchip_memory_MB'] = total_onchip_memory_MB
                        s_dict['round_latencies'] = round_latencies
                        s_dict['utilization'] = utilization
                        s_dict['per_round_utilization'] = per_round_utilization

    return sumcheck_core_stats
