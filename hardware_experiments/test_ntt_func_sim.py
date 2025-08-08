import random
import math
import sys
import argparse
import pickle
import os
from ntt_func_sim import ArchitectureSimulator
from ntt import ntt, ntt_dit_rn, ntt_dif_nr, bit_rev_shuffle
from ntt_utility import *
from util import calc_rate
from fourstep_ntt_perf_models import get_compute_latency
from params_ntt_v_sum import *
from plot_funcs import plot_pareto_frontier_from_pickle, plot_pareto_all_configs_from_pickle, plot_pareto_multi_bw_fixed_n

def get_area_stats(total_modmuls, total_modadds, total_num_words, bit_width=256):
    logic_area = total_modmuls*modmul_area + total_modadds*modadd_area
    memory_area = (total_num_words * bit_width / BITS_PER_MB) * MB_CONVERSION_FACTOR
    return logic_area, memory_area

def simulate_4step_all_onchip_one_pe(arch, mat, global_omega, modulus, output_scale=True, tags_only=True, skip_compute=False):
    num_rows = len(mat)
    num_cols = len(mat[0])
    total_length = num_rows * num_cols
    L = total_length   

    # Perform prefetch operation once before processing all columns
    arch.prefetch()

    output_matrix = [[0] * num_cols for _ in range(num_rows)]
    for idx in range(num_cols + 2):  # Only need +2 now since no prefetch stage
        if idx < num_cols:
            col = [mat[row][idx] for row in range(num_rows)] if not skip_compute else "don't care"
            tag = idx
        else:
            col = None
            tag = None
        arch.step(col, tag, tags_only=tags_only)
    
        if not skip_compute:
            out_data, out_tag = arch.out
            if idx >= 2:  # Output starts after 2 cycles (READ -> COMPUTE -> WRITE)
                for i in range(num_rows):
                    if output_scale:
                        output_matrix[i][out_tag] = (out_data[i] * pow(global_omega, (i * out_tag) % L, modulus)) % modulus
                    else:
                        output_matrix[i][out_tag] = out_data[i]
    if skip_compute:
        output_matrix = None

    return output_matrix, arch.cycle_time

def simulate_4step_all_onchip(arch, num_pes, mat, global_omega, modulus, output_scale=True, tags_only=True, skip_compute=False):
    num_rows = len(mat)
    num_cols = len(mat[0])
    total_length = num_rows * num_cols
    L = total_length

    assert num_cols % num_pes == 0, "Number of columns must be divisible by number of PEs."

    # Perform prefetch operation once before processing all columns
    arch.prefetch()

    output_matrix = [[0] * num_cols for _ in range(num_rows)]

    num_col_groups = num_cols // num_pes
    num_steps = num_cols // num_pes + 2
    for idx in range(num_steps):  # Only need +2 now since no prefetch stage
        if idx < num_col_groups:
            # Extract num_pes columns starting from idx * num_pes
            col = [[mat[row][idx * num_pes + pe] for pe in range(num_pes)] for row in range(num_rows)] if not skip_compute else "don't care"
            tags = [idx * num_pes + pe for pe in range(num_pes)]  # Individual column indices
        else:
            col = None
            tags = None
        arch.step(col, tags, tags_only=tags_only)
        if not skip_compute:
            out_data, out_tags = arch.out
            if idx >= 2:  # Output starts after 2 cycles (READ -> COMPUTE -> WRITE)
                if out_tags is not None:
                    if isinstance(out_tags, list):
                        # Multi-PE case: out_data is a list of columns, out_tags is a list of column indices
                        for pe in range(num_pes):
                            col_idx = out_tags[pe]
                            for i in range(num_rows):
                                if output_scale:
                                    output_matrix[i][col_idx] = (out_data[pe][i] * pow(global_omega, (i * col_idx) % L, modulus)) % modulus
                                else:
                                    output_matrix[i][col_idx] = out_data[pe][i]
                    else:
                        # Single PE case: out_data is a single column, out_tags is a single column index
                        col_idx = out_tags
                        for i in range(num_rows):
                            if output_scale:
                                output_matrix[i][col_idx] = (out_data[i] * pow(global_omega, (i * col_idx) % L, modulus)) % modulus
                            else:
                                output_matrix[i][col_idx] = out_data[i]
    if skip_compute:
        output_matrix = None

    return output_matrix, arch.cycle_time

def simulate_4step_notall_onchip(arch, mat, omegas_matrix, global_omega, modulus, output_scale=True, tags_only=True):
    num_rows = len(mat)
    num_cols = len(mat[0])
    total_length = num_rows * num_cols
    L = total_length   

    # Perform prefetch operation once before processing all columns
    arch.prefetch()
    arch.set_omegas(None)  # No omegas set for this case

    output_matrix = [[0] * num_cols for _ in range(num_rows)]
    for idx in range(num_cols + 2):  # Only need +2 now since no prefetch stage
        if idx < num_cols:
            col = [mat[row][idx] for row in range(num_rows)]
            tag = idx
        else:
            col = None
            tag = None
        arch.step(col, tag, tags_only=tags_only)
        out_data, out_tag = arch.out
        if idx >= 2:  # Output starts after 2 cycles (READ -> COMPUTE -> WRITE)
            for i in range(num_rows):
                if output_scale:
                    output_matrix[i][out_tag] = (out_data[i] * pow(global_omega, (i * out_tag) % L, modulus)) % modulus
                else:
                    output_matrix[i][out_tag] = out_data[i]

    return output_matrix, arch.cycle_time

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def flatten(matrix):
    return [elem for row in matrix for elem in row]

def get_read_latency(num_words, num_butterflies, max_read_rate):
    desired_read_rate = num_butterflies * 2
    actual_read_rate = min(desired_read_rate, max_read_rate)

    read_latency = int(math.ceil(num_words/actual_read_rate))
    return read_latency

def get_compute_latency_single_stage(ntt_len, num_butterflies, bf_latency, modadd_latency, output_scaled=False, stage="most"):

    if stage == "first":
        return modadd_latency + ntt_len/(num_butterflies*2) - 1
    elif stage == "most":
        return bf_latency + ntt_len/(num_butterflies*2) - 1
    elif stage == "last":
        if output_scaled:
            return 3*bf_latency + ntt_len/(num_butterflies*2) - 1  # 3 modmuls, 1 for butterfly, 2 for elementwise multiply
        else:
            return bf_latency + ntt_len/(num_butterflies*2) - 1
    
    return int(compute_latency)

def get_twiddle_factors(exponent, bit_width=256):
    n = exponent
    L = 2**n
    M, N = closest_powers_of_two(n)
    print(f"M = {M}, N = {N}")

    # Get the required modulus and omega
    modulus = find_a_modulus(L, bit_width)

    omegas_L = generate_twiddle_factors(L, modulus)
    omega_L = omegas_L[1]

    omegas_N = generate_twiddle_factors(N, modulus)
    omega_N = omegas_N[1]

    omegas_M = generate_twiddle_factors(M, modulus)
    omega_M = omegas_M[1]

    return M, N, omegas_L, omega_L, omegas_N, omega_N, omegas_M, omega_M, modulus

def get_latencies_and_rates(col_words, row_words, num_bfs, bit_width, available_bw, freq, modadd_latency=1, modmul_latency=20, bf_latency=21, output_scaled=True):
    max_read_rate = calc_rate(bit_width, available_bw, freq)  # Example: 1 GHz frequency, 1 TB/s
    
    # this gets latency accounting for desired rate as well
    mem_latency_cols = get_read_latency(col_words, num_bfs, max_read_rate)
    mem_latency_rows = get_read_latency(row_words, num_bfs, max_read_rate)

    compute_latency_cols = get_compute_latency(col_words, num_bfs, bf_latency, modadd_latency, output_scaled=True)
    compute_latency_rows = get_compute_latency(row_words, num_bfs, bf_latency, modadd_latency, output_scaled=False)

    prefetch_amt = 1.5 * col_words  # prefetching 1 column of local twiddles (M/2 words), and 1 column of global twiddles (M words)
    first_step_prefetch_latency = get_read_latency(prefetch_amt, num_bfs, max_read_rate)
    print(f"First step prefetch latency: {first_step_prefetch_latency} cycles")
    # print(f)
    return mem_latency_cols, mem_latency_rows, compute_latency_cols, compute_latency_rows, first_step_prefetch_latency

def run_fit_onchip(target_n=None, target_bw=None):

    random.seed(0)

    # sweep parameters: n, bandwidth, U, PEs

    bit_width = 256
    available_bw = 1024
    freq = 1e9

    modadd_latency = 1
    modmul_latency = 20
    bf_latency = modmul_latency + modadd_latency

    # Use target values if provided, otherwise use full sweep ranges
    if target_bw is not None:
        bandwidths = [target_bw]
    else:
        bandwidths = [2**i for i in range(6, 13)] # 64 GB/s to 4096 GB/s
    
    unroll_factors = [1, 2, 4, 8, 16, 32, 64]
    
    if target_n is not None:
        lengths = [target_n]
    else:
        lengths = range(16, 27)
    
    pe_counts = [1, 2, 4, 8, 16, 32, 64]
    # Four step NTT: L = M*N, M > N

    check_correctness = False
    skip_compute = True

    # Dictionary to store results indexed by (n, bandwidth, U, pe_amt)
    results = {}

    for n in lengths:

        # fixed for a given n
        M, N, omegas_L, omega_L, omegas_N, omega_N, omegas_M, omega_M, modulus = get_twiddle_factors(n, bit_width)

        # Generate random data and reshape it.
        data = [random.randint(0, modulus - 2) for _ in range(1<<n)]

        # Reshape data into M x N matrix (list of lists)
        matrix = [data[i*N:(i+1)*N] for i in range(M)]

        for available_bw in bandwidths:
            for U in unroll_factors:
                for pe_amt in pe_counts:

                    num_col_words = M*pe_amt
                    num_row_words = N*pe_amt
                    total_bfs = U*pe_amt

                    mem_latency_cols, mem_latency_rows, compute_latency_cols, compute_latency_rows, first_step_prefetch_latency = \
                        get_latencies_and_rates(num_col_words, num_row_words, total_bfs, bit_width, available_bw, freq, modadd_latency, modmul_latency, bf_latency)
                    # dont have to fetch global twiddles for row-wise NTTs, only omegas_N
                    fourth_step_prefetch_latency = mem_latency_rows

                    print("Simulating four-step NTT when mini NTT fits on-chip...")

                    arch_1 = ArchitectureSimulator(omegas_M, modulus, mem_latency_cols, compute_latency_cols, prefetch_latency=first_step_prefetch_latency, skip_compute=skip_compute)
                    temp_matrix, cycle_time_1 = simulate_4step_all_onchip(arch_1, pe_amt, matrix, omega_L, modulus, output_scale=True, skip_compute=skip_compute)

                    temp_matrix_T = transpose(temp_matrix) if not skip_compute else transpose(matrix)
                    arch_2 = ArchitectureSimulator(omegas_N, modulus, mem_latency_rows, compute_latency_rows, prefetch_latency=fourth_step_prefetch_latency, skip_compute=skip_compute)
                    final_matrix, cycle_time_2 = simulate_4step_all_onchip(arch_2, pe_amt, temp_matrix_T, omega_L, modulus, output_scale=False, skip_compute=skip_compute)

                    total_cycles = cycle_time_1 + cycle_time_2

                    total_num_words = 5.5 * M * pe_amt # M > N, so 4 buffers of M words for double buffered ping pong, 1/2 buffer for local twiddles, 1 buffer for global twiddles
                    total_modmuls = U*pe_amt
                    total_modadds = U*2*pe_amt

                    results[(n, available_bw, U, pe_amt)] = {
                        "total_cycles": total_cycles,
                        "total_modmuls": total_modmuls,
                        "total_modadds": total_modadds,
                        "total_num_words": total_num_words
                    }

                    # print(f"Cycle time: {cycle_time_1 + cycle_time_2}")

                    # if n < 13:
                    if check_correctness:
                        print(f"hw_config: n={n}, bw={available_bw}, U={U}, pe_amt={pe_amt}")
                        final_vector = flatten(final_matrix)

                        result_direct = bit_rev_shuffle(ntt_dif_nr(data, modulus, omegas_L))
                        result_direct = ntt_dit_rn(bit_rev_shuffle(data), modulus, omegas_L)

                        # print(f"NTT result fourstep  = {list(final_vector)}")
                        # print(f"NTT result direct    = {result_direct}")
                    
                        if final_vector != result_direct:
                            print("Mismatch between four-step NTT and direct NTT results!")
                            exit()
                        else:
                            print("Four-step NTT matches direct NTT results.")
                        print()

    # Print results
    print("Results:")
    for key, value in results.items():
        n, available_bw, U, pe_amt = key
        print(f"n={n}, bw={available_bw}, U={U}, pe_amt={pe_amt} -> total_cycles: {value['total_cycles']}, total_modmuls: {value['total_modmuls']}, total_num_words: {value['total_num_words']}")

    # Save results to pickle file
    output_dir = "pickle_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on arguments
    if target_n is not None and target_bw is not None:
        filename = f"results_n{target_n}_bw{target_bw}.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
    print(f"Results saved to {filepath}")


def run_notfit_onchip():

    U = 4
    bit_width = 256
    available_bw = 1024
    freq = 1e9    


    modadd_latency = 1
    modmul_latency = 20
    bf_latency = modmul_latency + modadd_latency

    n = 16
    L = 2**n
    M, N = closest_powers_of_two(n)
    print(f"M = {M}, N = {N}")

    # Get the required modulus and omega
    modulus = find_a_modulus(L, bit_width)
    random.seed(0)

    omegas_L = generate_twiddle_factors(L, modulus)
    omega_L = omegas_L[1]

    omegas_N = generate_twiddle_factors(N, modulus)
    omega_N = omegas_N[1]

    omegas_M = generate_twiddle_factors(M, modulus)
    omega_M = omegas_M[1]
    
    max_read_rate = calc_rate(bit_width, available_bw, freq)  # Example: 1 GHz frequency, 1 TB/s
    
    desired_read_rate = U * 2  # U banks that are double ported

    actual_read_rate = min(desired_read_rate, max_read_rate)
    mem_latency_cols = get_read_latency(M, U, actual_read_rate)
    mem_latency_rows = get_read_latency(N, U, actual_read_rate)
    compute_latency_cols = get_compute_latency(M, U, bf_latency, modadd_latency, output_scaled=True)
    compute_latency_rows = get_compute_latency(N, U, bf_latency, modadd_latency, output_scaled=False)

    # Four step NTT: L = M*N, M > N

    # Generate random data and reshape it.
    data = [random.randint(0, modulus - 2) for _ in range(L)]
    matrix = [data[i*N:(i+1)*N] for i in range(M)]
    # Reshape omegas_M into M x N matrix (list of lists)
    omegas_M_matrix = [omegas_M[i*N:(i+1)*N] for i in range(M)]
    # Reshape omegas_N into N x M matrix (list of lists)
    omegas_N_matrix = [omegas_N[i*M:(i+1)*M] for i in range(N)]

    print("Simulating four-step NTT when mini NTT does not fit on-chip...")

    arch_1 = ArchitectureSimulator(None, modulus, mem_latency_cols, compute_latency_cols, mem_latency_cols)
    temp_matrix, cycle_time_1 = simulate_4step_notall_onchip(arch_1, matrix, omegas_M_matrix, omega_L, modulus, output_scale=True)

    temp_matrix_T = transpose(temp_matrix)
    arch_2 = ArchitectureSimulator(None, modulus, mem_latency_rows, compute_latency_rows, mem_latency_rows)
    final_matrix, cycle_time_2 = simulate_4step_notall_onchip(arch_2, temp_matrix_T, omegas_N_matrix, omega_L, modulus, output_scale=False)

    # if n < 13:
    final_vector = flatten(final_matrix)

    result_direct = bit_rev_shuffle(ntt_dif_nr(data, modulus, omegas_L))
    result_direct = ntt_dit_rn(bit_rev_shuffle(data), modulus, omegas_L)

    # print(f"NTT result fourstep  = {list(final_vector)}")
    # print(f"NTT result direct    = {result_direct}")

    if final_vector != result_direct:
        print("Mismatch between four-step NTT and direct NTT results!")
    else:
        print("Four-step NTT matches direct NTT results.")

    print(f"Cycle time: {cycle_time_1 + cycle_time_2}")

    print("Simulating four-step NTT when mini NTT does not fit on-chip...")


def run_one_config_fit_onchip():

    random.seed(0)

    # sweep parameters: n, bandwidth, U, PEs

    bit_width = 256
    available_bw = 4096
    freq = 1e9

    modadd_latency = 1
    modmul_latency = 20
    bf_latency = modmul_latency + modadd_latency

    unroll_factors = [1, 2, 4, 8, 16, 32, 64]
    lengths = range(16, 27)
    pe_counts = [1, 2, 4, 8, 16, 32, 64]
    # Four step NTT: L = M*N, M > N

    check_correctness = False
    skip_compute = True

    # Dictionary to store results indexed by (n, bandwidth, U, pe_amt)
    results = {}

    n = 16
    U = 2
    pe_amt = 1

    # fixed for a given n
    M, N, omegas_L, omega_L, omegas_N, omega_N, omegas_M, omega_M, modulus = get_twiddle_factors(n, bit_width)

    # Generate random data and reshape it.
    data = [random.randint(0, modulus - 2) for _ in range(1<<n)]

    # Reshape data into M x N matrix (list of lists)
    matrix = [data[i*N:(i+1)*N] for i in range(M)]

    num_col_words = M*pe_amt
    num_row_words = N*pe_amt
    total_bfs = U*pe_amt

    mem_latency_cols, mem_latency_rows, compute_latency_cols, compute_latency_rows, first_step_prefetch_latency = \
        get_latencies_and_rates(num_col_words, num_row_words, total_bfs, bit_width, available_bw, freq, modadd_latency, modmul_latency, bf_latency)
    # dont have to fetch global twiddles for row-wise NTTs, only omegas_N
    fourth_step_prefetch_latency = mem_latency_rows

    print("Simulating four-step NTT when mini NTT fits on-chip...")

    arch_1 = ArchitectureSimulator(omegas_M, modulus, mem_latency_cols, compute_latency_cols, first_step_prefetch_latency, skip_compute=skip_compute)
    # arch_1.set_debug(True)  # Enable debug output
    temp_matrix, cycle_time_1 = simulate_4step_all_onchip(arch_1, pe_amt, matrix, omega_L, modulus, output_scale=True, skip_compute=skip_compute)

    temp_matrix_T = transpose(temp_matrix) if not skip_compute else transpose(matrix)
    arch_2 = ArchitectureSimulator(omegas_N, modulus, mem_latency_rows, compute_latency_rows, fourth_step_prefetch_latency, skip_compute=skip_compute)
    final_matrix, cycle_time_2 = simulate_4step_all_onchip(arch_2, pe_amt, temp_matrix_T, omega_L, modulus, output_scale=False, skip_compute=skip_compute)

    total_cycles = cycle_time_1 + cycle_time_2

    total_num_words = 5.5 * M * pe_amt# M > N, so 4 buffers of M words for double buffered ping pong, 1/2 buffer for local twiddles, 1 buffer for global twiddles
    total_modmuls = U*pe_amt
    total_modadds = U*2*pe_amt


    results[(n, available_bw, U, pe_amt)] = {
        "total_cycles": total_cycles,
        "total_modmuls": total_modmuls,
        "total_modadds": total_modadds,
        "total_num_words": total_num_words
    }

    # print(f"Cycle time: {cycle_time_1 + cycle_time_2}")

    # if n < 13:
    if check_correctness:
        print(f"hw_config: n={n}, bw={available_bw}, U={U}, pe_amt={pe_amt}")
        final_vector = flatten(final_matrix)

        result_direct = bit_rev_shuffle(ntt_dif_nr(data, modulus, omegas_L))
        result_direct = ntt_dit_rn(bit_rev_shuffle(data), modulus, omegas_L)

        # print(f"NTT result fourstep  = {list(final_vector)}")
        # print(f"NTT result direct    = {result_direct}")
    
        if final_vector != result_direct:
            print("Mismatch between four-step NTT and direct NTT results!")
            exit()
        else:
            print("Four-step NTT matches direct NTT results.")
        print()

    # Print results
    print("Results:")
    for key, value in results.items():
        n, available_bw, U, pe_amt = key
        print(f"n={n}, bw={available_bw}, U={U}, pe_amt={pe_amt} -> total_cycles: {value['total_cycles']}, total_modmuls: {value['total_modmuls']}, total_num_words: {value['total_num_words']}")

def run_pareto_analysis(n=None, bw=None, multi_bw=False):
    """
    Run Pareto frontier analysis on pickle results.
    
    Args:
        n: Problem size exponent (if None, analyzes all available results)
        bw: Bandwidth in GB/s (if None, analyzes all available results)
        multi_bw: If True and n is specified, plot multiple BWs for fixed n
    """
    print("Running Pareto frontier analysis...")
    
    if multi_bw and n is not None:
        # Plot multiple bandwidths for fixed n
        bw_values = [64, 128, 256, 512, 1024, 2048, 4096]
        print(f"Plotting Pareto frontiers for n={n} across bandwidths: {bw_values}")
        plot_pareto_multi_bw_fixed_n(n, bw_values)
    elif n is not None and bw is not None:
        # Plot for specific n and bw
        print(f"Plotting Pareto frontier for n={n}, bw={bw}")
        plot_pareto_frontier_from_pickle(n, bw)
    else:
        # Plot for all configurations
        pickle_dir = "pickle_results"
        if not os.path.exists(pickle_dir):
            print(f"Error: Directory {pickle_dir} not found!")
            return
        
        # Find available pickle files
        pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
        if not pickle_files:
            print(f"No pickle files found in {pickle_dir}")
            return
        
        print(f"Found {len(pickle_files)} pickle files:")
        for pkl_file in pickle_files:
            print(f"  {pkl_file}")
        
        # Plot the first file or a specific one
        if pickle_files:
            filepath = os.path.join(pickle_dir, pickle_files[0])
            print(f"Plotting Pareto frontier for all configurations in {pickle_files[0]}")
            plot_pareto_all_configs_from_pickle(filepath)

def print_results_for_n_bw(n, bw, pickle_dir="pickle_results"):
    """
    Print all results entries for a specific (n, bw) combination.
    
    Args:
        n: Problem size exponent (e.g., 16 for 2^16)
        bw: Bandwidth in GB/s (e.g., 1024)
        pickle_dir: Directory containing pickle files
    """
    # Construct filename based on n and bw
    filename = f"results_n{n}_bw{bw}.pkl"
    filepath = os.path.join(pickle_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Pickle file {filepath} not found!")
        return
    
    # Load results from pickle file
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    print(f"\nAll results for n={n} (2^{n} elements), bw={bw} GB/s:")
    print("=" * 80)
    print(f"{'Config':<25} {'Cycles':<12} {'ModMuls':<10} {'ModAdds':<10} {'Words':<12} {'Logic Area':<12} {'Mem Area':<12} {'Total Area':<12}")
    print("-" * 140)
    for k, v in results.items():
        print(k, v)
    exit()
    print(results)
    # Sort results for consistent output
    sorted_results = sorted(results.items())
    
    for key, value in sorted_results:
        result_n, result_bw, U, pe_amt = key
        if result_n == n and result_bw == bw:
            cycles = value['total_cycles']
            modmuls = value['total_modmuls']
            modadds = value['total_modadds'] if 'total_modadds' in value else modmuls * 2
            words = value['total_num_words']
            
            # Calculate area components
            logic_area, memory_area = get_area_stats(modmuls, modadds, words)
            total_area = logic_area + memory_area
            
            config_str = f"U={U}, PE={pe_amt}"
            print(f"{config_str:<25} {cycles:<12.0f} {modmuls:<10} {modadds:<10} {words:<12.0f} {logic_area:<12.2f} {memory_area:<12.2f} {total_area:<12.2f}")
    
    print("-" * 140)
    
    # Count total configurations
    matching_configs = sum(1 for key, _ in results.items() if key[0] == n and key[1] == bw)
    print(f"Total configurations for n={n}, bw={bw}: {matching_configs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTT Function Simulator')
    parser.add_argument('--n', type=int, help='Problem size exponent (e.g., 16 for 2^16)')
    parser.add_argument('--bw', '--bandwidth', type=int, help='Bandwidth in GB/s (e.g., 1024)')
    parser.add_argument('--mode', choices=['sweep', 'test', 'one_config', 'plot', 'print'], default='sweep',
                        help='Mode to run: sweep (parameter sweep), test (simple test), one_config (single configuration), plot (Pareto analysis), print (print results table)')
    parser.add_argument('--multi-bw', action='store_true', 
                        help='When in plot mode with --n specified, plot multiple bandwidths on the same chart')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        print("Running simple test...")
        run_simple_test()
    elif args.mode == 'one_config':
        print("Running single configuration test...")
        run_one_config_fit_onchip()
    elif args.mode == 'plot':
        print("Running Pareto frontier analysis...")
        run_pareto_analysis(args.n, args.bw, getattr(args, 'multi_bw', False))
    elif args.mode == 'print':
        if args.n is not None and args.bw is not None:
            print("Printing results table...")
            print_results_for_n_bw(args.n, args.bw)
        else:
            print("Error: --n and --bw arguments are required for print mode")
            print("Usage: python test_ntt_func_sim.py --mode print --n 20 --bw 1024")
    else:  # sweep mode
        if args.n is not None and args.bw is not None:
            print(f"Running sweep for n={args.n}, bw={args.bw}")
            run_fit_onchip(target_n=args.n, target_bw=args.bw)
        elif args.n is not None:
            print(f"Running sweep for n={args.n}, all bandwidths")
            run_fit_onchip(target_n=args.n)
        elif args.bw is not None:
            print(f"Running sweep for bw={args.bw}, all problem sizes")
            run_fit_onchip(target_bw=args.bw)
        else:
            print("Running full parameter sweep...")
            run_fit_onchip()
