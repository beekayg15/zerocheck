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
from fourstep_ntt_perf_models import get_compute_latency, get_compute_latency_with_sparsity
from params_ntt_v_sum import *
from plot_funcs import plot_pareto_frontier_from_pickle, plot_pareto_all_configs_from_pickle, plot_pareto_multi_bw_fixed_n
from poly_analyzer import analyze_polynomial, count_operations

import time
import numpy as np
from tqdm import tqdm

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

def get_compute_latency_single_stage(ntt_len, num_butterflies, bf_latency, modadd_latency, output_scaled=False, stage="most"):

    if stage == "first":
        return modadd_latency + ntt_len/(num_butterflies*2) - 1
    elif stage == "most":
        return bf_latency + ntt_len/(num_butterflies*2) - 1
    elif stage == "last":
        if output_scaled:
            stage_latency = bf_latency + ntt_len/(num_butterflies*2) - 1
            scale_latency = ntt_len/num_butterflies + (bf_latency - modadd_latency) - 1
            return stage_latency + scale_latency
        else:
            return bf_latency + ntt_len/(num_butterflies*2) - 1

def get_twiddle_factors(exponent, bit_width=256, debug=False):
    n = exponent
    L = 2**n
    M, N = closest_powers_of_two(n)
    print(f"M = {M}, N = {N}") if debug else None

    # Get the required modulus and omega
    modulus = find_a_modulus(L, bit_width)

    omegas_L = generate_twiddle_factors(L, modulus)
    omega_L = omegas_L[1]

    omegas_N = generate_twiddle_factors(N, modulus)
    omega_N = omegas_N[1]

    omegas_M = generate_twiddle_factors(M, modulus)
    omega_M = omegas_M[1]

    return M, N, omegas_L, omega_L, omegas_N, omega_N, omegas_M, omega_M, modulus

def get_read_latency(num_words, num_read_ports, max_read_rate):
    desired_read_rate = num_read_ports
    actual_read_rate = min(desired_read_rate, max_read_rate)

    read_latency = int(math.ceil(num_words/actual_read_rate))
    return read_latency

# col words = num words in a column (M), row words = num words in a row (N)
def get_latencies_and_rates(col_words, row_words, num_bfs, num_pes, bit_width, available_bw, freq, modadd_latency=1, modmul_latency=20, bf_latency=21, debug=False):
    max_read_rate = calc_rate(bit_width, available_bw, freq)  # Example: 1 GHz frequency, 1 TB/s
    
    num_read_ports_per_pe = num_bfs * 2

    # this gets latency accounting for desired rate as well
    r_or_w_mem_latency_cols = get_read_latency(col_words*num_pes, num_read_ports_per_pe*num_pes, max_read_rate)
    r_and_w_mem_latency_cols = get_read_latency(2*col_words*num_pes, 2*num_read_ports_per_pe*num_pes, max_read_rate)

    r_or_w_mem_latency_rows = get_read_latency(row_words*num_pes, num_read_ports_per_pe*num_pes, max_read_rate)
    r_and_w_mem_latency_rows = get_read_latency(2*row_words*num_pes, 2*num_read_ports_per_pe*num_pes, max_read_rate)

    # all PEs are synchronized, therefore it suffices to calculate the latency for 1 PE
    compute_latency_cols = get_compute_latency(col_words, num_bfs, bf_latency, modadd_latency, output_scaled=True)
    compute_latency_rows = get_compute_latency(row_words, num_bfs, bf_latency, modadd_latency, output_scaled=False)

    cols_local_twiddle_prefetch_words = col_words / 2
    # fetch (num_pes - 1) columns of global twiddles, and 1 column of global scale twiddles
    global_twiddle_prefetch_words = col_words * num_pes

    # U ports for local twiddle memory
    cols_local_twiddle_prefetch_latency = get_read_latency(cols_local_twiddle_prefetch_words, num_bfs, max_read_rate)
    
    # assume dual ported memory for global twiddles, so 2U ports
    global_twiddle_prefetch_latency = get_read_latency(global_twiddle_prefetch_words, num_read_ports_per_pe*num_pes, max_read_rate)

    # debug=True
    if debug:
        print(f"max read rate: {round(max_read_rate, 3)} elements/cycle")
        print(f"Local twiddle prefetch latency: {cols_local_twiddle_prefetch_latency} cycles")
        print(f"Global twiddle prefetch latency: {global_twiddle_prefetch_latency} cycles")

        print(f"Single-Transfer latency for columns: {r_or_w_mem_latency_cols} cycles")
        print(f"Single-Transfer latency for rows: {r_or_w_mem_latency_rows} cycles")
        print(f"Dual-Transfer latency for columns: {r_and_w_mem_latency_cols} cycles")
        print(f"Dual-Transfer latency for rows: {r_and_w_mem_latency_rows} cycles")
        print(f"Compute latency for columns: {compute_latency_cols} cycles")
        print(f"Compute latency for rows: {compute_latency_rows} cycles")

    first_step_prefetch_latency = cols_local_twiddle_prefetch_latency + global_twiddle_prefetch_latency

    # technically not needed if same local twiddles reused...
    rows_local_twiddle_prefetch_words = row_words / 2
    rows_local_twiddle_prefetch_latency = get_read_latency(rows_local_twiddle_prefetch_words, num_bfs, max_read_rate)
    fourth_step_prefetch_latency = rows_local_twiddle_prefetch_latency

    # 1.5 * col_words  # prefetching 1 column of local twiddles (M/2 words), and 1 column of global twiddles (M words)
    # prefetch_amt = local_twiddle_prefetch_words + global_scale_twiddle_prefetch_words + global_twiddle_prefetch_words
    # first_step_prefetch_latency = get_read_latency(prefetch_amt, num_bfs, max_read_rate)
    # print(f"First step prefetch latency: {first_step_prefetch_latency} cycles")
    # print(f)
    return r_or_w_mem_latency_cols, r_or_w_mem_latency_rows, r_and_w_mem_latency_cols, r_and_w_mem_latency_rows, \
        compute_latency_cols, compute_latency_rows, first_step_prefetch_latency, fourth_step_prefetch_latency


# col words = num words in a column (M), row words = num words in a row (N)
def get_latencies_and_rates_with_sparsity(col_words, row_words, num_bfs, num_pes, bit_width, available_bw, freq, modadd_latency=1, modmul_latency=20, bf_latency=21, debug=False, sparse_fraction=0):
    max_read_rate = calc_rate(bit_width, available_bw, freq)  # Example: 1 GHz frequency, 1 TB/s
    
    num_read_ports_per_pe = num_bfs * 2
    dense_fraction = 1 - sparse_fraction
    assert dense_fraction > 0

    # this gets latency accounting for desired rate as well
    read_mem_latency_cols = get_read_latency(dense_fraction*col_words*num_pes, num_read_ports_per_pe*num_pes, max_read_rate)
    write_mem_latency_cols = get_read_latency(col_words*num_pes, num_read_ports_per_pe*num_pes, max_read_rate)
    r_and_w_mem_latency_cols = get_read_latency((1 + dense_fraction)*col_words*num_pes, 2*num_read_ports_per_pe*num_pes, max_read_rate)

    # no longer sparse after the columnwise ntts
    read_mem_latency_rows = get_read_latency(row_words*num_pes, num_read_ports_per_pe*num_pes, max_read_rate)
    write_mem_latency_rows = get_read_latency(row_words*num_pes, num_read_ports_per_pe*num_pes, max_read_rate)
    r_and_w_mem_latency_rows = get_read_latency(2*row_words*num_pes, 2*num_read_ports_per_pe*num_pes, max_read_rate)

    # all PEs are synchronized, therefore it suffices to calculate the latency for 1 PE
    sparse_amplification = int(1/dense_fraction)

    compute_latency_cols = get_compute_latency_with_sparsity(col_words, num_bfs, bf_latency, modadd_latency, sparse_amplification, output_scaled=True, debug=debug)
    compute_latency_rows = get_compute_latency(row_words, num_bfs, bf_latency, modadd_latency, output_scaled=False)

    cols_local_twiddle_prefetch_words = col_words / 2
    # fetch (num_pes - 1) columns of global twiddles, and 1 column of global scale twiddles
    global_twiddle_prefetch_words = col_words * num_pes

    # U ports for local twiddle memory
    cols_local_twiddle_prefetch_latency = get_read_latency(cols_local_twiddle_prefetch_words, num_bfs, max_read_rate)
    
    # assume dual ported memory for global twiddles, so 2U ports
    global_twiddle_prefetch_latency = get_read_latency(global_twiddle_prefetch_words, num_read_ports_per_pe*num_pes, max_read_rate)

    # debug=True
    if debug:
        print("Sparse NTT read latencies")
        print(f"max read rate: {round(max_read_rate, 3)} elements/cycle")
        print(f"Local twiddle prefetch latency: {cols_local_twiddle_prefetch_latency} cycles")
        print(f"Global twiddle prefetch latency: {global_twiddle_prefetch_latency} cycles")

        print(f"Single-Transfer read latency for columns: {read_mem_latency_cols} cycles")
        print(f"Single-Transfer write latency for columns: {write_mem_latency_cols} cycles")
        print(f"Single-Transfer read latency for rows: {read_mem_latency_rows} cycles")
        print(f"Single-Transfer write latency for rows: {write_mem_latency_rows} cycles")
        print(f"Dual-Transfer latency for columns: {r_and_w_mem_latency_cols} cycles")
        print(f"Dual-Transfer latency for rows: {r_and_w_mem_latency_rows} cycles")
        print(f"Compute latency for columns: {compute_latency_cols} cycles")
        print(f"Compute latency for rows: {compute_latency_rows} cycles")
        print()

    first_step_prefetch_latency = cols_local_twiddle_prefetch_latency + global_twiddle_prefetch_latency

    rows_local_twiddle_prefetch_words = row_words / 2
    rows_local_twiddle_prefetch_latency = get_read_latency(rows_local_twiddle_prefetch_words, num_bfs, max_read_rate)
    fourth_step_prefetch_latency = rows_local_twiddle_prefetch_latency

    rw_latencies = read_mem_latency_cols, write_mem_latency_cols, r_and_w_mem_latency_cols, \
        read_mem_latency_rows, write_mem_latency_rows, r_and_w_mem_latency_rows

    compute_latencies = compute_latency_cols, compute_latency_rows
    prefetch_latencies = first_step_prefetch_latency, fourth_step_prefetch_latency

    return rw_latencies, compute_latencies, prefetch_latencies

# we should get this value from the simulator. this is the latency for a single NTT
def expected_latency(M, N, num_pes, prefetch_latencies, mem_latencies, compute_latencies):
    
    first_step_prefetch_latency, fourth_step_prefetch_latency = prefetch_latencies
    r_or_w_mem_latency_cols, r_or_w_mem_latency_rows, r_and_w_mem_latency_cols, r_and_w_mem_latency_rows = mem_latencies
    compute_latency_cols, compute_latency_rows = compute_latencies
    
    pipeline_steps = 3
    col_steps = N // num_pes + pipeline_steps - 1
    row_steps = M // num_pes + pipeline_steps - 1

    # num_steps = (num inputs + pipeline steps - 1) - 4. 4 comes from 1 step being read only, 1 step read and compute, 1 step compute and write, 1 step write only
    col_ntt_latency = first_step_prefetch_latency + 2*r_or_w_mem_latency_cols + 2*max(r_or_w_mem_latency_cols, compute_latency_cols) + (col_steps - 4)*max(compute_latency_cols, r_and_w_mem_latency_cols)
    row_ntt_latency = fourth_step_prefetch_latency + 2*r_or_w_mem_latency_rows + 2*max(r_or_w_mem_latency_rows, compute_latency_rows) + (row_steps- 4)*max(compute_latency_rows, r_and_w_mem_latency_rows)

    return col_ntt_latency + row_ntt_latency


def characterize_poly(polynomial, debug=False):
    unique_count, reused_count = analyze_polynomial(polynomial)

    if debug:
        print(f"Polynomial: {polynomial}")
        print(f"Unique entries: {unique_count}")
        print(f"Reused entries: {reused_count}")

    num_adds, num_products = count_operations(polynomial)

    return unique_count, reused_count, num_adds, num_products

# elementwise time is time to stream values on chip
def estimate_elementwise_latency(poly_features, mat_dims, butterflies_per_pe, num_pes, bit_width, available_bw, freq, modadd_latency=1, modmul_latency=20, bmax=5):

    max_read_rate = calc_rate(bit_width, available_bw, freq)
    
    # we have 2*(number of butterflies) read ports but only (number of butterfly) modmuls
    # so best we can do is read those many elements per cycle to feed into the modmuls
    num_read_ports = butterflies_per_pe
    
    
    # this is also equivalent to num_rows, num_cols = mat_dims
    col_words, row_words = mat_dims

    # read row-wise, this latency is for num_pes in parallel
    fetch_cycles = get_read_latency(row_words*num_pes, num_read_ports*num_pes, max_read_rate)
    # print(row_words)
    # print(max_read_rate)
    # print(fetch_cycles)
    num_unique_mles, num_reused_mles, num_adds, num_products = poly_features
    compute_cycles = modadd_latency*num_adds + modmul_latency*num_products

    # get the number of unique MLEs, and get the number of MLEs that are reused
    assert num_reused_mles <= bmax - 2

    num_groups = col_words/num_pes
    fetch_cycles *= num_groups
    fetch_cycles *= num_unique_mles

    # compute is pipelined with fetch. this is an approximation
    # TODO: verify this approximation
    elementwise_time = fetch_cycles + compute_cycles*num_groups

    return elementwise_time


def simulate_mini_ntt_onchip(ntt_len, num_butterflies, modadd_latency=1, modmul_latency=20, bit_width=256, sparse_fraction=0):
    """
    Simulate a mini NTT that fits on-chip. This is one core fitting one polynomial.
    Args:
        ntt_len: Length of the NTT (e.g., 1024)
        num_butterflies: Number of butterflies (unroll factor)
        modadd_latency: Latency of a modular addition (default 1)
        modmul_latency: Latency of a modular multiplication (default 20)
        bit_width: Bit width for memory calculation (default 256)
    Returns:
        A dict with total_cycles, total_modmuls, total_modadds, total_num_words
    """
    num_stages = int(math.log2(ntt_len))
    max_butterflies_per_stage = ntt_len // 2
    dense_fraction = 1 - sparse_fraction
    assert dense_fraction > 0, "dense_fraction must be > 0"
    sparse_amplified_factor = int(1/dense_fraction)

    # The maximum allowed butterflies is (ntt_len//2) * num_stages
    max_total_butterflies = max_butterflies_per_stage * num_stages
    if num_butterflies > max_total_butterflies:
        num_butterflies = max_total_butterflies

    # Case 1: one stage per round
    if num_butterflies <= max_butterflies_per_stage:
        output_scaled = False
        # total_cycles = get_compute_latency(
        #     ntt_len, num_butterflies, modmul_latency + modadd_latency, modadd_latency, output_scaled=output_scaled
        # )
        total_cycles = get_compute_latency_with_sparsity(
            ntt_len, num_butterflies, modmul_latency + modadd_latency, modadd_latency, sparse_amplified_factor, output_scaled
        )
    # # Case 2: k stages per round, k is integer, num_butterflies = k * (ntt_len//2)
    else:
        # Not a valid case: skip or raise error
        raise ValueError(
            f"num_butterflies ({num_butterflies}) must be <= ntt_len//2 or an integer multiple of ntt_len//2"
        )

    # Number of modmuls and modadds is equal to the number of butterflies (units), not the total number of operations
    total_modmuls = num_butterflies
    total_modadds = num_butterflies * 2  # modadd, modsub

    # On-chip memory: ping-pong buffer (ntt_len * 2) + local twiddle words (ntt_len / 2)
    ping_pong_buffer_words = ntt_len * 2
    local_twiddle_words = 0  # ntt_len / 2. Shared across cores
    total_num_words = ping_pong_buffer_words + local_twiddle_words

    return {
        "total_cycles": total_cycles,
        "total_modmuls": total_modmuls,
        "total_modadds": total_modadds,
        "total_num_words": total_num_words
    }


def get_step_radix_gate_degree(gate_degree):
    """
    Get the step radix of breaking a larger (deg-1) to two 2's power sum.
    Then we can run NTT on each size.

    - Example:
        - gate_degree = 1: returns [0]
        - gate_degree = 2: returns [1] (1*N NTT)
        - gate_degree = 3: returns [2] (2*N NTT)
        - gate_degree = 4: returns [2, 1] (2*N+1*N NTT)
        - gate_degree = 5: returns [4] (4*N NTT)
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


def run_miniNTT_fit_onchip(target_n:int, polynomial, modadd_latency=1, modmul_latency=20, bit_width=256, consider_sparsity=True):
    """
    iNTT, NTT, iNTT.
    For a given polynomial, run num_unique_mles miniNTT cores in parallel for NTT of length (d-1)*N,
    sweeping number of butterflies from 1 to the largest possible.
    Args:
        target_n: Problem size exponent (e.g., 16 for 2^16)
        polynomial: The polynomial (list of lists) to analyze. [["q1", "q2"], ["q3"]]
        modadd_latency: Latency of a modular addition (default 1)
        modmul_latency: Latency of a modular multiplication (default 20)
        bit_width: Bit width for memory calculation (default 256)
    Returns:
        A dict: key=(num_butterflies), value=dict of cost for that config
    """
    # Analyze polynomial to get number of unique MLEs and max degree
    poly_features = characterize_poly(polynomial)
    num_unique_mles = poly_features[0]
    num_reused_mles = poly_features[1]
    num_adds_poly = poly_features[2]
    num_products_poly = poly_features[3]

    input_ntt_len = 2 ** target_n
    max_degree = max(len(term) for term in polynomial)
    N = 2 ** target_n
    ntt_len = (max_degree - 1) * N

    # Sweep number of butterflies from 1 to ntt_len//2 (radix-2)
    max_butterflies = max(1, ntt_len // 2)
    num_stages = int(math.log2(ntt_len))
    # Sweep number of butterflies: dense for 64 to 2048 (step 64), otherwise powers of two
    sweep_butterflies = [2 ** i for i in range(int(math.log2(max_butterflies)) + 1) if 2 ** i <= max_butterflies]
    if 1 not in sweep_butterflies:
        sweep_butterflies = [1] + sweep_butterflies

    # Add denser sweep between 64 and 2048 (inclusive), step 256
    dense_min = 64
    dense_max = 4096
    dense_butterflies = [v for v in range(dense_min, min(dense_max, max_butterflies) + 1, 256)]
    sweep_butterflies = sorted(set(sweep_butterflies + dense_butterflies))
    # # Add extra sweep points: k * max_butterflies, for k = 2..num_stages
    # for k in range(2, num_stages + 1):
    #     val = k * max_butterflies
    #     if val <= num_stages * max_butterflies and val not in sweep_butterflies:
    #         sweep_butterflies.append(val)
    sweep_butterflies = sorted(sweep_butterflies)

    results = {}
    for num_butterflies in sweep_butterflies:
        # 1. Input iNTTs
        input_ntt_result = simulate_mini_ntt_onchip(
            input_ntt_len, min(num_butterflies, input_ntt_len // 2), modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
            sparse_fraction=0
        )
        total_cycles = input_ntt_result["total_cycles"]

        # 2. Bigger NTTs
        step_size = get_step_radix_gate_degree(max_degree)
        mini_ntt_result = simulate_mini_ntt_onchip(
            step_size[0] * ntt_len, min(num_butterflies, step_size[0] * ntt_len // 2),
            modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
            sparse_fraction=1 - 1/step_size[0] if consider_sparsity else 0,
        )
        if len(step_size) > 1:
            mini_ntt_result_b = simulate_mini_ntt_onchip(
                step_size[1] * ntt_len, min(num_butterflies, step_size[1] * ntt_len // 2),
                modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
                sparse_fraction=1 - 1/step_size[1] if consider_sparsity else 0,
            )
            mini_ntt_result["total_cycles"] += mini_ntt_result_b["total_cycles"]

        # Parallel execution: latency is that of one core, resources scale with num_unique_mles
        total_cycles += mini_ntt_result["total_cycles"]
        total_modmuls = mini_ntt_result["total_modmuls"] * num_unique_mles
        total_modadds = mini_ntt_result["total_modadds"] * num_unique_mles
        total_num_words = mini_ntt_result["total_num_words"] * num_unique_mles + ntt_len / 2  # pingpong, +local_twiddle_words

        # 3. q iNTT
        q_intt_result = simulate_mini_ntt_onchip(
            step_size[0] * ntt_len, min(num_butterflies, step_size[0] * ntt_len // 2),
            modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
            sparse_fraction=0,
        )
        if len(step_size) > 1:
            q_intt_result_b = simulate_mini_ntt_onchip(
                step_size[1] * ntt_len, min(num_butterflies, step_size[1] * ntt_len // 2),
                modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
                sparse_fraction=0,
            )
            q_intt_result["total_cycles"] += q_intt_result_b["total_cycles"]
        total_cycles += q_intt_result["total_cycles"]

        results[num_butterflies] = {
            "ntt_len": ntt_len,
            "num_unique_mles": num_unique_mles,
            "total_cycles": total_cycles,
            "total_modmuls": total_modmuls,
            "total_modadds": total_modadds,
            "total_num_words": total_num_words,
        }
    return results


def run_miniNTT_partial_onchip(target_n: int, polynomial, target_bw: int, modadd_latency=1, modmul_latency=20, bit_width=256, freq=1e9, consider_sparsity=True):
    """
    iNTT, NTT, iNTT.
    For a given polynomial, run num_unique_mles miniNTT cores in parallel for NTT of length (d-1)*N,
    sweeping number of butterflies from 1 to the largest possible, considering bandwidth.
    Args:
        target_n: Problem size exponent (e.g., 16 for 2^16)
        polynomial: The polynomial (list of lists) to analyze. [["q1", "q2"], ["q3"]]
        target_bw: Available bandwidth in GB/s
        modadd_latency: Latency of a modular addition (default 1)
        modmul_latency: Latency of a modular multiplication (default 20)
        bit_width: Bit width for memory calculation (default 256)
        freq: Frequency in Hz (default 1e9)
    Returns:
        A dict: key=(num_butterflies), value=dict of cost for that config
    """
    poly_features = characterize_poly(polynomial)
    num_unique_mles = poly_features[0]
    max_degree = max(len(term) for term in polynomial)
    N = 2 ** target_n
    ntt_len = (max_degree - 1) * N
    input_ntt_len = 2 ** target_n

    # Sweep number of butterflies from 1 to ntt_len//2 (radix-2)
    max_butterflies = max(1, ntt_len // 2)
    sweep_butterflies = [2 ** i for i in range(int(math.log2(max_butterflies)) + 1) if 2 ** i <= max_butterflies]
    if 1 not in sweep_butterflies:
        sweep_butterflies = [1] + sweep_butterflies

    # Add denser sweep between 64 and 8192 (inclusive), step 1024
    dense_min = 64
    dense_max = 8192
    dense_butterflies = [v for v in range(dense_min, min(dense_max, max_butterflies) + 1, 1024)]
    sweep_butterflies = sorted(set(sweep_butterflies + dense_butterflies))

    # Calculate cycles to fetch input_ntt_len elements, each of bit_width bits, over target_bw GB/s at freq Hz
    bw_words_per_sec = (target_bw * (1 << 30) * 8) // bit_width  # GB/s to word/s
    fetch_cycles = math.ceil(input_ntt_len / (bw_words_per_sec / freq))

    results = {}
    for num_butterflies in sweep_butterflies:
        # 1. Input iNTT (assume full input needs to be streamed in)
        input_cycles = simulate_mini_ntt_onchip(
            input_ntt_len, min(num_butterflies, input_ntt_len // 2),
            modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width
        )["total_cycles"]

        # 2. Bigger NTTs (high degree)
        step_size = get_step_radix_gate_degree(max_degree)
        mini_ntt_result = simulate_mini_ntt_onchip(
            step_size[0] * ntt_len, min(num_butterflies, step_size[0] * ntt_len // 2),
            modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
            sparse_fraction=1 - 1/step_size[0] if consider_sparsity else 0,
        )
        if len(step_size) > 1:
            mini_ntt_result_b = simulate_mini_ntt_onchip(
                step_size[1] * ntt_len, min(num_butterflies, step_size[1] * ntt_len // 2),
                modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
                sparse_fraction=1 - 1/step_size[1] if consider_sparsity else 0,
            )
            mini_ntt_result["total_cycles"] += mini_ntt_result_b["total_cycles"]
        
        input_mini_ntt_cycles = mini_ntt_result["total_cycles"] + input_cycles
        if fetch_cycles <= input_mini_ntt_cycles / 2:  # prefetch and WB are overlapped by compute
            total_cycles = fetch_cycles + input_mini_ntt_cycles * num_unique_mles
        elif fetch_cycles < input_mini_ntt_cycles:
            total_cycles = fetch_cycles * 2 * (num_unique_mles - 1) + input_mini_ntt_cycles + (input_mini_ntt_cycles - fetch_cycles)
        else:
            total_cycles = fetch_cycles * 2 * (num_unique_mles - 1) + fetch_cycles

        total_modmuls = mini_ntt_result["total_modmuls"]
        total_modadds = mini_ntt_result["total_modadds"]
        total_num_words = mini_ntt_result["total_num_words"] * 2 + ntt_len / 2  # pingpong+1double bf+1result, +local_twiddle_words

        # 3. q iNTT
        q_intt_result = simulate_mini_ntt_onchip(
            step_size[0] * ntt_len, min(num_butterflies, step_size[0] * ntt_len // 2),
            modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
            sparse_fraction=0,
        )
        if len(step_size) > 1:
            q_intt_result_b = simulate_mini_ntt_onchip(
                step_size[1] * ntt_len, min(num_butterflies, step_size[1] * ntt_len // 2),
                modadd_latency=modadd_latency, modmul_latency=modmul_latency, bit_width=bit_width,
                sparse_fraction=0,
            )
            q_intt_result["total_cycles"] += q_intt_result_b["total_cycles"]
        total_cycles += q_intt_result["total_cycles"]

        results[num_butterflies] = {
            "ntt_len": ntt_len,
            "num_unique_mles": num_unique_mles,
            "total_cycles": total_cycles,
            "total_modmuls": total_modmuls,
            "total_modadds": total_modadds,
            "total_num_words": total_num_words,
        }
    return results


def run_fit_onchip(target_n=None, target_bw=None, progress_print=False, polynomial=None, save_pkl=True, unroll_factors_pow=None):

    random.seed(0)

    if polynomial is None:
        polynomial = [["f"]]

    poly_features = characterize_poly(polynomial)
    num_unique_mles = poly_features[0]

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
    
    if unroll_factors_pow is None:
        unroll_factors_pow = range(0, math.ceil(target_n / 2)) if target_n is not None else range(0, 13)
    else:  # need this to match result keys
        unroll_factors_pow = range(0, unroll_factors_pow)
    unroll_factors = [2**i for i in unroll_factors_pow]
    
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

                    # num_col_words = M*pe_amt
                    # num_row_words = N*pe_amt
                    # total_bfs = U*pe_amt

                    r_or_w_mem_latency_cols, r_or_w_mem_latency_rows, r_and_w_mem_latency_cols, r_and_w_mem_latency_rows, \
                        compute_latency_cols, compute_latency_rows, first_step_prefetch_latency, fourth_step_prefetch_latency = \
                        get_latencies_and_rates(M, N, U, pe_amt, bit_width, available_bw, freq, modadd_latency, modmul_latency, bf_latency)
                    # dont have to fetch global twiddles for row-wise NTTs, only omegas_N

                    if progress_print:
                        print("Simulating four-step NTT when mini NTT fits on-chip...")

                    arch_1 = ArchitectureSimulator(omegas_M, modulus, r_or_w_mem_latency_cols, r_and_w_mem_latency_cols, compute_latency_cols, prefetch_latency=first_step_prefetch_latency, skip_compute=skip_compute)
                    temp_matrix, cycle_time_1 = simulate_4step_all_onchip(arch_1, pe_amt, matrix, omega_L, modulus, output_scale=True, skip_compute=skip_compute)

                    temp_matrix_T = transpose(temp_matrix) if not skip_compute else transpose(matrix)
                    arch_2 = ArchitectureSimulator(omegas_N, modulus, r_or_w_mem_latency_rows, r_and_w_mem_latency_rows, compute_latency_rows, prefetch_latency=fourth_step_prefetch_latency, skip_compute=skip_compute)
                    final_matrix, cycle_time_2 = simulate_4step_all_onchip(arch_2, pe_amt, temp_matrix_T, omega_L, modulus, output_scale=False, skip_compute=skip_compute)

                    single_ntt_cycles = cycle_time_1 + cycle_time_2

                    # Calculate expected latency for this configuration
                    expected_cycles = expected_latency(M, N, pe_amt, 
                                                      (first_step_prefetch_latency, fourth_step_prefetch_latency),
                                                      (r_or_w_mem_latency_cols, r_or_w_mem_latency_rows, r_and_w_mem_latency_cols, r_and_w_mem_latency_rows),
                                                      (compute_latency_cols, compute_latency_rows))

                    # Check for discrepancies
                    if single_ntt_cycles != expected_cycles:
                        print(f"Expected latency (cycles): {expected_cycles}")
                        print(f"Actual latency (cycles): {single_ntt_cycles}")
                        print("Mismatch between expected and actual latency!")
                        exit()

                    transposed_mat_dims = (N, M)

                    all_ntt_cycles = single_ntt_cycles*num_unique_mles
                    elementwise_cycles = estimate_elementwise_latency(poly_features, transposed_mat_dims, U, pe_amt, bit_width, available_bw, freq, modadd_latency=modadd_latency, modmul_latency=modmul_latency, bmax=5)

                    total_cycles = all_ntt_cycles + elementwise_cycles

                    # 5 buffers in each PE, each of length M
                    ping_pong_double_buffer_words = M*4*pe_amt

                    local_twiddle_words = M / 2     # shared among all PEs
                    global_scale_twiddle_words = M  # shared among all PEs
                    global_twiddle_words = M * pe_amt  # each PE computes its own global twiddle column

                    total_num_words = ping_pong_double_buffer_words + local_twiddle_words + global_scale_twiddle_words + global_twiddle_words

                    total_modmuls = U*pe_amt
                    total_modadds = U*2*pe_amt

                    results[(n, available_bw, U, pe_amt)] = {
                        "total_cycles": total_cycles,
                        "single_ntt_cycles": single_ntt_cycles,
                        "all_ntt_cycles": all_ntt_cycles,
                        "elementwise_cycles": elementwise_cycles,
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
        print(f"n={n}, bw={available_bw}, U={U}, pe_amt={pe_amt} -> "
              f"total_cycles: {value['total_cycles']}, "
              f"single_ntt_cycles: {value['single_ntt_cycles']}, "
              f"all_ntt_cycles: {value['all_ntt_cycles']}, "
              f"elementwise_cycles: {value['elementwise_cycles']}, "
              f"total_modmuls: {value['total_modmuls']}, "
              f"total_num_words: {value['total_num_words']}")

    # Save results to pickle file
    output_dir = "pickle_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on arguments
    if target_n is not None and target_bw is not None:
        if save_pkl:
            filename = f"results_n{target_n}_bw{target_bw}.pkl"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {filepath}")

    return results


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
    r_or_w_mem_latency_cols = get_read_latency(M, 2*U, actual_read_rate)
    r_or_w_mem_latency_rows = get_read_latency(N, 2*U, actual_read_rate)
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

    arch_1 = ArchitectureSimulator(None, modulus, r_or_w_mem_latency_cols, compute_latency_cols, r_or_w_mem_latency_cols)
    temp_matrix, cycle_time_1 = simulate_4step_notall_onchip(arch_1, matrix, omegas_M_matrix, omega_L, modulus, output_scale=True)

    temp_matrix_T = transpose(temp_matrix)
    arch_2 = ArchitectureSimulator(None, modulus, r_or_w_mem_latency_rows, compute_latency_rows, r_or_w_mem_latency_rows)
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


def run_fourstep_fit_on_chip(target_n, sparse_fraction, target_bw, polynomial, unroll_factors_pow=None, progress_print=False, single_config=False, single_config_params=None):

    skip_compute = True

    if polynomial is None:
        # polynomial = [["f"]]
        exit("must provide polynomial")

    poly_features = characterize_poly(polynomial)
    num_unique_mles = poly_features[0]

    bit_width = 256
    freq = 1e9

    modadd_latency = 1
    modmul_latency = 20
    bf_latency = modmul_latency + modadd_latency

    if unroll_factors_pow is None:
        unroll_factors_pow = range(0, math.ceil(target_n / 2)) if target_n is not None else range(0, 13)
    else:
        unroll_factors_pow = range(0, unroll_factors_pow)
    unroll_factors = [2**i for i in unroll_factors_pow]

    pe_counts = [1, 2, 4, 8, 16]  # [1, 2, 4, 8, 16, 32, 64]

    # fixed for a given n
    M, N, omegas_L, omega_L, omegas_N, omega_N, omegas_M, omega_M, modulus = get_twiddle_factors(target_n, bit_width, progress_print)

    # Generate random data and reshape it.
    start_time = time.time()
    # data = [random.randint(0, modulus - 2) for _ in range(1<<target_n)]
    data = np.zeros(1 << target_n, dtype=np.int64)
    
    end_time = time.time()
    print(f"Data generation time: {end_time - start_time:.2f} seconds") if progress_print else None

    # Reshape data into M x N matrix (list of lists)
    # matrix = [data[i*N:(i+1)*N] for i in range(M)]
    matrix = np.reshape(data, (M, N))

    # Initialize results dictionary
    results = {}

    if single_config:
        target_U, target_pe_amt = single_config_params
    

    for U in tqdm(unroll_factors, desc="Unroll factors (U)"):
        if single_config and U != target_U:
            continue
        
        for pe_amt in pe_counts:
            if single_config and pe_amt != target_pe_amt:
                continue
            # num_col_words = M*pe_amt
            # num_row_words = N*pe_amt
            # total_bfs = U*pe_amt

            start_time = time.time()
            rw_latencies, compute_latencies, prefetch_latencies = \
                get_latencies_and_rates_with_sparsity(M, N, U, pe_amt, bit_width, target_bw, freq, modadd_latency, modmul_latency, bf_latency, debug=progress_print, sparse_fraction=sparse_fraction)
            # dont have to fetch global twiddles for row-wise NTTs, only omegas_N

            end_time = time.time()
            print(f"get latency time for U={U}, pe_amt={pe_amt}: {end_time - start_time:.2f} seconds") if progress_print else None

            read_mem_latency_cols, write_mem_latency_cols, r_and_w_mem_latency_cols, read_mem_latency_rows, write_mem_latency_rows, r_and_w_mem_latency_rows = rw_latencies
            compute_latency_cols, compute_latency_rows = compute_latencies
            first_step_prefetch_latency, fourth_step_prefetch_latency = prefetch_latencies

            cols_sparse_latencies = read_mem_latency_cols, write_mem_latency_cols, r_and_w_mem_latency_cols
            rows_sparse_latencies = read_mem_latency_rows, write_mem_latency_rows, r_and_w_mem_latency_rows

            if progress_print:  # Removed progress_print parameter
                print("Simulating four-step NTT when mini NTT fits on-chip...")

            start_time = time.time()

            arch_1 = ArchitectureSimulator(omegas_M, modulus, None, None, compute_latency_cols, prefetch_latency=first_step_prefetch_latency, skip_compute=skip_compute, sparsity=True, sparse_latencies=cols_sparse_latencies)
            arch_1.set_debug(progress_print)
            temp_matrix, cycle_time_1 = simulate_4step_all_onchip(arch_1, pe_amt, matrix, omega_L, modulus, output_scale=True, skip_compute=skip_compute)

            end_time = time.time()
            print(f"Simulation time for arch_1: {end_time - start_time:.2f} seconds") if progress_print else None

            start_time = time.time()

            temp_matrix_T = transpose(temp_matrix) if not skip_compute else transpose(matrix)
            arch_2 = ArchitectureSimulator(omegas_N, modulus, None, None, compute_latency_rows, prefetch_latency=fourth_step_prefetch_latency, skip_compute=skip_compute, sparsity=True, sparse_latencies=rows_sparse_latencies)
            arch_2.set_debug(progress_print)
            final_matrix, cycle_time_2 = simulate_4step_all_onchip(arch_2, pe_amt, temp_matrix_T, omega_L, modulus, output_scale=False, skip_compute=skip_compute)

            end_time = time.time()
            print(f"Simulation time for arch_2: {end_time - start_time:.2f} seconds") if progress_print else None

            single_ntt_cycles = cycle_time_1 + cycle_time_2

            transposed_mat_dims = (N, M)

            all_ntt_cycles = single_ntt_cycles*num_unique_mles
            elementwise_cycles = estimate_elementwise_latency(poly_features, transposed_mat_dims, U, pe_amt, bit_width, target_bw, freq, modadd_latency=modadd_latency, modmul_latency=modmul_latency, bmax=5)

            total_cycles = all_ntt_cycles + elementwise_cycles

            # 5 buffers in each PE, each of length M
            ping_pong_double_buffer_words = M*3*pe_amt

            local_twiddle_words = M / 2     # shared among all PEs
            global_scale_twiddle_words = M  # shared among all PEs
            global_twiddle_words = M * pe_amt  # each PE computes its own global twiddle column

            total_num_words = ping_pong_double_buffer_words + local_twiddle_words + global_scale_twiddle_words + global_twiddle_words

            total_modmuls = U*pe_amt
            total_modadds = U*2*pe_amt

            data_to_store = {
                "total_cycles": total_cycles,
                "single_ntt_cycles": single_ntt_cycles,
                "col_ntt_cycles": cycle_time_1,
                "row_ntt_cycles": cycle_time_2,
                "all_ntt_cycles": all_ntt_cycles,
                "elementwise_cycles": elementwise_cycles,
                "total_modmuls": total_modmuls,
                "total_modadds": total_modadds,
                "total_num_words": total_num_words
            }
            if single_config:
                print(f"Config: n={target_n}, bw={target_bw}, U={U}, pe_amt={pe_amt}")
                print("Single configuration results:")
                for key, value in data_to_store.items():
                    print(f"  {key}: {value}")

            results[(target_n, target_bw, U, pe_amt)] = data_to_store


    return results


def run_one_config_fourstep_fit_onchip(target_n=20, target_bw=64, polynomial=[["f"]], U_in=4, pe_amt_in=8):
    """
    Test the run_fourstep_fit_on_chip function with a single configuration
    to verify its functionality and output.
    """
    
    random.seed(0)
    
    polynomial = [["g", "h", "s"], ["o"]]
    poly_features = characterize_poly(polynomial)
    num_unique_mles, num_reused_mles, num_adds, num_products = poly_features
    
    print("Testing fourstep fit on chip function...")
    print(f"Polynomial: {polynomial}")
    print(f"Polynomial features: unique_mles={num_unique_mles}, reused_mles={num_reused_mles}, adds={num_adds}, products={num_products}")
    
    # Test parameters
    sparsity_list = [0, 1/2, 3/4, 7/8]  # Test sparsity values: 0%, 50%, 75%, 87.5%
    unroll_factors_pow = 6  # Test unroll factors up to 2^6 = 64

    single_config = True
    single_config_params = (U_in, pe_amt_in)  # Test with U=8 and pe_amt=4

    print(f"Test configuration: n={target_n}, bandwidth={target_bw} GB/s")
    print(f"Testing unroll factors up to 2^{unroll_factors_pow-1} = {2**(unroll_factors_pow-1)}")
    print()

    for sparsity in sparsity_list:
        print(f"Testing with sparsity fraction: {sparsity*100:.0f}%")
        results = run_fourstep_fit_on_chip(
            target_n=target_n,
            sparse_fraction=sparsity,
            target_bw=target_bw,
            polynomial=polynomial,
            unroll_factors_pow=unroll_factors_pow,
            single_config=single_config,
            single_config_params=single_config_params
        )
        
        if results is None:
            print("Function completed successfully (no return value)")
        else:
            print(f"Function returned results with {len(results)} configurations")
            
            # Display a sample of results if available
            sample_configs = list(results.keys())[:3]  # Show first 3 configurations
            for config in sample_configs:
                n, bw, U, pe_amt = config
                result = results[config]
                print(f"Config (n={n}, bw={bw}, U={U}, pe_amt={pe_amt}):")
                print(f"  Total cycles: {result['total_cycles']}")
                print(f"  Single NTT cycles: {result['single_ntt_cycles']}")
                print(f"  Memory words: {result['total_num_words']}")
                print()
        
        print("✓ Fourstep fit on chip test completed successfully!\n")
        

def run_one_config_fit_onchip(target_n=20, target_bw=64, polynomial=[["f"]], U_in=4, pe_amt_in=8):

    random.seed(0)

    polynomial = [["g", "h", "s"], ["o"]]
    poly_features = characterize_poly(polynomial)
    num_unique_mles, num_reused_mles, num_adds, num_products = poly_features

    # sweep parameters: n, bandwidth, U, PEs

    bit_width = 256
    available_bw = target_bw
    freq = 1e9

    modadd_latency = 1
    modmul_latency = 20
    bf_latency = modmul_latency + modadd_latency

    check_correctness = False
    skip_compute = True

    # Dictionary to store results indexed by (n, bandwidth, U, pe_amt)
    results = {}

    n = target_n
    U = U_in
    pe_amt = pe_amt_in

    # fixed for a given n
    M, N, omegas_L, omega_L, omegas_N, omega_N, omegas_M, omega_M, modulus = get_twiddle_factors(n, bit_width)

    # Generate random data and reshape it.
    # data = [random.randint(0, modulus - 2) for _ in range(1<<n)]
    data = np.zeros(1 << target_n, dtype=np.int64)
    
    # Reshape data into M x N matrix (list of lists)
    # matrix = [data[i*N:(i+1)*N] for i in range(M)]
    matrix = np.reshape(data, (M, N))
    
    # num_col_words = M*pe_amt
    # num_row_words = N*pe_amt
    # total_bfs = U*pe_amt

    r_or_w_mem_latency_cols, r_or_w_mem_latency_rows, r_and_w_mem_latency_cols, r_and_w_mem_latency_rows, \
        compute_latency_cols, compute_latency_rows, first_step_prefetch_latency, fourth_step_prefetch_latency = \
        get_latencies_and_rates(M, N, U, pe_amt, bit_width, available_bw, freq, modadd_latency, modmul_latency, bf_latency, debug=True)

    print()
    print("Simulating four-step NTT when mini NTT fits on-chip...")

    arch_1 = ArchitectureSimulator(omegas_M, modulus, r_or_w_mem_latency_cols, r_and_w_mem_latency_cols, compute_latency_cols, first_step_prefetch_latency, skip_compute=skip_compute)
    arch_1.set_debug(True)  # Enable debug output
    temp_matrix, cycle_time_1 = simulate_4step_all_onchip(arch_1, pe_amt, matrix, omega_L, modulus, output_scale=True, skip_compute=skip_compute)

    print("#############################")
    print("now transpose and fourth step")
    print("#############################")
    print()

    temp_matrix_T = transpose(temp_matrix) if not skip_compute else transpose(matrix)
    arch_2 = ArchitectureSimulator(omegas_N, modulus, r_or_w_mem_latency_rows, r_and_w_mem_latency_rows, compute_latency_rows, fourth_step_prefetch_latency, skip_compute=skip_compute)
    arch_2.set_debug(True)  # Enable debug output
    final_matrix, cycle_time_2 = simulate_4step_all_onchip(arch_2, pe_amt, temp_matrix_T, omega_L, modulus, output_scale=False, skip_compute=skip_compute)

    single_ntt_cycles = cycle_time_1 + cycle_time_2

    # Calculate expected latency for this configuration
    expected_cycles = expected_latency(M, N, pe_amt, 
                                      (first_step_prefetch_latency, fourth_step_prefetch_latency),
                                      (r_or_w_mem_latency_cols, r_or_w_mem_latency_rows, r_and_w_mem_latency_cols, r_and_w_mem_latency_rows),
                                      (compute_latency_cols, compute_latency_rows))

    print(f"Expected latency (cycles): {expected_cycles}")
    print(f"Actual latency (cycles): {single_ntt_cycles}")

    # Check for discrepancies
    if single_ntt_cycles != expected_cycles:
        print("Mismatch between expected and actual latency!")
        exit()
   
    
    transposed_mat_dims = (N, M)

    all_ntt_cycles = single_ntt_cycles*num_unique_mles
    elementwise_cycles = estimate_elementwise_latency(poly_features, transposed_mat_dims, U, pe_amt, bit_width, available_bw, freq, modadd_latency=modadd_latency, modmul_latency=modmul_latency, bmax=5)

    total_cycles = all_ntt_cycles + elementwise_cycles


    ping_pong_double_buffer_words = M*4*pe_amt
    local_twiddle_words = M / 2     # shared among all PEs
    global_scale_twiddle_words = M  # shared among all PEs
    global_twiddle_words = M * pe_amt  # each PE computes its own global twiddle column


    total_num_words = ping_pong_double_buffer_words + local_twiddle_words + global_scale_twiddle_words + global_twiddle_words
    total_modmuls = U*pe_amt
    total_modadds = U*2*pe_amt


    results[(n, available_bw, U, pe_amt)] = {
        "total_cycles": total_cycles,
        "single_ntt_cycles": single_ntt_cycles,
        "all_ntt_cycles": all_ntt_cycles,
        "elementwise_cycles": elementwise_cycles,
        "total_modmuls": total_modmuls,
        "total_modadds": total_modadds,
        "total_num_words": total_num_words
    }

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
        print(f"n={n}, bw={available_bw}, U={U}, pe_amt={pe_amt} -> "
            f"total_cycles: {value['total_cycles']}, "
            f"single_ntt_cycles: {value['single_ntt_cycles']}, "
            f"all_ntt_cycles: {value['all_ntt_cycles']}, "
            f"elementwise_cycles: {value['elementwise_cycles']}, "
            f"total_modmuls: {value['total_modmuls']}, "
            f"total_num_words: {value['total_num_words']}")

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
    parser.add_argument('--mode', choices=['sweep', 'test', 'one_config', 'fourstep_test', 'plot', 'print'], default='sweep',
                        help='Mode to run: sweep (parameter sweep), test (simple test), one_config (single configuration), fourstep_test (test fourstep fit on chip), plot (Pareto analysis), print (print results table)')
    parser.add_argument('--multi-bw', action='store_true', 
                        help='When in plot mode with --n specified, plot multiple bandwidths on the same chart')
    
    args = parser.parse_args()
    
    polynomial = [["g", "h", "s"], ["o"]]
    target_n = 23
    target_bw = 1e99
    U_in=16
    pe_amt_in=1

    if args.mode == 'test':
        print("Running simple test...")
        run_simple_test()
    elif args.mode == 'one_config':
        print("Running single configuration test...")
        run_one_config_fit_onchip(target_n=target_n, target_bw=target_bw, polynomial=polynomial, U_in=U_in, pe_amt_in=pe_amt_in)
    elif args.mode == 'fourstep_test':
        print("Running fourstep fit on chip test...")
        run_one_config_fourstep_fit_onchip(target_n=target_n, target_bw=target_bw, polynomial=polynomial, U_in=U_in, pe_amt_in=pe_amt_in)
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
            run_fit_onchip(target_n=args.n, target_bw=args.bw, polynomial=polynomial)
        elif args.n is not None:
            print(f"Running sweep for n={args.n}, all bandwidths")
            run_fit_onchip(target_n=args.n, polynomial=polynomial)
        elif args.bw is not None:
            print(f"Running sweep for bw={args.bw}, all problem sizes")
            run_fit_onchip(target_bw=args.bw, polynomial=polynomial)
        else:
            print("Running full parameter sweep...")
            run_fit_onchip(polynomial=polynomial)
