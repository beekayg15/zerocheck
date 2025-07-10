import random
from ntt_func_sim import ArchitectureSimulator
from ntt import ntt, ntt_dit_rn, ntt_dif_nr, bit_rev_shuffle
from ntt_utility import *
from util import calc_rate
from fourstep_ntt_perf_models import get_compute_latency


def simulate_matrix_columns(arch, mat, global_omega, modulus, output_scale=True, tags_only=True):
    num_rows = len(mat)
    num_cols = len(mat[0])
    total_length = num_rows * num_cols
    L = total_length   

    output_matrix = [[0] * num_cols for _ in range(num_rows)]
    for idx in range(num_cols + 2):
        if idx < num_cols:
            col = [mat[row][idx] for row in range(num_rows)]
            tag = idx
        else:
            col = None
            tag = None
        arch.step(col, tag, tags_only=tags_only)
        out_data, out_tag = arch.out
        if idx >= 2:
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

def run():

    U = 4
    bit_width = 256
    available_bw = 1024
    freq = 1e9    


    modadd_latency = 3
    modmul_latency = 20
    bf_latency = modmul_latency + modadd_latency

    n = 20
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
    mem_latency_cols = get_read_latency(M, U, max_read_rate)
    mem_latency_rows = get_read_latency(N, U, max_read_rate)
    compute_latency_cols = get_compute_latency(M, U, bf_latency, modadd_latency, output_scaled=True)
    compute_latency_rows = get_compute_latency(N, U, bf_latency, modadd_latency, output_scaled=False)

    # Four step NTT: L = M*N, M > N

    # Generate random data and reshape it.
    data = [random.randint(0, modulus - 2) for _ in range(L)]
    # print(f"original data        = {list(data)}")

    # Reshape data into M x N matrix (list of lists)
    matrix = [data[i*N:(i+1)*N] for i in range(M)]
    arch_1 = ArchitectureSimulator(omegas_M, modulus, mem_latency_cols, compute_latency_cols)
    temp_matrix, cycle_time_1 = simulate_matrix_columns(arch_1, matrix, omega_L, modulus, output_scale=True)

    temp_matrix_T = transpose(temp_matrix)
    arch_2 = ArchitectureSimulator(omegas_N, modulus, mem_latency_rows, compute_latency_rows)
    final_matrix, cycle_time_2 = simulate_matrix_columns(arch_2, temp_matrix_T, omega_L, modulus, output_scale=False)

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

if __name__ == "__main__":
    run()
