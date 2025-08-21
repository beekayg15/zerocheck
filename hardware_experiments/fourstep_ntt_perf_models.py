import math
from util import *

def closest_powers_of_two(exponent):
    # If the exponent is even, it can be exactly divided by two
    if exponent % 2 == 0:
        return 2**(exponent // 2), 2**(exponent // 2)
    # If the exponent is odd, we return the two closest powers of 2
    else:
        return 2**((exponent // 2) + 1), 2**(exponent // 2)


# this function calculates the latency of a single four-step NTT. Simple estimate with no masking is assumed.
# Specifically, prefetching weights and fetching initial column of the matrix is done serially, and after all of the column NTTs
# the subsequent weight prefetch and initial transposed column fetch is also done serially.
# Estimating the latency with full masking will require more complex logic (based on BW)
def single_fourstep_ntt(M, N, num_pes, mem_latency_for_rows, compute_latency_for_rows, mem_latency_for_cols, compute_latency_for_cols):

    # num_pes is a power of 2
    assert num_pes & (num_pes - 1) == 0, "num_pes must be a power of 2"

    num_compute_steps_cols = (N / num_pes) # N = number of columns
    num_compute_steps_rows = (M / num_pes) # M = number of rows

    col_step_latency = max(mem_latency_for_cols, compute_latency_for_cols)
    row_step_latency = max(mem_latency_for_rows, compute_latency_for_rows)

    # Calculate the total latency for the NTT
    # 3x fetch latency for fetching weights, fetching initial matrix column, and writing back last output matrix column
    total_latency = (num_compute_steps_cols * col_step_latency + 3 * mem_latency_for_cols) + (num_compute_steps_rows * row_step_latency + 3 * mem_latency_for_rows)
    
    return int(total_latency)

# this assumes that each butterfly has only 1 modmul and 1 modadd
def get_compute_latency(ntt_len, num_butterflies, bf_latency, modadd_latency, output_scaled=False, debug=False):
    
    num_stages = math.log2(ntt_len)
    first_stage = modadd_latency + max(ntt_len/(num_butterflies*2), 1) - 1 # only modadd
    most_stages = bf_latency + max(ntt_len/(num_butterflies*2), 1) - 1     # modmul
    if output_scaled:
        last_stage = bf_latency + max(ntt_len/(num_butterflies*2), 1) - 1    # 3 modmuls, 1 for butterfly, 2 for elementwise multiply
        scaling_stage = max(ntt_len/num_butterflies, 1) + (bf_latency - modadd_latency) - 1
        if debug:
            print(f"Scaling stage: {scaling_stage}")
            print(f"Last stage: {last_stage}")
        last_stage += scaling_stage
    else:
        last_stage = most_stages
    
    assert num_stages > 2

    if debug:
        print(f"First stage: {first_stage}, Most stages: {most_stages}, Last stage: {last_stage}, Num stages: {num_stages}")

    compute_latency = first_stage + (num_stages - 2) * most_stages + last_stage
    return int(compute_latency)


# this model assumes that all PEs are simultaneously reading from from off-chip memory.
# if the bandwidth available is not enough, we may want to pipeline reads per PE, but that is
# not considered in this model.
def sweep_single_ntt(ntt_exp, num_butterflies_list, num_pes_list, supplemental_data):

    bits_per_element, available_bw, freq, bf_latency, modmul_area, modadd_latency, scale_factors = supplemental_data

    max_read_rate = calc_rate(bits_per_element, available_bw, freq)

    phy_area = 29.6
    # phy cost based on bandwidth
    if available_bw <= 512:
        # bad estimate for now
        phy_area = (available_bw/512)*14.9
    
    else:
        phy_area *= available_bw/1024

    data_dict = dict()

    # 5 dual-ported banks of T words (4 from ping-pong with double buffering, 1 from elementwise scale)
    # 1 single-ported bank of T/2 words (for twiddles)  
    # 5 * 2 * 1 + 1 * 1 * 0.5 = 10.5
    num_macro_banks = 10.5
    for num_pes in num_pes_list:

        for num_butterflies in num_butterflies_list:
            M, N = closest_powers_of_two(ntt_exp)

            # number of butterflies (U) is how many 2ported banks we need, meaning we have 2U read ports
            num_read_ports_per_pe = num_butterflies * 2
            desired_read_rate = num_read_ports_per_pe*num_pes # can read up to these many words per cycle

            num_words_read_cols = (M)*num_pes
            num_words_read_rows = (N)*num_pes
 
            actual_read_rate = min(desired_read_rate, max_read_rate)
            
            mem_latency_for_cols = int(math.ceil(num_words_read_cols/actual_read_rate))
            mem_latency_for_rows = int(math.ceil(num_words_read_rows/actual_read_rate))

            compute_latency_for_cols = get_compute_latency(M, num_butterflies, bf_latency, modadd_latency, output_scaled=True)
            compute_latency_for_rows = get_compute_latency(N, num_butterflies, bf_latency, modadd_latency, output_scaled=False)
            
 
            # Call the single_fourstep_ntt function
            latency = single_fourstep_ntt(M, N, num_pes, mem_latency_for_rows, compute_latency_for_rows, mem_latency_for_cols, compute_latency_for_cols)
            
            # U modmuls for butterfly logic
            num_modmuls = num_butterflies * num_pes

            total_mb_on_chip = num_words_read_cols*bits_per_element/BITS_PER_MB

            subbanking_scale_factor = scale_factors[int(math.log2(num_butterflies))]
            
            # size in MB of 1 macro bank of T = M words with single-ported read/write sub-banks
            size_of_1_macro_bank = M*bits_per_element/BITS_PER_MB

            # size in mm^2 of 1 macro bank with single-ported read/write sub-banks
            area_of_1_macro_bank = size_of_1_macro_bank*MB_CONVERSION_FACTOR*subbanking_scale_factor

            # size in mm^3 of 1 macro bank with single-ported read/write sub-banks + phy area
            mem_area = num_macro_banks*area_of_1_macro_bank*num_pes + phy_area
            
            data_dict[(num_pes, num_butterflies)] = [latency, num_modmuls, num_modmuls*modmul_area + mem_area]
    return data_dict


# these are areas defined for 2^13 words
# these memories are single port for write and read, and can pick only one or the other
# note that these are for 254b
# we will use just 1 memory bank for element-wise unless BW necessitates 2 or more banks
data_mem_area_254b = np.array([1*440822, 2*255128, 4*165018, 8*118900, 16*96615, 32*85536, 64*44958, 128*24669])/1e6*(1)

mem_area_1bank_256KB_mm2 = data_mem_area_254b[0]
scale_factors = data_mem_area_254b / mem_area_1bank_256KB_mm2

if __name__ == "__main__":

    ntt_exp = 20
    available_bw = 1024 # GB/s
    freq = 1e9
    num_butterflies_list = [1, 2, 4, 8, 16, 32, 64, 128]  # Example list of number of butterflies
    num_pes_list = [1, 2, 4, 8]  # Example list of number of PEs
    modmul_area = 0.264
    modadd_latency = 3
    modmul_latency = 20
    bf_latency = modmul_latency + modadd_latency
    
    bits_per_element = 256
    
    supplemental_data = bits_per_element, available_bw, freq, bf_latency, modmul_area, modadd_latency, scale_factors
    
    # Call the sweep_single_ntt function
    ntt_data = sweep_single_ntt(ntt_exp, num_butterflies_list, num_pes_list, supplemental_data)

    print()
    # Print the ntt_data dictionary
    for key, value in ntt_data.items():
        num_pes, num_butterflies = key
        latency, num_modmuls, area_cost = value
        print(f"Num PEs: {num_pes}, Num Butterflies: {num_butterflies}, Latency: {latency}, Num Modmuls: {num_modmuls}, Total Area: {area_cost}")



    

    




