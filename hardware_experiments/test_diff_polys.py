from sumcheck_models import *

from poly_list import *

test = [
    ["f"]
]

poly_list = [vanilla_gate, jellyfish_gate, vanilla_perm, jellyfish_perm, opencheck, spartan_1, spartan_2]
poly_list = [complete_addition_11]
# poly_list = [test]
# poly_list = [spartan_1]
# poly_list = [opencheck]

comp_type = "zkspeed_vanilla"  # nocap, zkspeed_vanilla, zkspeed_jellyfish

# sumcheck_pes_range = [(1 << i) for i in range(6)]
# eval_engines_range = range(2, 8)
# product_lanes_range = range(3, 9)
onchip_mle_sizes_range = [(1 << i) for i in range(10, 16)]
bw_ranges = [64, 128, 256, 512, 1024, 2048, 4096] # GB/s
bw_ranges = [4096] # GB/s



num_sumcheck_sram_buffers = 24
num_mle_combine_sram_buffers = 6
# 1 temp MLE with up to 32 extensions per MLE entry pair --> 32/2 = 16
multifunction_tree_sram_scale_factor = 16

if comp_type == "nocap":

    # nocap comp
    num_pes = 32
    num_eval_engines = 3
    num_product_lanes = 4
    onchip_mle_sizes_range = [2048]
    bw_ranges = [1024] # GB/s
    bits_per_element = 64
    mle_update_latency = 2
    extensions_latency = 10
    modmul_latency = 2
    modadd_latency = 1
    num_vars = 24
elif comp_type == "zkspeed_vanilla":
    # zkspeed comp
    num_pes = 16
    num_eval_engines = 2
    num_product_lanes = 5
    onchip_mle_sizes_range = [16384]
    bw_ranges = [2048] # GB/s
    bits_per_element = 256
    mle_update_latency = 10
    extensions_latency = 20
    modmul_latency = 10
    modadd_latency = 1
    num_vars = 24

elif comp_type == "zkspeed_jellyfish":
    # zkspeed comp
    num_pes = 16
    num_eval_engines = 2
    num_product_lanes = 7
    onchip_mle_sizes_range = [16384]
    bw_ranges = [2048] # GB/s
    bits_per_element = 256
    mle_update_latency = 10
    extensions_latency = 20
    modmul_latency = 10
    modadd_latency = 1
    num_vars = 22
    poly_list = [jellyfish_gate_hyperplonk]
    # poly_list = [jellyfish_perm_hyperplonk]
    # poly_list = [opencheck]


freq = 1e9

debug = True
debug_just_start = True
sweep_dict = {}
for onchip_mle_size in onchip_mle_sizes_range:
    for available_bw in bw_ranges:
        print(f"\nTesting with onchip_mle_size={onchip_mle_size}, available_bw={available_bw} GB/s")
        sumcheck_hardware_params = num_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, onchip_mle_size
        supplemental_data = bits_per_element, available_bw, freq

        poly_list_latencies = []

        for poly in poly_list:
            sumcheck_polynomial = poly
            num_build_mle = len({elem for sublist in poly for elem in sublist if isinstance(elem, str) and elem.startswith("fz")})
            print(f"Processing polynomial: {poly}, num_build_mle: {num_build_mle}")

            # Create the schedule and calculate total latency
            round_latencies, _ = create_sumcheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, num_build_mle, supplemental_data, debug)
            total_latency = sum(round_latencies)
            print(f"Total latency for {poly}: {total_latency:.2f}")
            poly_list_latencies.append((poly, total_latency))
        continue
        high_degree_latencies = []

        custom_poly = [
            ["q1", "w1"],
            ["q2", "w2"],
            ["q3", "w2"],
            ["qc"]
        ]

        for degree in range(1, 30):
            print(f"\nTesting custom polynomial with degree {degree + 2}")
            custom_poly[2] += ["w1"] 
            sumcheck_polynomial = custom_poly
            round_latencies, schedule = create_sumcheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, 0, supplemental_data, debug, debug_just_start=debug_just_start)
            # print_schedule(schedule[0])
            total_latency = sum(round_latencies)
            total_latency_s = total_latency / freq  # Convert cycles to milliseconds (assuming freq is in Hz and latency is in cycles)
            del round_latencies

            print()
            print("###############################################################################")
            print(f"Total latency for custom polynomial with degree {degree + 2}: {total_latency:.2f}")
            print("###############################################################################")
            print()
            # high_degree_latencies.append(total_latency)
            high_degree_latencies.append(total_latency_s)
    
        sweep_dict[(onchip_mle_size, available_bw)] = (poly_list_latencies, high_degree_latencies)

exit()

# Print the results
print("\nSweep Results:")
for (onchip_mle_size, available_bw), (latencies, high_degree_latencies) in sweep_dict.items():
    print(f"\nOnchip MLE Size: {onchip_mle_size}, Available BW: {available_bw} GB/s")
    for poly, latency in latencies:
        print(f"  Total Latency: {latency:.2f}")
    for latency in high_degree_latencies:
        print(f"  Total Latency: {latency:.2f}")
print("\nSweep completed.")

import matplotlib.pyplot as plt
print("Keys of sweep_dict:")
print(list(sweep_dict.keys()))
for onchip_mle_size in onchip_mle_sizes_range:
    plt.figure(figsize=(10, 6))
    for available_bw in bw_ranges:
        result = sweep_dict.get((onchip_mle_size, available_bw))
        _, latencies = result
        print(latencies)
        degrees = list(range(3, 32))  # Degree ranges from 3 to 31 inclusive
        plt.plot(degrees, latencies, marker='o', label=f'BW={available_bw} GB/s')

    plt.title(f'High Degree Latencies for Onchip MLE Size {onchip_mle_size}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Total Latency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/high_degree_latencies_onchip_{onchip_mle_size}.png')