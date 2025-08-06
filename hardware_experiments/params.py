import os
import pickle

BITS_PER_GB = 1<<33
BITS_PER_MB = 1<<23
MB_CONVERSION_FACTOR = 1.76 # mm^2 / MB
ADD_POLICY = "longest_queue"

# curve-specific parameters
bits_per_scalar = 255
bits_per_point_coord = 381
bits_per_point = 3*bits_per_point_coord
bits_per_point_reduced = 2*bits_per_point_coord # points are initially stored as (X, Y, 1)
bits_per_type = [bits_per_scalar, bits_per_point, bits_per_point_reduced]

dual_core_permcheck_scale_factor = (2*bits_per_scalar + bits_per_point_reduced) / (bits_per_scalar + bits_per_point_reduced) 

# chip-level parameters
freq = 1e9

# msm folder
msm_base_dir = "../zkspeed_v0/msm_sims/datapoints/double_fetch/bls12-381/"
padd_latency = 96

obrld_file_path = f"../zkspeed_v0/bucket_reduction_latencies_{padd_latency}_stages.pkl"
if os.path.exists(obrld_file_path):
    with open(obrld_file_path, 'rb') as f:
        optimal_bucket_reduction_latency_dict = pickle.load(f)

# msm_base_dir = "../zkspeed_v1/msm_sims/datapoints/double_fetch/bls12-381/"
# padd_latency = 123
# obrld_file_path = f"../zkspeed_v1/bucket_reduction_latencies_{padd_latency}_stages.pkl"
# if os.path.exists(obrld_file_path):
#     with open(obrld_file_path, 'rb') as f:
#         optimal_bucket_reduction_latency_dict = pickle.load(f)

# latencies 
mle_update_latency = 20
modmul_latency = 20
modadd_latency = 1
transcript_latency = 36
extensions_latency = 5

onchip_sram_penalty = False

# areas
modmuls_in_padd = 19
modmul_area_381b = .582
modmul_frac_in_hls = 0.93 # when we get better area estimate, we should update this
padd_area = modmuls_in_padd * modmul_area_381b / modmul_frac_in_hls

# 255b data
modmul_area = 0.264

modadd_area = 0.002
modinv_area = 0.027
reg_area = 580e-6
sha_area = 0.0212
hbm_area = 29.6

rr_ctrl_area = 49040e-6
sleep_ctrl_area = 50e-6
num_barycentric_regs = 32
barycentric_reg_area = num_barycentric_regs * reg_area

# delay buffers for sumcheck (up to 32 slots)
# assume wraparound after this point
# TODO: model the wraparound
per_pe_delay_buffer_count = 32

max_nv = 30
num_opencheck_build_mle_regs = max_nv*7 # to store 6 sets of sha challenges used by build mle before opencheck, and the final opencheck \mu-length point
opencheck_build_mle_reg_area = num_opencheck_build_mle_regs * reg_area

sparse_fraction_zeros = 0.45
sparse_fraction_ones = 0.45
sparse_fraction_dense = 0.1
avg_bits_per_witness_word = (sparse_fraction_dense*bits_per_scalar + (sparse_fraction_ones + sparse_fraction_zeros))

max_pl_offset = 9

scale_factor_22_to_7nm = 3.6
scale_factor_14_to_7nm = 2.0


# mle areas for vanilla gate
modmuls_per_mle_update_pe = 1
modmuls_for_mle_combine = 12
mle_combine_modmul_area = modmuls_for_mle_combine * modmul_area

# packaging data for later use
padd_stats = padd_latency, padd_area
primitive_stats = modmul_area, modadd_area, modinv_area, reg_area


# msm params
ws_list  = range(7, 11)
ppw_list = [2**i for i in range(10, 15)]
ocw_list = [2**i for i in range(6)]
qd_list  = [16]
ii_list  = [1]
max_frac_mle_units = 4

permcheck_mle_units_range = range(1, max_frac_mle_units + 1)

# programmable sumcheck sweep parameters

sumcheck_pes_range = [(1 << i) for i in range(6)]
eval_engines_range = range(2, 8)
product_lanes_range = range(3, 9)
onchip_mle_sizes_range = [(1 << i) for i in range(10, 16)]

# 16 of these are double buffered for reading in MLEs, 8 of them are writeback FIFOs for MLE Updates
# there are actually 16 FIFOs but of half the buffer size, so effectively 8 buffers
num_sumcheck_sram_buffers = 24
num_mle_combine_sram_buffers = 6

# 1 temp MLE with up to 32 extensions per MLE entry pair --> 32/2 = 16
multifunction_tree_sram_scale_factor = 16


vanilla_skip_none_fraction_dict = {
    "q1": 0,
    "q2": 0,
    "q3": 0,
    "qM": 0,
    "qc": 0
}

jellyfish_skip_none_fraction_dict = {
    "q1": 0,
    "q2": 0,
    "q3": 0,
    "q4": 0,
    "q5": 0,
    "qc": 0,
    "qM1": 0,
    "qM2": 0,
    "qH1": 0,
    "qH2": 0,
    "qH3": 0,
    "qH4": 0,
    "qECC": 0
}

all_skip_none_fraction_dict = {
    "q1": 0,
    "q2": 0,
    "q3": 0,
    "q4": 0,
    "q5": 0,
    "qM": 0,
    "qM1": 0,
    "qM2": 0,
    "qH1": 0,
    "qH2": 0,
    "qH3": 0,
    "qH4": 0,
    "qECC": 0,
    "qc": 0
}

vanilla_zerocheck_polynomial = [
        ["q1", "w1", "fz"],
        ["q2", "w2", "fz"],
        ["q3", "w3", "fz"],
        ["qM", "w1", "w2", "fz"],
        ["qc", "fz"],
    ]

vanilla_permcheck_polynomial = [
    ["pi", "fz"],
    ["p1", "p2", "fz"],
    ["phi", "d1", "d2", "d3", "fz"],
    ["n1", "n2", "n3", "fz"],
]

jellyfish_zerocheck_polynomial = [
    ["q1", "w1", "fz"],
    ["q2", "w2", "fz"],
    ["q3", "w3", "fz"],
    ["q4", "w4", "fz"],
    ["q5", "w5", "fz"],
    ["qM1", "w1", "w2", "fz"],
    ["qM2", "w3", "w4", "fz"],
    ["qH1", "w1", "w1", "w1", "w1", "w1", "fz"],
    ["qH2", "w2", "w2", "w2", "w2", "w2", "fz"],
    ["qH3", "w3", "w3", "w3", "w3", "w3", "fz"],
    ["qH4", "w4", "w4", "w4", "w4", "w4", "fz"],
    ["qECC", "w1", "w2", "w3", "w4", "fz"],
    ["qc", "fz"],
]

jellyfish_permcheck_polynomial = [
    ["pi", "fz"],
    ["p1", "p2", "fz"],
    ["phi", "d1", "d2", "d3", "d4", "d5", "fz"],
    ["n1", "n2", "n3", "n4", "n5", "fz"],
]

opencheck_polynomial = [
    ["y1", "fz1"],
    ["y2", "fz2"],
    ["y3", "fz3"],
    ["y4", "fz4"],
    ["y5", "fz5"],
    ["y6", "fz6"],
]



# # sandbox params
# max_frac_mle_units = 4
# ws_list  = [9]
# ppw_list = [1024]
# ocw_list = [16]
# permcheck_mle_units_range = range(3, max_frac_mle_units + 1)


# sumcheck_pes_range = [16]
# eval_engines_range = [7]
# product_lanes_range = [4]
# onchip_mle_sizes_range = [16384]

# # end sandbox params

# experiments
# arbitrary prime
# modmul_area = .478
# padd_area = 23.77
# onchip_sram_penalty = True




def generate_custom_zerocheck_polynomial(d):
    if d < 2:
        raise ValueError("d must be at least 2")
    return [
        ["q1", "w1", "fz"],
        ["q2", "w2", "fz"],
        ["qM"] + ["w1"] * (d - 1) + ["w2"] + ["fz"],
        ["qc", "fz"],
    ]

custom_vanilla_permcheck_polynomial = [
    ["pi", "fz"],
    ["p1", "p2", "fz"],
    ["phi", "d1", "d2", "fz"],
    ["n1", "n2", "fz"],
]

# extensions_latency = 25
# modmul_area = .478
# padd_area = 23.77