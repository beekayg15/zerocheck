
vanilla_gate = [
    ["q1", "w1"],
    ["q2", "w2"],
    ["q3", "w3"],
    ["qM", "w1", "w2"],
    ["qc"],
]

jellyfish_gate = [
    ["q1", "w1"],
    ["q2", "w2"],
    ["q3", "w3"],
    ["q4", "w4"],
    ["q5", "w5"],
    ["qM1", "w1", "w2"],
    ["qM2", "w3", "w4"],
    ["qH1", "w1", "w1", "w1", "w1", "w1"],
    ["qH2", "w2", "w2", "w2", "w2", "w2"],
    ["qH3", "w3", "w3", "w3", "w3", "w3"],
    ["qH4", "w4", "w4", "w4", "w4", "w4"],
    ["qECC", "w1", "w2", "w3", "w4"],
    ["qc"],
]

vanilla_perm = [
    ["pi"],
    ["p1", "p2"],
    ["phi", "d1", "d2", "d3"],
    ["n1", "n2", "n3"],
]

jellyfish_perm = [
    ["pi"],
    ["p1", "p2"],
    ["phi", "d1", "d2", "d3", "d4", "d5"],
    ["n1", "n2", "n3", "n4", "n5"],
]

opencheck = [
    ["y1", "fz1"],
    ["y2", "fz2"],
    ["y3", "fz3"],
    ["y4", "fz4"],
    ["y5", "fz5"],
    ["y6", "fz6"],
]


verifiable_asics = [
    ["qadd", "a"],
    ["qadd", "b"],
    ["qmul", "a", "b"]
]

spartan_1 = [
    ["A", "B", "fz"],
    ["C", "fz"]
]

spartan_2 = [
    ["ABC", "Z"]
]
spartan_2 = [
    ["A", "Z"],
    ["B", "Z"],
    ["C", "Z"]
]
vanilla_gate_zkspeed = [
    ["q1", "w1", "fz"],
    ["q2", "w2", "fz"],
    ["q3", "w3", "fz"],
    ["qM", "w1", "w2", "fz"],
    ["qc", "fz"],
]

vanilla_perm_zkspeed = [
    ["pi", "fz"],
    ["p1", "p2", "fz"],
    ["phi", "d1", "d2", "d3", "fz"],
    ["n1", "n2", "n3", "fz"],
]

jellyfish_gate_hyperplonk = [
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

jellyfish_perm_hyperplonk = [
    ["pi", "fz"],
    ["p1", "p2", "fz"],
    ["phi", "d1", "d2", "d3", "d4", "d5", "fz"],
    ["n1", "n2", "n3", "n4", "n5", "fz"],
]



custom_poly = [
    ["q1", "w1"],
    ["q2", "w2"],
    ["q3", "w2"],
    ["qc"]
]

witness_non_id_point = [
    ["q_non-id-point", "y", "y"],
    ["q_non-id-point", "x", "x", "x"],
    ["q_non-id-point"]
]

witness_id_point_1 = [
    ["q_point", "x", "y", "y"],
    ["q_point", "x", "x", "x"],
    ["q_point", "x"]
]

witness_id_point_2 = [
    ["q_point", "y", "y", "y"],
    ["q_point", "y", "x", "x", "x"],
    ["q_point", "y"]
]

incomplete_addition_1 = [
    ["q_add-incomplete", "x_r", "x_p", "x_p"],
    ["q_add-incomplete", "x_r", "x_p", "x_q"],
    ["q_add-incomplete", "x_r", "x_q", "x_q"],

    ["q_add-incomplete", "x_q", "x_p", "x_p"],
    ["q_add-incomplete", "x_q", "x_p", "x_q"],
    ["q_add-incomplete", "x_q", "x_q", "x_q"],

    ["q_add-incomplete", "x_p", "x_p", "x_p"],
    ["q_add-incomplete", "x_p", "x_p", "x_q"],
    ["q_add-incomplete", "x_p", "x_q", "x_q"],

    ["q_add-incomplete", "y_p", "y_p"],
    ["q_add-incomplete", "y_p", "y_q"],
    ["q_add-incomplete", "y_q", "y_q"]
]
incomplete_addition_2 = [
    ["q_add-incomplete", "y_r", "x_p"],
    ["q_add-incomplete", "y_r", "x_q"],
    ["q_add-incomplete", "y_q", "x_p"],
    ["q_add-incomplete", "y_q", "x_q"],

    ["q_add-incomplete", "y_p", "x_q"],
    ["q_add-incomplete", "y_p", "x_r"],
    ["q_add-incomplete", "y_q", "x_q"],
    ["q_add-incomplete", "y_q", "x_r"]
]

complete_addition_1 = [
    ["q_add", "x_q", "x_q", "lambda"],
    ["q_add", "x_q", "x_p", "lambda"],
    ["q_add", "x_p", "x_p", "lambda"],
    ["q_add", "x_q", "y_q"],
    ["q_add", "x_q", "y_p"],
    ["q_add", "x_p", "y_q"],
    ["q_add", "x_p", "y_p"]
]

complete_addition_2 = [
    ["q_add", "y_p", "lambda"],
    ["q_add", "x_p", "x_p"],

    ["q_add", "x_q", "alpha", "y_p", "lambda"],
    ["q_add", "x_q", "alpha", "x_p", "x_p"],
    ["q_add", "x_p", "alpha", "y_p", "lambda"],
    ["q_add", "x_p", "alpha", "x_p", "x_p"],
]

complete_addition_3 = [
    ["q_add", "x_p", "x_q", "x_q", "lambda", "lambda"],
    ["q_add", "x_p", "x_q", "x_p"],
    ["q_add", "x_p", "x_q", "x_q"],
    ["q_add", "x_p", "x_q", "x_r"],

]

complete_addition_4 = [
    ["q_add", "x_p", "x_q", "lambda", "x_p"],
    ["q_add", "x_p", "x_q", "lambda", "x_r"],
    ["q_add", "x_p", "x_q", "y_p"],
    ["q_add", "x_p", "x_q", "y_r"],
]

complete_addition_5 = [
    ["q_add", "x_p", "x_q", "y_q", "lambda", "lambda"],
    ["q_add", "x_p", "x_q", "y_p", "lambda", "lambda"],
    ["q_add", "x_p", "x_q", "y_q", "x_p"],
    ["q_add", "x_p", "x_q", "y_q", "x_q"],
    ["q_add", "x_p", "x_q", "y_q", "x_r"],
    ["q_add", "x_p", "x_q", "y_p", "x_p"],
    ["q_add", "x_p", "x_q", "y_p", "x_q"],
    ["q_add", "x_p", "x_q", "y_p", "x_r"],
]

complete_addition_6 = [
    ["q_add", "x_p", "x_q", "y_q", "lambda", "x_p"],
    ["q_add", "x_p", "x_q", "y_q", "lambda", "x_r"],
    ["q_add", "x_p", "x_q", "y_q", "y_p"],
    ["q_add", "x_p", "x_q", "y_q", "y_r"],
    ["q_add", "x_p", "x_q", "y_p", "lambda", "x_p"],
    ["q_add", "x_p", "x_q", "y_p", "lambda", "x_r"],
    ["q_add", "x_p", "x_q", "y_p", "y_p"],
    ["q_add", "x_p", "x_q", "y_p", "y_r"]
]

complete_addition_7 = [
    ["q_add", "x_r"],
    ["q_add", "x_q"],
    ["q_add", "x_p", "beta", "x_r"],
    ["q_add", "x_p", "beta", "x_q"],
]

complete_addition_8 = [
    ["q_add", "y_r"],
    ["q_add", "y_q"],
    ["q_add", "x_p", "beta", "y_r"],
    ["q_add", "x_p", "beta", "y_q"],
]

complete_addition_9 = [
    ["q_add", "x_r"],
    ["q_add", "x_p"],
    ["q_add", "x_q", "gamma", "x_r"],
    ["q_add", "x_q", "gamma", "x_p"],
]

complete_addition_10 = [
    ["q_add", "y_r"],
    ["q_add", "y_p"],
    ["q_add", "x_q", "gamma", "y_r"],
    ["q_add", "x_q", "gamma", "y_p"],

]

complete_addition_11 = [
    ["q_add", "x_r"],
    ["q_add", "x_q", "x_p", "alpha", "x_r"],
    ["q_add", "y_q", "y_p", "delta", "x_r"],
]

complete_addition_12 = [
    ["q_add", "y_r"],
    ["q_add", "x_q", "x_p", "alpha", "y_r"],
    ["q_add", "y_q", "y_p", "delta", "y_r"],
]

# DO NOT USE
jellyfish_gate_zkspeed = [
    ["q1", "w1", "fz"],
    ["q2", "w2", "fz"],
    ["q3", "w3", "fz"],
    ["q4", "w4", "fz"],
    ["q5", "w5", "fz"],
    ["qM1", "w1", "w2", "fz"],
    ["qM2", "w3", "w4", "fz"],
    ["qH1", "w1", "w1", "w1", "fz"],
    ["qH2", "w2", "w2", "w2", "fz"],
    ["qH3", "w3", "w3", "w3", "fz"],
    ["qH4", "w4", "w4", "w4", "fz"],
    ["qECC", "w1", "w2", "w3", "w4", "fz"],
    ["qc", "fz"],
]

jellyfish_perm_zkspeed = [
    ["pi", "fz"],
    ["p1", "p2", "fz"],
    ["phi", "d1", "d2", "d3", "d4", "d5", "fz"],
    ["n1", "n2", "n3", "n4", "n5", "fz"],
]
