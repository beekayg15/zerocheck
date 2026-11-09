"""Apply the Scenario III element-wise fix to a saved four-step NTT sweep without re-simulating.

The old `process_gate_n_bw` summed 2k+1 `run_fourstep_fit_on_chip(polynomial=[["f"]])` results, each of which
includes its own element-wise read pass, so that pass was counted 2k+1 times. The fixed version (now in
sumcheck_NTT_sweep.py) strips those passes and charges one element-wise stage for the real gate.

`estimate_elementwise_latency` does not depend on the simulation, so every per-call pass can be recomputed from
(n, bw, U, pe_amt) alone; the script checks each recomputed pass against the `elementwise_cycles` column that the
sweep saved, then rewrites `total_latency`. Run from hardware_experiments/:

    python fix_fourstep_elementwise.py outplot_mo/n_32/general_runs_fourstep_ntt.xlsx
"""
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

from ntt_utility import closest_powers_of_two
from poly_list import *  # noqa: F401,F403  (gate helpers)
from sumcheck_NTT_sweep import fourstep_gate_elementwise_cycles, gate_to_string
from test_ntt_func_sim import characterize_poly, estimate_elementwise_latency, get_step_radix_gate_degree

POLYNOMIALS = [
    [["g1", "g2"]],
    [["g1", "g2", "g3"]],
    [["g1", "g2", "g3", "g4"]],
    [["g1", "g2"], ["g3"]],
    [["g1", "g2"], ["g1"], ["g3"]],
    [["g1", "g2"], ["g1"], ["g2"]],
    [["g1", "g2"], ["g1"], ["g2"], ["g3"]],
    [["g1", "g2"], ["g3"], ["g4"], ["g5"]],
    [["g1", "g2"], ["g3"], ["g4"]],
    [["g1", "g2"], ["g3"], ["g4"], ["g5"], ["g6"]],
]


def per_call_pass(n_pow, bw, U, pe_amt):
    """Element-wise pass that one run_fourstep_fit_on_chip(polynomial=[["f"]]) call adds at size 2^n_pow."""
    M, N = closest_powers_of_two(int(n_pow))
    return estimate_elementwise_latency(characterize_poly([["f"]]), (N, M), U, pe_amt, 256, bw, 1e9)


def main(path):
    path = Path(path)
    df = pd.read_excel(path)
    gates = {gate_to_string(g): g for g in POLYNOMIALS}
    new_total, new_ew, ntt_only = [], [], []
    mismatches = 0
    for _, r in df.iterrows():
        gate = gates[r["gate_name"]]
        k, n, bw, U, pe = int(r["gate_num_unique_mle"]), int(r["n"]), int(r["available_bw"]), int(r["unroll_factor"]), int(r["pe_amt"])
        steps = get_step_radix_gate_degree(int(r["gate_degree"]))
        pass_input = per_call_pass(n, bw, U, pe)                                    # input iNTT at size N
        pass_chunks = [per_call_pass(n + math.log2(s), bw, U, pe) for s in steps]   # one pass per step-radix chunk
        # the sweep saved the pass of the first chunk of `res` (the second chunk was not summed in)
        if pass_chunks[0] != r["elementwise_cycles"]:
            mismatches += 1
        old_passes = k * pass_input + (k + 1) * sum(pass_chunks)
        ntt = r["total_latency"] - old_passes
        ew = fourstep_gate_elementwise_cycles(gate, n, bw, U, pe)
        ntt_only.append(ntt); new_ew.append(ew); new_total.append(ntt + ew)
    print(f"rows: {len(df)}, per-call pass mismatches vs saved elementwise_cycles: {mismatches}")
    assert mismatches == 0, "recomputed element-wise pass does not reproduce the saved column; aborting"

    backup = path.with_name(path.stem + "_pre_ewfix.xlsx")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup: {backup}")
    df["total_latency_pre_ewfix"] = df["total_latency"]
    df["elementwise_cycles_per_call"] = df["elementwise_cycles"]
    df["ntt_cycles"] = ntt_only
    df["elementwise_cycles"] = new_ew
    df["total_latency"] = new_total
    df.to_excel(path, index=False)
    print(f"written: {path}")

    c = df[(df.gate_name == "g1 g2 + g3 + g4") & (df.available_bw == 1024) & (df.pe_amt == 2) & (df.unroll_factor.isin([66, 98]))]
    print(c[["unroll_factor", "pe_amt", "total_latency_pre_ewfix", "ntt_cycles", "elementwise_cycles", "total_latency", "total_area"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outplot_mo/n_32/general_runs_fourstep_ntt.xlsx")
