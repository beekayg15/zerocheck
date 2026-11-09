"""What-if: tie the Scenario II SumCheck SRAM budget to the new (two-accumulator) NTT SRAM model,
instead of the old 4.5 x NTT-size budget hard-coded in sweep_onchip_sumcheck_configs. Writes only to the scratchpad."""
import os, sys
HW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # hardware_experiments/
OUT = sys.argv[1]
os.chdir(HW)
sys.path.insert(0, HW)
import matplotlib
matplotlib.use("Agg")

src = open("sumcheck_miniNTT_partial_sweep.py").read()
old = "num_word_in_ntt = ntt_length * 4 + ntt_length / 2"
new = ("num_word_in_ntt = ntt_length * 3 + 2 * sum(get_step_radix_gate_degree(gate_degree - 1)) * (2**num_vars) + ntt_length / 2")
assert src.count(old) == 1
src = src.replace(old, new).split('if __name__ == "__main__":')[0]
ns = {"__name__": "variant"}
exec(compile(src, "sumcheck_miniNTT_partial_sweep_variant.py", "exec"), ns)

from sumcheck_miniNTT_sweep import load_results
_, ntt_df = load_results("outplot_mo_part_onchip/n_20/general_runs_stream")
bw = [64, 256, 1024]
row1 = [[["g1", "g2"]], [["g1", "g2"], ["g3"], ["g4"]], [["g1", "g2"], ["g3"], ["g4"], ["g5"]]]
row2 = [[["g1", "g2", "g3"]], [["g1", "g2", "g3", "g4"]]]
sc_df = ns["sweep_onchip_sumcheck_configs"](num_var_list=[20], available_bw_list=bw, polynomial_list=row1 + row2)
sc_df.to_excel(f"{OUT}/sc_budget_variant_sc.xlsx", index=False)
ns["plot_gate_acrx_bw_grid_1x3"](sc_df, ntt_df, f"{OUT}/sc_budget_variant", row1, bw, [0, 2, 3], row_idx=0)
ns["plot_gate_acrx_bw_grid_1x3"](sc_df, ntt_df, f"{OUT}/sc_budget_variant", row2, bw, [4, 5], row_idx=1)
