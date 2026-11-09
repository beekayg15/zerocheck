"""Regenerate the Fig. 9 right panel (degree-3/4 gates) from the on-chip sweep results.
Mirrors the commented-out block at sumcheck_miniNTT_sweep.py:1176-1186."""
import sys
from pathlib import Path
HW = str(Path(__file__).resolve().parents[2])  # hardware_experiments/
sys.path.insert(0, HW)
import os
os.chdir(HW)
from sumcheck_miniNTT_sweep import load_results, plot_gate_acrx_group_single

n_values = 17
output_dir = Path(f"outplot_mo_onchip/n_{n_values}")
poly_style_name = "general_runs"
sc_results_df, ntt_result_df = load_results(output_dir.joinpath(poly_style_name))

polynomial_list = [
    [["g1", "g2", "g3"]],  # a gate of degree 3
    [["g1", "g2", "g3", "g4"]],
]
polynomial_color_idx = [4, 5]
plot_gate_acrx_group_single(
    sc_df=sc_results_df,
    ntt_df=ntt_result_df,
    filename=output_dir.joinpath(f"{poly_style_name}_n{n_values}_high_degree"),
    poly_group=polynomial_list,
    color_idx=polynomial_color_idx,
)
