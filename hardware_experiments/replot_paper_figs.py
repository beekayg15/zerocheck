"""Redraw the Pareto plots of Figs. 8, 9, 11 and 12 from the saved sweep results (no re-simulation).

Run from hardware_experiments/ after sumcheck_miniNTT_sweep.py (n=17) and sumcheck_miniNTT_partial_sweep.py (n=20)
have written their xlsx files. Fig. 13 is redrawn by sumcheck_NTT_sweep.py, which already loads its saved results.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import sumcheck_miniNTT_sweep as onchip
import sumcheck_miniNTT_partial_sweep as partial

# Scenario I (N = 2^17): Fig. 8 (more terms, with inset) and Fig. 9 (higher degree)
n_values = 17
output_dir = Path(f"outplot_mo_onchip/n_{n_values}")
poly_style_name = "general_runs"
sc_df, ntt_df = onchip.load_results(output_dir.joinpath(poly_style_name))
onchip.plot_gate_acrx_group_inset(
    sc_df=sc_df,
    ntt_df=ntt_df,
    filename=output_dir.joinpath(f"{poly_style_name}_n{n_values}_inset"),
    poly_group=[[["g1", "g2"]], [["g1", "g2"], ["g3"]], [["g1", "g2"], ["g3"], ["g4"]]],
    color_idx=[0, 1, 2],
)
onchip.plot_gate_acrx_group_single(
    sc_df=sc_df,
    ntt_df=ntt_df,
    filename=output_dir.joinpath(f"{poly_style_name}_n{n_values}_high_degree"),
    poly_group=[[["g1", "g2", "g3"]], [["g1", "g2", "g3", "g4"]]],
    color_idx=[4, 5],
)

# Scenario II (N = 2^20): Fig. 11 (row 1) and Fig. 12 (row 2)
n_values = 20
bw_values = [64, 256, 1024]
output_dir = Path(f"outplot_mo_part_onchip/n_{n_values}")
poly_style_name = "general_runs_stream"
sc_df, ntt_df = partial.load_results(output_dir.joinpath(poly_style_name))
polynomial_list = [
    [[["g1", "g2"]], [["g1", "g2"], ["g3"], ["g4"]], [["g1", "g2"], ["g3"], ["g4"], ["g5"]]],
    [[["g1", "g2", "g3"]], [["g1", "g2", "g3", "g4"]]],
]
color_list = [[0, 2, 3], [4, 5]]
for row_idx in (0, 1):
    partial.plot_gate_acrx_bw_grid_1x3(
        sc_df=sc_df,
        ntt_df=ntt_df,
        filename=output_dir.joinpath(f"{poly_style_name}_row{row_idx + 1}_1x3"),
        poly_group=polynomial_list[row_idx],
        bw_list=bw_values,
        color_idx=color_list[row_idx],
        row_idx=row_idx,
    )
