from helper_funcs import sumcheck_only_sweep
from itertools import product
import params
from poly_list import *
import pandas as pd
from util import is_pareto_efficient
from test_ntt_func_sim import run_fourstep_fit_on_chip, get_step_radix_gate_degree, characterize_poly, get_twiddle_factors
from ntt_utility import closest_powers_of_two
import math
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import itertools
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def ntt_sram_MB(case: str, n: int, num_unique_mles: int, max_degree: int) -> float:
    """
    Estimate the SRAM (in MB) required for a given NTT configuration.
    Args:
        case: str, one of "Four-step", "Stream", "All onchip".
        n: int, the power.
        gate: list of str, each representing a gate (e.g., ["g1", "g2", "g3"])
    Returns:
        sram_MB: float, estimated SRAM in MB
    """
    deg_ntt = get_step_radix_gate_degree(max_degree)[0]
    if case == "All onchip":
        return (2 * (deg_ntt * 2 ** n) * num_unique_mles + 0.5 * (deg_ntt * 2 ** n)) * (32 / 1024 / 1024)
    elif case == "Stream":
        return (4 * (deg_ntt * 2 ** n) * 1 + 0.5 * (deg_ntt * 2 ** n)) * (32 / 1024 / 1024)
    elif case == "Four-step":
        closest_n = closest_powers_of_two(n)[0]
        return ((3 + 0.5 + 1 + 1) * closest_n) * (32 / 1024 / 1024)
    else:
        raise ValueError("Invalid case, must be one of 'fourstep', 'stream', 'all_onchip'")


def sumcheck_sram_MB(n: int, num_unique_mles: int, max_degree: int) -> float:
    """
    Estimate the SRAM (in MB) required for a given all onchip sumcheck configuration.
    Args:
        n: int, the power.
        num_unique_mles: int, number of unique MLEs.
        max_degree: int, maximum degree of the polynomial.
    Returns:
        sram_MB: float, estimated SRAM in MB
    """
    max_degree = max_degree + 1  # fz
    num_unique_mles = num_unique_mles + 1  # fz
    if max_degree <= 3:
        return ((2 ** n) * num_unique_mles) * (32 / 1024 / 1024)
    else:
        return ((2 ** n) * num_unique_mles + (max_degree + 1) * (2 ** n) / 2) * (32 / 1024 / 1024)


def ntt_plot_degree_vs_sram_n18(max_MB, filename: str = None):
    """
    For n=18, sweep degree from 2 to 9 and unique MLE from 1 to 6,
    compute SRAM (MB) using ntt_sram_MB for three cases,
    and plot a 3D surface: x=unique_mle, y=degree, z=SRAM (MB), color by case.
    """
    n = 18
    degrees = range(2, 7)
    unique_mles_list = range(1, 7)
    cases = ["All onchip", "Stream", "Four-step"]
    color_maps = {
        "All onchip": cm.Blues,
        "Stream": cm.Oranges,
        "Four-step": cm.Greens
    }

    # Collect data
    data = []
    for deg in degrees:
        for num_mle in unique_mles_list:
            for case in cases:
                try:
                    sram_mb = ntt_sram_MB(case, n, num_mle, deg)
                except Exception:
                    sram_mb = float('nan')
                data.append({
                    "degree": deg,
                    "unique_mle": num_mle,
                    "sram_mb": sram_mb,
                    "case": case
                })

    df = pd.DataFrame(data)

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(10, -130)  # Adjust view

    for case in cases:
        sub_df = df[df["case"] == case]
        X, Y = np.meshgrid(
            sorted(sub_df["unique_mle"].unique()), 
            sorted(sub_df["degree"].unique())
        )

        Z = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = sub_df[
                    (sub_df["unique_mle"] == X[i, j]) & 
                    (sub_df["degree"] == Y[i, j])
                ]["sram_mb"]
                Z[i, j] = val.values[0] if not val.empty else np.nan

        # Use original color map
        if case == "Four-step":
            norm = Normalize(Y.min(), Y.max())
            facecolors = cm.Greens(norm(Y))
            surf = ax.plot_surface(
                X, Y, Z,
                facecolors=facecolors,
                alpha=0.9,
                linewidth=0,
                antialiased=True,
                shade=False
            )
        else:
            surf = ax.plot_surface(
                X, Y, Z,
                cmap=color_maps[case],
                alpha=0.6 if case == "Stream" else 0.9,
                linewidth=0,
                antialiased=True,
                shade=True
            )

    ax.set_xlabel("Unique MLE")
    ax.set_ylabel("Polynomial Degree")
    ax.set_zlabel("SRAM (MB)")
    ax.set_title(f"SRAM vs Unique MLE/Degree (n={n})")

    # Set tick step to 1 for unique_mle and degree
    ax.set_xticks(list(unique_mles_list))
    ax.set_yticks(list(degrees))

    if max_MB is not None:
        ax.set_zlim(0, max_MB)

    # Custom legend
    legend_patches = [Patch(color=color_maps[case](0.7), label=case) for case in cases]
    ax.legend(handles=legend_patches, loc='best')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    plt.show()


def ntt_plot_degree_vs_sram_n28(max_MB, filename: str = None):
    """
    For a given n, sweep degree from 2 to 9 and unique MLE from 1 to 6, 
    compute SRAM (MB) using ntt_sram_MB for three cases, and plot a 3D surface:
    x=unique_mle, y=degree, z=SRAM (MB), color by case.
    """
    n = 28

    degrees = range(2, 7)
    unique_mles_list = range(1, 7)

    cases = ["All onchip", "Stream", "Four-step"]
    color_maps = {
        "All onchip": cm.Blues,
        "Stream": cm.Oranges,
        "Four-step": cm.Greens
    }

    # Collect data
    data = []
    for deg in degrees:
        for num_mle in unique_mles_list:
            for case in cases:
                # skip deg > 3 for All onchip
                if case == "All onchip" and deg > 3:
                    continue

                try:
                    sram_mb = ntt_sram_MB(case, n, num_mle, deg)
                except Exception:
                    sram_mb = float('nan')
                data.append({
                    "degree": deg,
                    "unique_mle": num_mle,
                    "sram_mb": sram_mb,
                    "case": case
                })

    df = pd.DataFrame(data)

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(10, -130)  # Adjust view

    fixed_alpha = 0.9  # fixed transparency for surfaces

    for case in cases:
        sub_df = df[df["case"] == case]
        X, Y = np.meshgrid(
            sorted(sub_df["unique_mle"].unique()), 
            sorted(sub_df["degree"].unique())
        )

        Z = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = sub_df[
                    (sub_df["unique_mle"] == X[i, j]) & 
                    (sub_df["degree"] == Y[i, j])
                ]["sram_mb"]
                Z[i, j] = val.values[0] if not val.empty else np.nan

        # Clip values above max_MB
        Z_clipped = np.clip(Z, 0, max_MB) if max_MB is not None else Z

        if case == "Four-step":
            # Use Y for coloring, unchanged
            norm = Normalize(Y.min(), Y.max())
            facecolors = cm.Greens(norm(Y))
            surf = ax.plot_surface(
                X, Y, Z_clipped,
                facecolors=facecolors,
                alpha=fixed_alpha,
                linewidth=0,
                antialiased=True,
                shade=False
            )
        else:
            # Stream and All onchip: color by Z_clipped
            norm = Normalize(0, max_MB) if max_MB is not None else Normalize(Z.min(), Z.max())
            facecolors = color_maps[case](norm(Z_clipped))
            if max_MB is not None:
                # Make clipped region fully transparent
                facecolors[Z > max_MB, 3] = 0
                # Apply fixed alpha for visible part
                facecolors[Z <= max_MB, 3] = fixed_alpha

            surf = ax.plot_surface(
                X, Y, Z_clipped,
                facecolors=facecolors,
                linewidth=0,
                antialiased=True,
                shade=True
            )

    ax.set_xlabel("Unique MLE")
    ax.set_ylabel("Polynomial Degree")
    ax.set_zlabel("SRAM (MB)")
    ax.set_title(f"SRAM vs Unique MLE/Degree (n={n})")

    # Set tick step to 1 for unique_mle and degree
    ax.set_xticks(list(unique_mles_list))
    ax.set_yticks(list(degrees))

    if max_MB is not None:
        ax.set_zlim(0, max_MB)

    # Custom legend
    legend_patches = [Patch(color=color_maps[case](0.7), label=case) for case in cases]
    ax.legend(handles=legend_patches, loc='best')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    plt.show()


def sumcheck_plot_degree_vs_sram_n18(max_MB, filename: str = None):
    """
    For n=18, sweep degree from 2 to 9 and unique MLE from 1 to 6,
    compute SRAM (MB) using sumcheck_sram_MB for "All onchip" case,
    and plot a 3D surface: x=unique_mle, y=degree, z=SRAM (MB), color by degree (Blues colormap).
    """
    n = 18
    degrees = range(2, 7)
    unique_mles_list = range(1, 7)
    case = "All onchip"
    color_map = cm.Blues

    # Collect data
    data = []
    for deg in degrees:
        for num_mle in unique_mles_list:
            try:
                sram_mb = sumcheck_sram_MB(n, num_mle, deg)
            except Exception:
                sram_mb = float('nan')
            data.append({
                "degree": deg,
                "unique_mle": num_mle,
                "sram_mb": sram_mb,
            })

    df = pd.DataFrame(data)

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(10, -130)  # Adjust view

    fixed_alpha = 0.9  # fixed transparency for surfaces

    X, Y = np.meshgrid(
        sorted(df["unique_mle"].unique()), 
        sorted(df["degree"].unique())
    )

    Z = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val = df[
                (df["unique_mle"] == X[i, j]) & 
                (df["degree"] == Y[i, j])
            ]["sram_mb"]
            Z[i, j] = val.values[0] if not val.empty else np.nan

    # Mask out-of-range values
    if max_MB is not None:
        Z = np.where(Z > max_MB, np.nan, Z)

    # Color by degree (Y) for visual consistency with NTT plots
    norm = Normalize(Y.min(), Y.max())
    facecolors = color_map(norm(Y))

    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        alpha=fixed_alpha,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    ax.set_xlabel("Unique MLE")
    ax.set_ylabel("Polynomial Degree")
    ax.set_zlabel("SRAM (MB)")
    ax.set_title(f"SumCheck SRAM vs Unique MLE/Degree (n={n})")

    # Set tick step to 1 for unique_mle and degree
    ax.set_xticks(list(unique_mles_list))
    ax.set_yticks(list(degrees))

    if max_MB is not None:
        ax.set_zlim(0, max_MB)

    # Custom legend
    legend_patches = [Patch(color=color_map(0.7), label=case)]
    ax.legend(handles=legend_patches, loc='best')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    plt.show()


def plot_n17_ntt_sumcheck_allonchip(ax, max_MB=None, filename=None):
    """
    For n=17, sweep degree from 2 to 9 and unique MLE from 1 to 6,
    plot both NTT and SumCheck "All onchip" SRAM (MB) as 3D surfaces on the given ax.
    """
    marker_styles = ['o', 's', '^', 'X', 'D', 'P', '*', 'v', '<', '>']
    n = 17
    degrees = range(2, 7)
    unique_mles_list = range(1, 7)
    color_ntt = cm.Blues
    color_sumcheck = cm.Greens

    # Collect NTT data
    ntt_data = []
    for deg in degrees:
        for num_mle in unique_mles_list:
            try:
                sram_mb = ntt_sram_MB("All onchip", n, num_mle, deg)
            except Exception:
                sram_mb = float('nan')
            ntt_data.append({
                "degree": deg,
                "unique_mle": num_mle,
                "sram_mb": sram_mb,
            })
    ntt_df = pd.DataFrame(ntt_data)

    # Collect SumCheck data
    sumcheck_data = []
    for deg in degrees:
        for num_mle in unique_mles_list:
            try:
                sram_mb = sumcheck_sram_MB(n, num_mle, deg)
            except Exception:
                sram_mb = float('nan')
            sumcheck_data.append({
                "degree": deg,
                "unique_mle": num_mle,
                "sram_mb": sram_mb,
            })
    sumcheck_df = pd.DataFrame(sumcheck_data)

    # Set 3D view
    ax.view_init(10, -130)

    # Prepare mesh for NTT
    X_ntt, Y_ntt = np.meshgrid(
        sorted(ntt_df["unique_mle"].unique()),
        sorted(ntt_df["degree"].unique())
    )
    Z_ntt = np.empty_like(X_ntt, dtype=float)
    for i in range(X_ntt.shape[0]):
        for j in range(X_ntt.shape[1]):
            val = ntt_df[
                (ntt_df["unique_mle"] == X_ntt[i, j]) &
                (ntt_df["degree"] == Y_ntt[i, j])
            ]["sram_mb"]
            Z_ntt[i, j] = val.values[0] if not val.empty else np.nan
    if max_MB is not None:
        Z_ntt = np.where(Z_ntt > max_MB, np.nan, Z_ntt)
    # Use a colormap range that avoids the lightest colors (e.g., map [0,1] to [0.3,1] of the colormap)
    norm_ntt = Normalize(Y_ntt.min(), Y_ntt.max())
    # Scale normalized values to [0.3, 1.0] to avoid the lightest part of the colormap
    scaled_norm = 0.3 + 0.6 * norm_ntt(Y_ntt)
    facecolors_ntt = color_ntt(scaled_norm)
    surf_ntt = ax.plot_surface(
        X_ntt, Y_ntt, Z_ntt,
        facecolors=facecolors_ntt,
        alpha=0.8,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    # Prepare mesh for SumCheck
    X_sc, Y_sc = np.meshgrid(
        sorted(sumcheck_df["unique_mle"].unique()),
        sorted(sumcheck_df["degree"].unique())
    )
    Z_sc = np.empty_like(X_sc, dtype=float)
    for i in range(X_sc.shape[0]):
        for j in range(X_sc.shape[1]):
            val = sumcheck_df[
                (sumcheck_df["unique_mle"] == X_sc[i, j]) &
                (sumcheck_df["degree"] == Y_sc[i, j])
            ]["sram_mb"]
            Z_sc[i, j] = val.values[0] if not val.empty else np.nan
    if max_MB is not None:
        Z_sc = np.where(Z_sc > max_MB, np.nan, Z_sc)
    norm_sc = Normalize(Y_sc.min(), Y_sc.max())
    scaled_norm = 0.3 + 0.6 * norm_sc(Y_sc)
    facecolors_sc = color_sumcheck(scaled_norm)
    surf_sc = ax.plot_surface(
        X_sc, Y_sc, Z_sc,
        facecolors=facecolors_sc,
        alpha=0.6,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    ax.set_xlabel("Unique Polynomial", fontsize=14)
    ax.set_ylabel("Polynomial Degree", fontsize=14)
    ax.set_zlabel("SRAM (MB)", fontsize=14)

    ax.set_xticks(list(unique_mles_list))
    ax.set_yticks(list(degrees))
    if max_MB is not None:
        ax.set_zlim(0, max_MB)

    # Set fontsize for all tick labels
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    legend_patches = [
        Patch(color=color_ntt(0.7), label="NTT All onchip"),
        Patch(color=color_sumcheck(0.7), label="SumCheck All onchip")
    ]
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(0.06, 0.7), frameon=True)


def plot_n17_n20_n28_ntt_sumcheck_1x3(max_MB=400, filename=None):
    """
    Draw a 1x3 grid:
      - Left: n=17, All onchip NTT + All onchip SumCheck
      - Middle: n=20, Streaming NTT + All onchip SumCheck
      - Right: n=28, Four-step NTT + All onchip SumCheck
    Each subplot: x=unique_mle, y=degree, z=SRAM (MB), two surfaces per subplot.
    """
    n_list = [17, 20, 28]
    ntt_cases = ["All onchip", "Stream", "Four-step"]
    ntt_cmaps = [cm.Blues, cm.Oranges, cm.Purples]
    sumcheck_cmap = cm.Greens
    degrees = range(2, 7)
    unique_mles_list = range(1, 7)

    fig = plt.figure(figsize=(18, 6))
    for idx, (n, ntt_case, ntt_cmap) in enumerate(zip(n_list, ntt_cases, ntt_cmaps)):
        # NTT data
        ntt_data = []
        for deg in degrees:
            for num_mle in unique_mles_list:
                try:
                    sram_mb = ntt_sram_MB(ntt_case, n, num_mle, deg)
                except Exception:
                    sram_mb = float('nan')
                ntt_data.append({
                    "degree": deg,
                    "unique_mle": num_mle,
                    "sram_mb": sram_mb,
                })
        ntt_df = pd.DataFrame(ntt_data)

        # SumCheck data (always all onchip)
        sumcheck_data = []
        for deg in degrees:
            for num_mle in unique_mles_list:
                try:
                    sram_mb = sumcheck_sram_MB(n, num_mle, deg)
                except Exception:
                    sram_mb = float('nan')
                sumcheck_data.append({
                    "degree": deg,
                    "unique_mle": num_mle,
                    "sram_mb": sram_mb,
                })
        sumcheck_df = pd.DataFrame(sumcheck_data)

        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.view_init(10, -130)

        # NTT surface
        X_ntt, Y_ntt = np.meshgrid(
            sorted(ntt_df["unique_mle"].unique()),
            sorted(ntt_df["degree"].unique())
        )
        Z_ntt = np.empty_like(X_ntt, dtype=float)
        for i in range(X_ntt.shape[0]):
            for j in range(X_ntt.shape[1]):
                val = ntt_df[
                    (ntt_df["unique_mle"] == X_ntt[i, j]) &
                    (ntt_df["degree"] == Y_ntt[i, j])
                ]["sram_mb"]
                Z_ntt[i, j] = val.values[0] if not val.empty else np.nan
        if max_MB is not None:
            Z_ntt = np.where(Z_ntt > max_MB, np.nan, Z_ntt)
        norm_ntt = Normalize(Y_ntt.min(), Y_ntt.max())
        facecolors_ntt = ntt_cmap(norm_ntt(Y_ntt))
        ax.plot_surface(
            X_ntt, Y_ntt, Z_ntt,
            facecolors=facecolors_ntt,
            alpha=0.8,
            linewidth=0,
            antialiased=True,
            shade=False
        )

        # SumCheck surface
        X_sc, Y_sc = np.meshgrid(
            sorted(sumcheck_df["unique_mle"].unique()),
            sorted(sumcheck_df["degree"].unique())
        )
        Z_sc = np.empty_like(X_sc, dtype=float)
        for i in range(X_sc.shape[0]):
            for j in range(X_sc.shape[1]):
                val = sumcheck_df[
                    (sumcheck_df["unique_mle"] == X_sc[i, j]) &
                    (sumcheck_df["degree"] == Y_sc[i, j])
                ]["sram_mb"]
                Z_sc[i, j] = val.values[0] if not val.empty else np.nan
        if max_MB is not None:
            Z_sc = np.where(Z_sc > max_MB, np.nan, Z_sc)
        norm_sc = Normalize(Y_sc.min(), Y_sc.max())
        facecolors_sc = sumcheck_cmap(norm_sc(Y_sc))
        ax.plot_surface(
            X_sc, Y_sc, Z_sc,
            facecolors=facecolors_sc,
            alpha=0.6,
            linewidth=0,
            antialiased=True,
            shade=False
        )

        ax.set_xlabel("Unique MLE", fontsize=13)
        ax.set_ylabel("Polynomial Degree", fontsize=13)
        ax.set_zlabel("SRAM (MB)", fontsize=13)
        ax.set_xticks(list(unique_mles_list))
        ax.set_yticks(list(degrees))
        if max_MB is not None:
            ax.set_zlim(0, max_MB)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        ax.set_title(f"n={n}, {ntt_case} NTT + All onchip SumCheck", fontsize=14)
        legend_patches = [
            Patch(color=ntt_cmap(0.7), label=f"{ntt_case} NTT"),
            Patch(color=sumcheck_cmap(0.7), label="SumCheck All onchip")
        ]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=11)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    plt.show()


if __name__ == "__main__":
    
    ################################################
    # n_values = 18
    # output_dir = Path(f"outplot_mem/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # ntt_plot_degree_vs_sram_n18(max_MB=400, filename=output_dir / f"sram_vs_degree_n{n_values}_ntt.png")

    # n_values = 28
    # output_dir = Path(f"outplot_mem/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # ntt_plot_degree_vs_sram_n28(max_MB=None, filename=output_dir / f"sram_vs_degree_n{n_values}_ntt.png")

    ################################################
    # n_values = 18
    # output_dir = Path(f"outplot_mem/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # sumcheck_plot_degree_vs_sram_n18(max_MB=400, filename=output_dir / f"sram_vs_degree_n{n_values}_sumcheck.png")

    ################################################
    # n_values = 17
    # output_dir = Path(f"outplot_mem/n_{n_values}")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # plot_n17_ntt_sumcheck_allonchip(max_MB=200, filename=output_dir / f"sram_vs_degree_n{n_values}_all_onchip.pdf")

    n_values = 28
    output_dir = Path(f"outplot_mem/n_{n_values}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plot_n17_n20_n28_ntt_sumcheck_1x3(max_MB=None, filename=output_dir / f"sram_vs_degree_n17_n20_n28_ntt_sumcheck_1x3.pdf")

    ################################################

    print("End...")

