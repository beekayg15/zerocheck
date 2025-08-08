import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from util import is_pareto_efficient
from params_ntt_v_sum import modmul_area, modadd_area, BITS_PER_MB, MB_CONVERSION_FACTOR

def get_area_stats(total_modmuls, total_modadds, total_num_words, bit_width=256):
    """Calculate logic and memory area in mm^2"""
    logic_area = total_modmuls * modmul_area + total_modadds * modadd_area
    memory_area = (total_num_words * bit_width / BITS_PER_MB) * MB_CONVERSION_FACTOR
    total_area = logic_area + memory_area
    return logic_area, memory_area, total_area

def plot_pareto_frontier_from_pickle(n, bw, pickle_dir="pickle_results"):
    """
    Read results from pickle file for a fixed N and bandwidth, then plot the Pareto frontier.
    
    Args:
        n: Problem size exponent (e.g., 16 for 2^16)
        bw: Bandwidth in GB/s (e.g., 1024)
        pickle_dir: Directory containing pickle files
    """
    # Construct filename based on n and bw
    filename = f"results_n{n}_bw{bw}.pkl"
    filepath = os.path.join(pickle_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Pickle file {filepath} not found!")
        return
    
    # Load results from pickle file
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    # Extract data for the specified n and bw
    data_points = []
    labels = []
    
    for key, value in results.items():
        result_n, result_bw, U, pe_amt = key
        if result_n == n and result_bw == bw:
            # Extract metrics
            cycles = value['total_cycles']
            modmuls = value['total_modmuls']
            modadds = value['total_modadds'] if 'total_modadds' in value else modmuls * 2  # Fallback if not available
            words = value['total_num_words']
            
            # Calculate total area (logic + memory)
            logic_area, memory_area, total_area = get_area_stats(modmuls, modadds, words)
            
            # For Pareto analysis: [cycles (runtime), total_area] (both to minimize)
            data_points.append([cycles, total_area])
            labels.append(f"U={U}, PE={pe_amt}")
    
    if not data_points:
        print(f"No data found for n={n}, bw={bw}")
        return
    
    data_points = np.array(data_points)
    
    # Find Pareto efficient points
    pareto_mask = is_pareto_efficient(data_points)
    pareto_points = data_points[pareto_mask]
    pareto_labels = [labels[i] for i in range(len(labels)) if pareto_mask[i]]
    
    # Create the plot
    plt.figure(figsize=(7, 5))
    
    # Plot all points
    plt.scatter(data_points[:, 0], data_points[:, 1], 
               alpha=0.6, s=50, color='lightblue', label='All configurations')
    
    # Highlight Pareto efficient points
    plt.scatter(pareto_points[:, 0], pareto_points[:, 1], 
               alpha=0.8, s=100, color='red', label='Pareto efficient')
    
    # Sort Pareto points by x-coordinate for line plotting
    sorted_indices = np.argsort(pareto_points[:, 0])
    sorted_pareto = pareto_points[sorted_indices]
    
    # Draw Pareto frontier line
    plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
             'r--', alpha=0.7, linewidth=2, label='Pareto frontier')
    
    # Annotate Pareto efficient points
    for i, (point, label) in enumerate(zip(pareto_points, pareto_labels)):
        plt.annotate(label, (point[0], point[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.xlabel('Runtime (cycles)')
    plt.ylabel('Total Area (mm²)')
    plt.title(f'Pareto Frontier: Runtime vs Area\nn={n} (2^{n} elements), Bandwidth={bw} GB/s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use log scale if the range is large
    if np.max(data_points[:, 0]) / np.min(data_points[:, 0]) > 100:
        plt.xscale('log')
    if np.max(data_points[:, 1]) / np.min(data_points[:, 1]) > 100:
        plt.yscale('log')
    
    plt.tight_layout()
    # Save the figure to a path, create the directory if it doesn't exist
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pareto_n{n}_bw{bw}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()
    
    # Print summary
    print(f"\nPareto Frontier Analysis for n={n}, bw={bw} GB/s:")
    print(f"Total configurations: {len(data_points)}")
    print(f"Pareto efficient configurations: {len(pareto_points)}")
    print("\nPareto efficient points:")
    for point, label in zip(pareto_points, pareto_labels):
        print(f"  {label}: {point[0]:.0f} cycles, {point[1]:.2f} mm²")

def plot_pareto_all_configs_from_pickle(pickle_file, metric_x='total_cycles', metric_y='total_area'):
    """
    Read results from a pickle file and plot Pareto frontier for all configurations.
    
    Args:
        pickle_file: Path to the pickle file
        metric_x: X-axis metric ('total_cycles', 'total_modmuls', 'total_num_words', 'total_area')
        metric_y: Y-axis metric ('total_cycles', 'total_modmuls', 'total_num_words', 'total_area')
    """
    if not os.path.exists(pickle_file):
        print(f"Error: Pickle file {pickle_file} not found!")
        return
    
    # Load results from pickle file
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    
    # Extract data points
    data_points = []
    labels = []
    
    for key, value in results.items():
        n, bw, U, pe_amt = key
        
        # Calculate total area if needed
        if metric_x == 'total_area' or metric_y == 'total_area':
            modmuls = value['total_modmuls']
            modadds = value['total_modadds'] if 'total_modadds' in value else modmuls * 2
            words = value['total_num_words']
            _, _, total_area = get_area_stats(modmuls, modadds, words)
            value['total_area'] = total_area
        
        x_val = value[metric_x]
        y_val = value[metric_y]
        
        # For modmuls, we want to maximize (so negate for Pareto analysis)
        if metric_x == 'total_modmuls':
            x_val = -x_val
        if metric_y == 'total_modmuls':
            y_val = -y_val
            
        data_points.append([x_val, y_val])
        labels.append(f"n={n}, bw={bw}, U={U}, PE={pe_amt}")
    
    if not data_points:
        print("No data found in pickle file")
        return
    
    data_points = np.array(data_points)
    
    # Find Pareto efficient points
    pareto_mask = is_pareto_efficient(data_points)
    pareto_points = data_points[pareto_mask]
    pareto_labels = [labels[i] for i in range(len(labels)) if pareto_mask[i]]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    plt.scatter(data_points[:, 0], data_points[:, 1], 
               alpha=0.6, s=30, color='lightblue', label='All configurations')
    
    # Highlight Pareto efficient points
    plt.scatter(pareto_points[:, 0], pareto_points[:, 1], 
               alpha=0.8, s=80, color='red', label='Pareto efficient')
    
    # Sort Pareto points by x-coordinate for line plotting
    sorted_indices = np.argsort(pareto_points[:, 0])
    sorted_pareto = pareto_points[sorted_indices]
    
    # Draw Pareto frontier line
    plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
             'r--', alpha=0.7, linewidth=2, label='Pareto frontier')
    
    # Set axis labels (handle negated modmuls and area)
    x_label = metric_x.replace('_', ' ').title()
    y_label = metric_y.replace('_', ' ').title()
    
    if metric_x == 'total_modmuls':
        x_label += ' (maximize)'
    elif metric_x == 'total_area':
        x_label = 'Total Area (mm²)'
    elif metric_x == 'total_cycles':
        x_label = 'Runtime (cycles)'
    
    if metric_y == 'total_modmuls':
        y_label += ' (maximize)'
    elif metric_y == 'total_area':
        y_label = 'Total Area (mm²)'
    elif metric_y == 'total_cycles':
        y_label = 'Runtime (cycles)'
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Pareto Frontier: {x_label} vs {y_label}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nPareto Frontier Analysis:")
    print(f"Total configurations: {len(data_points)}")
    print(f"Pareto efficient configurations: {len(pareto_points)}")
    print(f"\nTop 10 Pareto efficient points:")
    for i, (point, label) in enumerate(zip(pareto_points[:10], pareto_labels[:10])):
        x_display = -point[0] if metric_x == 'total_modmuls' else point[0]
        y_display = -point[1] if metric_y == 'total_modmuls' else point[1]
        print(f"  {label}: {x_display:.0f}, {y_display:.0f}")

def plot_pareto_multi_bw_fixed_n(n, bw_values, pickle_dir="pickle_results"):
    """
    Plot Pareto frontiers for a fixed N across multiple bandwidth values on the same plot.
    
    Args:
        n: Problem size exponent (e.g., 16 for 2^16)
        bw_values: List of bandwidth values in GB/s (e.g., [64, 128, 256, 512, 1024, 2048, 4096])
        pickle_dir: Directory containing pickle files
    """
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    plt.figure(figsize=(7, 5))
    
    # Define colors for different bandwidths
    colormap = cm.get_cmap('viridis')
    color_norm = colors.Normalize(vmin=min(bw_values), vmax=max(bw_values))
    # colors = ["#EE8183", "#B9E0A5", "#6399E8", "#B19EE1", "#F4B07F"]
    colors = ['#A87C62', "#778898", "#6399E8", "#F4B07F", "#6CCBAB", "#B19EE1", "#EE8183"]

    all_pareto_points = []
    bw_found = []
    
    for i, bw in enumerate(bw_values):
        # Construct filename based on n and bw
        filename = f"results_n{n}_bw{bw}.pkl"
        filepath = os.path.join(pickle_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Pickle file {filepath} not found, skipping bw={bw}")
            continue
        
        # Load results from pickle file
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        # Extract data for the specified n and bw
        data_points = []
        labels = []
        
        for key, value in results.items():
            result_n, result_bw, U, pe_amt = key
            if result_n == n and result_bw == bw:
                # Extract metrics
                cycles = value['total_cycles']
                modmuls = value['total_modmuls']
                modadds = value['total_modadds'] if 'total_modadds' in value else modmuls * 2
                words = value['total_num_words']
                
                # Calculate total area (logic + memory)
                logic_area, memory_area, total_area = get_area_stats(modmuls, modadds, words)
                
                # For Pareto analysis: [cycles (runtime), total_area] (both to minimize)
                data_points.append([cycles, total_area])
                labels.append(f"U={U}, PE={pe_amt}")
        
        if not data_points:
            print(f"No data found for n={n}, bw={bw}")
            continue
        
        bw_found.append(bw)
        data_points = np.array(data_points)
        
        # Find Pareto efficient points
        pareto_mask = is_pareto_efficient(data_points)
        pareto_points = data_points[pareto_mask]
        
        if len(pareto_points) == 0:
            continue
        
        # Store for overall summary
        all_pareto_points.extend(pareto_points)
        
        # Get color for this bandwidth
        # color = colormap(color_norm(bw))
        color = colors[i % len(colors)]

        # Plot all points for this bandwidth (smaller, more transparent)
        plt.scatter(data_points[:, 0], data_points[:, 1], 
                   alpha=0.3, s=20, color=color, 
                   label=f'All configs (bw={bw} GB/s)')
        
        # Highlight Pareto efficient points
        plt.scatter(pareto_points[:, 0], pareto_points[:, 1], 
                   alpha=0.8, s=60, color=color, 
                   label=f'Pareto efficient (bw={bw} GB/s)', marker='o', edgecolors='black', linewidth=0.5)
        
        # Sort Pareto points by x-coordinate for line plotting
        sorted_indices = np.argsort(pareto_points[:, 0])
        sorted_pareto = pareto_points[sorted_indices]
        
        # Draw Pareto frontier line for this bandwidth
        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                 color=color, alpha=0.8, linewidth=2, linestyle='--')
    
    
    # Create custom legend (only show Pareto efficient points to avoid clutter)
    legend_elements = []
    for i, bw in enumerate(bw_found):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8,
                                        markeredgecolor='black', markeredgewidth=0.5,
                                        label=f'BW={bw} GB/s'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Use log scale if the range is large
    if len(all_pareto_points) > 0:
        all_points = np.array(all_pareto_points)
        # if np.max(all_points[:, 0]) / np.min(all_points[:, 0]) > 100:
        #     plt.xscale('log')
        if np.max(all_points[:, 1]) / np.min(all_points[:, 1]) > 100:
            plt.yscale('log')
    
    # # Set x-axis in terms of millions (1e6)
    ax = plt.gca()
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([f"{x/1e6:.1f}" for x in x_ticks])
    # plt.tight_layout()

    plt.xlabel('Runtime (1e6 cycles)')
    plt.ylabel('Total Area (mm²)')
    plt.title(f'Pareto Frontiers: Runtime vs Area for n={n} (2^{n} elements)\nAcross Multiple Bandwidths')

    # Save the figure
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pareto_multi_bw_n{n}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()
    
    # Print summary
    print(f"\nMulti-BW Pareto Frontier Analysis for n={n}:")
    print(f"Bandwidths analyzed: {bw_found}")
    print(f"Total Pareto efficient points across all BWs: {len(all_pareto_points)}")

