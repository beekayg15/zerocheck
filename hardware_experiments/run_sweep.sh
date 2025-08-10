#!/bin/bash

# NTT Function Simulator Parameter Sweep Script
# This script runs the simulator for all combinations of n and bw values
# and saves the output to separate log files in debug_logs folder

# Create debug_logs directory if it doesn't exist
mkdir -p debug_logs

# Define parameter ranges
n_values=(16 17 18 19 20 21 22 23 24 25 26)
bw_values=(64 128 256 512 1024 2048 4096)

echo "Starting parameter sweep..."
echo "n values: ${n_values[@]}"
echo "bw values: ${bw_values[@]}"
echo "Total combinations: $((${#n_values[@]} * ${#bw_values[@]}))"
echo ""

# Counter for progress tracking
total_combinations=$((${#n_values[@]} * ${#bw_values[@]}))
current_combination=0

# Loop through all combinations
for n in "${n_values[@]}"; do
    for bw in "${bw_values[@]}"; do
        current_combination=$((current_combination + 1))
        
        # Create filename
        output_file="debug_logs/sweep_n${n}_bw${bw}.txt"
        
        echo "[$current_combination/$total_combinations] Running n=$n, bw=$bw -> $output_file"
        
        # Run the simulation and pipe output to file
        python test_ntt_func_sim.py --n $n --bw $bw --mode sweep > "$output_file" 2>&1 &
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "  ✓ Success"
        else
            echo "  ✗ Failed (see $output_file for details)"
        fi
    done
done

echo ""
echo "Parameter sweep completed!"
echo "Results saved in debug_logs/ folder:"
ls -la debug_logs/sweep_*.txt | wc -l | xargs echo "Total files created:"
echo ""
echo "Example files:"
ls debug_logs/sweep_*.txt | head -5
