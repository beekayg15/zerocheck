
#!/bin/bash

n_values=(16 17 18 19 20 21 22 23 24 25 26)
bw_values=(64 128 256 512 1024 2048 4096)

# echo "Generating individual Pareto plots for each (n, bw) combination..."
# # Plot Pareto frontier for specific n and bw
# for n in "${n_values[@]}"; do
#     for bw in "${bw_values[@]}"; do
#         python test_ntt_func_sim.py --mode plot --n $n --bw $bw &
#     done
# done

# Wait for all individual plots to complete
# wait

echo "Generating multi-bandwidth plots for each n..."
# Plot multiple bandwidths for each n on the same chart
for n in "${n_values[@]}"; do
    python test_ntt_func_sim.py --mode plot --n $n --multi-bw &
done

# # Wait for all multi-BW plots to complete
# wait

echo "All plots completed!"

