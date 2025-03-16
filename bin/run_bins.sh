#!/bin/bash

# Configuration variables
THREADS=1
# THREADS=16


TEST_FILE="univar_opt_bench_multhr"

REPEAT=10
MIN_SIZE=8
MAX_SIZE=16

# Construct and run the command

# CMD="RAYON_NUM_THREADS=$THREADS cargo run --release --bin $TEST_FILE -- --repeat=$REPEAT --min-size=$MIN_SIZE
#  --max-size=$MAX_SIZE >> output_log/${TEST_FILE}_${THREADS}.log "
# echo "Running: $CMD"
# eval $CMD



##############################
# RAYON_NUM_THREADS=1 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=8 --max-size=8

# CMDS1="RAYON_NUM_THREADS=1 cargo run --release --bin univar_opt_bench_multhr -- --repeat=10 --min-size=11 --max-size=12 >> output_log/univar_opt_bench_multhr_1.log "
# CMDS2="RAYON_NUM_THREADS=1 cargo run --release --bin univar_opt_bench_multhr -- --repeat=10 --min-size=13 --max-size=18 >> output_log/univar_opt_bench_multhr_1.log "

CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=10 --min-size=12 --max-size=12 --prepare-threads=64 --run-threads=1 | tee -a output_log/univar_opt_bench_multhr_1.log "

for cmd in "${CMDS[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done
