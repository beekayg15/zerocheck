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

# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=10 --min-size=12 --max-size=12 --prepare-threads=64 --run-threads=1 | tee -a output_log/univar_opt_bench_multhr_1.log "

# for cmd in "${CMDS[@]}"; do
#     echo "Running: $cmd"
#     eval "$cmd"
# done


##############################

# CMDS1="RAYON_NUM_THREADS=1 cargo run --release --bin mullin_naive_bench_multhr -- --repeat=10 --min-size=5 --max-size=12 | tee -a output_log/mullin_naive_bench_multhr_1.log "
# CMDS1="RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 | tee -a output_log/univar_opt_bench_multhr_64.log "
# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 --batch-opening-threads=4 | tee -a output_log/univar_opt_bench_run_1_open_4.log "
# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=26 --max-size=28 --prepare-threads=64 --run-threads=1 --batch-opening-threads=5 | tee -a output_log/univar_opt_bench_1_26_28_run_1_open_5.log "
# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 --batch-opening-threads=8 | tee -a output_log/univar_opt_bench_run_1_open_8.log "

# CMDS1="cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=1 | tee -a output_log/mullin_opt_bench_run_1_open_1.log "
CMDS1="cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=26 --max-size=28 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=4 | tee -a output_log/mullin_opt_bench_1_26_28_run_1_open_4.log "


echo "Running: $CMDS1"
eval "$CMDS1"
