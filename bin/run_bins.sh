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
#  --max-size=$MAX_SIZE > output_log/${TEST_FILE}_${THREADS}.log "
# echo "Running: $CMD"
# eval $CMD

# Polynomials for testing
POLY="q1*q2*q3"
POLY="q1*q2*q3*q4*q5"
POLY="q1*q2*q3*q4*q5*q6*q7*q8*q*q"
POLY="q1*q2*q3 + q1*q2*q4"
POLY="q1*q2*q3 + q1*q2*q4 + q1*q3*q4"
POLY="q1*q2*q3 + q1*q2*q4 + q1*q3*q4 + q2*q3*q4"
POLY="q1*q2 + q1*q2*q3"
POLY="q1*q2*q3 + q1*q2*q4 + q1*q3*q4"
POLY="q1*q2*q3 + q1*q3*q4 + q2*q3*q4 + q1*q2*q5"
POLY="q1*q2*q3 + q1*q2*q4 + q1*q2*q5 + q1*q2*q6 + q1*q2*q7 + q1*q2*q8 + q1*q2*q9"




############################################################
# RAYON_NUM_THREADS=1 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=8 --max-size=8

# CMDS1="RAYON_NUM_THREADS=1 cargo run --release --bin univar_opt_bench_multhr -- --repeat=10 --min-size=11 --max-size=12 > output_log/univar_opt_bench_multhr_1.log "
# CMDS2="RAYON_NUM_THREADS=1 cargo run --release --bin univar_opt_bench_multhr -- --repeat=10 --min-size=13 --max-size=18 > output_log/univar_opt_bench_multhr_1.log "

# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=10 --min-size=12 --max-size=12 --prepare-threads=64 --run-threads=1 | tee -a output_log/univar_opt_bench_multhr_1.log "

# for cmd in "${CMDS[@]}"; do
#     echo "Running: $cmd"
#     eval "$cmd"
# done


############################################################

# CMDS1="RAYON_NUM_THREADS=1 cargo run --release --bin mullin_naive_bench_multhr -- --repeat=10 --min-size=5 --max-size=12 | tee -a output_log/mullin_naive_bench_multhr_1.log "
# CMDS1="RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 | tee -a output_log/univar_opt_bench_multhr_64.log "
# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 --batch-opening-threads=4 | tee -a output_log/univar_opt_bench_run_1_open_4.log "
# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=26 --max-size=28 --prepare-threads=64 --run-threads=1 --batch-opening-threads=5 | tee -a output_log/univar_opt_bench_1_26_28_run_1_open_5.log "
# CMDS1="cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 --batch-opening-threads=8 | tee -a output_log/univar_opt_bench_run_1_open_8.log "

# CMDS1="cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=18 --max-size=18 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=1 | tee -a output_log/mullin_opt_bench_run_1_open_1.log "
# CMDS1="cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=26 --max-size=28 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=4 | tee -a output_log/mullin_opt_bench_1_26_28_run_1_open_4.log "


# echo "Running: $CMDS1"
# eval "$CMDS1"

############################################################
####### multi-linear (hyrax) #######
# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=4; } > output_log/mullin_opt_bench_16_16_run_1_hyrax_open_4.log 2>&1

# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=hyrax --batch-opening-threads=4; } > output_log/mullin_opt_bench_24_24_run_1_hyrax_open_4.log 2>&1

{ echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=hyrax --batch-opening-threads=48"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=hyrax --batch-opening-threads=48; } > output_log/mullin_opt_bench_24_24_run_12_hyrax_open_48.log 2>&1

####### multi-linear (kzg) #######
# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=4; } > output_log/mullin_opt_bench_16_16_run_1_kzg_open_4.log 2>&1

# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=4; } > output_log/mullin_opt_bench_24_24_run_1_kzg_open_4.log 2>&1

{ echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=kzg --batch-opening-threads=48"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=kzg --batch-opening-threads=48; } > output_log/mullin_opt_bench_24_24_run_12_kzg_open_48.log 2>&1

####### multi-linear (ligero) #######
# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=4; } > output_log/mullin_opt_bench_16_16_run_1_ligero_open_4.log 2>&1

# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=4; } > output_log/mullin_opt_bench_24_24_run_1_ligero_open_4.log 2>&1

{ echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero --batch-opening-threads=48"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero --batch-opening-threads=48; } > output_log/mullin_opt_bench_24_24_run_12_ligero_open_48.log 2>&1

####### multi-linear (ligero_poseidon) #######
# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=4; } > output_log/mullin_opt_bench_16_16_run_1_ligeroposeidon_open_4.log 2>&1

# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=4"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=4; } > output_log/mullin_opt_bench_24_24_run_1_ligeroposeidon_open_4.log 2>&1

{ echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=48"; RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=48; } > output_log/mullin_opt_bench_24_24_run_12_ligeroposeidon_open_48.log 2>&1

####### univar (kzg) #######
# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=5" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=5; } > output_log/univar_opt_bench_16_16_run_1_kzg_open_5.log 2>&1

# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=5" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=kzg --batch-opening-threads=5; } > output_log/univar_opt_bench_24_24_run_1_kzg_open_5.log 2>&1

{ echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=kzg --batch-opening-threads=60" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=kzg --batch-opening-threads=60; } > output_log/univar_opt_bench_24_24_run_12_kzg_open_60.log 2>&1

####### univar (ligero) #######
# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=5" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=5; } > output_log/univar_opt_bench_16_16_run_1_ligero_open_5.log 2>&1

# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=5" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero --batch-opening-threads=5; } > output_log/univar_opt_bench_24_24_run_1_ligero_open_5.log 2>&1

{ echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero --batch-opening-threads=60" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero --batch-opening-threads=60; } > output_log/univar_opt_bench_24_24_run_12_ligero_open_60.log 2>&1

####### univar (ligero_poseidon) #######
# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=5" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=1 --min-size=16 --max-size=16 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=5; } > output_log/univar_opt_bench_16_16_run_1_ligeroposeidon_open_5.log 2>&1

# { echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=5" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=1 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=5; } > output_log/univar_opt_bench_24_24_run_1_ligeroposeidon_open_5.log 2>&1

{ echo "Running: RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=60" ; RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=24 --max-size=24 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero_poseidon --batch-opening-threads=60; } > output_log/univar_opt_bench_24_24_run_12_ligeroposeidon_open_60.log 2>&1
