#!/usr/bin/env bash
set -o pipefail

mkdir -p output_log

run_and_log() {
    local log_file="$1"
    shift

    {
        printf "Running:"
        printf " %q" "$@"
        printf "\n\n"
        "$@"
    } 2>&1 | tee "$log_file"

    return "${PIPESTATUS[0]}"
}

UNIVAR_CMD=(
    env RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr --
    --repeat=1 --min-size=20 --max-size=20 --prepare-threads=64
    --run-threads=64 --poly-commit-scheme=ligero
    --batch-opening-threads=64 "--f=g1*g2 + g3 + g4"
)

MULLIN_CMD=(
    env RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr --
    --repeat=1 --min-size=20 --max-size=20 --prepare-threads=64
    --run-threads=64 --poly-commit-scheme=ligero
    --batch-opening-threads=64 "--f=g1*g2 + g3 + g4"
)

#########################################

run_and_log \
    output_log/univar_opt_bench_20_20_run_64_ligero_open_64.log \
    "${UNIVAR_CMD[@]}"

# run_and_log \
#     output_log/mullin_opt_bench_20_20_run_128_ligero_open_64.log \
#     "${MULLIN_CMD[@]}"
