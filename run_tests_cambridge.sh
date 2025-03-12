#!/bin/bash

# Configuration variables (edit these as needed)
# THREADS=16
# PACKAGE="zerocheck"
# TEST_FILE="opt_uni_zc_tests_multhr"
# TEST_FUNCTION="tests::bench_opt_uni_zc"

# Construct and run the command
# CMD="RAYON_NUM_THREADS=$THREADS cargo test --package $PACKAGE --test $TEST_FILE -- $TEST_FUNCTION --exact --show-output"

# CMD="RAYON_NUM_THREADS=1 cargo test --package zerocheck --test opt_uni_zc_tests_multhr -- tests::bench_opt_uni_zc --exact --show-output"
# CMD="RAYON_NUM_THREADS=16 cargo test --package zerocheck --test opt_uni_zc_tests_multhr -- tests::bench_opt_uni_zc --exact --show-output"
CMD="RAYON_NUM_THREADS=1 cargo test --package zerocheck --test opt_uni_zc_tests -- tests::bench_opt_uni_zc --exact --show-output"
# CMD="RAYON_NUM_THREADS=16 cargo test --package zerocheck --test opt_uni_zc_tests -- tests::bench_opt_uni_zc --exact --show-output"


echo "Running: $CMD"
eval $CMD
