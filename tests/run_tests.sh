#!/bin/bash

# Configuration variables
THREADS=1
# THREADS=16

PACKAGE="zerocheck"

TEST_FILE="opt_uni_zc_tests_multhr"
# TEST_FILE="opt_uni_zc_tests"

TEST_FUNCTION="tests::bench_opt_uni_zc"

# Construct and run the command
CMD="RAYON_NUM_THREADS=$THREADS cargo test --package $PACKAGE --test $TEST_FILE
 -- $TEST_FUNCTION --exact --show-output >> output_log/output_${TEST_FILE}_${THREADS}.log "

# CMD="RAYON_NUM_THREADS=1 cargo test --package zerocheck --test opt_uni_zc_tests_multhr
#  -- tests::bench_opt_uni_zc --exact --show-output >> output_log/output_opt_uni_zc_tests_multhr_1.log "
# CMD="RAYON_NUM_THREADS=$THREADS cargo test --package zerocheck --test opt_uni_zc_tests_multhr
#  -- tests::bench_opt_uni_zc --exact --show-output >> output_log/output_opt_uni_zc_tests_multhr_$THREADS.log "
# CMD="RAYON_NUM_THREADS=1 cargo test --package zerocheck --test opt_uni_zc_tests
#  -- tests::bench_opt_uni_zc --exact --show-output >> output_log/opt_uni_zc_tests.log "
# CMD="RAYON_NUM_THREADS=16 cargo test --package zerocheck --test opt_uni_zc_tests
#  -- tests::bench_opt_uni_zc --exact --show-output >> output_log/output_opt_uni_zc_tests.log "


echo "Running: $CMD"
eval $CMD
