# ZeroCheck Implementation

## 1. Software Baseline

1. The ZeroCheck protocol implementation can be found under `/src`, which includes both univariate zerocheck and multilinear zerocheck.
2. Several tests are provided and can be run in `/src`. For concrete software benchmarks, see `/bin`.
   - Example benchmark script: `/bin/run_bins.sh`

For multilinear:
```
RAYON_NUM_THREADS=64 cargo run --release --bin mullin_opt_bench_multhr -- --repeat=2 --min-size=10 --max-size=10 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero --batch-opening-threads=48
```

For univariate:
```
RAYON_NUM_THREADS=64 cargo run --release --bin univar_opt_bench_multhr -- --repeat=2 --min-size=10 --max-size=10 --prepare-threads=64 --run-threads=12 --poly-commit-scheme=ligero --batch-opening-threads=60
```

## 2. Hardware Architecture Simulator

The hardware architecture simulator is located in `/hardware_experiments`.

There are two main parts:

- **SumCheck simulation:** Uses `helper_funcs.py`, which provides simulation and sweeping for SumCheck.
- **NTT simulation:**
  - On-chip NTT (mini-NTT) simulation: see `test_ntt_func_sim.py` (`run_miniNTT_fit_onchip` function).
  - Large NTT (four-step NTT) simulation: see `test_ntt_func_sim.py` (`run_fourstep_fit_on_chip` function).


