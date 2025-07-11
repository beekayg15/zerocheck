use criterion::criterion_main;

mod benches;

criterion_main! {
    benches::uni_naive_bench::benchmarks,
    benches::uni_opt_bench::benchmarks,
}
