# Relational filter profiling

The relational v2 Kalman filters now reuse the scalar engine's ordered sparse
transition products for state dimensions of at least 20. Smaller systems keep
dense multiplication. The full-trace filter and final-state filter both use
this dispatch; the scalar fitting engine's arithmetic is unchanged.

## Evidence

Local measurements on DuckDB **1.5.5**, one thread, 150 deterministic
observations (`sin(t)`, every nineteenth observation missing), one parameter
probe. Each query was prepared once, warmed once, then executed seven times;
the table reports median wall time for execution and fetching the final state.
System-matrix construction and macro loading are excluded. These are filter
microbenchmarks, not end-to-end fit speedups or larger-than-memory benchmarks.

| State dimension | Before | After | Speedup |
|---|---:|---:|---:|
| 2 (ARMA) | 151 ms | 156 ms | 0.97× |
| 14 (seasonal MA) | 213 ms | 211 ms | 1.01× |
| 27 (integrated seasonal MA) | 456 ms | 254 ms | 1.79× |

The small-model differences are within ordinary timing variation. Preparation
increased from about 3 ms to 7 ms because the query now contains two recursive
plans. All returned final-state values matched the previous library exactly
on these inputs.

`EXPLAIN (ANALYZE, FORMAT JSON)` identified matrix projections as the dominant
value work for the 27-state model. The projection computing `T*P` and `T*a`
dropped from 141 ms to 30 ms; the projection computing `TP*T'` and the outer
product dropped from 128 ms to 21 ms. The largest join changed from about 8 ms to 7 ms.
Operator times are instrumented measurements, separate from the unprofiled
wall-time medians. Recursive operator timings can include child work and must
not be added together.

## Implementation

The shared sparse helpers live in `sql/00_linalg.sql`, so both filtering and
estimation can load them in dependency order. Each relational filter partitions
systems by state dimension before starting separate dense and sparse recursive
plans. A constant flag lets the optimizer remove the unused arithmetic from
each plan. Merely putting a dimension-dependent CASE inside one recursive plan
added noticeable expression overhead to small systems.

The sparse path caches each transition matrix's nonzero column indices per
system, preserves ascending accumulation order, and retains the existing dense
fallback for NULL/nonfinite matrix entries. This cache scales with the system
matrices, not the observation timeline. Mixed-dimension probe batches are
covered by exact comparisons against the dense implementation at one and four
threads. Seasonal likelihood comparisons cover missing observations, trend,
exogenous regressors, diffuse initialization, and concentrated scale.

## Reproduce

Using a Python environment with the test dependencies installed:

```bash
python tools/profile_filter.py --output scratch/profile --repeats 7
```

The command writes a JSON `EXPLAIN ANALYZE` plan and complete final-state result
for each case, plus a summary containing preparation time, execution samples,
and the most expensive operators. `--rows` and `--threads` adjust the workload.

To compare with a saved pre-change assembled library:

```bash
python tools/profile_filter.py --macros /path/to/before.sql --output scratch/before --repeats 7
python tools/profile_filter.py --output scratch/after --repeats 7
```

Run these sequentially with no competing test/benchmark process. Compare the
`*-result.json` files as well as timings. No machine-specific timing assertion
is added to the correctness suite.
