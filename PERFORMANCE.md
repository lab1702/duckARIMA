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

## Follow-up: compute intercept lookups before recursion

A second pass against commit `e48ca4a` moves the invariant intercept lookup
out of each recursive step. The original filter joined observations twice per
step: once for the target and once for the current or next timestep's trend
intercept. Both v2 filters now materialize `(probe_id, t, yd, ct)` with a single
relational time-key join before starting their recursion. The intermediate can
spill; it is not an input-sized LIST. This trades additional relational storage
for fewer repeated scans and hash joins.

DuckDB 1.5.5, one thread, **500 observations**, five measured executions after
one warmup (same deterministic inputs as above):

| State dimension | Previous main | Precomputed intercepts | Speedup |
|---|---:|---:|---:|
| 2 | 548 ms | 317 ms | 1.73× |
| 14 | 800 ms | 466 ms | 1.72× |
| 27 | 875 ms | 513 ms | 1.70× |

These are additional gains over the sparse transition optimization, measured
at a different observation count from the first table. All final-state values
matched the baseline exactly. The recursive LEFT hash join disappears from the
new plan; the intercept join now executes outside the time recursion. Remaining
observation access inside the recursion can still scale poorly with long
series, so this is not a claim of linear overall runtime.

The profiling tool also accepts `--mode trace` for the full trace and
`--mode likelihood` for the relational likelihood including system construction
and observation adjustment. Each mode emits plans and result files for exact
baseline comparisons; timings include fetching the selected result. The default
`--mode state` continues to exclude system construction. For example:

```bash
python tools/profile_filter.py --mode likelihood --rows 500 --repeats 5 --output scratch/likelihood
```

The full relational likelihood comparison at the same 500 rows and five
repetitions includes initialization, which limits the overall gain:

| State dimension | Previous main | Precomputed intercepts | Time reduction |
|---|---:|---:|---:|
| 2 | 518 ms | 317 ms | 39% |
| 14 | 2,667 ms | 2,376 ms | 11% |
| 27 | 2,867 ms | 2,449 ms | 15% |

Every likelihood result matched exactly. Validation: 64 existing targeted tests
passed, including v2 fixtures, live statsmodels comparisons, mixed dense/sparse
systems, assembly, and the 20 MB relational likelihood regression; one added
regression also passed for shifted/unshifted intercepts, missing targets, a gap
in time keys, and a probe with no observations. No golden fixtures changed.

## Follow-up: finite covariance initialization for pure MA transitions

Profiling the full likelihood after commit `79fb874` put about 1.85 seconds in
the initial covariance projection for the 14-state seasonal MA model. The
previous implementation solved a 196-variable Kronecker system even though its
transition matrix is a pure shift.

For an exact shift matrix, `T^k = 0`, so the stationary covariance is the finite
series `Q + T Q T' + ... + T^(k-1) Q (T')^(k-1)`. Each output element is an
ascending sum along one diagonal of Q. The new path uses O(k³) value work and
O(k²) output storage instead of constructing the O(k⁴) augmented system and
performing O(k⁶) dense elimination. Ascending summation matches the old solver's
pivot-order arithmetic on the validated finite inputs.

The macro checks the matrix entries, not the model's name or a near-zero
tolerance. General and near-shift transitions retain the original solver,
now named `_sarimax_lyap_dense`. NULL/nonfinite inputs and shortcut results
containing overflow or negative zero also retain the dense path. The scalar
estimation engine's existing doubling initialization is unchanged.

Use `--params COEF1 COEF2 VARIANCE` to vary the profiling inputs. The two
coefficients mean AR/MA for the k2 case and ordinary/seasonal MA for k14/k27;
they default to `0.3 0.2 1.0` and are recorded in the summary JSON.

Fresh full-likelihood measurements on DuckDB 1.5.5, one thread, 500 observations,
five measured executions after one warmup, against `79fb874`:

| State dimension | Previous main | Shift covariance | Speedup |
|---|---:|---:|---:|
| 2 (general ARMA) | 317 ms | 323 ms | 0.98× |
| 14 (seasonal MA) | 2,340 ms | 488 ms | 4.80× |
| 27 (integrated seasonal MA) | 2,445 ms | 528 ms | 4.63× |

The 2-state difference is ordinary timing variation; its covariance uses the
general solver. For an additional negative-coefficient check (`--params -0.3
-0.2 1.0`, 150 observations, three measured executions), k14 improved from
1,997 ms to 150 ms and k27 from 2,014 ms to 157 ms. The larger relative gain at
150 rows reflects initialization's fixed cost; these are likelihood-call
measurements, not full optimizer-fit speedups.

All six baseline/tuned likelihood comparisons were exactly equal. Validation:
364 targeted tests passed and nine non-applicable exogenous-data checks skipped.
This includes the new shift-versus-dense covariance comparisons, general-solver
fixtures, both filter generations, live statsmodels checks, low-memory
likelihood checks, and assembly. No fixtures or numerical tolerances changed.
