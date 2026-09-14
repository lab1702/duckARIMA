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

## Follow-up: copy products for pure-shift filters

After `588f345`, the 14-state seasonal filter again spent most of its value work
in the `T*P`, `T*a`, and `TP*T'` projections. A generic sparse-threshold sweep at
500 rows found that switching 4–8-state systems to the existing sparse products
was slower than dense multiplication. That change was not retained.

The new path specializes exact pure-shift transitions of dimension **14 or
larger**. `T*P` copies adjacent rows, `T*a` copies adjacent elements, and `P*T'`
copies adjacent columns, with zeros at the boundaries. These products take
O(k²), O(k), and O(k²) value work respectively. The kernels retain the dense
fallback for NULL/nonfinite operands and preserve the leading zero addition
used by the ordered dense fold, including signed-zero behavior in the tests.

The dispatcher now creates separate dense, generic sparse, and pure-shift
recursive plans. The shift flag is checked once from exact matrix entries;
near-shift matrices do not qualify. A constant flag in each recursive plan lets
DuckDB discard unused expressions. The general sparse cutoff remains 20 and
the scalar engine keeps its existing behavior. The extra plan adds a few
milliseconds to preparation and small per-system intermediates, not an
input-sized LIST.

The profiling tool's `--cases` option selects state dimensions to investigate:
`k2 k4 k6 k8 k14 k27` (the default remains `k2 k14 k27`). Example:

```bash
python tools/profile_filter.py --mode state --cases k2 k4 k6 k8 k14 k27 --rows 500 --repeats 5
```

Final full-likelihood comparison against `588f345`: DuckDB 1.5.5, one thread,
500 observations, seven measured executions after one warmup:

| State dimension | Previous main | Specialized shift path | Time reduction |
|---|---:|---:|---:|
| 2 | 327 ms | 324 ms | 1% |
| 14 | 494 ms | 386 ms | 22% |
| 27 | 570 ms | 561 ms | 2% |

The small differences for k2/k27 are timing variation. For k14, the instrumented
`T*P`/`T*a` projection fell from 110 ms to 20 ms and the `TP*T'`/outer-product
projection from 85 ms to 17 ms. A separate full-trace comparison (five measured
executions, same inputs) improved from 468 ms to 371 ms, with all trace values
exactly equal. All three full-likelihood results also matched exactly.

Validation: 80 targeted tests passed, including copy-versus-dense arithmetic
with signed zero and nonfinite inputs, mixed dense/sparse/shift/near-shift
systems at one and four threads, v2 fixtures, live statsmodels comparisons,
low-memory likelihoods, and assembly. The strengthened mixed-system test was
also rerun after adding its near-shift case. No fixtures or tolerances changed.

## Follow-up: combine covariance subtraction and process noise

After `1c31131`, each v2 filter step still built separate lists for the outer
product, the covariance subtraction, and the process-noise addition. The new
projection computes each entry as `(TPT'[i,j] - tpz[i]*tpz[j]/F) + RQR[i,j]`
directly, skipping the subtraction when the observation is missing. It retains
the original floating-point operation order and the separate symmetrization
pass. The full-trace and compact relational filters both use this projection;
scalar fitting is unchanged.

This removes two intermediate lists and one projection stage per recursive
step. Two broader candidates were slower in the same workload and were not
retained: carrying system matrices in the recursive state to remove a join,
and combining symmetrization with the covariance update. The latter duplicated
entry arithmetic despite reducing the number of list passes.

Full-likelihood measurements against `1c31131`: DuckDB 1.5.5, one thread,
500 observations, seven measured executions after one warmup. Runs were
sequential in baseline → tuned → baseline order, with no competing tests.

| State dimension | First baseline | Tuned | Repeated baseline | Reduction vs first baseline |
|---|---:|---:|---:|---:|
| 2 | 319 ms | 300 ms | 324 ms | 6% |
| 14 | 373 ms | 355 ms | 390 ms | 5% |
| 27 | 563 ms | 523 ms | 550 ms | 7% |

The tuned medians were lower than both baseline runs in all cases. These are
modest local improvements, not full-fit speedup estimates. Separate full-trace
runs (three measured executions) changed from 319/362/536 ms to 293/350/484 ms
for k2/k14/k27. All three likelihood results and all three complete traces
matched the baseline exactly.

Validation: the existing 80 targeted tests passed, covering transition
arithmetic, v2 fixtures, mixed transition structures, missing observations,
concentrated scale, live statsmodels checks, the low-memory relational path,
and generated assembly. No fixtures or tolerances changed.

## Follow-up: retain system columns in the compact recursive state

Against `56f0041`, the compact filter still joined each recursive row to the
same system row on every timestep. It now carries `k`, the intercept/burn
indices, transition matrices, sparse support, and process covariance as ordinary
columns beside the current Kalman state. The recursive step needs only the
observation join. These extra columns are omitted from the returned result.

`USING KEY (probe_id)` replaces the current row on every step, so the added
storage scales with probes and state dimension, not observation count. It does
increase the constant amount of retained per-probe state. The full-trace filter
is unchanged, avoiding duplication of invariant matrices across saved timesteps.
The scalar fitting path is also unchanged.

The ordinary-column representation matters: a previous struct-based version
added extraction overhead and was slower. This pass also tested bounded batches
of 16 and 128 observations using fixed-size aggregate slots; both were slower
than the original filter and were discarded. There is no batched observation
list in the retained implementation.

Full-likelihood measurements on DuckDB 1.5.5, one thread, against `56f0041`:
500-row measurements used seven timed executions after a warmup, while the
2,000-row measurements used three. Baseline and tuned runs were sequential,
with no competing tests. The table reports medians.

| Rows | State dimension | Previous main | Retained system columns | Time reduction |
|---|---|---:|---:|---:|
| 500 | 2 | 303 ms | 296 ms | 2% |
| 500 | 14 | 360 ms | 332 ms | 8% |
| 500 | 27 | 530 ms | 486 ms | 8% |
| 2,000 | 2 | 1,194 ms | 1,181 ms | 1% |
| 2,000 | 14 | 1,468 ms | 1,366 ms | 7% |
| 2,000 | 27 | 2,099 ms | 1,961 ms | 7% |

The k2 differences are within timing variation. Separate 500-row final-state
benchmarks (five repetitions) changed from 296/351/499 ms to 270/324/469 ms for
k2/k14/k27. Every final-state field and every measured likelihood matched the
baseline exactly. The analyzed recursive plans contain one hash join rather
than two. Observation access still occurs per timestep, so this does not remove
the filter's long-series scaling limitations.

Validation: 80 targeted tests passed, including the 20 MB relational likelihood
regression, mixed state dimensions and transition types, time gaps and empty
probes, v2 fixtures, live statsmodels checks, and generated assembly. No fixtures,
public result columns, or numerical tolerances changed.
