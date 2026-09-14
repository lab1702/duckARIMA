"""Reproducible EXPLAIN ANALYZE profiles of the relational Kalman filter.

Run with the test environment: python tools/profile_filter.py --output scratch/profile
Use --macros to compare an earlier assembled library, holding inputs/settings fixed.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--macros', type=Path, default=ROOT / 'sarimax_macros.sql')
    parser.add_argument('--output', type=Path, default=ROOT / 'scratch/profile')
    parser.add_argument('--mode', choices=['state', 'trace', 'likelihood'], default='state')
    parser.add_argument('--rows', type=int, default=150)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.rows, args.repeats, args.threads) < 1:
        parser.error('rows, repeats and threads must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute(f'SET threads = {args.threads}')
    c.execute(args.macros.read_text())
    c.execute('CREATE TABLE probes AS SELECT 1 AS probe_id, [0.3, 0.2, 1.0]::DOUBLE[] AS params')
    c.execute('CREATE TABLE obs AS SELECT 1 AS probe_id, i AS t, '
              'CASE WHEN i % 19 = 0 THEN NULL ELSE sin(i::DOUBLE) END AS yd, '
              '0e0 AS ct FROM range(1, ?) r(i)', [args.rows + 1])
    c.execute('CREATE VIEW y AS SELECT t, yd AS y FROM obs')
    c.execute('CREATE TABLE x(t BIGINT, j BIGINT, x DOUBLE)')
    c.execute('CREATE TABLE degrees(idx BIGINT, degree BIGINT)')
    summary = {'duckdb': duckdb.__version__, 'mode': args.mode, 'rows': args.rows,
               'threads': args.threads, 'cases': {}}
    for name, orders in [('k2', '0,1,1,0,0,1,0,0,0,false'),
                         ('k14', '0,0,1,0,1,12,0,0,0,false'),
                         ('k27', '0,0,1,0,1,12,1,1,0,false')]:
        c.execute(f"CREATE OR REPLACE TABLE sys AS SELECT * FROM _sarimax_systems_v2('probes', {orders})")
        if args.mode == 'likelihood':
            query = ("SELECT _sarimax_ll_c_ooc_v2([0.3,0.2,1.0]::DOUBLE[], "
                     f"'y', 'x', 'degrees', {orders})")
        elif args.mode == 'trace':
            query = "SELECT * FROM _sarimax_kfilter_v2('obs', 'sys') ORDER BY probe_id, t"
        else:
            query = "SELECT * FROM _sarimax_kfilter_state_v2('obs', 'sys')"
        start = time.perf_counter()
        c.execute('PREPARE filter_query AS ' + query)
        prepare_s = time.perf_counter() - start
        result = c.execute('EXECUTE filter_query').fetchall()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            assert c.execute('EXECUTE filter_query').fetchall() == result
            times.append(time.perf_counter() - start)
        profile = json.loads(c.execute('EXPLAIN (ANALYZE, FORMAT JSON) EXECUTE filter_query').fetchone()[1])
        (args.output / f'{name}.json').write_text(json.dumps(profile, indent=2))
        (args.output / f'{name}-result.json').write_text(json.dumps(result))
        operators = []
        def walk(node):
            if 'operator_name' in node:
                operators.append({'operator': node['operator_name'],
                                  'seconds': node['operator_timing'],
                                  'rows': node['operator_cardinality'],
                                  'rows_scanned': node.get('operator_rows_scanned', 0),
                                  'detail': node.get('extra_info', {})})
            for child in node.get('children', []):
                walk(child)
        walk(profile)
        summary['cases'][name] = {'prepare_s': prepare_s, 'execution_s': times,
                                  'median_s': statistics.median(times),
                                  'top_operators': sorted(operators, key=lambda x: -x['seconds'])[:8]}
        print(name, f'prepare={prepare_s:.3f}s execute={statistics.median(times):.3f}s', flush=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    c.close()


if __name__ == '__main__':
    main()
