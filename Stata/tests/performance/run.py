"""Compare a Git baseline and the working-tree ado in fresh Stata processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import textwrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def stata(executable, dofile, folder, *arguments, timeout=1200):
    subprocess.run([executable, '-bq', 'do', str(dofile), *map(str, arguments)],
                   cwd=folder, timeout=timeout, check=True)


def compare(left, right):
    def read(path):
        return dict(line.split('\t', 1) for line in path.read_text().splitlines())

    before, after = read(left), read(right)
    if before.keys() != after.keys():
        raise AssertionError('Returned result names differ.')
    differing, maximum = 0, 0.0
    for key, value in before.items():
        if value == after[key]:
            continue
        x, y = float(value), float(after[key])
        difference = abs(x-y)/(1+abs(x))
        if not difference <= 1e-12:
            raise AssertionError((key, value, after[key], difference))
        differing += 1
        maximum = max(maximum, difference)
    return dict(entries=len(before), differing=differing,
                maximum_scaled_difference=maximum)


def run_case(executable, case, variant, source, folder, data):
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ['success', 'results.tsv']:
        (folder / filename).unlink(missing_ok=True)
    options = textwrap.wrap(case['options'] + ' graph_off _no_updates',
                            width=68, break_long_words=False,
                            break_on_hyphens=False)
    continued = '\n'.join('        ' + line +
                          (' ///' if i < len(options)-1 else '')
                          for i, line in enumerate(options))
    code = f'''version 16.0
clear all
set more off
set seed 91823
set sortseed 91823
set maxvar 32767
use "{data / (case['data'] + '.dta')}", clear
{case['prep']}
quietly run "{source}"

timer clear
timer on 1
capture noisily did_multiplegt_dyn y id time treatment, ///
{continued}
local command_rc = _rc
timer off 1
quietly timer list 1
scalar elapsed = r(t1)
file open timing using "{folder}/time.txt", write replace
file write timing %12.6f (elapsed) _n %9.0f (`command_rc') _n
file close timing
if `command_rc' != 0 exit `command_rc'

local snapshot "{folder}/results.tsv"
include "{HERE}/snapshot.do"
sort id time y
save "{folder}/after.dta", replace
file open done using "{folder}/success", write replace
file write done "ok"
file close done
exit, clear
'''
    dofile = folder / 'estimate.do'
    dofile.write_text(code)
    stata(executable, dofile, folder,
          timeout=7200 if case.get('stress_dates') == 3000 else 1200)
    if not (folder / 'success').exists():
        raise RuntimeError(f'{case["name"]}/{variant}: see {folder}/estimate.log')
    seconds = float((folder / 'time.txt').read_text().split()[0])
    print(f'{case["name"]} {variant}: {seconds:.3f}s', flush=True)
    return seconds


def compare_data(executable, baseline, candidate, folder):
    (folder / 'data_passed').unlink(missing_ok=True)
    dofile = folder / 'compare_data.do'
    dofile.write_text(f'''version 16.0
clear all
set more off
use "{baseline}/after.dta", clear
cf _all using "{candidate}/after.dta", all
file open done using "{folder}/data_passed", write replace
file write done "ok"
file close done
exit, clear
''')
    stata(executable, dofile, folder)
    if not (folder / 'data_passed').exists():
        raise AssertionError(f'Restored data differ: {folder}/compare_data.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stata', default=os.environ.get('STATA_BIN', 'stata-mp'))
    parser.add_argument('--base', default='upstream/main')
    parser.add_argument('--benchmarks', action='store_true')
    parser.add_argument('--stress-dates', type=int, choices=[1000, 3000])
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--output', type=Path, default=HERE / '.artifacts')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    executable = shutil.which(args.stata)
    if executable is None:
        parser.error('Set STATA_BIN or --stata to a licensed Stata executable.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = output / 'data'
    data.mkdir(exist_ok=True)
    baseline = output / 'baseline.ado'
    original = subprocess.check_output(
        ['git', 'show', f'{args.base}:Stata/did_multiplegt_dyn.ado'], cwd=REPO)
    baseline.write_bytes(original)
    candidate = REPO / 'Stata/did_multiplegt_dyn.ado'
    cases = json.loads((HERE / 'cases.json').read_text())
    cases = [c for c in cases if
             (c.get('stress_dates') == args.stress_dates if c.get('stress_dates')
              else args.benchmarks or not c['benchmark']) and
             (not args.cases or c['name'] in args.cases)]
    if not cases:
        parser.error('No matching scenarios.')
    setup = output / 'setup.do'
    generators = ['reversals.do', 'small.do']
    if any(c['benchmark'] and not c.get('stress_dates') for c in cases):
        generators.append('staggered.do')
    setup.write_text('\n'.join(f'do "{HERE / name}" "{data}"'
                               for name in generators) + '\n' +
                     (f'do "{HERE}/stress.do" "{data}" {args.stress_dates}\n'
                      if any(c.get('stress_dates') for c in cases) else '') +
                     'exit, clear\n')
    stata(executable, setup, output)
    for case in cases:
        if not (data / (case['data'] + '.dta')).exists():
            raise AssertionError(f'Fixture missing; see {output}/setup.log')

    metadata = dict(base_ref=args.base,
                    baseline_sha256=hashlib.sha256(original).hexdigest(),
                    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    records = []
    for case_index, case in enumerate(cases):
        repetitions = args.repetitions if case['timed'] else 1
        for rep in range(repetitions):
            folder = output / case['name'] / str(rep)
            paths = {v: folder / v for v in ['baseline', 'candidate']}
            order = ['baseline', 'candidate']
            if (case_index+rep) % 2:
                order.reverse()
            sources = dict(baseline=baseline, candidate=candidate)
            times = {v: run_case(executable, case, v, sources[v], paths[v], data)
                     for v in order}
            result = compare(paths['baseline'] / 'results.tsv',
                             paths['candidate'] / 'results.tsv')
            compare_data(executable, paths['baseline'], paths['candidate'], folder)
            records.append(dict(case=case['name'], rep=rep, options=case['options'],
                                **times, comparison=result, restored_data_equal=True))
            summaries = []
            for name in dict.fromkeys(r['case'] for r in records):
                rows = [r for r in records if r['case'] == name]
                a = statistics.median(r['baseline'] for r in rows)
                b = statistics.median(r['candidate'] for r in rows)
                summaries.append(dict(case=name, repetitions=len(rows),
                                      baseline_seconds=a, candidate_seconds=b,
                                      speedup=a/b))
            (output / 'results.json').write_text(json.dumps(
                dict(**metadata, records=records, summary=summaries), indent=2)+'\n')
            print(f'  PASS: {result}', flush=True)
    print(f'{len(records)} paired runs passed. Results: {output}/results.json')


if __name__ == '__main__':
    main()
