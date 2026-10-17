Stata performance regression tests

Requirements
  Licensed Stata/SE or Stata/MP 16 or newer, installed gtools, Python 3.8+
  (standard library only), and Git. The command keeps its version 12.0 target;
  only Stata/MP 16.1 was executed for this contribution.

From the repository root, set STATA_BIN to the Stata executable, then run:
  python3 Stata/tests/performance/run.py --base upstream/main

The default suite runs 34 synthetic-data scenarios once. To include the
four large-panel benchmarks and repeat timing scenarios three times:
  python3 Stata/tests/performance/run.py --base upstream/main \
      --benchmarks --repetitions 3

Use --cases with names from cases.json to select scenarios, --stata to
specify an executable, and --output to choose an artifact directory.
If the upstream remote has a different name, pass its revision to --base.
Pin --base to a commit when reproducing a published result. The baseline
ado is obtained with git show; neither installed ado files nor the Git
baseline are edited. Generated data, logs, and results are ignored by Git.
No binary datasets or machine-specific logs are committed.

Each command runs in a fresh Stata process with fixed data/RNG/sort seeds.
Both ado files are loaded before the estimation timer. Variants run
sequentially, with their order alternating between scenarios/repetitions.
Data generation, loading, and result export are outside the timer.
graph_off and _no_updates are supplied to both commands.

The runner compares all e() scalars, matrix entries, matrix stripes,
e() macros, and every cell in mat_res_XX, at 17 significant digits.
Result names, missing values, and text must agree. Numeric acceptance:
  abs(candidate - baseline)/(1 + abs(baseline)) <= 1e-12
It also checks returned column names/storage types and uses Stata cf to
compare every data value after sorting by id/time/outcome. Missing or
failed estimations, output mismatches, and data differences fail the run.
Timings, source hashes, comparisons, and medians are in results.json.

Coverage
  Repeated binary treatment reversals in both directions; multivalued
  treatment crossing above/below baseline; reset(0), reset(3), reset(5),
  and a reset threshold that makes no splits; original-unit and coarser
  clustering; switchers(in/out); controls; weights; normalization and
  normalized weights; same_switchers and same_switchers_pl; missing and
  unbalanced observations; saved samples; linear/nonparametric trends;
  continuous treatment; seeded bootstrap; by(); by_path(); double storage;
  granular demeaning; repeated observations requiring collapse; and
  heterogeneity predictions.

The reversal fixtures have 400 units and 40 periods (16,000 rows). Spells
change every seven periods, with eight staggered first-switch dates. Both
initial switch directions are represented. One fixture has no never-
switchers, so reset creates additional switching segments.
The multivalued paths cross both sides of baseline treatment. Outcomes
depend on current treatment and its first two lags.

The optional large grid uses 12 effects and 6 placebos, clustered SEs:
  Rows      Units  Periods  Adoption dates
  100,000   1,000  100       70
  200,000   2,000  100        5
  200,000   2,000  100       70
  200,000   1,000  200      160
Twenty percent of units never adopt; all treated cohorts have enough pre-
and post-treatment observations for the requested windows. See the fixture
assertions and cases.json for exact definitions and command options.

Results are synthetic-data validation, not a proof for every dataset or
option combination. Graph rendering and external design/date exports are
not exhaustively compared. The separate unavailable-horizon reset failure
is not counted as a successful estimation or included in timing medians.

For the Mata helper precision checks, run in Stata from the repository root:
  do Stata/tests/performance/helper_checks.do Stata/did_multiplegt_dyn.ado
A successful run prints HELPER_CHECKS_PASSED.

Optional long-calendar stress tests (one effect, no placebos):
  python3 Stata/tests/performance/run.py --base upstream/main \
      --stress-dates 1000 --cases stress_1000
  python3 Stata/tests/performance/run.py --base upstream/main \
      --stress-dates 3000 --cases stress_3000
The 1,000-date case has 1,250 units, 1,040 periods, and 1.3 million rows.
The 3,000-date case has 3,750 units, 3,040 periods, and 11.4 million rows.
These optional cases are excluded from --benchmarks unless explicitly
selected with --stress-dates. Upstream fills the complete group/time grid,
so shortening unit-specific observation windows would not avoid that
internal expansion. The 3,000-date baseline can take tens of minutes.
