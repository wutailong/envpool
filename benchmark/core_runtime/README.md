# Core runtime: measured throughput and correctness

> Historical first-round report. For the cumulative fixes, current experimental
> branch, validation limits and later adverse measurements, start with
> [the core runtime review / 分支导航](REVIEW.md). The numbers below predate those
> follow-up repairs and are preserved as historical evidence.

These results show **configuration-dependent gains and regressions**, not a
universal speedup. All 234 final timing samples and all 18 configuration/affinity
comparisons are included. Exploratory candidate-v1 samples are excluded.

The strongest repeatable-looking case here is CartPole with 256 environments,
full batches and one worker: +21.6% versus the installed original and +13.5%
versus the same-flags rebuilt original when pinned; +17.5% and +16.9% with the
default scheduler. Small changes and overlapping ranges are inconclusive in this
shared-cloud experiment. There are only five pinned or three default-scheduler
samples per variant/case; no confidence interval or universal speed claim is made.

Important regressions include pinned CartPole 20/20/1 (−4.5% versus installed
original), 64/64/1 (−8.6%), and 1024/1024/8 (−2.9%). Default-scheduler CartPole
256/256/4 is −9.1% versus original and −14.3% versus rebuilt; 1024/256/8 is
−13.0% versus original. HalfCheetah results are mixed, including −6.1% for
default-scheduler 64/64/4 versus original. This microbenchmark does not establish
end-to-end training speed.

## Complete median results

N = environments, B = returned batch size, T = worker threads. Rates below are
**environment responses per second**, including normal next-step autoreset
responses. They are neither vector calls/s nor pure non-reset physics steps/s.
Vector calls/s = responses/s divided by B. The historical raw keys `env_steps`
and `env_steps_per_s` retain this response-count meaning.

| Environment | N | B | T | Affinity | Original | Rebuilt | Candidate | vs original | vs rebuilt |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CartPole-v1 | 1 | 1 | 1 | pinned | 29,218 | 28,546 | 30,239 | +3.5% | +5.9% |
| CartPole-v1 | 20 | 20 | 1 | pinned | 368,804 | 367,884 | 352,356 | -4.5% | -4.2% |
| CartPole-v1 | 64 | 64 | 1 | pinned | 797,476 | 730,546 | 728,684 | -8.6% | -0.3% |
| CartPole-v1 | 256 | 256 | 1 | pinned | 1,223,963 | 1,311,224 | 1,487,769 | +21.6% | +13.5% |
| CartPole-v1 | 256 | 256 | 4 | pinned | 1,402,295 | 1,319,814 | 1,533,120 | +9.3% | +16.2% |
| CartPole-v1 | 1024 | 1024 | 8 | pinned | 2,506,349 | 2,515,914 | 2,433,869 | -2.9% | -3.3% |
| CartPole-v1 | 256 | 64 | 4 | pinned | 1,533,659 | 1,434,184 | 1,592,564 | +3.8% | +11.0% |
| CartPole-v1 | 1024 | 256 | 8 | pinned | 2,741,997 | 3,001,614 | 3,233,427 | +17.9% | +7.7% |
| HalfCheetah-v4 | 20 | 20 | 1 | pinned | 23,230 | 22,782 | 22,951 | -1.2% | +0.7% |
| HalfCheetah-v4 | 64 | 64 | 4 | pinned | 75,791 | 76,672 | 79,028 | +4.3% | +3.1% |
| HalfCheetah-v4 | 256 | 64 | 4 | pinned | 88,801 | 90,856 | 90,105 | +1.5% | -0.8% |
| HalfCheetah-v4 | 256 | 256 | 8 | pinned | 124,641 | 122,795 | 127,423 | +2.2% | +3.8% |
| CartPole-v1 | 20 | 20 | 1 | default | 373,045 | 349,696 | 366,839 | -1.7% | +4.9% |
| CartPole-v1 | 256 | 256 | 1 | default | 1,238,657 | 1,245,645 | 1,455,615 | +17.5% | +16.9% |
| CartPole-v1 | 256 | 256 | 4 | default | 1,494,701 | 1,585,291 | 1,358,156 | -9.1% | -14.3% |
| CartPole-v1 | 1024 | 256 | 8 | default | 3,044,633 | 2,651,479 | 2,649,575 | -13.0% | -0.1% |
| HalfCheetah-v4 | 64 | 64 | 4 | default | 76,483 | 75,149 | 71,826 | -6.1% | -4.4% |
| HalfCheetah-v4 | 256 | 64 | 4 | default | 88,975 | 93,655 | 94,474 | +6.2% | +0.9% |

All 54 variant/case ranges, sample counts, coefficients of variation (sample
standard deviation divided by mean), and median vector-call latencies are in
[results/dispersion.txt](results/dispersion.txt). Unrounded summary numbers,
including CPU/wall-time ratios, are in [results/summary.json](results/summary.json).
Min/max are observed sample ranges, not confidence intervals.

## Measurement method and machine

- Linux 6.18.44, x86-64, AMD EPYC 9V74; nine logical CPUs available (0–8).
- Python 3.12.14, EnvPool 1.2.7, NumPy 2.5.3, Gymnasium 1.3.0.
- Public Gymnasium API, seed 42, constant preallocated zero actions. Imports,
  environment creation, initial resets, and 400 warmup vector calls are untimed.
- Full batch: `reset()` then `step()`. Partial batch: `async_reset()`, `recv()`,
  then repeated `send()`/`recv()` using the returned environment IDs.
- Each sample runs at least two seconds, checking the wall clock every 128 vector
  calls. Each runs in its own Python process. Variants run serially in
  original/rebuilt/candidate order on even repetitions and reversed order on odd
  repetitions. No builds or correctness tests ran concurrently with these samples.
- Pinned: caller CPU 0; worker CPUs 1 through T. Caller pinning happens after pool
  construction, so the background stock-buffer allocation thread retains its
  normal scheduling. Default: the OS chooses CPUs from the available set.
- `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Process CPU seconds include
  threads. Peak RSS is Linux KiB. Raw `affinity` is the caller's effective CPU set,
  not a measurement of each worker or allocator thread's affinity.
- Pinned classic: 8 cases × 3 variants × 5 samples = 120 rows. Pinned heavy:
  4 × 3 × 5 = 60. Default confirmation: 6 × 3 × 3 = 54. Total: 234.

The three `results/final-*.jsonl` files retain every timing, count, memory, label,
and affinity field from the recorded final rows. Only the absolute imported
module filesystem path (`module`) was removed for publication. Imports were
verified locally before the measurements. The published runner avoids emitting
private module paths and verifies an explicit runtime root when supplied.

## Controls and build limitations

`original` is the unchanged installed original. `rebuilt` is unchanged source at
commit `9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3`, rebuilt with the same compiler
and flags as `candidate`. The same-flags control is important: the installed
original alone cannot isolate code changes from build-toolchain effects.

The two rebuilt timed production clients are classic-control and MuJoCo Gym.
This was a standalone native-client rebuild with GCC 14.2.0, C++17, and flags:

```text
-O3 -g0 -DNDEBUG -fPIC -fvisibility=hidden -fno-omit-frame-pointer
-ffunction-sections -fdata-sections -fstack-protector
-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1 -pthread
```

Original Bazel link parameter files were reused with rebuilt EnvPool objects and
redirected outputs. Unchanged external native dependencies were reused read-only.
No installed original binary or original checkout was replaced. All imports were
from distinct runtime copies outside the source checkout. Other unrebuilt modules
in those copies remained original and do not provide performance evidence for
those families. Dummy bindings were rebuilt for regression tests only.

The original machine's Bazel cache and native dependencies are not included.
These tools run against independently built installations; they do not recreate
that private build layout. Build both source revisions with the same toolchain,
flags, assets, and dependency versions before a new comparison. Record those
build details with your new results. Full clean Bazel wheel/release builds,
all-family coverage, other operating systems, and CUDA/GPU XLA remain outside
this validation. The measured hardware and scheduling variability also limit
transfer to other machines.

## Run a fresh comparison

The scripts are standalone Python 3.12+ programs. The interpreter(s) must already
have NumPy, Gymnasium, the chosen EnvPool native modules and required assets
installed. Do not benchmark a source checkout lacking compiled extensions.
Run from an otherwise idle machine. Different Python dependency versions would
confound the build comparison.

Set `PYTHON` to an explicit interpreter path. `ORIGINAL_ROOT`, `REBUILT_ROOT`,
and `CANDIDATE_ROOT` are directories **containing** each build's `envpool` package,
not the package directory itself. Set `BENCH` to this benchmark directory and
`OUT` to a fresh results directory. Quoted variables support spaces in paths.

```sh
mkdir -p "$OUT"
"$PYTHON" "$BENCH/run_matrix.py" --python "$PYTHON"   --original-root "$ORIGINAL_ROOT" --rebuilt-root "$REBUILT_ROOT"   --candidate-root "$CANDIDATE_ROOT" --matrix classic --pin   --reps 5 --seconds 2 --out "$OUT/final-pinned-classic.jsonl"
"$PYTHON" "$BENCH/run_matrix.py" --python "$PYTHON"   --original-root "$ORIGINAL_ROOT" --rebuilt-root "$REBUILT_ROOT"   --candidate-root "$CANDIDATE_ROOT" --matrix heavy --pin   --reps 5 --seconds 2 --out "$OUT/final-pinned-heavy.jsonl"
"$PYTHON" "$BENCH/run_matrix.py" --python "$PYTHON"   --original-root "$ORIGINAL_ROOT" --rebuilt-root "$REBUILT_ROOT"   --candidate-root "$CANDIDATE_ROOT" --matrix confirm   --reps 3 --seconds 2 --out "$OUT/final-default-confirm.jsonl"
"$PYTHON" "$BENCH/summarize.py"   "$OUT/final-pinned-classic.jsonl" "$OUT/final-pinned-heavy.jsonl"   "$OUT/final-default-confirm.jsonl" --json-out "$OUT/summary.json"   --dispersion-out "$OUT/dispersion.txt"
```

Alternatively omit the root flags and use three installed environments via
`--original-python`, `--rebuilt-python`, and `--candidate-python`. `--python` is
still required as the fallback interpreter. The launcher removes `PYTHONPATH`
and uses a temporary working directory, but preserves explicitly configured
asset paths. Pinning is Linux-only and validates requested CPUs against the
process's allowed set. Customize `--caller-cpu` and `--worker-offset` for your
machine; that changes the affinity configuration from the published experiment.

Output files must be fresh; the launcher refuses to append to an old experiment.
Use `--dry-run` to inspect commands without importing EnvPool or benchmarking.
The summary requires explicit input files and rejects duplicate case/variant/rep
identities, preventing accidental pooling of repeated experiments. Keep different
candidate revisions in separate experiment directories.

## Correctness and allocation evidence

The local original/rebuilt/candidate rollout check compared 5,499 arrays exactly,
covering CartPole, Acrobot, Pendulum and HalfCheetah: 12 full-stream configurations,
240 responses per environment, episodes capped at 79 steps, retained output
batches, and additional shuffled/subset resets with 100 steps per phase. Async
streams are grouped by environment and sequence, not nondeterministic arrival
order. The runnable recorder and dtype/shape/byte comparator are included;
large NPZ archives are intentionally excluded.

```sh
"$PYTHON" "$BENCH/check_rollouts.py" record --variant original   --envpool-root "$ORIGINAL_ROOT" --out "$OUT/original.npz"
"$PYTHON" "$BENCH/check_rollouts.py" record --variant rebuilt   --envpool-root "$REBUILT_ROOT" --out "$OUT/rebuilt.npz"
"$PYTHON" "$BENCH/check_rollouts.py" record --variant candidate   --envpool-root "$CANDIDATE_ROOT" --out "$OUT/candidate.npz"
"$PYTHON" "$BENCH/check_rollouts.py" compare "$OUT/original.npz"   "$OUT/rebuilt.npz" "$OUT/candidate.npz"
```

The accompanying core validation passed 31 native tests, 11 dummy Python tests,
5 classic-control suites and 22 focused ASan/UBSan tests. Leak detection had to
be disabled because LeakSanitizer cannot inspect threads in this sandbox;
this is not a leak-clean claim. Additional bounded evidence is documented in:

- [CPU PPO parity](ppo/README.md), with identical training/evaluation traces and
  checkpoints; instrumented concurrent-run timing is not speed evidence.
- [CPU XLA/JAX parity](xla/README.md), including jitted and asynchronous paths.
- [ToyText/MiniGrid rebuilt-family coverage](families/README.md).

`count_allocations.cc` is the standalone diagnostic source used for thread-local
scalar `operator new` counting. Use the optional Bazel target in each source checkout (copy this benchmark
folder into the unchanged checkout if needed):

```sh
bazel run -c opt //benchmark/core_runtime:count_allocations
```

This target uses the repository's normal native dependencies; its full Bazel build
was not rerun during publication packaging. The recorded probe itself was compiled
and run against each revision using the existing read-only native dependencies. It is diagnostic instrumentation, not a replacement allocator for use in
production, and does not count every possible allocation API.

Across 10,000 measured CartPole `EnvStep` calls after 10 warmups, allocation calls
fell from 108,062 to 68,062 (−37.0%). Across 10,000 direct completed
`StateBuffer::Wait` calls, metadata allocations fell from 100,000 to 0. The
`allocations-control.jsonl` and `allocations-candidate.jsonl` files retain both
allocation counts and byte totals. `recv_alloc_*` measures queue-level receive:
it includes scheduling-dependent fallback buffer construction and must not be
interpreted as a fixed per-call allocation cost or an end-to-end speedup.

## Tool smoke tests

```sh
"$PYTHON" "$BENCH/test_tools.py"
ruff check "$BENCH"/*.py
ruff format --check "$BENCH"/*.py
cpplint "$BENCH/count_allocations.cc"
clang-format --style=file --dry-run --Werror "$BENCH/count_allocations.cc"
```

The smoke tests use the checked-in timing rows and synthetic arrays; they do not
run performance benchmarks or real environment rollouts. Reformatting and CLI
parameterization of the measurement scripts do not constitute new timing samples.
