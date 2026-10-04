# Direct Python receive list: not retained

**Decision: do not apply this production change.** One vector allocation is
reliably removed, but the fixed throughput study below does not establish a
useful speed benefit and has an adverse multi-player Dummy signal. The
[prototype](prototypes/direct_list.patch) remains unapplied; current production
core is unchanged from ce1c47f2. This report and its tests preserve the finding
without adding generated raw records to the repository.

This focused candidate-header probe compares the old `ToNumpy` plus
`std::vector<py::array>` return caster against the actual candidate
`PyEnvPool<FakePool>::PyRecv`. It is an allocation and public-contract audit,
not a throughput benchmark. The existing native `ToNumpy` helper is used
unchanged by the comparator.

## Probe scope

The probe binds the actual candidate `PyEnvPool<FakePool>::PyRecv`, alongside
an exact copy of the old vector-return body and its STL list caster. Fake
`Recv` moves pre-staged positive-shaped arrays, so setup allocation is excluded.
Six tests cover list/order/dtype/shape/value/mutability/zero-copy contracts, GIL
release, retained primitive and Container views, exact-once payload reclamation,
allocation deltas, and partial-conversion cleanup.

Only calling-thread ordinary C++ new/new[] requested bytes are counted. Python,
NumPy, malloc, aligned allocation and other threads are outside the counter.
The fault sweep covers only intercepted allocation sites, not process-wide
leaks or Python/NumPy out-of-memory behavior. No malformed or empty arrays are
constructed. Counters are diagnostic, not a timing benchmark.

## Build and run

Apply the prototype only to a separate, clean experimental checkout of this
branch, not to the retained runtime used as control:

```sh
git apply --check --unidiff-zero benchmark/core_runtime/recv_list/prototypes/direct_list.patch
git apply --unidiff-zero benchmark/core_runtime/recv_list/prototypes/direct_list.patch
```

Use a separate output directory outside that checkout. The extension needs
C++17, Python and pybind11 headers, the candidate checkout root, Abseil headers
and its usual logging link dependencies, and XLA FFI headers. It does not need
an environment implementation, concurrentqueue, ThreadPool, MuJoCo, or OpenCV.
Build without LTO. A static assertion rejects pre-change `PyRecv` headers.

On Linux/ELF, `-Wl,-Bsymbolic-functions` is required to bind this probe's own
operator-new/delete calls to its replacements. Import using Python's ordinary
local extension loading; do not use `LD_PRELOAD` or `RTLD_GLOBAL` for the probe.
Do not link these replacements into a production extension. A self-test verifies one explicitly requested 37-byte allocation.

For example, using matched existing dependency paths and PIC Abseil archives:

```sh
mkdir -p "$OUTPUT"
"$CXX" -std=c++17 -O3 -g0 -fPIC -fvisibility=hidden -pthread \
  $("$PYTHON" -m pybind11 --includes) \
  -I"$SOURCE" -I"$ABSEIL_HEADERS" -I"$XLA_FFI_HEADERS" \
  -c "$SOURCE/benchmark/core_runtime/recv_list/recv_list_probe.cc" \
  -o "$OUTPUT/recv_list_probe.o"
"$CXX" -shared -Wl,-Bsymbolic-functions -pthread \
  "$OUTPUT/recv_list_probe.o" \
  -Wl,--start-group $ABSEIL_PIC_ARCHIVES -Wl,--end-group \
  -o "$OUTPUT/recv_list_probe.so"
PYTHONPATH="$OUTPUT" "$PYTHON" -m unittest discover \
  -s "$SOURCE/benchmark/core_runtime/recv_list" -p 'test_recv_list.py' -v
PYTHONPATH="$OUTPUT" "$PYTHON" \
  "$SOURCE/benchmark/core_runtime/recv_list/bench_allocations.py" \
  --iterations 1000 > "$OUTPUT/allocations.json"
```

If pybind11 is available only as existing build-tree headers, replace the
`python -m pybind11 --includes` substitution with the matching Python and
pybind11 include flags from that build. Preserve matched compiler, Python,
NumPy, and dependency versions in any reported results. A sanitizer build
adds `-O1 -g1 -fsanitize=address,undefined` at compile and link; the Python
process may require its matching ASan and C++ runtimes preloaded first.
Sanitizer observations do not change the bounded allocation-counter scope.

## Verified correctness and allocation results (2026-10-04)

The archived candidate patch changes only `envpool/core/py_envpool.h`. Its comparator
is the retained cumulative [ce1c47f2 core](https://github.com/wutailong/envpool/commit/ce1c47f238a069454f732a70089d1b9857dcc6de), also present in
[73977d9f](https://github.com/wutailong/envpool/commit/73977d9f75313c1b261b1b726591e9076aeabba4).
These are incremental comparisons against that retained implementation, not
speed ratios against original main.

The pinned pybind11 3.1.0 STL caster first creates the final Python list, then
increments every array reference while the temporary vector subsequently
decrements each reference. Direct list construction eliminates the vector and
those per-array pairs. Casting the final `py::list` itself still has one ordinary
list INCREF/DECREF pair, so the isolated net refcount reduction is N-1 pairs for
N fields. The GIL-release region and array/capsule conversion are unchanged.
In the unapplied prototype, `ToNumpy(vector*)` is preserved and `PyRecv` has
C++ return type `py::list`.
This is not a promise of source compatibility for external callers explicitly
expecting `std::vector<py::array>`, or a mixed-header ABI guarantee. The private
Python `_recv` still returns an ordinary list; its generated type annotation
may be less specific.

Each positive-shape probe case ran 1,000 receives in each mode:

| Fields | Native allocation calls / receive, old → new | Requested bytes saved / receive |
| ---: | ---: | ---: |
| 1 | 4 → 3 | 8 |
| 4 | 13 → 12 | 32 |
| 8 | 25 → 24 | 64 |
| 16 | 49 → 48 | 128 |
| 32 | 97 → 96 | 256 |

All consumed checksums agree. This verifies exactly one temporary-vector
allocation removed per receive in this compiler/probe, not a change in NumPy
payload size, process memory, or training speed.

Passed checks:

- Six new binding/allocation/lifetime tests on NumPy 2.5.3 and 1.26.4.
- The same six tests in three ASan/UBSan runs (NumPy 1.26.4). Controlled
  allocation failures cover 23 legacy and 22 direct-list allocation sites,
  plus a successful sentinel for each; all tracked allocations and payloads
  are reclaimed exactly once.
- Eleven existing Dummy and five Classic Control Python tests; 5,499 exact
  rollout arrays across CartPole, Acrobot, Pendulum and HalfCheetah.
- Eight exact CPU XLA records for CartPole/HalfCheetah, including JIT, scan and
  canonicalized per-environment asynchronous streams.
- Full synchronous CPU CartPole PPO: 256,000 responses, 100 updates and 8,000
  optimizer steps; 101 checkpoints, 223 arrays, 4,925 tensors and 106,238 scalars
  equal the retained recording exactly, maximum difference zero.
- ToyText/MiniGrid: 59 cross-runtime cases, 6,575 arrays and 65,304,189 bytes
  exactly equal; 17 ToyText tests and two MiniGrid determinism methods across
  82 MiniGrid-prefixed IDs pass. BabyAI cross-runtime coverage is one of 96
  registered tasks. Render comparison is DoorKey only, using 12 frame arrays
  across 84 selected environment frames.
- Twenty-three existing experiment-helper tests; Ruff, clang-format, cpplint
  and `git diff --check` for the changed files.

Classic Control, MuJoCo Gym and Dummy binding translation units were rebuilt
with matching C++17/O3 flags. ToyText and the explicit MiniGrid binding were
also rebuilt; MiniGrid's 16 unchanged project units were reused only after
checking their source hashes and dependency files exclude `py_envpool.h`.
Unchanged renderer/external archives remain reused. This is five validated
families across separate frozen runtime roots, not a rebuilt all-family wheel.
The 71 shared runtime Python files match byte-for-byte; candidate source
staging also includes 110 test/oracle helper files absent from the installed
comparator, which ordinary runtime registration does not import.

No queue, worker, scheduling, ownership implementation or XLA implementation
changed. ASan/UBSan instrumented the focused conversion module; Python,
NumPy and external dependencies were not all instrumented, and LeakSanitizer
was disabled. No new TSan run, other-platform, GPU, Box2D, all-BabyAI, upstream
MiniGrid-oracle or full-release validation is implied. Asynchronous arrival
order and general asynchronous PPO trajectories are not required to match.

The first probe fixture incorrectly nested a captured storage owner behind an
Array control block with weak observers; both paths consequently failed its
lifetime counters. The fixture was corrected to alias actual typed storage
with non-cycling weak counter observers before the results above were recorded.
No production ownership change was made to accommodate the test.

## Fixed throughput study

The predeclared trial compares duplicate same-binary control labels a/b with
candidate c/d. Eight blocks use orders acdb, cabd, bdca, dbac twice, randomized
case ordering with seed 20261003, seed 42 actions, 400 untimed warmup calls and
at least three measured seconds per fresh process. Four cases in
[cases.json](cases.json) give 128 samples; the unchanged two-case Dummy driver
adds 64. The separate fixed four-block PPO driver gives 16 full-budget runs,
with two discarded priming updates followed by 256,000 responses, 100 updates
and 8,000 optimizer steps. Environment rates count responses including normal
autoresets, not vector calls or exclusively physics transitions.

All performance work is serial, with no simultaneous build, regression tests
or desktop sampling. Production header and native hashes are frozen before
and after. Paired log-rate effects and 10,000 complete-block bootstrap
intervals are descriptive in this shared cloud, not significance guarantees.
A/A effects, negative blocks, sample dispersion and PPO results must be
considered together. The decision plan does not add samples to search for
favorable results.

### Completed result and decision

All 192 environment samples and all 16 PPO trials completed; all 42 native
manifest entries across the two timed roots and all production header hashes
were unchanged before/after measurement. Only three clients in each timed
candidate root are new builds; the manifest count includes reused support
modules. Every PPO semantic fingerprint matches.

Positive effects below mean faster candidate **relative to ce1**, using paired
block log-rate contrasts rather than ratios of independent medians:

| Case: environments / batch / threads | Paired effect | Descriptive 95% interval | Negative blocks | A/A control / candidate effect |
| --- | ---: | --- | ---: | ---: |
| CartPole-v1 20/20/1 | +0.11% | -4.21% to +5.38% | 4/8 | -6.34% / -5.56% |
| CartPole-v1 256/256/4 | -3.97% | -14.67% to +8.59% | 5/8 | +17.37% / +1.45% |
| CartPole-v1 1024/256/8 | -5.32% | -14.23% to +4.85% | 6/8 | +1.97% / +0.04% |
| HalfCheetah-v4 256/64/4 | +0.19% | -2.30% to +2.97% | 4/8 | -4.94% / -2.71% |
| DummyPlayers1 64/64/4 | +3.46% | -1.13% to +8.76% | 3/8 | +2.65% / -4.61% |
| DummyPlayers4 64/64/4 | -8.30% | -14.35% to -0.99% | 7/8 | +4.10% / -3.82% |

For dispersion, each implementation has 16 individual samples per row. Rates
are environment responses/s; brackets are observed min/max, not confidence
intervals. CV is sample standard deviation divided by the mean.

| Case | Control median [range] | Candidate median [range] | CV control / candidate |
| --- | ---: | ---: | ---: |
| CartPole-v1 20/20/1 | 416,901 [332,551, 503,600] | 411,544 [344,709, 486,083] | 13.01% / 10.20% |
| CartPole-v1 256/256/4 | 2,248,752 [1,507,789, 2,588,593] | 2,171,454 [1,424,796, 2,601,809] | 17.51% / 20.62% |
| CartPole-v1 1024/256/8 | 3,776,993 [3,064,771, 5,374,869] | 3,643,729 [3,241,984, 5,190,121] | 15.58% / 12.81% |
| HalfCheetah-v4 256/64/4 | 97,074 [80,039, 103,391] | 95,824 [82,575, 102,342] | 6.76% / 5.13% |
| DummyPlayers1 64/64/4 | 742,290 [484,139, 804,837] | 723,855 [494,519, 829,762] | 16.68% / 14.33% |
| DummyPlayers4 64/64/4 | 495,524 [360,869, 570,665] | 452,513 [322,556, 537,568] | 13.32% / 15.10% |

Full-budget PPO paired rate effect is **-0.63%**, interval **-2.49% to +0.86%**;
two of four blocks are negative. Control median training time is 20.186 s
(range 19.169–20.480), candidate 20.014 s (19.531–21.111). The median-time
comparison and paired log-rate estimand are different, explaining their
different signs. Same-binary effects are -1.33% for control and +3.84% for
candidate; the candidate A/A interval itself is +2.26% to +5.38% despite
identical code, underscoring the limited causal meaning of these small-window
intervals. No faster-training claim is supported.

Multi-player Dummy is negative in seven of eight blocks, with an adverse
interval in this window. Large A/A variation prevents calling -8.30% a proven
portable code-caused slowdown; it also does not justify dismissing the adverse
result as entirely noise. The other cases do not establish a useful net speed
improvement. Given the small per-receive allocation saving and changed C++
return type, retain the current implementation and reject this prototype.
No additional samples were added to search for a favorable outcome. The next
independent candidate targets the larger temporary ActionSlice batch, without
carrying this change forward.

Raw samples, manifests, checks and initial setup failures remain in the bounded
local experiment records. This branch publishes conclusions, variance, methods,
input configuration, tests and the unapplied patch, not generated run files.
Historical evidence links elsewhere still refer to their original revisions;
they are not evidence for this candidate. Hosted CI status is reported at
publication; local passes alone are not a hosted CI pass.



### Reproduce the fixed timing plan

Build and freeze independent retained-control and candidate runtime roots first,
matching the native toolchain, Python dependencies and assets in [the review](../REVIEW.md#复现前核对).
`CONTROL` and `CANDIDATE` below contain their respective `envpool/` directories;
`OUTPUT` is a new directory outside either source checkout. From the repository
root, with the general and PPO Python interpreters supplied explicitly:

```sh
STUDY=benchmark/core_runtime
mkdir "$OUTPUT"
"$PYTHON" "$STUDY/experiments/run_blocks.py" --python "$PYTHON" \
  --variant a="$CONTROL" --variant b="$CONTROL" \
  --variant c="$CANDIDATE" --variant d="$CANDIDATE" --paired-replicates \
  --cases "$STUDY/recv_list/cases.json" --blocks 8 --seconds 3 \
  --warmup 400 --affinities default --out "$OUTPUT/general.jsonl"
"$PYTHON" "$STUDY/experiments/summarize_paired_blocks.py" \
  "$OUTPUT/general.jsonl" --plan "$STUDY/recv_list/plan.json" \
  --json-out "$OUTPUT/general-summary.json"
"$PYTHON" "$STUDY/container_ownership/run_container_blocks.py" \
  --python "$PYTHON" --baseline-root "$CONTROL" \
  --candidate-root "$CANDIDATE" --out "$OUTPUT/container.jsonl"
"$PYTHON" "$STUDY/experiments/summarize_paired_blocks.py" \
  "$OUTPUT/container.jsonl" --plan "$OUTPUT/container.jsonl.plan.json" \
  --json-out "$OUTPUT/container-summary.json"
"$PPO_PYTHON" "$STUDY/state_tuple/run_paired_ppo.py" \
  --python "$PPO_PYTHON" --baseline-root "$CONTROL" \
  --candidate-root "$CANDIDATE" --output "$OUTPUT/ppo"
"$PYTHON" "$STUDY/state_tuple/summarize_paired_ppo.py" \
  "$OUTPUT/ppo" --output "$OUTPUT/ppo-summary.json"
```

Record selected module/source hashes before and after your new run. General
sample rows do not embed native hashes; the plan alone does not prove binary
identity. Dummy and PPO additionally verify native identity in their samples.
Keep raw runs locally, including adverse/failed attempts, rather than adding
new generated logs, binaries, checkpoints or sample files to this branch.
