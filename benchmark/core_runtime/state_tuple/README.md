# Direct typed state tuples

Retain this candidate for its exact per-state allocation reduction and
workload-specific gains in the fixed measurement window below. Small CartPole,
HalfCheetah and full PPO timings are inconclusive; there is no universal speedup
claim. The source is frozen at local code commit
`d1bdd646d667c83d4865bb117849bbcc46b285aa`. Independent source/concurrency and final measurement reviews found no blocking
issue within the stated scope.

The baseline is the published Container-storage runtime
[`bf2f16c0d724d01c480668703012342ab1f0fd4e`](https://github.com/wutailong/envpool/commit/bf2f16c0d724d01c480668703012342ab1f0fd4e)
on `perf/core-container-storage`. Earlier experiments' timings are not pooled
with this experiment.

## Mechanism and compatibility boundaries

`AllocateTuple<Values>` constructs typed Array views inline, removing the
intermediate `std::vector<Array>` allocation. The vector-returning
`StateBuffer::Allocate` and `StateBufferQueue::Allocate` remain supported. Both
paths share reservation/view helpers; `Env::Allocate` moves typed views into its
State and initializes Container fields through tuple references. Only the
completion callback remains in the Env's saved slice.

- Mixed vector/tuple callers retain quota accounting, ordered single-player
  views, queue selection and explicit completion behavior.
- Reservation is not transactional: exceptions after reservation do not roll
  back counters or offsets. Discarding a slice does not complete it.
- Tuple-arity mismatch is a fatal `CHECK_EQ`, not a recoverable exception. It
  precedes StateBuffer reservation, but the queue ticket is already selected.
- Compatibility is source-level for rebuilt clients; there is no old/new
  mixed-header ABI guarantee.

## Measured allocation audit

The inherited [CartPole probe](../action_metadata/prototypes/count_action_allocations.cc)
was built before the edit against the source-equivalent published baseline and
again after it. After ten warmup steps, its ordinary thread-local `operator new`
counter measures 10,000 `EnvStep` calls. Both runs classify 969 as resets and
9,031 as non-resets:

| Step class | Count | Allocation calls before / after | Requested bytes before / after |
| --- | ---: | ---: | ---: |
| Reset | 969 | 5 / 4 | 608 / 32 |
| Non-reset | 9,031 | 7 / 6 | 624 / 48 |
| Total | 10,000 | 68,062 / 58,062 | 6,224,496 / 464,496 |

Exactly one allocation and 576 requested bytes disappear per measured EnvStep:
10,000 calls and 5,760,000 bytes in total. These are compiler-specific requested
allocation traffic, not retained memory, allocator footprint, RSS or throughput.
Wait-metadata allocation counts remain zero. Variable receive-caller allocation
counts are excluded from causal comparisons. See the [comparison](results/allocation-comparison.json)
and raw [control](results/allocation-control.jsonl)/[candidate](results/allocation-candidate.jsonl).

## Executed correctness gates

- 76 native core/Dummy tests, including 12 new tuple-path regressions
- 74 tests in each of three ASan/UBSan repeats and three TSan repeats
- 16 existing Python tests; two normal NumPy lifetime/partial-conversion cases
  plus 50 sanitized case executions
- 5,499 exact rollout arrays and eight exact CPU XLA records
- Full synchronous PPO parity: 101 checkpoints, 223 arrays, 4,925 tensors and
  106,238 scalars, with maximum absolute difference zero
- Four Box2D cases: 260 identical recursive records, 1,040 inner Container arrays
  and eight retained payloads after pool/outer deletion
- 23 inherited benchmark-harness tests

Classic Control, MuJoCo Gym, Dummy and Box2D clients were rebuilt; both Box2D
clients use `ENVPOOL_TEST` for diagnostic Container outputs. The 12 new tests
cover view layouts, empty/player slices, ordering, mixed reservations, fatal
arity checks, partial queue cycles, delayed completion, retirement and Container
initialization. Two existing long Dummy stress tests pass natively but are
excluded from sanitizer runs. LeakSanitizer is off, dependencies/Python are not
all instrumented, and the NumPy sanitizer conversion probe uses NumPy 1.26.4.
All-family release builds, other platforms, GPU or Container-valued XLA, and
general asynchronous PPO are outside this validation.

## Fixed runtime experiment

Following the [preceding methodology](../container_storage/README.md), five
general/Box2D configurations and two Dummy-player configurations each use eight
balanced blocks: identical control labels `a/b`, identical candidate labels
`c/d`, orders `acdb`, `cabd`, `bdca`, `dbac`, seed 42, 400 untimed warmup calls
and at least three measured seconds per fresh process. All 160 general and 64
Dummy samples are retained. Rates count environment responses, including
autoresets; Dummy includes normal Python Container conversion. No builds, other
benchmarks, sanitizers or desktop sampling ran during measurement.

Effects are paired block log-rate contrasts, positive as faster. The 95%
intervals use 10,000 complete-block bootstrap resamples and are descriptive,
not significance guarantees.

| Configuration (environments / batch / threads) | Rate effect | Exploratory 95% interval |
| --- | ---: | ---: |
| CartPole 20 / 20 / 1 | +3.41% | -4.72% to +10.10% |
| CartPole 256 / 256 / 4 | +26.10% | +18.31% to +36.26% |
| CartPole 1024 / 256 / 8 | +10.11% | +3.08% to +17.97% |
| HalfCheetah 256 / 64 / 4 | -0.13% | -3.92% to +3.88% |
| BipedalWalker test mode 64 / 16 / 4 | +5.02% | +3.66% to +6.61% |
| Dummy, max players 1, 64 / 64 / 4 | +17.30% | +9.39% to +25.62% |
| Dummy, max players 4, 64 / 64 / 4 | +9.47% | +1.76% to +17.88% |

CartPole 256 and BipedalWalker are positive in all eight blocks. Substantial
same-binary variation remains: candidate `d/c` effects are +8.69% for small
CartPole, -6.27% for eight-thread CartPole and +10.21% for one-player Dummy.
General rows lack embedded native hashes, so pair identity relies on separate
frozen-runtime provenance; Dummy and PPO verify hashes in their records. Source
and all 84 native manifest entries (including reused modules, not 84 rebuilds) are unchanged before and after measurement.
These results support workload-specific gains in this window, not portability
of their magnitude or absence of regressions elsewhere.

### Initial full-budget PPO trial

The [driver](run_paired_ppo.py) runs exactly four fixed blocks in the same four
orders: 16 fresh processes, each with 256,000 responses, 100 updates and 8,000
optimizer steps after two discarded priming updates. Control median is 20.026 s
(range 19.468–20.685 s); candidate median is 20.014 s (19.658–21.128 s). The primary
paired rate effect is -0.55%, interval -2.30% to +1.22%; this is inconclusive.
Same-binary `b/a` and `d/c` effects are -0.46% and +2.00%. All 16 semantic
fingerprints agree. This was the initial predeclared trial, with no additional
confirmation, adaptive sampling or discarded adverse timing window.

## Reproduction and evidence

The [original plan](results/plan.json) declares `expected_general_samples=160`.
The [analysis plan](results/analysis-plan.json) only aliases it as
`expected_samples` for the unchanged inherited summarizer; sampling did not
change. Keep both records. Recompute summaries into fresh output files:

```sh
D=benchmark/core_runtime/state_tuple
R="$D/results"
python "$D/../experiments/summarize_paired_blocks.py" "$R/performance.jsonl" --plan "$R/analysis-plan.json" --json-out general-recomputed.json
python "$D/../experiments/summarize_paired_blocks.py" "$R/container-performance.jsonl" --plan "$R/container-performance.jsonl.plan.json" --binary-hash-field binary_sha256 --json-out container-recomputed.json
python "$D/summarize_paired_ppo.py" "$R/ppo-throughput" --output ppo-recomputed.json
python "$D/run_paired_ppo.py" --python "$PPO_PYTHON" --baseline-root "$CONTROL_ROOT" --candidate-root "$CANDIDATE_ROOT" --output ppo-fresh
```

Key records: [validation](results/validation.json), [native](results/native.xml),
[ASan/UBSan](results/address.log), [TSan](results/thread.log),
[general throughput](results/performance-summary.json), [Dummy throughput](results/container-performance-summary.json),
[PPO throughput](results/ppo-throughput/summary.json), [build commands](results/build-commands.json),
[freeze before](results/freeze-before.json) and [freeze after](results/freeze-after.json).
Raw rows accompany each summary. Published evidence excludes private machine
paths, binaries, weights, credentials and other private artifacts.
