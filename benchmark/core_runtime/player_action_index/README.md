# Rejected experiment: lazy player-action indices

**Do not apply the production patch as a performance recommendation.**
This research branch keeps the published
[direct-state-tuple core, ce1c47f2](https://github.com/wutailong/envpool/commit/ce1c47f238a069454f732a70089d1b9857dcc6de)
unchanged. It adds useful action-value/ownership regressions, an interleaved
player workload and the complete evidence for a rejected optimization.
The [candidate patch](prototypes/lazy_player_indices.patch) is archived unapplied.

The candidate reduced allocations correctly, but the intended normal multiplayer
workload showed no reliable speed benefit and several ordinary configurations
had adverse results in the fixed trial. No extra sampling was added to seek a
favorable result. This is a conservative engineering rejection, not proof of an
inherent exact slowdown or a claim that noise explains every difference.

## Hypothesis and allocation evidence

`Env::ParseAction` first built a vector of matching player-row indices, even
when those rows formed one contiguous slice and the vector was then discarded.
The candidate tracked count/start/end without allocating. Only a later matching
row separated by a gap caused the earlier contiguous indices to be materialized.
It used the original push_back sequence, without reserve, and preserved row order.
A foreign suffix alone did not trigger gathering. Single-player handling,
zero-player owning arrays, per-field gathering and input-batch lifetime stayed
unchanged. No members, persistent caches or queue synchronization changed.

The [allocation probe](allocation_probe.cc) exercises real EnvStep through Action
construction, capturing a thread-local ordinary new/new[] counter at the first
statement of Step. Each of ten cases uses ten warmups and 10,000 measured
non-reset calls. Inputs, checks, state allocation, completion, Wait, background
threads and printing are outside the counted region. Every action value, shape
and source-row order is checked. Per-call counts are constant in these cases:

| Selection | Calls before / after | Requested bytes before / after |
| --- | ---: | ---: |
| Single-player control | 4 / 4 | 48 / 48 |
| Contiguous 1 player, beginning or end | 5 / 4 | 52 / 48 |
| Contiguous 4 players, middle | 7 / 4 | 76 / 48 |
| Contiguous 16 players, beginning or end | 9 / 4 | 172 / 48 |
| Scattered 4 players | 21 / 21 | 548 / 548 |
| Interleaved 16 players | 47 / 47 | 1,700 / 1,700 |
| Zero matches or empty player list | 7 / 7 | 168 / 168 |

All ten action checksums agree. These are requested allocation bytes, not RSS,
live memory or throughput. After whitespace-only formatting of the probe, its
exact final source was also rebuilt against immutable control headers; all
control records matched the earlier run. Both control runs are retained.

## Correctness gates

Local rejected code commit: `4969498a2e15c73fdcbd7b16607cb4c07f2676e2`.
Independent review found no correctness blocker in the small parser change.
The twelve new tests pass on both the unmodified control and candidate. They
cover shared/player scalar and multidimensional payload values, source-row
order, slice versus gathered/zero ownership, batch lifetime, repeated transitions,
a safe empty-inner-dimension slice and unchanged single-player selection.

Candidate gates, distinct from this branch's unchanged production code:

- 88 native core/Dummy tests
- 86 tests in each of three ASan/UBSan repeats and three TSan repeats
- 16 existing Python tests, two normal conversion cases and 50 sanitized cases
- 5,499 exact rollout arrays and eight exact CPU XLA records
- Full synchronous PPO: 101 checkpoints, 223 arrays, 4,925 tensors and
  106,238 scalars; maximum absolute difference zero
- Four Box2D diagnostic cases: 260 equal records, 1,040 inner Container arrays
  and eight retained payloads after pool/outer deletion
- 23 inherited harness tests and ten new interleaving-helper tests
- Real interleaved-workload smoke checks on both frozen runtimes; each warmup
  observed p=1,2,3 and sent 260 genuinely scattered p>1 action batches

Four clients were rebuilt: Classic Control, MuJoCo Gym, Dummy and Box2D with
ENVPOOL_TEST. Two long Dummy tests pass natively but are excluded from sanitizer
runs. LeakSanitizer is off; dependencies/Python are not all instrumented; the
conversion sanitizer uses NumPy 1.26.4. This is not all-family/platform validation,
GPU or Container-valued XLA, general asynchronous PPO, or whole-program leak
freedom. Published tests are active; the candidate env.h patch is not.

## Fixed runtime results

The [plan](results/plan.json) fixes 160 general/Box2D, 64 normal Dummy and 32
interleaved Dummy samples, plus sixteen full-budget PPO runs. Each throughput
case has eight balanced blocks, a/b identical control and c/d identical candidate,
orders acdb/cabd/bdca/dbac, 400 untimed warmup calls and at least three measured
seconds. Runs are serial, without concurrent builds/tests or desktop sampling.
All samples and adverse blocks are retained. Rates count environment responses,
including autoresets, rather than vector calls or player rows.

The [interleaved stress workload](SCATTERED.md) explicitly sets all 64 env seeds
to 42, then applies one precomputed permutation to all three player fields.
This synchronized workload stays separate from normal-seed Dummy. Its identical
NumPy indexing/copy work is included in both variants. It is not an isolated
C++ parser microbenchmark. Box2D below also uses its diagnostic test-mode spec.

Paired block log-rate effects, positive as faster; intervals are descriptive
10,000-block-bootstrap intervals, not significance or non-regression guarantees:

| Configuration (environments / batch / threads) | Rate effect | Exploratory 95% interval |
| --- | ---: | ---: |
| CartPole 20 / 20 / 1 | -4.04% | -7.10% to -1.16% |
| CartPole 256 / 256 / 4 | -0.80% | -4.14% to +2.38% |
| CartPole 1024 / 256 / 8 | -4.45% | -7.21% to -1.57% |
| HalfCheetah 256 / 64 / 4 | +1.08% | +0.12% to +1.94% |
| BipedalWalker test mode 64 / 16 / 4 | -1.22% | -2.16% to -0.27% |
| Dummy max players 1, 64 / 64 / 4 | -0.27% | -7.29% to +7.13% |
| Dummy max players 4, 64 / 64 / 4 | -3.98% | -8.78% to +0.91% |
| Interleaved Dummy max players 4, 64 / 64 / 4 | -1.13% | -8.66% to +7.36% |

CartPole 20, eight-thread CartPole and diagnostic Box2D are negative in 6/8, 7/8
and 6/8 blocks. HalfCheetah's favorable result is also retained. The targeted
normal multiplayer and scattered workloads do not establish a gain. A/A
variation is substantial: normal Dummy 4 control b/a is +10.26%; interleaved
control b/a is +6.66%. These observations do not identify the cause of the
adverse ordinary results. Unchanged single-player source logic is not proof of
unchanged compiled-code performance.

PPO uses four fixed four-label blocks, 256,000 responses, 100 updates and 8,000
optimizer steps per run after two discarded priming updates. Control median is
19.727 s (19.387–20.253 s); candidate 19.903 s (19.203–20.327 s). The paired rate effect
is -0.72%, interval -1.39% to +0.21%, inconclusive; all 16 semantic fingerprints
match. Candidate same-binary d/c itself is -1.04%, interval -1.70% to -0.35%.
This reinforces the need for caution without proving a noise-only explanation.

## Reproduce and inspect

The archived patch applies to the baseline env.h; leave it unapplied to use this
research branch's production core. Use the recorded compiler flags and separate
runtime roots. Candidate source and benchmark helpers plus 84 native manifest
entries were checked before/after timing; reused modules are not rebuilds.
General rows rely on frozen root mapping/manifests; Dummy, interleaved and PPO
also verify row-level hashes. Short functional smoke timings are not pooled
into the fixed performance trial.

```sh
D=benchmark/core_runtime/player_action_index
R="$D/results"
python "$D/../experiments/summarize_paired_blocks.py" "$R/performance.jsonl" --plan "$R/plan.json" --json-out general-recomputed.json
python "$D/../experiments/summarize_paired_blocks.py" "$R/container-performance.jsonl" --plan "$R/container-performance.jsonl.plan.json" --json-out container-recomputed.json
python "$D/../experiments/summarize_paired_blocks.py" "$R/interleaved-performance.jsonl" --plan "$R/interleaved-performance.jsonl.plan.json" --json-out interleaved-recomputed.json
python "$D/summarize_paired_ppo.py" "$R/ppo-throughput" --output ppo-recomputed.json
```

Evidence: [allocation](results/allocation-comparison.json), [validation](results/validation.json),
[native](results/native.xml), [ASan/UBSan](results/address.log), [TSan](results/thread.log),
[general](results/performance-summary.json), [normal Dummy](results/container-performance-summary.json),
[interleaved](results/interleaved-performance-summary.json), [PPO](results/ppo-throughput/summary.json),
[build commands](results/build-commands.json), [provenance](results/runtime-provenance.json).
Raw records accompany every summary. No binaries, weights, credentials or
private machine paths are published. The useful deliverable is reproducible
coverage and a documented rejection; the production recommendation stays with
the earlier validated core version and its stated scope.
