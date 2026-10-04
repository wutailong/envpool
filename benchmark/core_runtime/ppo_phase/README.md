# Where reference PPO spends its time

Raw run files have been removed from the current tree. This report retains the
conclusions, adverse findings, method and limitations; evidence links below point
to the immutable pre-cleanup commit. See [archive and replay instructions](../ARCHIVE.md).

For this fixed CPU CartPole workload, the retained runtime spends about **67.43%
inside PPO update, 24.73% in remaining collection, and 7.79% inside the Python
EnvPool step/reset boundary**. That boundary includes adaptation, conversion,
native execution/waiting and observer cost; it is not pure C++ time.

This branch changes no production EnvPool code, training algorithm or
hyperparameters. It adds observation tools and complete diagnostic evidence.
The retained native runtime is
[ce1c47f2](https://github.com/wutailong/envpool/commit/ce1c47f238a069454f732a70089d1b9857dcc6de);
the comparator is the untouched original
[9c31c547](https://github.com/wutailong/envpool/commit/9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3).
The preceding [player-index candidate](../player_action_index/README.md) remains
unapplied. This is one workload/window, not a universal EnvPool cost breakdown.

## Fixed design and validation

The reference remains CartPole-v1, 20 synchronous environments, two EnvPool
threads, one Torch/intra-op and inter-op thread, deterministic CPU PPO,
100 updates, 256,000 responses and 8,000 Adam steps per run. Two disposable
untimed updates prime libraries/JIT; policy, optimizer, pool and collector are
then recreated after reseeding. Setup, initial reset, fingerprints and output
writing are excluded from the measured training interval. Episode/subset resets
inside collection are included.

The [runner](run_profile_blocks.py) executes exactly 32 fresh processes:

| Labels | Native runtime | Observation |
| --- | --- | --- |
| a / b | Original | Untouched plain harness |
| c / d | Original | Phase profiling |
| e / f | Retained | Untouched plain harness |
| g / h | Retained | Phase profiling |

Four fixed orders are aceghfdb, dfhbagec, egacdbhf and hbdfecag. Each condition's
two replicas bracket the block center; each condition occupies every slot once
across four blocks, and each label appears early/late twice. These are genuine
same-native/same-script A/A pairs within each condition. This balances positions
and linear log-time drift, not arbitrary carryover, thermal or scheduling effects.
There were no concurrent builds, tests or desktop sampling during measurement.
All runs are retained; no favorable-result sampling was added.

- 18 helper tests pass, including exact AST fidelity after removing declared
  observer insertions, fake-clock nesting, proxy forwarding, plan and analysis guards.
- Four short original/retained × plain/profile pilots match semantic fingerprints.
  Their two-update timings are functional checks, not pooled performance evidence.
- All 32 formal runs match configuration, budgets and semantic fingerprints
  covering initial/final policy, optimizer, RNG, return normalization and recorded
  collection/loss metrics. This does not replace a full per-step journal comparison.
- Every profile records 100 calls to each of policy.train, collect, PPO update and
  buffer reset; 12,800 env.step calls and 829 episode/subset env.reset calls.
- All integer nested residuals are nonnegative and every disjoint partition equals
  its rounded wall interval. No profile required the declared 1 µs envelope clamp.
- All measurement-source hashes and 41 native manifest entries remain unchanged.
  These are 20 original and 21 retained entries, not 41 rebuilt modules.

[Instrumentation details](INSTRUMENTATION.md) explain the exact observation diff,
proxy boundaries and failure checks. Local observation-code commit:
`065557d4afcb1a0bbb4f89091de0b0115fc8e47a`.

## Retained-runtime phase breakdown

Each percentage uses that profiled run's own total wall time. The table gives
independently computed medians across eight retained-profile runs, so its medians
need not sum exactly to 100%; each individual run's partition does.

| Disjoint region | Median seconds | Median wall share |
| --- | ---: | ---: |
| PPO update, including processing/learning/optimizer work | 13.494 | 67.434% |
| Collection excluding EnvPool calls | 4.953 | 24.730% |
| EnvPool step calls | 1.493 | 7.498% |
| EnvPool episode/subset reset calls | 0.057 | 0.287% |
| policy.train | 0.010 | 0.050% |
| Buffer reset | 0.009 | 0.043% |
| Loop/other residual | 0.003 | 0.017% |

Combined env-call share has median **7.785%**, range **7.431–8.495%**; combined
elapsed time has median 1.551 s. Original-profile env-call share is 7.827%, range
7.331–8.868%. Original/retained PPO-update shares are 67.418%/67.434%, and remaining
collection shares 24.657%/24.730%. These are descriptive observed phases, not
exclusive native CPU utilization or a causal comparison of every component.

Inclusive collect already contains step/reset. Do not add inclusive collect to
those nested times. Remaining collection includes policy inference, buffering
and other work; it is not synonymous with pure Python or non-core work.
Whole-process CPU/wall medians are 1.027 original and 1.023 retained, including
worker threads; these ratios cannot identify exclusive core CPU time.

## Plain performance and observer cost

Only untouched plain runs provide the primary native-runtime comparison.
Original plain median is 19.812 s (19.601–20.248 s); retained plain median is
19.785 s (19.431–19.967 s). The paired block geometric-mean rate effect is
**+0.56%**, with descriptive bootstrap interval **+0.31% to +0.86%**. All four
block effects are positive: +0.48%, +0.57%, +0.23% and +0.98%. The paired effect
and pooled medians are different statistics, not interchangeable estimates.

This is a small favorable signal in one four-block window, not a guaranteed PPO
speedup or evidence that raw stepping gains transfer unchanged to training.
It compares the installed original with the cumulative retained native build,
not a newly rebuilt same-flags original. The inherited
[build limitations](../README.md#controls-and-build-limitations) apply: the
contrast does not isolate one patch from all cumulative code/build differences.
Same-binary variation remains material. Intervals below use 10,000 complete-block
resamples, seed 738; updates/calls are not independent experimental repetitions.

| Contrast | Effect | Exploratory 95% interval |
| --- | ---: | ---: |
| Retained vs original, plain rate | +0.564% | +0.313% to +0.857% |
| Retained vs original, profiled rate (diagnostic only) | -0.432% | -1.345% to +0.488% |
| Original profile/plain elapsed-time overhead | +0.124% | -0.393% to +0.641% |
| Retained profile/plain elapsed-time overhead | +1.127% | +0.649% to +1.890% |
| Original plain b/a A/A rate | +0.444% | -1.264% to +1.976% |
| Original profile d/c A/A rate | +0.328% | -2.679% to +3.347% |
| Retained plain f/e A/A rate | -0.330% | -1.395% to +1.182% |
| Retained profile h/g A/A rate | +0.306% | -0.922% to +1.548% |

Profiling changes the runtime point estimate's sign. Do not substitute profiled
wall times for the plain comparison, claim negative observed overhead is a free
speedup, or subtract total profiler overhead mechanically from individual phases.
All adverse and negative-overhead blocks remain in the raw data.

## Implication for further core work

Keep training math and hyperparameters unchanged. Within the authorized core
scope, prioritize measured marshaling/call-boundary/synchronization costs and
correctness, and verify end-to-end effects instead of assuming a raw vector-step
speedup applies to this small-batch training loop. Large-batch stepping results
remain workload-specific; the earlier 256-environment benchmark is not this
20-environment/two-thread PPO configuration.

A conditional illustration uses each profile's env-call fraction f: if an
improvement were confined entirely to that region, with all other time unchanged,
a region speedup s would give total speedup 1/(1-f+f/s). On retained profiles,
halving that region models median **+4.05% total throughput** (about 3.89% less
elapsed time); making the entire region free models +8.44%. These are not hard
core ceilings or predictions: the boundary includes non-native work and observer
cost, while background workers, spinning, allocation and cache effects can affect
other phases. No algorithm change follows from this thought experiment.

## Reproduce and inspect

```sh
D=benchmark/core_runtime/ppo_phase
python -B "$D/run_profile_blocks.py" --python "$PPO_PYTHON" --original-package "$ORIGINAL_PACKAGE" --retained-package "$RETAINED_PACKAGE" --output new-trial
python -B "$D/summarize_profile_blocks.py" new-trial --output new-summary.json
```

Here `ORIGINAL_PACKAGE` and `RETAINED_PACKAGE` are the actual `envpool/`
directories, not their parents. `PPO_PYTHON` is the interpreter containing the
shared RL dependencies: recorded Python 3.12.14, NumPy 1.26.4, Torch 2.5.1+cpu,
Tianshou 0.5.1, Gymnasium 1.3.0 and Numba 0.68.0. See the observed
[dependency snapshot](../ppo/requirements.txt) and
[reproduction checklist](../REVIEW.md#复现前核对).

To recompute the **published historical trial** without running any training,
use this separate command instead:

```sh
: "${ARCHIVE_ROOT:?Extract the archive as described in ../ARCHIVE.md}"
python -B "$D/summarize_profile_blocks.py" "$ARCHIVE_ROOT/benchmark/core_runtime/ppo_phase/results/trial" --output archived-summary-recomputed.json
```

Outputs must be fresh paths. Keep the same pinned dependencies and separate frozen
packages; the runner supplies independently measured native hashes. No weights,
checkpoints or rollout arrays are written. This study does not retest every
family/platform, GPU or asynchronous PPO, nor prove general training equivalence.

Evidence: [summary and raw measurements](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/trial/summary.json),
[raw rows](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/trial/samples.jsonl), [predeclared plan](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/trial/plan.json),
[completion](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/trial/complete.json), [validation](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/validation.json),
[helper tests](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/all-helper-tests.log), [pilot checks](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/pilot-validation.json),
[source freeze](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/source-freeze.json), [native freeze](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/runtime-freeze.json),
[before](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/freeze-before.json) and [after](https://github.com/wutailong/envpool/blob/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime/ppo_phase/results/freeze-after.json).
Private machine paths, credentials, binaries and model artifacts are not published.
