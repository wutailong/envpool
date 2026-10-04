# Max-player metadata cache: deferred

**Decision: leave the prototype unapplied.** Repeated config construction is
removed and positive correctness checks pass, but the fixed runtime study does
not establish a net benefit. All five environment intervals cross zero;
CartPole 256/256/4 and full PPO have adverse point estimates. This is not proof
of a precise regression or zero effect. No extra timing sought a favorable result.
Production remains the [97688bda input-owner repair](https://github.com/wutailong/envpool/commit/97688bdad104058ae2e4fd08be5579efce4dacac).

## Reason and mechanism

The preceding [binding study](../binding_cost/README.md) identified wrapper work
at microsecond scale. A bounded baseline cProfile pass on 50,000 warmed positive
CartPole `_from` calls observes exactly 50,000 `config` calls. They reconstruct
an eleven-field dict through native metadata access. Under the profiler, config
accounts for 0.123 s of 0.343 s in `_from`; these perturbed timings do not predict
an uninstrumented whole-step gain.

The [unapplied patch](prototypes/max_num_players_cache.patch) caches only the native
construction-time `max_num_players` integer per pool. It preserves the fresh
public config dict, input snapshots, recv player-ID copies, action-spec reads,
and normalization/inference algorithm. Both existing metadata-read sites call
a guarded helper: ordinary class-level config overrides still execute each time,
including overrides installed/removed after warming or between the two reads.

The native spec snapshot is read-only through the ordinary Python API. The guard
is not a general contract for custom mutable fake specs, private cache mutation,
or arbitrary `__getattribute__` interception retaining the original descriptor.
No array/native-resource owner is added. Repeated positive native Dummy inference
constructs one config view across eight calls instead of repeated views; explicit
player mappings bypass this lookup. This is an operation-count saving, not a
whole-process memory or speed claim.

## Fixed runtime experiment

Control: frozen retained wrapper/native runtime at 97688bda. Candidate: exactly
its native files, with only `envpool/python/envpool.py` and `protocol.py` changed.
No native source or binary is rebuilt. The two package roots each have 21 native
entries; five carry the prior repair-family builds, others remain inherited
support modules. Merely importing a module is not executing/testing that family.

**160 environment samples:** eight blocks, five cases, `a/b` identical control,
`c/d` identical candidate, orders `acdb/cabd/bdca/dbac`. Case-order seed 20261003;
400 warmup calls then at least three measured seconds, seed 42, default scheduling
on nine logical CPUs. Python 3.12.14, NumPy 2.5.3; BLAS/OMP/MKL threads=1.
Timing runs serially without concurrent builds/tests/screen sampling. No rows
are discarded. Rates count environment responses including normal autoresets,
not vector calls. Positive paired effects mean faster.

Each block compares duplicate-label mean log rates; intervals resample eight
complete blocks 10,000 times with seed 738 and are descriptive, not significance
guarantees. Actual loaded native/Python module hashes are recorded after the
unchanged measured loop; 202 Python/native entries per root remain unchanged
before/after the entire experiment. A/A columns expose same-version dispersion.

| Environment, N/B/T | Paired rate effect | Descriptive 95% interval | Adverse blocks | A/A control / candidate |
| --- | ---: | ---: | ---: | ---: |
| CartPole-v1, 20/20/1 | +2.29% | -3.45% to +8.15% | 3/8 | -5.09% / -0.97% |
| CartPole-v1, 20/20/2 | +3.33% | -3.44% to +10.92% | 3/8 | +1.34% / +3.70% |
| CartPole-v1, 256/256/4 | -2.95% | -7.75% to +1.77% | 5/8 | +1.26% / +6.66% |
| CartPole-v1, 1024/256/8 | +0.47% | -4.74% to +6.83% | 5/8 | +10.63% / -10.91% |
| HalfCheetah-v4, 256/64/4 | +1.55% | -1.48% to +4.56% | 2/8 | +0.96% / +1.20% |

| N/B/T | Control median [range] responses/s; CV | Candidate median [range] responses/s; CV |
| --- | ---: | ---: |
| 20/20/1 | 422,182 [355,534–466,271]; 6.96% | 420,487 [353,264–507,640]; 10.50% |
| 20/20/2 | 434,377 [347,772–528,404]; 10.66% | 442,374 [351,637–585,142]; 13.91% |
| 256/256/4 | 2,000,212 [1,565,306–2,405,587]; 13.96% | 1,953,168 [1,473,298–2,416,597]; 13.52% |
| 1024/256/8 | 3,548,894 [2,819,580–4,662,355]; 13.65% | 3,461,956 [2,452,312–4,620,851]; 14.62% |
| 256/64/4 | 93,294 [77,864–97,941]; 7.84% | 94,223 [79,355–99,309]; 6.26% |

The last row is HalfCheetah. Native-only Dummy throughput was deliberately not
rerun: that existing benchmark bypasses the changed wrapper. Positive multi-player
DM/Gymnasium mapping checks cover the mechanism instead.

## Full-budget PPO

Sixteen fresh-process runs use four balanced blocks, unchanged synchronous CPU
CartPole PPO: 20 environments, two workers, 256,000 responses, 100 updates and
8,000 optimizer steps, after two discarded priming updates. NumPy 1.26.4,
Torch 2.5.1+cpu, Tianshou 0.5.1, Numba 0.68.0. All semantic fingerprints match.

The paired rate estimate is **-0.81%**, interval **-2.61% to +1.48%**, with three
of four adverse blocks. Control median is **20.431 s** (20.213–22.936); candidate
**21.059 s** (20.233–21.616). Keep that slower candidate median visible; it is
not the same estimator as the block-weighted rate effect. Same-version A/A is
+2.04% for control and -2.63% for candidate; the latter's interval is -3.82% to
-0.88%. These data cannot support a training speedup or regression-free claim.

## Correctness and useful retained tooling

- Twenty positive scalar/config/mapping tests pass, including the exact valid
  player maximum, independent pools, fresh-dict mutation isolation, dynamic
  class-property overrides and ordinary generated DM/Gymnasium wrappers.
- Existing 13 Dummy and five Classic Control tests pass.
- 5,499 rollout arrays match both the original 9c31 record and retained 97688bda
  record exactly across CartPole, Acrobot, Pendulum and HalfCheetah. Async streams
  compare per environment, not global completion order.
- Eight CPU XLA records match the retained reference; first-use tracing, scan
  and asynchronous send/recv paths remain covered for CartPole/HalfCheetah.
- Fresh, separate-process full PPO journals match exactly: 101 checkpoints,
  223 arrays, 4,925 tensors and 106,238 scalars, maximum difference zero.
- Fifty-nine ToyText/MiniGrid scenarios match 6,575 arrays / 65,304,189 bytes;
  17 ToyText tests and two MiniGrid methods covering 82 IDs pass. BabyAI/render
  retain the prior representative-task scope, not all-task coverage.

The existing PPO comparator assumed different native hashes. That correctly
rejected this Python-only comparison before comparing journals, so it now has
an explicit [`--python-only` mode](../ppo/verify_ppo_parity.py): equal native
hashes, distinct packages/native paths, equal loaded wrapper-module sets and
actually different wrapper-content hashes are all required. Train records
actual imported paths/origins/hashes and verifies them again afterward. The
ordinary distinct-native guard and old-record compatibility remain. Publication
formatting only parenthesizes the final provenance assertion; the full harness
AST is identical to the version used for the two fresh journal runs. Twenty-one
[focused identity tests](../ppo/test_python_only.py) pass; training AST/config
and numerical comparison logic are unchanged apart from provenance/identity.

An initial package assertion also rejected an inherited native symlink before
training. With all candidate processes stopped, identical native bytes were
materialized inside the candidate. Controls were untouched; failed partial
outputs were preserved. Two new full runs with the adapted comparator establish
the stated equality. Neither initial harness failure was a trajectory mismatch.

No new native sanitizer campaign is implied: this candidate adds only Python
scalar state and uses unchanged native bytes. These results do not prove all
families/platforms, GPU XLA, general PPO or asynchronous training equivalence.

## Replay

From an isolated checkout of this report's
[public parent 84846c49](https://github.com/wutailong/envpool/commit/84846c49632049ee6639d28b6ce72660a171994d), apply the
prototype patch; do not alter the retained runtime. The [positive checker](prototypes/check_cache.py)
requires `--source /path/to/candidate/envpool/python --runtime /path/to/runtime`
and optionally `--native`. Use an independently built retained runtime and a
separate package copy containing the two patched Python files. Materialize its
Classic Control binary locally if using the strict PPO journal harness.

The [fixed plan](plan.json) supplies cases/orders/budgets. Use the existing
`experiments/run_blocks.py --paired-replicates`, paired-block summary, and
`state_tuple/run_paired_ppo.py` with separate frozen roots. For Python-only work,
record actual wrapper hashes as well as native hashes; native identity alone
cannot distinguish these variants. The local timed loop was unchanged; a
read-only post-loop wrapper recorded loaded-module provenance. The reusable
`experiments/bench_runtime.py` now records the actual native pool and wrapper
hashes after its clocks, so fresh runs carry that distinction directly. This
report's fixed results still belong to the earlier unchanged loop and post-loop
recorder. Keep generated records outside the repository.

For exact journal comparison use `ppo/verify_ppo_parity.py run --python-only`
with `--original` and `--candidate` pointing to their actual `envpool/` package
directories, an explicit unchanged PPO config and a fresh `--output` directory.
The prototype remains available for future investigation with better measurement
identifiability; this cycle does not publish it as a retained speed optimization.
