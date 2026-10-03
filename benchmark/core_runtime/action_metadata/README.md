# Rejected single-player action-metadata reuse experiment

**Rejected: the allocation saving did not establish a robust net runtime gain.**
No production action cache or cache-specific regression test from this trial is
retained. The complete [prototype patch](prototypes/action-metadata.patch) is
archived only as evidence, including its `Array` friendship, `Env` changes,
`BUILD` target, and new `env_action_test.cc`. The retained production headers do not apply this patch.

The tested control is the initialized shutdown baseline published as
[`6685e510c1fad428b02152c1a378d6289880c4df`](https://github.com/wutailong/envpool/commit/6685e510c1fad428b02152c1a378d6289880c4df).
Its tree equals local base `dffcc5fd36d88e7e0bd1a50e53653c36e0e46507`:
`b2ee911677b39d4344b33500bc0d2293f948a37f`, as recorded in
[baseline publication provenance](results/baseline-publication.json).
The immutable local prototype is `f3e015c307b2ef20bba077bcd94915a552c8a132`.
This trial predates the separate player-discount correctness repair; these
measurements and validation results must not be attributed to that repair.

## Mechanism and allocation audit

The prototype rebinds single-player non-owning action views while retaining
shape-vector capacity. After `Step`, it moves field metadata back into each
`Env` cache and clears borrowed data pointers before completion publication.
It preserves the multiplayer path and does not make borrowed action copies own
or extend the lifetime of the input batch. The added tests cover mixed types,
batch reordering, replacement storage, rank growth/shrink, empty tails,
independent copied metadata, borrowed lifetime, reset/throw recovery, bounds,
and immediate receive/send reuse across worker threads.

The [audit](results/allocation-comparison.json) intercepts global scalar
`operator new` during direct serial CartPole `EnvStep`, after 10 warmup calls.
It is neither a throughput measurement nor a whole-process allocation census:

- Each of 9,031 non-reset steps: **7 to 5 calls**, 624 to 608 bytes.
- Each of 969 reset steps: unchanged at 5 calls and 608 bytes.
- All 10,000 measured calls: **68,062 to 50,000 allocations**;
  **6,224,496 to 6,080,000 bytes**.

[Control](results/defined-allocations.jsonl) and
[candidate](results/candidate-allocations.jsonl) raw counters are retained,
along with the [audit source](prototypes/count_action_allocations.cc).
The packaged audit source was license-header/format normalized and rebuilt
against both frozen sources; all EnvStep counters were reproduced exactly
([control rerun](results/public-audit-defined.jsonl),
[prototype rerun](results/public-audit-candidate.jsonl)).
The separate `Wait` allocation counts fluctuate with background state-buffer
refill availability; their difference is **not attributable to this patch**.

## Balanced throughput trial and rejection

[The fixed plan](results/plan.json) has seven configurations, eight blocks,
400 untimed warmup calls, seed 42, at least three measured seconds in each fresh
process, and default scheduling on nine available logical CPUs. All **224 raw
samples** are in [throughput.jsonl](results/throughput.jsonl); no configurations
or blocks were discarded, and no extra sampling sought a favorable result.

Labels `a/b` use one control runtime, and `c/d` use one prototype runtime.
Orders `acdb`, `cabd`, `bdca`, `dbac` provide treatment-level ABBA/BAAB and rotate
all labels through every position. The primary contrast is the block-average
of `mean(log(c), log(d)) - mean(log(a), log(b))`, with equal block weights.
The existing 10,000-resample complete-block intervals are descriptive, not
significance guarantees. Rates count environment responses, including normal
autoresets. These are shared-host observations, not dedicated-machine results.

| Environment | N/B/T | Prototype vs control | Descriptive 95% interval |
| --- | --- | ---: | ---: |
| CartPole | 20/20/1 | -0.140% | -4.443% to +4.786% |
| CartPole | 20/20/2 | +0.811% | -2.811% to +4.798% |
| CartPole | 256/256/1 | -2.081% | -6.214% to +2.032% |
| CartPole | 256/256/4 | -0.649% | -7.840% to +7.439% |
| CartPole | 1024/256/8 | +1.060% | -3.075% to +4.269% |
| HalfCheetah | 64/64/4 | +1.229% | +0.061% to +2.405% |
| HalfCheetah | 256/64/4 | +1.257% | -0.086% to +2.927% |

All five CartPole intervals cross zero. HalfCheetah 64/64/4 is barely positive,
but its same-binary `b/a` and `d/c` effects are +1.438% and +2.453%, respectively.
The latter itself has a positive descriptive interval. Other A/A effects are
larger still: CartPole 20/20/1 control `b/a` is +9.385%. Thus the planned trial
does not establish enough robust benefit to justify the added cache complexity.
This does not prove every true effect is zero or explain every timing sample.
[The complete existing summary](results/paired-summary.json) retains every block,
A/A contrast, range, coefficient of variation, CPU cost, and scheduler statistic.

### Full-budget synchronous PPO

Eight separate throughput runs use two ABBA/BAAB blocks. Each discards two
untimed priming updates, then starts a fresh seeded 256,000-response run with
100 updates and 8,000 optimizer steps. All model/optimizer/RNG/metric fingerprints
are identical. Control median: **19.892 s**, range 19.823–19.926 s; prototype
median: **19.949 s**, range 19.883–20.211 s. Overlapping ranges and the slower
prototype median do not support a training speedup. All runs remain in
[raw samples](results/ppo-throughput.jsonl) and the
[existing summary](results/ppo-throughput-summary.json).

## Correctness and provenance

The rejected prototype passed the recorded gates:

- 59 native core/Dummy tests: [XML](results/native.xml)
- 57 tests in each of three ASan/UBSan repeats: [XML](results/address.xml),
  [all-repeat log](results/address-log.txt)
- 57 tests in each of three TSan repeats: [XML](results/thread.xml),
  [all-repeat log](results/thread-log.txt)
- 11 Dummy and five Classic Control Python tests:
  [Dummy log](results/dummy-python-log.txt), [Classic log](results/classic-python-log.txt)
- [5,499 exact rollout arrays](results/rollout-comparison.json) and
  [eight exact CPU XLA cases](results/xla-comparison.json)
- [Full PPO exact parity](results/ppo-parity.json): 101 checkpoints, 223 arrays,
  4,925 tensors, 106,238 scalars, maximum absolute difference zero

PPO parity covers one fixed-seed synchronous CPU configuration, not general or
asynchronous trajectory equivalence. The two original long Dummy tests passed
natively and were omitted only from sanitizer runs; short sync/async/multiplayer
coverage remained. LeakSanitizer was disabled because sandbox thread inspection
was unavailable. Dependencies were not all rebuilt with sanitizers.
The 23 experiment-harness tests are inherited baseline validation, **not newly
rerun here**. See the bounded [validation record](results/validation.json).

[Source](results/source-freeze.json) and [header](results/header-freeze.json)
SHA-256 manifests, plus both complete 21-module runtime manifests
([control](results/defined-runtime-freeze.json),
[prototype](results/candidate-runtime-freeze.json)), were reverified after timing.
Only Classic Control, MuJoCo Gym, and Dummy clients were rebuilt; the other 18
modules are read-only control links. No GPU, all-family, full-release, or
cross-platform validation is claimed. Throughput rows contain no measured binary
hashes, so row-level identity remains unverified; see
[runtime provenance](results/runtime-provenance.json) for the separate evidence.

## Reproduce in isolation

The [actual native build argv](results/native-build-commands.json) records the
Linux x86-64 C++17/O3 clients using existing read-only Bazel dependencies.
MuJoCo's compiler garbage-collector parameters limit build memory, without
changing runtime optimization flags. Location prefixes in evidence are replaced
by `SOURCE`, `OUTPUT`, `BAZEL_EXECROOT`, `BASELINE_RUNTIME`, `CANDIDATE_RUNTIME`,
`PYTHON_ENV`, `PPO_PYTHON_ENV`, or `WORKSPACE`; resolve these before reuse.
Other evidence values, hashes, and recorded timestamps are unchanged.

Apply the zero-context patch only to an isolated checkout of the recorded base:

```bash
git apply --index --unidiff-zero /path/to/action-metadata.patch
```

Build separate control/prototype runtime roots; never overwrite an imported
binary. Use `experiments/run_blocks.py` with the saved seven-case plan,
`--blocks 8 --seconds 3 --warmup 400 --affinities default --paired-replicates`,
and labels `a/b` for control and `c/d` for prototype. Use
`experiments/run_ppo_blocks.py --blocks 2` for the separate full-budget PPO trial.
The existing summaries were copied, not recomputed while packaging this evidence.
