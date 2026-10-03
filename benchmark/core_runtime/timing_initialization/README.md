# Initialize timing accumulators; reject an inconclusive cursor optimization

The retained production change initializes all three protected duration counters
in `AsyncEnvPool` to zero. Previously, `dur_send_` and `dur_recv_` were read by
`+=` without an initialized floating-point value. The unused `dur_send_all_`
is initialized consistently; merely declaring that unused field is not itself
an executed undefined read. This is a correctness fix, with **no speed claim**.

The baseline is the shutdown fix published as
[`9fff99ef555cb0199748bbda58446d669803fa64`](https://github.com/wutailong/envpool/commit/9fff99ef555cb0199748bbda58446d669803fa64).
The existing fields, their protected visibility, native layout, and clock calls
are preserved. Removing the telemetry was considered but rejected here because
external C++ subclasses may use those protected members. This change does not
establish a new concurrent-public-call contract or make arbitrary simultaneous
`Send` calls safe.

## Retained tests

The constructor regression checks all three counters before any Reset/Send/Recv.
Short Dummy sync, async, and multiplayer tests make these execution paths
practical under sanitizers. The constructor check can coincidentally pass old
code when its storage happens to be zero; it is not a deterministic negative
control, and ASan/UBSan do not establish absence of uninitialized reads.

Queue coverage now includes bounded multiple producers/consumers, exact-once
payload delivery, repeated small-ring reuse, non-power-of-two bulk wrap,
concurrent bulk contiguity, and stop-aware shutdown. All started test threads
are joined. The delayed-consumer case protects an already-copied value, rather
than forcing a pause inside the dequeue operation. The unsigned-counter test
covers one selected safe rollover alignment, **not arbitrary uint64 rollover**:
the pre-existing non-power-of-two ring mapping can repeat slots at other
alignments after uint64 wrap. The cursor experiment does not change that limit.

## Rejected cursor experiment

The [prototype patch](prototypes/cursor.patch) replaces two atomic `fetch_add`
operations with relaxed load/store pairs **inside the existing enqueue/dequeue
semaphore guards**. It keeps atomic fields for approximate readers and changes
no semaphore, wake-up, payload-copy, or shutdown admission ordering. Independent
review found no delta-specific concurrency blocker, and all executed correctness
gates passed. Nevertheless, the bounded throughput experiment did not establish
a useful robust gain, so this patch is not applied to the retained source.

The initialized source is the comparison control: this avoids using undefined
counter reads as the experimental baseline. Labels `a` and `b` use the identical
control runtime; `c` and `d` use the identical cursor runtime. Both complete
native manifests were verified before and after timing. See
[runtime provenance](results/runtime-provenance.json). The raw timing format has
no per-row binary hashes; the analysis explicitly leaves that row-level claim
unverified and relies on the separate frozen-runtime provenance.

## Bounded measurement and outcome

[The saved plan](results/plan.json) specifies seven configurations, eight blocks,
400 untimed warmup calls, seed 42, at least three measured seconds per fresh
process, and default scheduling on nine available logical CPUs. All 224 samples
are retained in [raw JSONL](results/cursor-throughput.jsonl). No build, environment
test, or other benchmark ran concurrently. The host is shared and nonstationary.

The orders `acdb`, `cabd`, `bdca`, `dbac` give treatment-level ABBA/BAAB and rotate
every individual label through every position. The primary effect averages
`log(c),log(d)` minus `log(a),log(b)` within each block, then averages blocks with
equal weight. Separate `b/a` and `d/c` contrasts expose same-binary variation.
The 10,000-resample block intervals are descriptive, not significance guarantees.
Rates count environment responses, including ordinary autoresets, not vector
calls or exclusively non-reset physics transitions.

| Environment | N/B/T | Cursor vs initialized control | Descriptive 95% interval |
| --- | --- | ---: | ---: |
| CartPole | 20/20/1 | +0.79% | -4.11% to +7.28% |
| CartPole | 20/20/2 | -0.70% | -6.36% to +5.48% |
| CartPole | 256/256/1 | +1.74% | -2.88% to +6.65% |
| CartPole | 256/256/4 | +3.78% | -7.45% to +14.76% |
| CartPole | 1024/256/8 | +3.35% | -2.02% to +9.83% |
| HalfCheetah | 64/64/4 | +0.34% | -1.81% to +2.44% |
| HalfCheetah | 256/64/4 | +0.70% | -0.62% to +2.00% |

Every interval crosses zero. Same-binary block effects include +7.31% for the
20/20/2 control pair and +10.72% for the 1024/256/8 cursor pair. These observations
show why small positive point estimates are insufficient here; they do not prove
that all real differences are zero or that scheduling explains every sample.
See [the complete paired summary](results/cursor-paired-summary.json) for every
block, sample range, coefficient of variation, CPU cost, and scheduler statistic.
No additional sampling was used to search for a favorable result.

## Correctness validation

Both initialized control and rejected cursor source passed:

- 50 native core/Dummy tests
- 48 tests in each of three ASan/UBSan repeats
- 48 tests in each of three TSan repeats
- 11 Dummy and five Classic Control Python tests
- 5,499 rollout arrays exactly equal to the published shutdown recording
- Eight CPU XLA case records exactly equal for CartPole and HalfCheetah

The two long original Dummy SinglePlayer/MultiPlayers tests were omitted only
from sanitizer runs; both passed natively. LeakSanitizer is disabled because
thread inspection is unavailable in this sandbox, so no leak-clean claim is
made. Dependencies were not all rebuilt with sanitizers. Only Classic Control,
MuJoCo Gym, and Dummy native clients were rebuilt. Other families, GPU XLA,
other platforms, and the complete release build are not newly validated.

The [source hashes](results/source-freeze.json), native test XML, frozen binary
manifests, and [normalized native build commands](results/native-build-commands.json)
record the actual selected-source artifacts. Build method and toolchain match
[the preceding shutdown validation](../shutdown/README.md#actual-build-and-execution-method):
GCC 14.2, Linux x86-64, C++17/O3 native clients, Python 3.12.14, existing read-only
Bazel dependencies. MuJoCo compilation uses GCC garbage-collector parameters to
limit compiler memory, without changing runtime optimization flags.


### Full synchronous PPO

Both variants also match the published shutdown recording exactly for the
fixed-seed 256,000-response CartPole run: 100 updates, 8,000 optimizer steps,
101 checkpoints, 223 arrays, 4,925 tensors, and 106,238 scalars. Maximum absolute
difference is zero. This is one deterministic synchronous CPU configuration,
not a promise about asynchronous training trajectories. Reports are
[initialized control](results/defined-ppo-parity.json) and
[cursor prototype](results/cursor-ppo-parity.json).

Eight additional full-budget PPO throughput runs use two ABBA/BAAB blocks,
with two untimed priming updates discarded before each freshly seeded run.
All final model/optimizer/RNG/metric fingerprints are equal. The initialized
control median is 20.217 s (range 20.090–20.346); cursor median is 20.164 s
(range 20.049–20.523). The overlapping ranges do not establish a training
speedup. [Raw samples](results/ppo-throughput.jsonl) and
[summary](results/ppo-throughput-summary.json) retain all eight runs.

The 23 lightweight experiment-helper tests, Python Ruff checks, C++ formatting,
and cpplint passed. All 104 native modules in the five earlier control runtimes
retain their recorded hashes, and their five source checkouts remain clean.
See [the validation summary](results/validation.json).

## Reproduce the comparison

Use independent runtime roots built from the initialized source and the
prototype applied to a separate checkout. Do not replace an imported binary.
The normalized build manifest uses SOURCE/OUTPUT/WORKSPACE/BAZEL_EXECROOT-style
location placeholders, not ready-to-run local paths. Standard repository Bazel
core/Dummy tests are an alternative; the recorded standalone commands are the
ones actually run here.

```bash
# In the isolated experimental checkout only:
git apply --index --unidiff-zero benchmark/core_runtime/timing_initialization/prototypes/cursor.patch

# From the retained checkout, with two independently built runtime roots:
python benchmark/core_runtime/experiments/run_blocks.py \
  --python "$PYTHON" --variant a="$CONTROL" --variant b="$CONTROL" \
  --variant c="$CURSOR" --variant d="$CURSOR" \
  --cases benchmark/core_runtime/experiments/cursor-cases.json \
  --blocks 8 --seconds 3 --warmup 400 --affinities default \
  --paired-replicates --out new-samples.jsonl
python benchmark/core_runtime/experiments/summarize_paired_blocks.py \
  new-samples.jsonl \
  --plan benchmark/core_runtime/timing_initialization/results/plan.json \
  --json-out new-summary.json
python -m unittest discover \
  -s benchmark/core_runtime/experiments -p 'test*blocks.py'
```

The helper rejects incomplete/subset experiments, duplicate slots, altered
orders/settings, and contradictory measured hashes when supplied. Existing
single-label summary behavior remains available for older experiments.
