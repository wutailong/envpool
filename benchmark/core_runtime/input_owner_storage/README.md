# Direct Python input owner storage: not retained

**Decision: leave the prototype unapplied.** It removes one native allocation
per input conversion, but the completed runtime study has unresolved tradeoffs:
CartPole 256/256/4 is adverse, four-player Dummy is favorable, and full PPO does
not establish a gain. This is not a correctness rejection or proof of slowdown.
The retained production baseline remains the
[97688bda leak repair](https://github.com/wutailong/envpool/commit/97688bdad104058ae2e4fd08be5579efce4dacac).

## Mechanism and allocation result

The [prototype](prototypes/direct_owner.patch) keeps `ArrayT` on the stack, then
moves its reference into an explicitly typed generic `py::object` captured by
the existing shared-pointer deleter. Shape and mutable-data access finish before
the move. The callback acquires the GIL and clears that object at final strong
release, before a surviving weak control block is destroyed. A generic null
object must be used: default-constructing `ArrayT` could create another array.

Conversion flags, rejection behavior, shape construction and accepted alias/copy
semantics are unchanged. No extra Python reference or shared control block is
introduced. The scoped [actual-header probe](../numpy_input_owner/README.md)
counts calling-thread ordinary C++ new/new[] traffic, excluding Python/NumPy
payload allocation, unrelated threads, RSS and input preparation.

| Positive shape, int32 | Allocation calls, baseline → candidate | Requested bytes, baseline → candidate |
| --- | ---: | ---: |
| 1 | 4 → 3 | 52 → 44 |
| 20 | 4 → 3 | 52 → 44 |
| 256 | 4 → 3 | 52 → 44 |
| 20 × 2 | 4 → 3 | 64 → 56 |
| 2 × 3 × 4 | 4 → 3 | 76 → 68 |

Every tested shape saves exactly one allocation and eight requested bytes on
this x86-64 build. All actual fault sites and successful sentinels reclaim their
tracked allocations and preserve source reference counts. The patch updates
the allocation-cost expectation from four sites to three, keeping the ownership
checks intact.

Actual release Classic Control int-deleter code also becomes smaller:
`_M_dispose` is 119 → 87 bytes and `_M_destroy` 95 → 79 bytes. Code size and
allocation counts are mechanism evidence, not a timing result or explanation
for the end-to-end effects below.

## Correctness and build scope

- Eight focused tests pass with NumPy 2.5.3 and 1.26.4. Six ownership/lifetime
  tests pass three ASan/UBSan runs, including native-thread final strong release
  followed by weak-only destruction. Fault injection is normal-build only;
  sanitizer mode has no replacement allocators. LeakSanitizer is disabled and
  external dependencies are not all instrumented. No new TSan run is claimed.
- 13 Dummy and five Classic Control Python tests pass, including the permanent
  positive read-only reset/send rejection regressions.
- 5,499 rollout arrays and eight CPU XLA case records match the repaired control.
- Full synchronous CartPole PPO matches exactly: 256,000 responses, 100 updates,
  8,000 optimizer steps, 101 checkpoints, 223 arrays, 4,925 tensors and 106,238
  scalars; maximum difference zero.
- 59 ToyText/MiniGrid comparisons give 6,575 equal arrays and 65,304,189 equal
  bytes. Seventeen ToyText tests and two determinism methods across 82
  MiniGrid-prefixed IDs pass. BabyAI covers one of 96 tasks; render parity is
  DoorKey only.

Five binding clients were rebuilt with matching GCC 14.2/C++17/O3 settings:
Classic Control, MuJoCo Gym, Dummy, ToyText and MiniGrid. Unchanged dependencies
and renderer/project archives are reused; MiniGrid reuses 16 unaffected project
units whose dependencies exclude the changed header. This is not an all-family
wheel, GPU, cross-platform, upstream-oracle or general async-PPO equivalence
claim. Normal async comparisons preserve per-environment order, not arrival order.

A copied local build driver initially retained the previous output-root prefix.
Exclusive output-directory creation stopped it before compilation or native
writes, but one prior generated source manifest had been refreshed. It was
restored from the prior freeze and an independently matching baseline header
snapshot. Independent review confirmed the restored contents and unchanged
control native hashes. The corrected isolated build supplied all results here.

## Fixed runtime study

The control is **97688bda**, not ce1 and not the original main branch. Do not
chain these estimates with an earlier trial to infer a different comparison.
Same-binary labels a/b are control, c/d candidate. Eight blocks use acdb, cabd,
bdca and dbac twice, seed 42, 400 warmups, at least three measured seconds and
all nine allowed CPUs without worker pinning. Four general cases give 128
samples; one-/four-player Dummy add 64. Four blocks of full-budget PPO add 16
trials, each after two discarded priming updates. No extra samples were added.

All timing is serial without builds, tests or desktop sampling. All 42 timed
native entries and 17 production headers match before/after. General rows do
not embed binary hashes: attribution also relies on the saved explicit-root
invocation and separate freeze. Dummy and PPO verify native hashes in rows.
All 16 PPO semantic fingerprints agree.

Positive means faster candidate relative to the repaired control. Intervals
are descriptive complete-block bootstrap ranges, not significance guarantees.

| Environments / batch / threads | Effect | Descriptive 95% interval | Negative blocks | A/A control / candidate |
| --- | ---: | --- | ---: | ---: |
| CartPole-v1 20/20/1 | -0.46% | -4.65% to +3.64% | 3/8 | -3.14% / +2.50% |
| CartPole-v1 256/256/4 | -6.88% | -14.54% to +0.97% | 5/8 | +2.71% / -0.82% |
| CartPole-v1 1024/256/8 | +5.33% | -2.52% to +14.05% | 3/8 | -4.59% / +2.60% |
| HalfCheetah-v4 256/64/4 | -1.13% | -4.93% to +2.26% | 5/8 | +1.25% / +3.42% |
| DummyPlayers1 64/64/4 | -2.26% | -6.24% to +1.35% | 5/8 | -0.56% / -6.46% |
| DummyPlayers4 64/64/4 | +9.59% | +3.30% to +16.36% | 1/8 | -2.91% / -0.03% |

Each implementation has 16 samples per row. Rates count environment responses
including autoresets, not vector calls or individual player rows. Ranges below
are observed min/max; CV is sample standard deviation divided by mean.

| Case | Control median [range] | Candidate median [range] | CV control / candidate |
| --- | ---: | ---: | ---: |
| CartPole-v1 20/20/1 | 435,706 [360,442, 468,626] | 434,674 [377,924, 467,367] | 6.78% / 6.90% |
| CartPole-v1 256/256/4 | 2,258,260 [1,932,438, 2,521,834] | 2,115,022 [1,556,402, 2,603,517] | 8.41% / 14.63% |
| CartPole-v1 1024/256/8 | 3,340,159 [2,903,582, 4,986,228] | 3,830,122 [2,874,379, 5,077,673] | 20.86% / 17.25% |
| HalfCheetah-v4 256/64/4 | 82,983 [70,108, 93,671] | 82,398 [67,218, 98,738] | 8.14% / 10.20% |
| DummyPlayers1 64/64/4 | 667,216 [541,667, 724,645] | 650,427 [496,584, 755,849] | 7.94% / 9.37% |
| DummyPlayers4 64/64/4 | 430,487 [302,595, 508,434] | 462,935 [400,101, 475,594] | 13.64% / 4.73% |

Full PPO is **-0.38%**, interval **-2.22% to +1.50%**, with two of four blocks
adverse. Median training time is 21.080 s for control (20.431–22.302) and
21.363 s for candidate (20.728–21.543). A/A effects are -1.56% and -1.36%; the
candidate A/A is adverse in all four blocks with an interval below zero despite
identical binaries. This demonstrates the limits of these small-block intervals.
Ratios of independent medians are not the paired-block estimand.

The allocation reduction is real, and DummyPlayers4 is favorable in seven of
eight blocks. However, common synchronous CartPole 256 is adverse in five,
with block effects from -25.08% to +12.91%. Its aggregate -6.88% cannot be
silently dismissed as noise, nor established as a portable causal slowdown.
CPU/runnable-wait observations and smaller deleter code do not resolve that
tradeoff. Full PPO is inconclusive. Therefore this candidate is not promoted
as a general runtime optimization or labeled regression-free. Keep the safe
published repair and this unapplied source/test patch for future investigation.

## Reproduce without changing the control

Set `REPORT_CHECKOUT` to the absolute checkout containing this report. Use a
separate, full experimental clone at the exact repaired baseline:

```sh
git clone --branch docs/core-current-family-coverage https://github.com/wutailong/envpool.git envpool-owner-storage
cd envpool-owner-storage
git checkout --detach 97688bdad104058ae2e4fd08be5579efce4dacac
git apply --check "$REPORT_CHECKOUT/benchmark/core_runtime/input_owner_storage/prototypes/direct_owner.patch"
git apply "$REPORT_CHECKOUT/benchmark/core_runtime/input_owner_storage/prototypes/direct_owner.patch"
```

Run the [input-owner probe build and two-mode test commands](../numpy_input_owner/README.md#build-and-run)
against separate repaired-control and candidate headers. The patch supplies the
candidate's three-allocation test expectation. The older report's intentional
negative control concerns pre-repair code, not this study's repaired comparator.
Rebuild the actual clients used
in your workloads before timing; verify effective source/dependency and native
hashes, not merely package version or path names.

Use the unchanged [eight-block general/Dummy and 16-trial PPO commands](../recv_list/README.md#reproduce-the-fixed-timing-plan),
with `CONTROL` pointing to 97688bda and `CANDIDATE` to this applied prototype.
For the general summary, replace its historical plan argument with
`--plan "$REPORT_CHECKOUT/benchmark/core_runtime/input_owner_storage/plan.json"`; this [input plan](plan.json)
records the correct baseline and unchanged eight-block settings.
Record fresh invocations and hashes and write raw output outside the checkout.
This branch keeps the patch, methods, variance and conclusions, not generated
samples, logs, checkpoints or disassembly. Local checks are not a hosted CI pass.
