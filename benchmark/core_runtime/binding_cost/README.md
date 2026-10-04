# Binding costs and measurement limits

**Historical outcome: prioritize public-wrapper metadata work; no production change.**
The subsequent [metadata-cache trial](../wrapper_metadata/README.md) was deferred.
The [follow-on native send study](send_profile/README.md) now localizes a larger
work-permit signaling cost, with explicit observer limits; production is unchanged.
The complete positive-only diagnostic study separates native input conversion,
real Python/native boundaries, and observer overhead. It does not rehabilitate
the [rejected direct-owner prototype](../input_owner_storage/README.md).
The current production source remains the
[97688bda input-owner repair](https://github.com/wutailong/envpool/commit/97688bdad104058ae2e4fd08be5579efce4dacac).

## Main findings

- In the retained runtime, bypassing public wrapper work lowers calling-thread
  CPU in all 16 case/placement/block contrasts, by 8.1–22.6 microseconds per
  vector call. Small-batch wall medians are about 47 microseconds for public
  `step`, versus 32–36 for the bypass. This is useful localization, not a
  behavior-preserving implementation or guaranteed recoverable saving.
- The rejected native-owner prototype saves only about 6–8 nanoseconds per
  same-thread conversion in the focused loop. Three input arrays are converted
  by these ordinary calls, but release/GIL context differs in real production,
  so multiplying that result does not predict whole-step improvement.
- Timed-hook instrumentation reduces small-case throughput by about 12–17%.
  Instrumented phase timings are not uninstrumented exclusive C++ costs.
- Logical CPU placement did not consistently improve repeatability. A/A effects
  remain substantial, including about -23% in one medium observer-control cell.
  These observations do not uniquely separate wrapper work, synchronization,
  and host interference.

## Fixed method and identities

Binding study: **256 fresh processes**, four blocks, `acdb/cabd/bdca/dbac` orders;
`a/b` are identical controls and `c/d` identical comparison modes. Four
comparisons × two cases × two placements × four blocks × four labels.
Case order is shuffled by seed 2026100414. Every sample warms up 400 calls then
executes exactly 5,000 calls, seed 42, synchronous CartPole-v1, all-zero actions.
Small is N/B/T=20/20/1; medium is 256/256/4. Calls count normal autoreset responses.
One vector call produces N environment responses. These cases change both N
and T, so their difference is not an isolated batch-size experiment.

All timing is serial with no concurrent builds, other benchmarks or screenshot
sampling. Environment: Linux x86-64, Python 3.12.14, NumPy 2.5.3, GCC 14.2,
release C++17/O3; BLAS/OMP/MKL threads=1, PYTHONHASHSEED=0, LD_PRELOAD unset.
Original is the installed main `9c31c547`; retained is the rebuilt cumulative
`97688bda`. The two wrapper files, loader and driver hashes are equal/verified
as appropriate; native SHA-256s are checked for every process. This compares
cumulative source/builds, not one isolated patch or a new same-flags original
rebuild. Equality checks do not hash every Python dependency.

Pinned placement requests caller CPU 0 and worker CPUs 1..T, then verifies
actual `/proc` masks. Nine logical CPUs are available. The additional stock
buffer allocator remains allowed on 0–8; this is not physical-core isolation.
Only read-only topology/load/clock observations and ordinary per-thread affinity
are used. Physical sibling topology, frequency controls and cgroup quota files
are unavailable here; perf is not installed. No security/system settings change.

Each table effect is reciprocal-time **rate change**, computed from the mean
within-block log contrast, with equal block weights. Intervals resample four
complete blocks 10,000 times, seed 738. They are descriptive, not significance
or equivalence guarantees. A/A columns use the same method. All planned rows
are included, with no favorable stopping or follow-up sampling.

## Real binding comparisons

`raw` prepares `_from` and `_check_action` once, then repeatedly calls native
`_send` and `_recv`. It still performs native input conversion on every call.
It omits public conversion/copy, checks, dispatch, receive metadata bookkeeping,
output wrapping and some destruction, and changes scheduling. Writable input
buffers are fixed and never mutated. This mode is a diagnostic bypass only.
Original/retained comparison uses public mode; all other comparisons use the
retained runtime. Therefore these phase probes cannot explain which phase caused
the original-to-retained difference.

| Comparison | N/T, placement | Median control → comparison, µs/call | Rate effect [descriptive 95% interval] | A/A control / comparison |
| --- | --- | ---: | ---: | ---: |
| Original → retained | 256/4, default | 144.32 → 112.24 | +31.79% [+23.91, +38.74] | -1.51% / -1.84% |
| Original → retained | 256/4, pinned | 145.90 → 117.38 | +23.52% [+18.15, +29.59] | +9.23% / -11.89% |
| Original → retained | 20/1, default | 48.47 → 46.33 | +6.38% [-1.02, +14.34] | +0.55% / +5.95% |
| Original → retained | 20/1, pinned | 54.95 → 48.90 | +18.51% [+9.07, +33.41] | +3.15% / +13.93% |
| Public → raw | 256/4, default | 112.75 → 100.67 | +12.15% [+3.82, +21.15] | +3.54% / +8.60% |
| Public → raw | 256/4, pinned | 114.45 → 100.78 | +12.49% [+7.47, +17.23] | +6.37% / +7.26% |
| Public → raw | 20/1, default | 47.32 → 31.64 | +45.69% [+30.01, +62.59] | +5.10% / +10.44% |
| Public → raw | 20/1, pinned | 47.24 → 36.00 | +35.20% [+27.11, +43.86] | +9.05% / -0.69% |
| Public → timed hooks | 256/4, default | 121.71 → 121.51 | +6.30% [-1.92, +15.22] | -23.31% / -1.86% |
| Public → timed hooks | 256/4, pinned | 119.87 → 124.32 | -0.01% [-8.60, +15.90] | -20.03% / -9.32% |
| Public → timed hooks | 20/1, default | 46.90 → 55.43 | -17.04% [-21.91, -11.73] | -10.07% / -6.59% |
| Public → timed hooks | 20/1, pinned | 49.53 → 56.00 | -11.71% [-17.72, -3.74] | -5.51% / -0.55% |
| Public → no-clock hooks | 256/4, default | 116.03 → 111.81 | -2.56% [-13.32, +6.86] | +10.63% / -0.83% |
| Public → no-clock hooks | 256/4, pinned | 112.66 → 114.20 | -2.25% [-5.74, +1.36] | -0.05% / +6.35% |
| Public → no-clock hooks | 20/1, default | 50.37 → 53.75 | -3.86% [-11.23, +5.92] | +7.71% / -4.32% |
| Public → no-clock hooks | 20/1, pinned | 49.17 → 53.48 | -8.97% [-23.00, +0.26] | -10.17% / +4.26% |

For public → raw, rate +35.20–45.69% in small cases corresponds to about
26.0–31.4% lower block-geometric latency; medium +12.15–12.50% corresponds to
10.8–11.1%. These are not whole-application speedups. Medium/default has one
adverse wall-time block, despite positive calling-thread CPU contrasts.

Original → retained public latency ranges and coefficients of variation:

| N/T, placement | Original range µs; CV | Retained range µs; CV |
| --- | ---: | ---: |
| 256/4, default | 125.38–179.17; 12.74% | 106.31–120.88; 4.66% |
| 256/4, pinned | 131.77–201.77; 14.06% | 111.27–153.46; 11.98% |
| 20/1, default | 45.60–57.38; 9.18% | 40.47–56.24; 10.18% |
| 20/1, pinned | 47.02–71.94; 14.96% | 42.03–57.29; 10.36% |

## Instrumented boundaries, not exclusive work

Five non-nested hooks time `_from`, `_check_action`, `_send`, `_recv`, `_to`.
Thirty clock reads per vector call plus hooks/counters/accumulation perturb the
work. No-clock hooks also omit accumulation; observer contrasts are contextual,
not additive corrections. No overhead is subtracted.

| N/T, placement | `_from` | `_send` | `_recv` | `_to` | Unhooked + observer residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256/4, default | 6.53 | 30.22 | 61.96 | 6.69 | 16.32 |
| 256/4, pinned | 6.89 | 30.75 | 62.36 | 6.91 | 16.47 |
| 20/1, default | 5.91 | 7.88 | 24.21 | 5.03 | 12.90 |
| 20/1, pinned | 5.23 | 7.71 | 24.61 | 4.82 | 12.27 |

Values are median instrumented wall microseconds/call. `_check_action` is
0.20–0.26 microseconds after warmup. Medians of separate components need not sum
to the median total. `_recv` includes native waiting, GIL reacquisition and
native-to-NumPy wrapping. `_send` includes input conversion, queueing and possible
release of the previous action batch. Some returned-object destruction and
receive bookkeeping happen outside these leaves. Worker CPU overlaps them.

Caller CPU differs from wall elapsed; process CPU includes workers and allocator.
Scheduler snapshots bracket a slightly wider interval than the clocks.
Retained public caller runnable-wait medians are only 0.03–0.13 µs/call here,
while caller off-CPU elapsed is about 11 µs small and 43–44 µs medium. The latter
includes blocked synchronization/GIL time, not just runnable scheduling delay.
Pinned allocator CPU is about 11.6 µs small / 14.8 µs medium per call. These are
observed costs, not a causal allocation-removal speed prediction. VM-wide steal
counters are nonzero but cannot be attributed to this process or subtracted.

## Native conversion diagnostic

**104 fixed samples:** 96 conversion samples and eight consumer calibrations.
Three contrasts × default/caller-CPU0 placement × four balanced blocks × four
labels; shuffled group seed 2026100413. Each converts positive writable C-contiguous
owned int32 shape `(20,)` one million times after 10,000 warmups. Actual selected
`NumpyToArrayIncRef<int>` headers are compiled O3/DNDEBUG, no LTO. A separately
compiled noinline consumer reads size, rank and endpoint values; exact uint64
checksums and balanced Python references are checked outside clocks.

Each final native owner releases in the calling thread while it already holds
the GIL. Production retains batches across steps and may release prior owners
after `PySend` releases the GIL. There is no cross-thread/GIL-contention inference.
Allocator interception and `-Bsymbolic-functions` are absent; existing `-Bsymbolic`
remains. All recorded LD_PRELOAD values are unset. Native source/dependencies,
archives and binary hashes are frozen. The tested prototype remains unapplied.

| Contrast, placement | Median caller CPU ns/conversion, control → comparison | Rate effect [descriptive 95% interval] | A/A control / comparison |
| --- | ---: | ---: | ---: |
| ce1_to_repaired, cpu0 | 90.68 → 91.20 | +1.97% [-4.65, +12.79] | -3.84% / +1.17% |
| ce1_to_repaired, default | 91.72 → 96.14 | -2.21% [-5.69, +4.37] | -8.30% / +0.02% |
| original_to_repaired, cpu0 | 89.02 → 90.99 | -3.53% [-5.93, -1.17] | +1.94% / -0.32% |
| original_to_repaired, default | 97.08 → 97.18 | +1.11% [-3.47, +5.63] | +2.60% / -2.95% |
| repaired_to_prototype, cpu0 | 93.84 → 87.61 | +7.88% [+6.53, +9.45] | +0.74% / -1.45% |
| repaired_to_prototype, default | 96.00 → 87.96 | +7.57% [+0.87, +14.13] | +2.16% / +2.97% |

`ce1` is the prior cumulative tuple version, `repaired` is 97688bda, and
`prototype` is the rejected direct-owner-storage patch on that repair. This
microbenchmark is not the earlier broad runtime trial. Wall and caller CPU are
similar in this tight loop; substantial variation remains in CPU readings too.
Consumer-only median is about 1.7 ns, but one default-placement sample is 17.9 ns.
Calibration is context only: the compiled loops differ; do not subtract it.

## Validation, reproduction and limits

For each of installed original and retained runtime separately, four modes ×
two cases × 32 steps from fresh matched-seed pools produce identical public
leaves, native arrays, dtypes/shapes/bytes, key order and metadata across modes.
Phase clocks are disabled during these exact comparisons; they do not establish
cross-runtime parity or exact clock-enabled profiling parity. Normal autoresets occur.
Input hashes remain unchanged; pools and worker threads are reclaimed after
validation. Every timing row verifies frozen identities and stable thread sets;
every pinned row verifies actual masks. Independent review recomputed all 66
cell/clock point estimates and medians, phase medians and record hashes.

[Portable binding probe and commands](REPRODUCE.md) retain the measured loop,
hooks and validation with configurable runtime selection and explicit native
hash checks. The public helper is separately validated; original measurements
used the earlier path-locked helper. [Native sources/build scope](native/README.md)
retain the focused conversion diagnostic without generated binaries or logs.
No new production patch, native sanitizer campaign, asynchronous environment,
GPU/XLA comparison or full PPO trial is claimed by this diagnostic cycle.
The preceding repair/prototype studies retain their own separate validation.

A concrete next candidate is eliminating repeated construction-time scalar
metadata conversion in `_infer_players_env_id`, while preserving public config,
subclass overrides, action snapshots and output/player-ID copies. It requires a
separate mechanism check, semantic tests and balanced end-to-end experiment.
