# Caller-side send and enqueue localization

**Outcome: investigate work-permit signaling next; no production change.**
The retained core remains [97688bda](https://github.com/wutailong/envpool/commit/97688bdad104058ae2e4fd08be5579efce4dacac).
This consolidates three bounded diagnostic studies. Shared action ownership is
not removed, and the queue's permit counts, publication order and stop-aware
shutdown remain unchanged. The patches here are **unapplied instrumentation**.

**Observer sensitivity is material.** In the final same-binary comparison,
fine mode’s observed paired 20-environment/four-worker rate is +10.30% relative to coarse
(descriptive interval +0.22% to +20.18%; one adverse block). This is not negative
clock overhead or a production improvement. It prevents treating the measured
phase shares as removable production-time ceilings. No overhead is subtracted.

## What was learned

The first coarse study compared the retained binary with diagnostic OFF, then
OFF with coarse ON: 64 fixed samples, N/B/T=20/20/1 and 256/256/4. For 256/4,
SetAction, action slices, prior-owner release and synchronous accounting together
have a median 8.54 microseconds wall / 8.55 caller CPU per vector call, and a
median same-run public-step share of 7.05%. The small case is 1.30 microseconds /
2.68%. This composite did not satisfy the predeclared roughly 10-microsecond and
10%-of-step gate for pursuing an ownership rewrite. It is not pure refcount cost:
replacing the last prior action owner may reacquire the GIL. Coarse ON versus
OFF itself changes reciprocal-time rate by -6.37% at 20/1 (interval -16.74% to
+4.61%) and -3.01% at 256/4 (-5.60% to -0.67%). Retained versus diagnostic OFF
changes layout/codegen too; it is not a same-code-layout calibration. The original
gate also required observer disturbance clearly smaller than the target cost.

Coarse enqueue is larger (22.80 microseconds in 256/4), but these initial cases
change both batch size and worker count. A separate 32-sample **coarse-clock-OFF**
study used the existing enqueue-duration counter to separate those factors:

| Environments / workers | Existing enqueue median, µs/call [range] | CV |
| --- | ---: | ---: |
| 20 / 1 | 5.75 [5.04, 6.98] | 11.47% |
| 20 / 4 | 20.78 [18.80, 23.34] | 7.29% |
| 256 / 1 | 6.55 [5.68, 7.75] | 10.58% |
| 256 / 4 | 24.11 [20.35, 28.06] | 11.52% |

Paired enqueue **cost** rises about 252% at N=20 and 264% at N=256 when moving
from one to four workers; every block has the same sign. Increasing N alone is
much smaller, about 14%/18% for one/four workers. Yet 256/4 has lower whole-step
wall cost than 256/1 (paired -19.01%, interval -27.40% to -13.93%). This does not
recommend minimizing the worker count for every workload.

The final study splits enqueue while retaining all queue operations. Its four
fine phases are producer guard/reservation, slot fill, work-permit signal, and
producer-guard release. Values below are medians of eight fine-mode processes;
shares are computed within each sample before aggregation.

| N/T | Signal wall / caller CPU, µs | Signal wall range | Fill wall, µs | Signal share of enqueue / whole step | Positive release requests/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20/1 | 5.10 / 5.13 | 4.66–5.82 | 0.22 | 79.97% / 10.42% | 0.9975 |
| 20/4 | 17.88 / 17.92 | 15.66–21.30 | 0.24 | 92.67% / 39.33% | 3.9880 |
| 256/1 | 5.62 / 5.61 | 4.62–6.28 | 0.88 | 73.03% / 3.30% | 0.9982 |
| 256/4 | 20.19 / 20.21 | 18.08–23.08 | 1.33 | 88.34% / 15.59% | 3.9927 |

Guard/reservation wall medians are 0.25–0.32 microseconds, guard-release medians
0.27–0.46. These include paired-clock costs. The signal interval includes the
existing atomic permit update, release-count arithmetic and backend call, with
caller CPU supporting this localization. It does **not** isolate atomic/cache
contention, POSIX posting, kernel work or scheduler effects from one another.

The pinned concurrentqueue 1.0.5 implementation computes the existing positive
`toRelease` from its signed count, then the POSIX backend loops `sem_post`.
A diagnostic clone returns that already-computed positive count without adding
another atomic load or changing permits. The counter is release-request demand
under instrumentation, not actual post attempts/retries, futex calls, kernel
wakes or sleeping-worker counts. It excludes other queue/guard/shutdown signals.

## Final observer comparison, including unfavorable evidence

This 64-sample study has coarse and fine modes of **the same V2 binary**. It has
no V2 OFF timing. The frozen plan's shorthand “on/off observer disturbance” means
this fine-versus-coarse contrast only; earlier V1 OFF data is separate evidence.

| N/T | Coarse → fine median whole-step µs [ranges] | Paired reciprocal-time rate change [descriptive 95% interval] | A/A coarse / fine |
| --- | ---: | ---: | ---: |
| 20/1 | 51.06 [42.91, 66.95] → 47.73 [43.64, 63.15] | +4.41% [-4.23, +14.47] | -4.77% / +6.61% |
| 20/4 | 49.62 [45.67, 58.41] → 45.28 [38.71, 60.21] | +10.30% [+0.22, +20.18] | +4.88% / +1.05% |
| 256/1 | 156.22 [133.75, 207.16] → 174.62 [135.40, 210.64] | -0.04% [-5.38, +6.26] | +4.50% / +1.52% |
| 256/4 | 139.87 [107.61, 176.89] → 128.83 [106.78, 182.39] | +5.86% [-2.86, +15.36] | +3.01% / +0.27% |

The 256/1 medians are visibly unfavorable despite a near-zero paired estimate.
Medians across processes and equal-weight within-block log contrasts are different
estimators under this variation; neither is discarded. Fine mode has additional
code, clocks and counters and can change spin/park state, scheduling and codegen.
A favorable contrast does not make those clocks a proposed optimization.

Coarse-mode existing enqueue medians are 5.29, 19.47, 5.80 and 21.68 microseconds
for the rows above; fine medians are 6.10, 19.00, 7.36 and 22.39. Their rough scale
corroborates the earlier OFF study's direction, not an additive overhead model.
No favorable extra samples or production candidate suite followed these results.

## Method and identities

All samples use synchronous public Gymnasium CartPole-v1, seed 42, preallocated
zero actions, 400 untimed warmups and exactly 5,000 vector calls. N=B. One vector
call returns N environment responses, including normal autoresets. No Python
phase hooks are used. Default OS placement allows nine logical CPUs; no frequency,
security or scheduling settings change. Runs are fresh, serial processes without
concurrent compilation, tests or screenshot sampling. Linux x86-64, Python 3.12.14,
NumPy 2.5.3, Gymnasium 1.3.0, GCC 14.2, C++17/O3; BLAS/OMP/MKL threads=1,
PYTHONHASHSEED=0, LD_PRELOAD unset. VM topology/host load remain uncontrolled.

- Coarse study: two contrasts × two cases × four blocks × four labels = 64.
  Fixed orders `acdb/cabd/bdca/dbac`, group-order seed 2026100416.
- Orthogonal OFF study: four cases × four blocks × two identical replicas = 32.
  Each block rotates the first-half case order and reverses it for the second
  half, balancing each case around the block center.
- Fine study: four cases × four blocks × four labels = 64. `a/b` coarse,
  `c/d` fine; same four orders, case-order seed 2026100417. Each label occupies
  each temporal slot once. Four blocks, not 64 independent replicates per case.

Rate effects average within-block log contrasts, then exponentiate. Descriptive
intervals resample four whole blocks 10,000 times (fine summary seed 739). They
are not significance/equivalence guarantees. Fixed call counts yield different
window lengths across workloads. Caller CPU excludes concurrent workers;
process CPU includes them. Per-thread scheduler snapshots bracket a slightly
wider interval than the clocks and cannot yield exclusive phase accounting.

Fine phases use five batch-boundary pairs of CLOCK_MONOTONIC then
CLOCK_THREAD_CPUTIME_ID, nested inside four coarse boundary pairs. There are no
per-element clocks. Middle clocks extend producer-guard occupancy. Fine counter
recording is inside parent enqueue and the existing legacy timer, outside fine
phase sums. Top-level recording and final local destruction are outside parent
phase sums, as are outer Python conversion/GIL boundaries. Never add nested
phases twice or interpret sequential clock pairs as simultaneous reads.

| Artifact | SHA256 |
| --- | --- |
| Retained ClassicControl | `c066bf09f34c22f86fc41bb2c2d4c26e1ca886d63a6f0b70f17ad2c0baf3c74f` |
| Coarse V1 ClassicControl | `e33505ee93cb20028999526e6a9802249f50eb134d7b86e811c821b63ecf05e4` |
| Fine V2 ClassicControl | `a671519301eb11d9f18d994c0edc3ac18b7a2f572fe84f38ee5ebb26806066b4` |

Each diagnostic required one focused ClassicControl compile/link with identical
release flags. V2 uses a private copy of the pinned semaphore headers. No queue
or semaphore members/virtual layout change; added pool counters/branches/layout
and changed code placement are observer differences. Original methods, PySend,
GIL boundaries and ownership remain unchanged. Source removal/equivalence checks
recover the original operations byte-for-byte. All four project semaphore users
resolve the intended copied header; 51 reused archives define no queue/pool/
semaphore implementation. The Classic archive contains only its renderer object,
whose unchanged source/header do not depend on core. Other native modules remain
inherited support modules, not newly profiled clients. All native/wrapper/source,
shared dependency and archive hashes stay fixed before/after.

## Validation and next decision

V1 OFF/ON has 912 exact array comparisons against retained production; the two
new orthogonal cases add 456. V2 OFF/coarse/fine × four cases gives 12 positive
32-step rollouts and **2,736 exact arrays** against matched-seed retained output.
After portable packaging, the same 12 positive cases are repeated and all 2,736
arrays match the measured driver; no performance sampling is repeated.
Observations, rewards, termination flags and info leaves retain dtype, shape and
bytes; changing positive actions stay unchanged. Counters/clock status, nested
bounds, expected submitted actions, pool reclamation and no leftover threads are
checked. Fine release requests are bounded by calls × workers. Timers really are
active for the ON/fine parity checks. All 160 planned timing records complete.
Independent recomputation of 1,932 final-summary numeric entries agrees within
floating-point roundoff (absolute tolerance 1e-10 / relative 1e-11).

This cycle does not add async, full PPO, XLA, sanitizer, other-family/platform or
clean-release validation. Prior production correctness evidence is unchanged;
these are diagnostic builds only. Hosted CI may not trigger on the retained docs
branch; lack of a run is not a pass.

The next step is a narrow compatibility/design study of the signal path: identify
avoidable work while preserving every permit, release/acquire publication,
spin/park race behavior, worker progress and stop-aware shutdown. If no credible
mechanism survives that review, stop this direction. Neither naive wake reduction,
longer spinning, ownership removal nor a global semaphore rewrite follows from
these measurements. A later candidate needs an uninstrumented end-to-end gate.

## Reproduce the final fine/coarse diagnostic

The portable tools preserve the measured step/validation loops and summary math;
path selection, CLI gates and the two boundary-description corrections are
packaging changes. The packaged summary reproduces the existing final summary
byte-for-byte. Historical V1 measurements are context, not regenerated by this
V2 runner. Raw samples/build logs are intentionally not added to the repository.

Use separate fresh source and concurrentqueue 1.0.5 snapshots. Apply
`project-diagnostic.patch` to the project snapshot at production 97688bda, and
`concurrentqueue-diagnostic.patch` to the copied dependency root. Both patches
apply/reverse with byte-exact source parity. They are not enabled in this branch.
Only ClassicControl is instrumented, on the Linux/POSIX backend. The pinned
upstream archive SHA256 is
`4d6368a27492d86011fde5ca0cf386dce7c49cd425aa3d9b063ca6ec373a6ef3`.

Build a separate ClassicControl client with the same compiler, ABI, dependency
versions and flags as the retained control. The actual focused build reused its
existing compile/link recipe and renderer/external archives, changing only source,
project/dependency include roots and output operands. Flags were:
`-std=c++17 -O3 -g0 -DNDEBUG -fPIC -fvisibility=hidden
-fno-omit-frame-pointer -ffunction-sections -fdata-sections -fstack-protector
-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1 -pthread`.
Make every project `lightweightsemaphore.h` include resolve to the private copy,
and audit linked archive members to exclude stale core definitions. These tools
accept an **already-built** diagnostic package; they do not supply a new clean
Bazel/dependency installation or the private build cache. Do not reuse the older
ce1-pinned fresh-rebuild helper as if it compiled this newer patched source.

Stage a fresh package with unchanged Python wrappers/dependencies/support modules
and the new ClassicControl extension. Record its own native hash (different build
paths can legitimately yield different bytes). Never replace a loaded binary.
`RUNTIME` and `BASELINE` below contain their `envpool/` package, `TOOLS` is this
directory, `PYTHON` supplies the recorded dependencies, and all output directories
are fresh and outside the repository. `PROJECT` and `CQ` are the frozen diagnostic
source snapshots actually used by your build. Supply minimal, read-only source snapshots,
not directories containing .git, runtime packages, build caches or outputs: the
runner fingerprints their files recursively. Configure/snapshot use one caller
and must not overlap any sends.

```sh
unset PYTHONPATH LD_PRELOAD ENVPOOL_ASSETS_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$VALIDATION"
for case in s1 s4 m1 m4; do
  "$PYTHON" -B "$TOOLS/probe_send.py" --root "$BASELINE" --native-sha256 "$BASELINE_HASH" --mode baseline --case "$case" --save-rollout "$VALIDATION/baseline-$case.npz" > "$VALIDATION/baseline-$case.json"
  for mode in off coarse fine; do
    "$PYTHON" -B "$TOOLS/probe_send.py" --root "$RUNTIME" --native-sha256 "$NATIVE_HASH" --mode "$mode" --case "$case" --save-rollout "$VALIDATION/$mode-$case.npz" > "$VALIDATION/$mode-$case.json"
  done
done
"$PYTHON" -B - "$VALIDATION" <<'PYCOMPARE'
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
for case in ("s1", "s4", "m1", "m4"):
    with np.load(root / f"baseline-{case}.npz") as a:
        for mode in ("off", "coarse", "fine"):
            with np.load(root / f"{mode}-{case}.npz") as b:
                assert a.files == b.files
                for key in a.files:
                    x, y = a[key], b[key]
                    assert x.shape == y.shape and x.dtype == y.dtype
                    assert x.tobytes() == y.tobytes(), (case, mode, key)
print("Positive rollouts match exactly")
PYCOMPARE
"$PYTHON" -B "$TOOLS/run_profile_blocks.py" --root "$RUNTIME" --native-sha256 "$NATIVE_HASH" --validation-dir "$VALIDATION" --project-source "$PROJECT" --concurrentqueue-source "$CQ" --output "$OUT" --timing-authorized
"$PYTHON" -B "$TOOLS/summarize_profile.py" "$OUT" --output "$SUMMARY"
```

The runner records hashes, its fixed plan, all samples and completion checks to
your selected local output; it runs no baseline timing and no V2 OFF timing. Retain
unfavorable rows and A/A variation. Do not pool a new trial with this historical
window or publish incidental validation elapsed times as performance results.
