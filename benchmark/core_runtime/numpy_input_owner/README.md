# NumPy input owner exception safety

This focused probe calls the actual `NumpyToArrayIncRef` in
`envpool/core/py_envpool.h`. It uses only small, ordinary positive-sized NumPy
arrays of ranks 1–3. It does not import the EnvPool Python package, instantiate
environments, construct empty arrays, or exercise malformed array metadata.

The regression is same-dtype, C-contiguous, read-only input: conversion keeps
the existing array, and `mutable_data()` rejects it. A raw owning `ArrayT*`
allocated before that rejection is leaked. Fixed behavior must continue to
reject the input while reclaiming the wrapper and its Python reference.

## Observed defect and repair

The original installed 9c31 baseline and retained ce1 runtime both reproduce
the leak through the real CartPole `_reset` binding: 16 rejections of a
four-element read-only `int32` ID array raise the same ValueError but increase
its reference count from 2 to 18. Dropping the caller reference does not reclaim
it. The rebuilt repair preserves all 16 errors, keeps the count at 2 and allows
the weak reference to expire. Two permanent Dummy tests additionally cover
reset rejection and a failing final action field after earlier conversions.
Both tests fail on the retained baseline and pass with the repair.

The implementation keeps a `unique_ptr<ArrayT>` through conversion and shape
setup, then moves it into the existing shared-owner callback. The callback
explicitly resets that owner while holding the GIL. This also prevents a
surviving weak control block from delaying Python reference release.
Conversion flags, input rejection and accepted alias/copy semantics remain
unchanged. This minimal repair does not reduce successful-path allocations.

A bounded positive rank-three allocation audit finds four allocations totaling
76 requested bytes in both baseline and repair. Failing allocation two or three
leaks one wrapper and Python reference in the baseline; every failure site and
the success sentinel reclaim all tracked owners in the repair.

## Coverage

- Repeated read-only rejection preserves reference counts; rejected source
  weak references expire when their final Python reference is dropped.
- Writable `int32` and `float64` input preserves its pointer, shape, dtype,
  two-way mutation, and one Python owning reference across multiple native
  strong aliases and a returned NumPy view.
- Forced dtype and noncontiguous input keep existing copy semantics, including
  read-only inputs whose converted copy is writable. Original Python objects
  can be reclaimed while the converted values remain alive.
- The final native strong owner is released on a new `std::thread` while the
  calling thread releases the GIL. A Python weakref callback must run on that
  worker, before the still-surviving native `weak_ptr` is destroyed. A second,
  newly created native thread destroys that weak owner with no further Python
  release. Thread identifiers may be reused by the OS between these sequential
  worker lifetimes.
- In the normal build, a scoped, bounded replacement-new audit rejects each
  of the four native allocations in the small writable rank-three conversion:
  wrapper, two
  shape vectors, and shared-owner control block. Every failure must preserve
  the source reference count and reclaim all tracked allocations. A success
  sentinel follows the final failure site.

The audit covers C++ `operator new`/`new[]` calls bound inside this extension
only, not Python/NumPy allocations, process RSS, or arbitrary allocation sites
in external libraries. Type-cache warmup and source creation happen outside
its scope. Do not link the replacements into production extensions.

## Validation status and scope

The focused normal suite has eight tests and passes with NumPy 2.5.3 and
1.26.4. The six behavior/lifetime methods also pass three ASan/UBSan runs on
NumPy 1.26.4. The four accepted-input/lifetime methods pass separately against
the unchanged baseline. Allocation fault injection is **normal-build only**;
all four sites plus the success sentinel are checked, with no expected failure
in the fixed suite. ASan leak detection is disabled and dependent libraries are
not all instrumented. This is not a new TSan or other-platform result.

The initial combined ASan/interception probe failed during module import:
pybind11's fixed `__pybind11_module_cache` key allocated through an external
C++ string path, while the probe's locally bound replacement delete used free.
This allocator-pairing conflict preceded input conversion. The final sanitizer
mode removes replacement allocators entirely; the normal mode retains the
bounded failure audit. No alloc/dealloc diagnostic is suppressed, and disabled
audit APIs explicitly reject calls rather than returning misleading counts.

Five scoped families were rebuilt with GCC 14.2, C++17/O3 and matching release
options: Classic Control, MuJoCo Gym, Dummy, ToyText and MiniGrid. Only the
binding translation units depend on this header; unchanged renderer/project
archives and external dependencies are reused after source/dependency checks.
MiniGrid's 16 unchanged project units remain reused; this header-only change
does not require replacing archive members. The final-source rebuild produces
the same five native binaries as the preceding code-equivalent repair build.

Final checks against frozen ce1 controls pass:

- 13 Dummy Python tests, including the two new permanent regressions, and five
  Classic Control tests.
- 5,499 exact rollout arrays across CartPole, Acrobot, Pendulum and HalfCheetah.
- Eight exact CPU XLA records for CartPole/HalfCheetah.
- Full synchronous CartPole PPO: 256,000 responses, 100 updates and 8,000
  optimizer steps; 101 checkpoints, 223 arrays, 4,925 tensors and 106,238
  scalars match exactly, maximum difference zero.
- 59 ToyText/MiniGrid comparisons: 6,575 arrays and 65,304,189 bytes match.
  Seventeen ToyText tests and two MiniGrid determinism methods covering 82
  MiniGrid-prefixed IDs pass. BabyAI covers one of 96 tasks; render parity is
  DoorKey only. All 21 native entries in this family runtime stay unchanged
  through the tests; only five families are rebuilt, not an all-family wheel.

Python is 3.12.14. General checks use NumPy 2.5.3; PPO/sanitizer runs use NumPy
1.26.4, and PPO uses Torch 2.5.1 CPU/Tianshou 0.5.1. Existing
[PPO configuration and parity method](../ppo/README.md) are unchanged. The
release clients, not the allocator-intercepting probe, are used for training
and throughput. Other platforms, GPU, all-family packaging, upstream oracles,
and general asynchronous PPO equivalence are not established. Async state
comparisons use per-environment order, not arrival order.

## Fixed throughput regression screen

This correctness repair is evaluated separately from the two rejected
performance prototypes. Control labels a/b use ce1; c/d use the minimal RAII
repair. Four complete blocks use acdb, cabd, bdca and dbac, with the four
[existing cases](../recv_list/cases.json), seed 42, 400 warmups, at least three
measured seconds, and all nine allowed CPUs without worker pinning. This gives
64 fresh-process environment samples. The unchanged four-block PPO driver adds
16 full-budget trials, each after two discarded priming updates.

Source/native identities are frozen before/after; timing is serial with no
concurrent builds, tests or desktop sampling. The plan is a bounded regression
screen, not a promise of speedup or equivalence. Complete-block log contrasts,
descriptive bootstrap intervals, A/A dispersion and adverse blocks are retained
together, without adding samples in response to an unfavorable estimate.

## Completed timing result and retention decision

**Retain for the demonstrated leak repair, not as a speed optimization.**
The 20-environment single-thread case is adverse in three of four blocks,
at **-5.86%**, with its descriptive interval entirely below zero. This result
remains visible; the repair is not labeled performance-regression-free.

| Environments / batch / threads | Effect vs ce1 | Descriptive 95% interval | Negative blocks | A/A control / candidate |
| --- | ---: | --- | ---: | ---: |
| CartPole-v1 20/20/1 | -5.86% | -10.94% to -0.49% | 3/4 | -1.09% / -9.72% |
| CartPole-v1 256/256/4 | -2.88% | -5.73% to +0.05% | 3/4 | +13.85% / +13.35% |
| CartPole-v1 1024/256/8 | -2.30% | -14.00% to +7.34% | 1/4 | -7.76% / -13.98% |
| HalfCheetah-v4 256/64/4 | +0.00% | -9.32% to +9.11% | 1/4 | -4.91% / +2.77% |

Each row has eight individual samples per implementation. Rates below are
environment responses/s including autoresets; ranges are observed min/max,
and CV is sample standard deviation divided by mean.

| Case | Control median [range] | Candidate median [range] | CV control / candidate |
| --- | ---: | ---: | ---: |
| CartPole-v1 20/20/1 | 402,658 [386,775, 430,904] | 374,439 [317,803, 479,269] | 3.57% / 12.02% |
| CartPole-v1 256/256/4 | 2,276,711 [1,894,135, 2,883,171] | 2,222,243 [1,702,740, 2,821,269] | 12.53% / 14.93% |
| CartPole-v1 1024/256/8 | 3,012,304 [2,277,289, 4,155,742] | 3,114,450 [1,994,200, 3,754,498] | 18.68% / 20.66% |
| HalfCheetah-v4 256/64/4 | 89,336 [71,494, 97,841] | 87,742 [70,304, 99,690] | 12.26% / 12.11% |

Full-budget PPO gives **+1.88%**, descriptive interval **+1.03% to +3.48%**,
with all four paired blocks favorable. Control median is 21.409 s
(19.817–23.264), candidate 21.103 s (19.855–21.581). However, same-binary rate
effects are -3.18% for control and -2.48% for candidate; the latter interval is
itself wholly negative (-3.92% to -1.17%). These controls demonstrate why the
four-block intervals do not establish a portable gain or performance equivalence.
Ratios of independent medians are not the paired-block effect. All 16 PPO
semantic fingerprints and all 42 timed native manifest entries remain stable.

### Successful-path mechanism and remaining risk

There is no additional shared-owner/control-block layer: the small rank-three
conversion still allocates 8, 12, 24 and 32 bytes, and retains one Python owning
reference. Shared-pointer copying and the callback GIL acquisition/release are
unchanged. However, the successful machine code is not identical. The new
deleter clears its captured unique pointer at strong release, then its stored
owner destructor checks the now-null pointer when the control block is freed.

In the actual GCC 14.2 release Classic Control binary, the int conversion
control block `_M_dispose` symbol grows from 103 to 119 bytes, and `_M_destroy`
from 10 to 95 bytes. Much of the latter is a nonempty-owner cleanup path skipped
after the explicit reset. These code sizes are not timing measurements. The
additional store/check and code-layout changes can introduce overhead; the
available experiment does not causally assign the observed -5.86% to them, nor
justify dismissing that signal as entirely host noise.

The concrete exception-path leak, unchanged accepted-input semantics and
reviewed GIL/weak-owner lifetime justify retaining this minimal correctness
repair with the tradeoff documented. No extra samples were added to obtain
a favorable estimate. Reducing the existing wrapper allocation is a separate
future proposal, to be tested against this repaired baseline rather than mixed
into these measurements. Raw logs, checkpoints and disassembly stay local.

The general rows do not embed loaded-module hashes: the paired summary marks
that identity unverified from rows alone. This run's attribution also uses its
saved explicit runtime-root invocation and separate before/after hashes. PPO
additionally checks the native hash inside every trial. A path or label alone
is not identity proof.

To reproduce the fixed screen, first build/freeze independent ce1 `CONTROL`
and repaired `CANDIDATE` roots, each containing `envpool/`; match dependencies
and assets described in the [review](../REVIEW.md#复现前核对). Set `STUDY` to this
checkout's absolute `benchmark/core_runtime` path and `OUTPUT` to a new external
directory. These commands generate fresh local records, not archived results:

```sh
mkdir "$OUTPUT"
"$PYTHON" "$STUDY/experiments/run_blocks.py" --python "$PYTHON" \
  --variant a="$CONTROL" --variant b="$CONTROL" \
  --variant c="$CANDIDATE" --variant d="$CANDIDATE" --paired-replicates \
  --cases "$STUDY/recv_list/cases.json" --blocks 4 --seconds 3 \
  --warmup 400 --affinities default --out "$OUTPUT/general.jsonl"
"$PYTHON" "$STUDY/experiments/summarize_paired_blocks.py" \
  "$OUTPUT/general.jsonl" --plan "$STUDY/numpy_input_owner/plan.json" \
  --json-out "$OUTPUT/general-summary.json"
"$PPO_PYTHON" "$STUDY/state_tuple/run_paired_ppo.py" \
  --python "$PPO_PYTHON" --baseline-root "$CONTROL" \
  --candidate-root "$CANDIDATE" --output "$OUTPUT/ppo"
"$PYTHON" "$STUDY/state_tuple/summarize_paired_ppo.py" \
  "$OUTPUT/ppo" --output "$OUTPUT/ppo-summary.json"
```

Record selected source/native hashes before and after your run and keep the
exact invocation with the raw samples outside the checkout. The permanent
Dummy regressions run through the existing `dummy_py_envpool_test` target;
run Python package checks outside source checkouts with the desired runtime
on `PYTHONPATH`. Hosted CI is checked separately at publication; local results
do not imply a hosted all-platform pass.

## Build and run

Build the same source twice, selecting independent baseline and fixed header
roots through `-I`. Keep the module name `numpy_input_owner_probe` and use
separate output directories and Python processes. Both builds need C++17,
matching Python/pybind11 headers, Abseil headers and PIC logging archives, and
XLA FFI headers. No environment implementation is needed.

On Linux/ELF, `-Wl,-Bsymbolic-functions` binds replacement new/delete to this
probe, instead of interposing on Python or other extension modules. Use normal
local Python extension loading, never `LD_PRELOAD` or `RTLD_GLOBAL` for this
probe. Build without LTO; the interception self-test confirms one explicitly
requested allocation is seen.

```sh
PROBE="$SOURCE/benchmark/core_runtime/numpy_input_owner"
mkdir -p "$OUTPUT"
"$CXX" -std=c++17 -O3 -g0 -fPIC -fvisibility=hidden -pthread \
  $("$PYTHON" -m pybind11 --includes) \
  -I"$HEADER_ROOT" -I"$ABSEIL_HEADERS" -I"$XLA_FFI_HEADERS" \
  -c "$PROBE/numpy_input_owner_probe.cc" \
  -o "$OUTPUT/numpy_input_owner_probe.o"
"$CXX" -shared -Wl,-Bsymbolic-functions -pthread \
  "$OUTPUT/numpy_input_owner_probe.o" \
  -Wl,--start-group $ABSEIL_PIC_ARCHIVES -Wl,--end-group \
  -o "$OUTPUT/numpy_input_owner_probe.so"
PYTHONPATH="$OUTPUT" "$PYTHON" -m unittest discover \
  -s "$PROBE" -p 'test_numpy_input_owner.py' -v
```

If pybind11 exists only in a matched build tree, use its header flags and the
corresponding Python include flags instead of `python -m pybind11 --includes`.
Use an external output directory for binaries, logs and sanitizer records.

## Separate sanitizer mode

Compile the same source with
`-DNUMPY_INPUT_OWNER_DISABLE_ALLOCATION_AUDIT` and
`-O1 -g1 -fsanitize=address,undefined`; use those sanitizer flags when linking
and omit `-Wl,-Bsymbolic-functions`. This mode removes the replacement
new/delete functions, exposes `audit_enabled = false`, and makes the two
audit APIs throw an explicit unsupported-build error. The actual conversion
and all ownership fixtures remain unchanged.

Run the six behavior/lifetime tests explicitly:

```sh
PYTHONPATH="$ASAN_OUTPUT:$PROBE" "$PYTHON" -m unittest -v \
  test_numpy_input_owner.NumpyInputOwnerTest
```

The normal build exposes `audit_enabled = true`; ordinary test discovery runs
all eight tests, including the two in `AllocationAuditTest`. Do not skip tests
or suppress `alloc_dealloc_mismatch` to run the audit under ASan. Locally bound
replacement delete calls `free`, while an out-of-line libstdc++ allocation
can use ASan's operator new; that allocator mismatch can occur during pybind11
module initialization before the conversion under test runs.

Failure injection is therefore normal-build coverage only. Sanitizer coverage
includes actual-header rejection cleanup, aliases, converted copies, and native
strong/weak-owner teardown. Preload the matching ASan/C++ runtimes for Python
if necessary, and state whether leak detection and dependent-library
instrumentation were used.

## Separate baseline negative control

Run this in its own bounded process with a baseline-built probe. Failure is
intentional evidence of the original defect, not a passing test or part of a
fixed-build result:

```sh
PYTHONPATH="$BASELINE_OUTPUT:$PROBE" "$PYTHON" -m unittest -v \
  test_numpy_input_owner.NumpyInputOwnerTest.test_readonly_rejection_refcount_stays_stable \
  test_numpy_input_owner.NumpyInputOwnerTest.test_readonly_rejection_owner_is_reclaimed
```

The baseline attempts retain only the small test inputs. Do not amplify the
leak or combine this expected-failure run with ordinary fixed-build testing.
The normal fixed-build suite has no expected failures or skipped tests.
