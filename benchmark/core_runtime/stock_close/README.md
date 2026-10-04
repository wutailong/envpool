# Defensive stock shutdown and constructor cleanup

**Retained correctness repairs; no new speed claim.**
The initial shutdown study below describes
[2eb7d289](https://github.com/wutailong/envpool/commit/2eb7d2892d7b9829771d6a3f73c805c4735c589d).
The current source additionally contains the
[constructor-launch cleanup](#constructor-launch-cleanup-follow-up) described at
the end; earlier measurements keep their original binary attribution.

The source-level shutdown review found that teardown relied on background stock
allocators supplying items for a fixed consumer drain. That assumption is not
part of the allocator's stop contract. The repair cancels producer waits and
joins them directly, without requiring future stock publication.

This is a source-inferred lifecycle risk, not a measured original-version hang.
No direct original-failure reproduction or failure-frequency estimate is claimed.
The tests cover the repaired positive-capacity contract and normal API behavior.

## Minimal change and contract

- `CircularBuffer::PutOrStop` acquires a producer permit, then checks a monotonic
  stop flag before reserving a slot, moving input or publishing a consumer item.
  False leaves input ownership with the caller and requires permanent exit.
- `WakeProducersForStop` supplies producer-side terminal permits only. Every
  participating producer must use `PutOrStop`; ordinary `Put` cannot resume on
  a terminally awakened queue. Existing published values remain drainable.
- `StateBufferQueue` checks quit at creator-loop entry and uses `PutOrStop` under
  its existing full producer-mutex scope. Destruction stores stop, wakes enough
  creators and joins them. It no longer performs a blocking stock consumer drain.
- A genuinely admitted pre-stop write may finish after the stop store. All queue,
  flag, mutex and payload storage remain alive through joins. Artificial terminal
  permits never authorize a slot write or consumer publication.

Existing `Put`, `Get`, `TryGet`, state receive/allocation selection, factories,
queue/stock capacities, member layout and stepping-worker shutdown are unchanged.
No stock-reserve reduction or rejected broadcast backend is included. Normal
stock publication adds a post-wait atomic stop check; the previous quit check
moves to loop entry. There are no added allocation requests on this path, but
this is not a measured zero-overhead or throughput-regression-free claim.

The contract requires fully constructed objects, quiescent public callers,
returning allocations/factories and eventual thread scheduling. It does not make
concurrent API use against destruction valid. Partial-construction unwind,
thread-creation failure, allocation exceptions and generic throwing value
assignment remain outside this narrow repair.

## Checks performed

The new [`stock_close_test.cc`](../../../envpool/core/stock_close_test.cc) has
nine bounded positive tests, registered as `//envpool/core:stock_close_test`.
They cover FIFO/wrap, full-capacity cancellation, unchanged canceled ownership
and consumer publication, serialized creators, late entrants, in-flight allocation
ownership, terminal draining, valid pre-stop admission, immediate destruction,
and completed normal receives. Gate-arrival/result waits have 10-second deadlines;
a 30-second fixture watchdog covers all remaining waits and joins. No original-gap
timing sequence is used.

All nine cases passed under each of:
- Optimized GCC build
- AddressSanitizer + UndefinedBehaviorSanitizer
- ThreadSanitizer

That is 27 case executions, without retries or suppressions. ASan options were
`detect_leaks=0:halt_on_error=1:abort_on_error=1`; UBSan and TSan both halt on error.
**LeakSanitizer is not covered:** this host's earlier LSan exit checks reported
unsupported execution under ptrace. Disabling LSan does not establish leak freedom.

Two existing positive typed-Container cases were also rebuilt against this same
retained core: `ContainerOutputTest.ReceivedOutputLivesUntilLastArrayOwner` and
`ContainerOutputTest.ExtractedPayloadOwnsItsStorage`. An explicit GTest filter
selected only those cases; both passed once in optimized, ASan+UBSan and TSan
(another six executions, with 300-second process deadlines). They verify normal
received-output ownership and an extracted payload after pool destruction, using
positive inner shape `{1}`. All 14 consumed project files matched the retained
repair and 612 inputs remained frozen. Matching instrumented GoogleTest objects
were reused after provenance checks; trusted Abseil archives remain uninstrumented.
No full Container suite, forced shutdown schedule, stress or LSan coverage is
implied. No production or test-source changes were needed for these checks.

Fresh ClassicControl and MuJoCo Gym clients then passed ordinary API regressions
against the prior retained runtime, whose relevant source is public
[97688bda](https://github.com/wutailong/envpool/commit/97688bdad104058ae2e4fd08be5579efce4dacac):

- **5,499 exact rollout arrays:** CartPole, Acrobot, Pendulum and HalfCheetah;
  synchronous and per-environment asynchronous streams, positive subset/reordered
  resets, normal termination/autoreset, and retained outputs after pool release.
- **Eight CPU XLA records:** CartPole/HalfCheetah JIT step, separate JIT send/recv,
  scan and canonical per-env asynchronous streams. Each compiled path matches
  ordinary NumPy stepping; retained/candidate record digests also match.
- **One full fixed-config synchronous CPU PPO pair:** 101 checkpoints, 4,925
  tensors, 223 arrays and 106,238 scalar comparisons; maximum absolute difference
  zero. The existing [PPO configuration](../ppo/config.json) uses 100 updates,
  20 training environments, two EnvPool workers and single-thread Torch. This is
  one matched-seed training budget, not general or asynchronous training equivalence.

The PPO harness calls its baseline field `original`; here it means the previous
retained runtime, **not original main**. Runtime imports and actual Classic/MuJoCo
native hashes are checked in isolated processes. The 410 validation inputs remain
frozen. Wall times and serialization bytes are not treated as semantic evidence.

## Build identities and scope

Linux x86-64, GCC 14.2.0, C++17/release O3; Python 3.12.14. Ordinary rollout and
CPU-XLA checks use NumPy 2.5.3; PPO uses the separate recorded NumPy 1.26.4,
Torch 2.5.1+cpu, Tianshou 0.5.1 and Numba 0.68.0 environment. JAX/jaxlib is 0.11.1
on CPU. No Python wrapper or dependency update accompanies the repair.

| Client | Previous retained SHA256 | Repaired SHA256 |
| --- | --- | --- |
| ClassicControl | `c066bf09f34c22f86fc41bb2c2d4c26e1ca886d63a6f0b70f17ad2c0baf3c74f` | `5ecb6f7660e56d0d5f541338e46edd7e67f2d07da3d6f764cfeff05b05ad95c6` |
| MuJoCo Gym | `a9c1f44200764b27d9fb2b976c1c86e8ef983dfcfcee9b930dc5e856cf9b3428` | `c837c2e1e0bd929ef0e0f93488e98f5dc7751a2fcb86dcbcb65607e38d461f5f` |

The focused build preprocesses actual dependencies before compilation: 12,338
inputs are frozen, including 752 compiler-enumerated files and 462 system headers.
The two release dependency audits cover 611 and 640 files. Existing external and
renderer archives are reused after member/source/object checks; no changed core
implementation is supplied by a stale archive. Cached GoogleTest 1.18.0 supplies
the ordinary test harness; sanitizer test units are compiled with instrumentation.
Final tracked code and test files are byte-identical to the tested candidate.

An inherited provenance check first stopped on two already-committed documentation
updates. They were recorded explicitly in the new freeze; all other inherited
records stayed unchanged. A later create-once evidence filename collision stopped
after successful ClassicControl compile/link. A separately recorded continuation
verified and reused those outputs, then compiled only MuJoCo. Neither bookkeeping
stop was a native failure; no result or binary was overwritten or silently retried.

Only these two release clients were rebuilt. The staged package retains 19 other
unchanged native support modules for import. No new all-family, GPU, cross-platform,
clean-wheel or performance coverage is claimed. Existing OpenXLA enum-formatting
compiler warnings remain; this was not a warning-free build. Hosted CI on this
non-main branch may not run; absence of runs is not a pass.

## Re-run the public tests

With a normal configured project/dependency environment, the registered target is:

```sh
bazel test --config=release --jobs=1 --test_output=errors //envpool/core:stock_close_test
```

The recorded validation used focused matching GCC compile/link commands and
existing dependencies, rather than a fresh all-dependency Bazel build. For native
sanitizer checks, compile the test and its GoogleTest units with `-O1 -g1` plus
`-fsanitize=address,undefined` or separately `-fsanitize=thread`; never combine
ASan and TSan in one executable. Use bounded process deadlines and preserve errors.
Only disable leak detection when that coverage is explicitly excluded.

Build every desired environment client against the selected source revision; do
not treat a source checkout as an installed package or replace a loaded binary.
Use the existing [rollout](../check_rollouts.py), [CPU XLA](../xla/check_xla.py)
and [PPO](../ppo/README.md) tools with distinct explicit runtime roots/packages
and freshly recorded hashes. Original raw records, model weights and binaries
remain outside this repository; the retained source, regression test, configuration
and concise results are sufficient to guide a fresh validation in its own setup.

## Follow-on reserve accounting: defer capacity changes

A fixed, four-process positive-size accounting check used this unchanged repair
([2eb7d289](https://github.com/wutailong/envpool/commit/2eb7d2892d7b9829771d6a3f73c805c4735c589d)).
A counting state factory delegated to ordinary `Array(spec)`, recording completed
backing requests by constructor, background allocator and receive caller. Each
case filled and checked 32 normal batches, released returned arrays and exited
scope normally; internal/external deadlines were 20/30 seconds. All four passed
once, with source/dependency hashes unchanged. No candidate or timing study ran.

For `Q = 2 * (floor(N/B) + 2)` and `A = max(1, floor(hardware_concurrency/64))`, source
ownership predicts `2Q+A` settled idle sets. This host reported nine hardware
threads, hence `A=1`. Counted startup and post-32-receive construction totals were:

| N/B | One positive field | Settled startup | Settled after 32 receives | Caller fallback requests |
| --- | --- | ---: | ---: | ---: |
| 1/1 | float[1] | 13 | 45 | 26 |
| 16/4 | float[4] | 25 | 57 | 19 |
| 17/8 | float[4] | 17 | 49 | 15 |
| 256/256 | uint8[4,84,84] | 13 | 45 | 8 |

The last row is **synthetic Atari-observation-shaped storage only**, not an Atari
runtime or full state recipe. Its set requests 7,225,344 bytes (6.890625 MiB);
13 settled sets imply 89.578125 MiB of requested backing. The 45 cumulative
requests total 310.078125 MiB, **not live memory or RSS**. Metadata, allocator
overhead and environment assets are excluded. Factory counters affect scheduling;
the fallback split is one tight-loop observation, not a workload fallback rate or
a guarantee about ready burst coverage. No generic multi-allocator run was made.

Admission before construction could save `A` speculative sets at matched settled
checkpoints, but not the one-new-buffer-per-receive slope. Doing construction
inside the existing publication mutex would serialize currently parallel factories
when `A >= 2`; preserving parallelism requires additional permit ownership,
rollback and terminal-cancellation rules. Lowering stock capacity instead removes
ready burst buffers and can move work into the receive caller. Neither tradeoff
is justified by these counts. Both changes are deferred pending a concrete
memory-constrained workload and representative stock-occupancy evidence.

## Constructor-launch cleanup follow-up

**Retained defensive change; the exceptional launch branches are source-reviewed,
not directly exercised by these tests.** A later synchronous thread-launch
exception could leave earlier recorded threads without the class destructor's
cleanup, because a failed constructor does not call that destructor. This is a
source-derived finding; no original-version abort, resource exhaustion or launch
failure was reproduced.

The two core constructors now reserve handle-vector capacity before launching and
use body-local catches. Their private stop/wake/join helpers are shared with normal
destruction, signal only the successfully recorded core handles, and run while
queues, flags and environment storage remain alive. After successful cleanup,
a bare rethrow preserves the original exception. The initialization ThreadPool
dependency receives the same bounded rollback pattern through a checked-in patch
after `invoke_result.patch`; its stop-under-mutex, notification and normal task
drain behavior remain intact. No cache file, task algorithm, member layout, queue
capacity, public API or destructor exception specification was changed.

Reserve alone is insufficient: thread creation can still fail after reservation.
Cleanup failures, exceptions escaping background work/factories, invalid
configurations and universal constructor exception safety remain outside scope.
The stock allocator's later-launch case requires at least two creators under the
current hardware-count formula; this host has one. Normal worker progress and
returning factories remain prerequisites for joining.

### Final-candidate checks

- **45 positive native executions:** the nine stock-close cases, the two selected
  typed-output cases above, and four new ThreadPool lifecycle cases, each in
  optimized, ASan+UBSan and TSan builds. All passed. New cases use one/three workers,
  verify task results and completion by destruction. They do not guarantee pending
  tasks at destructor entry or exercise constructor rollback branches.
- **5,499 exact rollout arrays**, **eight CPU XLA records**, and **one full fixed
  synchronous PPO pair** matched the prior retained stock-close runtime: 101
  checkpoints, 4,925 tensors, 223 arrays, 106,238 scalar comparisons, maximum
  absolute difference zero. The previous PPO configuration is unchanged; this is
  a fresh correctness comparison, not a throughput measurement.
- Final touched C++ files pass project-config clang-format and cpplint. Native
  tests were rebuilt after a two-line formatting-only reflow; earlier outputs
  remain separately identified and are not added to the final coverage count.

Comparison baseline is **2eb7d289**, not original main or the earlier NumPy-owner
control. Frozen native identities for this comparison are:

| Client | Baseline SHA256 | Constructor-repair SHA256 |
| --- | --- | --- |
| ClassicControl | `5ecb6f7660e56d0d5f541338e46edd7e67f2d07da3d6f764cfeff05b05ad95c6` | `abe524a823c57dc16e3bbc749ace8bba6186c7bb4f329801dc505f7af447b435` |
| MuJoCo Gym | `c837c2e1e0bd929ef0e0f93488e98f5dc7751a2fcb86dcbcb65607e38d461f5f` | `6dea5c2e984703102df1d3dc1c060a402eb350e3ba94c7e62e23ceca86109e24` |

Matching GCC 14.2/C++17/O3 clients were freshly built; 7,881 build inputs and
410 semantic-validation inputs stayed frozen. Actual compiler dependencies prove
the final core snapshot and derived ThreadPool header were selected. The pinned
[upstream source](https://github.com/progschj/ThreadPool/blob/9a42ec1329f259a5f4881a291db1dcb8f2ad9040/ThreadPool.h)
was independently fetched and checked against its Git blob; applying the existing
patch then the new patch exactly matched the derived header. The archive ZIP was
unavailable, so no fresh ZIP checksum verification is claimed. Existing-patch
line offsets were accepted with zero fuzz; the new patch needed no offsets.

Cached link inputs were unchanged. The 55 Bazel archives had 207 member/object/
dependency freshness checks; four retained OpenCV archives lacked historical
per-member dependency files, so their evidence is limited to prior hashes, member
inventory and defined-symbol checks. Core/test and GoogleTest sanitizer units
were freshly instrumented; external archives were not all instrumented. LSan
remains excluded. Only ClassicControl and MuJoCo Gym release clients were rebuilt;
the 19 other support modules were unchanged. There is no new all-family, GPU,
cross-platform, clean-wheel, speed or asynchronous-training claim.

The new ordinary test target is `//envpool/core:threadpool_lifecycle_test`. The
selected native suite additionally uses `//envpool/core:stock_close_test` and
`//envpool/core:container_output_test`, with the latter restricted to the two
positive tests named above. The recorded build used focused GCC commands, not a
fresh Bazel dependency build. CI workflows trigger on main pushes or pull requests
(and release tags where configured); this non-main branch update alone does not
produce hosted CI evidence. Raw records and binaries remain outside the repository.
