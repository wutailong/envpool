# Generated ActionSlice batches: not retained

**Decision: reject this as a general core optimization.** It removes one batch
allocation and shows a favorable eight-thread CartPole signal, but the heavier
HalfCheetah workload is adverse in seven of eight blocks. The
[unapplied prototype](prototypes/generated_enqueue.patch) includes its ten
regression tests and Bazel target. Production core remains ce1c47f2.

The fixed runtime comparison is complete. The retained comparator is the
cumulative [ce1c47f2 core](https://github.com/wutailong/envpool/commit/ce1c47f238a069454f732a70089d1b9857dcc6de), also present in
[ad4a2926](https://github.com/wutailong/envpool/commit/ad4a29269024ae209212bb0e45477b607771400d).
The preceding recv-list prototype was rejected and is **not** part of this
candidate. Results below are workload-specific evidence, not a universal speedup.

## Reason and scope

`SendImpl` and `Reset` formerly built temporary `vector<ActionSlice>` batches,
then copied them into the queue. `EnqueueBulkGenerated` constructs the same
three scalar fields directly in the reserved ring slots. It invokes the getter
synchronously and retains no callback or input pointer.

- The vector API keeps its original signature, including member pointers.
- Producer serialization, cursor arithmetic, single bulk publication, dequeue
  synchronization, capacity preconditions and shutdown are unchanged.
- Every `SetAction` and synchronous accounting update still precedes enqueue.
  The local batch owner keeps borrowed IDs alive until generation finishes.
- SFINAE requires a nonthrowing getter with a nonthrowing ActionSlice result;
  invocation uses the matching rvalue index category. Getters must not block,
  reenter the queue or publish work themselves.
- The previous negative-narrowed-count `length_error` rejection is preserved
  before mutation. Reset retains first-axis indexing for positive multidimensional
  ID arrays rather than accidentally flattening them.
- Generation now occurs inside the enqueue interval. Internal `dur_send_`
  therefore includes that work; it is not the metric used to decide retention.

No Python binding, environment physics, training algorithm or hyperparameter
changes accompany this candidate. There is no new backpressure or guarantee
for callers that violate the queue's existing outstanding-action bound.

## Measured allocation traffic

[allocation_probe.cc](allocation_probe.cc) exercises the actual public methods
on a deterministic native Env. Configurations are 1/1/1, 20/20/1, 256/256/4 and
256/64/4 (environments / returned batch / threads). Each method/configuration
has three repeats, each with eight excluded warmups and 128 measured calls.
The async case submits all 256 IDs per call, then drains four batches outside
the counter; it must not be described as measuring a 64-ID submission.

All 48 paired records have matching checksums and returned state validation.
The calling-thread ordinary-new counter excludes input preparation, Recv,
worker/background threads and destruction. Per-call totals are constant within
each record:

| Public method | Allocation calls before → after |
| --- | ---: |
| Send(const vector&) | 6 → 5 |
| Send(vector&&) | 2 → 1 |
| Send(const Action&) | 8 → 7 |
| Reset(const Array&) | 2 → 1 |

Every measured call removes exactly one `ActionSlice` vector allocation.
Here `sizeof(ActionSlice)` is 12, saving 12, 240 or 3,072 requested bytes for
1, 20 or 256 submitted IDs respectively. These are allocation traffic counts,
not RSS, total process allocation, or throughput improvements. Both versions
use identical C++17/O3 settings, without LTO; source-buffer allocation is outside
the counted region and every returned ID/action/step/common state is checked.

## Correctness checks

- 98 native core/Dummy tests, including 10 new generated-enqueue tests.
- 96 tests in each of three ASan/UBSan and three TSan runs. The two existing
  long Dummy stress methods pass natively and are omitted only under sanitizers.
- After a test-only lint refactor, the final 10-test target was separately
  rebuilt and repeated three times in all three modes.
- New cases cover vector/generated parity, mixed bulk APIs, ring wrap and
  unsigned rollover, bounded multi-producer/consumer delivery, whole-bulk
  publication, delayed consumers, shutdown, getter constraints, all Send
  overloads, retained action ownership and reordered Reset IDs of ranks 1–3.
- Eleven existing Dummy and five Classic Control Python tests; 5,499 exact
  rollout arrays; eight exact CPU XLA records for CartPole and HalfCheetah.
- Full synchronous CartPole PPO parity: 256,000 responses, 100 updates, 8,000
  optimizer steps, 101 checkpoints, 223 arrays, 4,925 tensors and 106,238 scalars.
  Maximum difference is zero against the retained recording.
- Twenty-three experiment-helper tests, C++ formatting, cpplint and diff checks.

LeakSanitizer is disabled; external dependencies are not all instrumented.
This does not establish all-family, other-platform, GPU or general asynchronous
PPO equivalence. Async output comparisons normalize per-environment streams;
completion arrival order is not required to match.

Final supplemental coverage also passes: 59 ToyText/MiniGrid cases, 6,575
arrays and 65,304,189 bytes equal exactly; 17 ToyText tests and two determinism
methods covering 82 MiniGrid-prefixed IDs pass. BabyAI covers one representative
of 96 registered tasks; render parity is DoorKey only.

The first MiniGrid link accidentally used its previous archive despite compiling
two affected archive members. Its initial passing rollouts were **not accepted**
as complete candidate coverage. After timing finished, a new archive was built
and all 15 member hashes verified: two current rebuilt members and 13 unchanged
members. That archive and two current binding objects were linked into a new,
separate runtime, then the family comparison and tests were rerun. Final native
hashes stayed unchanged throughout those tests. The three timed clients were
unaffected: their reused renderer archive dependency files exclude both modified
headers. No stale MiniGrid artifact was substituted into the timing experiment.

## Fixed timing plan

Use the same four [general cases](../recv_list/cases.json) and
[8-block plan](../recv_list/plan.json) as the preceding study, but fresh frozen
runtimes and new output paths. No earlier samples are reused. Identical control
labels a/b and candidate c/d use orders acdb, cabd, bdca, dbac twice, seed 42,
400 warmups and at least three measured seconds per fresh process. The four
cases give 128 samples; the unchanged one-/four-player Dummy driver adds 64.
The separate four-block full-budget PPO driver gives 16 runs with two discarded
priming updates before each freshly seeded training run.

All timing is serial, without builds, regression tests or desktop sampling.
Native and production-source hashes are frozen before/after. Rates count
environment responses including autoresets. Paired log-rate effects, complete-
block bootstrap intervals, adverse blocks and A/A dispersion are reported
together; no samples were added merely to obtain a favorable estimate.

## Completed results and decision

All 192 environment samples and 16 PPO trials completed; all 42 native entries
across the two timed roots and all production header hashes matched before and
after. This count includes reused support modules, not 42 rebuilt clients.
All PPO semantic fingerprints agree. Positive effects mean faster candidate
**relative to ce1**, from paired block log-rate contrasts:

| Environments / batch / threads | Effect | Descriptive 95% interval | Negative blocks | A/A control / candidate |
| --- | ---: | --- | ---: | ---: |
| CartPole-v1 20/20/1 | +2.71% | -0.79% to +6.15% | 3/8 | +5.68% / +0.15% |
| CartPole-v1 256/256/4 | +5.30% | -3.98% to +15.43% | 2/8 | +8.24% / -11.58% |
| CartPole-v1 1024/256/8 | +11.00% | +2.31% to +20.69% | 1/8 | +0.23% / -4.23% |
| HalfCheetah-v4 256/64/4 | -4.44% | -8.48% to -0.82% | 7/8 | -2.97% / -0.50% |
| DummyPlayers1 64/64/4 | +5.02% | +0.09% to +10.70% | 2/8 | -2.05% / -2.05% |
| DummyPlayers4 64/64/4 | -1.07% | -6.04% to +3.70% | 3/8 | -3.73% / -4.45% |

Each implementation has 16 individual samples per environment row. Below,
rates are responses/s, brackets are observed min/max, and CV is sample standard
deviation divided by mean. These ranges are not confidence intervals.

| Case | Control median [range] | Candidate median [range] | CV control / candidate |
| --- | ---: | ---: | ---: |
| CartPole-v1 20/20/1 | 416,562 [339,248, 482,185] | 425,697 [381,595, 501,987] | 9.74% / 7.16% |
| CartPole-v1 256/256/4 | 1,951,828 [1,494,479, 2,576,437] | 2,004,794 [1,546,148, 2,732,299] | 18.96% / 15.72% |
| CartPole-v1 1024/256/8 | 3,530,733 [2,915,968, 4,744,260] | 3,928,228 [3,477,632, 4,393,557] | 15.92% / 6.98% |
| HalfCheetah-v4 256/64/4 | 80,587 [74,557, 98,139] | 78,348 [69,767, 98,525] | 9.71% / 9.31% |
| DummyPlayers1 64/64/4 | 649,788 [477,110, 683,999] | 673,805 [529,348, 729,241] | 9.12% / 7.52% |
| DummyPlayers4 64/64/4 | 451,112 [370,743, 480,596] | 433,965 [364,068, 489,175] | 8.43% / 8.38% |

The separate full-budget PPO estimate is **+0.27%**, interval **-0.72% to
+1.15%**, with one of four blocks negative. Control median training time is
20.765 s (20.339–21.606), candidate 20.597 s (20.148–22.166). Same-binary rate
effects are +1.82% for control and -1.10% for candidate. This does not establish
a training speedup; ratios of independent medians are not the paired estimand.

The eight-thread CartPole signal is favorable in seven of eight blocks, and
single-player Dummy is favorable in six. However, HalfCheetah is adverse in
seven blocks, while ordinary four-player Dummy is inconclusive. A/A variation
is still substantial. Existing CPU/runnable-wait counters do not identify the
cause of the HalfCheetah difference, nor justify dismissing it as entirely
scheduler noise. No portable +11.00% gain or fixed -4.44% slowdown is claimed.

A targeted confirmation was considered, then declined: the completed fixed
study already exposes the relevant cross-workload risk. No extra timing samples
or conditional batch-size tuning were added. The measured allocation saving and
CartPole signal are useful findings, but insufficient to change the general
retained core under this evidence. Keep the prototype unapplied and retain both
the favorable and adverse conclusions. Raw runs remain local; this branch adds
source/tests and conclusions, not generated sample or checkpoint files.

## Reproduction

First use the exact pre-report checkout, then apply the prototype there.
Its core is ce1, but its test BUILD layout includes later retained targets;
do not apply this zero-context patch directly to the historical ce1 checkout.

```sh
git clone --branch docs/core-current-family-coverage https://github.com/wutailong/envpool.git envpool-generated-enqueue
cd envpool-generated-enqueue
git checkout --detach ad4a29269024ae209212bb0e45477b607771400d
# Set REPORT_CHECKOUT to the checkout containing this report and prototype.
git apply --check --unidiff-zero "$REPORT_CHECKOUT/benchmark/core_runtime/generated_enqueue/prototypes/generated_enqueue.patch"
git apply --unidiff-zero "$REPORT_CHECKOUT/benchmark/core_runtime/generated_enqueue/prototypes/generated_enqueue.patch"
```

This also adds `envpool/core/generated_enqueue_test.cc` and its Bazel target.
Keep the control checkout unmodified. Build each revision in a separate root,
matching compiler/options/dependencies;
record actual source and native identities, since the package version alone
cannot distinguish them. The probe needs core headers, ThreadPool,
concurrentqueue, Abseil logging/check libraries and pthreads, with no Python or
assets. Its header documents the measurement boundary. Write its JSON output
outside the source tree and compare records by method, configuration and repeat.

The native target is `//envpool/core:generated_enqueue_test`; run it alongside
the existing core/Dummy tests, with the relevant sanitizer configuration.
The local run used the same focused GCC 14.2 standalone build method as earlier
studies, plus read-only external Bazel libraries. A clean Bazel/dependency rebuild
or an all-family wheel is not claimed.

For fresh runtime trials, follow the unchanged [timing commands](../recv_list/README.md#reproduce-the-fixed-timing-plan)
with independently built ce1 control and this candidate. Keep all fresh raw
samples, manifests and failures locally. This branch retains useful tests,
methods and conclusions rather than generated logs, checkpoints or binaries.
