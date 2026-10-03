# Shutdown wake-up correctness

The selected fix uses one stop-aware dequeue check and wakes workers without
writing shutdown actions into the ring. It fixes the reproduced teardown race;
it does **not** establish a throughput improvement. The exact subset of pending
work canceled at shutdown may change, as explained below.

The baseline source tree is the one published as
[`f1728fec30423f5651584a849a4ea3f3c985674e`](https://github.com/wutailong/envpool/commit/f1728fec30423f5651584a849a4ea3f3c985674e).
The preceding completion-lifetime fix is already present in that baseline and
remains a separate change. This branch is `fix/core-shutdown-wakeup`.

## Reproduced failure

[reproduce_pending_reset.cc](reproduce_pending_reset.cc) repeatedly creates a
one-environment, two-worker dummy pool, calls `Reset`, and immediately destroys
it without calling `Recv`. The worker count is explicitly larger than the
environment count (`num_envs=1`, `num_threads=2`); there is no concurrent public
API call. Default `num_threads=0` selects the smaller of batch size and hardware
thread count, and the default batch size is the environment count. This
demonstrates the explicit oversubscribed configuration, not a race on every
ordinary shutdown. It is separate from the earlier 20-environment/1-worker and
1,024-environment/8-worker performance observations.

The [sanitized baseline TSan report](results/baseline-tsan.txt) identifies a
write in `ActionBufferQueue::EnqueueBulk` from the destructor racing a worker's
payload copy in `Dequeue`. The ring has `2 * num_envs` slots. After a real reset
was queued, enqueueing one sentinel per worker could wrap around and overwrite
a slot still being copied. The old worker stop check happened after that copy.

## Fix and admission boundary

1. The destructor publishes `stop_`, then calls
   `WakeForShutdown(workers_.size())`. This signals permits only: no sentinel
   vector allocation, action-slot writes, or queue-cursor advances.
2. Workers use `DequeueOrStop`. After acquiring an item permit, it checks stop
   with acquire ordering before reserving or reading a slot. Returning `false`
   leaves the output unchanged and makes the worker exit.
3. Passing that check **admits one real action**. The worker copies and executes
   it even if shutdown begins between the check and the copy. There is no
   post-copy stop check. The destructor joins this work before releasing any
   environment, queue, or backing storage.

A worker admitted before stop can finish copying safely because shutdown never
writes payloads. Wake-only permits cannot become actions after stop. The
original `Dequeue` API is retained for ordinary queue clients.

This moves the admission boundary earlier than the old post-copy check. The
exact canceled subset of pending work is intentionally not preserved; an
already-admitted action may execute where the old worker would have discarded
it. This is consistent with the supported shutdown contract, which does not
promise draining or a particular cancellation subset. Removing the sentinel
allocation does not itself imply faster stepping or training.

## Supported shutdown contract

- The owner must finish public operations and prevent new ones before starting
  destruction. Concurrent destruction with `Send`, `Reset`, `Recv`, rendering,
  or another public operation is unsupported.
- Destruction supports idle workers and outstanding work from already-returned
  calls, including pending reset without receive. It may abandon unadmitted
  work; it does not guarantee draining and delivering every result.
- Admitted/in-flight environment work must eventually return. Destruction joins
  it. This is not cancellation of an indefinitely blocked environment callback.
- For standalone queue users, `WakeForShutdown` is terminal: quiesce producers,
  publish nonzero stop before waking, use `DequeueOrStop` for every consumer,
  wake at least all live consumers, and retain queue/flag/output storage until
  all consumers exit. Do not mix shutdown permits with ordinary `Dequeue`,
  clear stop and reuse the queue, or treat wake permits as actions.

## Validation of the selected source

Classic Control, MuJoCo Gym, and dummy native clients were rebuilt from the
selected one-check source. The final source/runtime hashes were checked against
the frozen validation snapshots. Earlier prototype passes were not substituted
for these results. See the [validation summary](results/validation.json) and
[selected PPO parity report](results/one-check-ppo-parity.json).

| Check | Selected one-check result |
| --- | --- |
| Native core/dummy | 38/38 tests passed |
| ASan/UBSan | 36 tests passed in each of five repeats |
| TSan | 36 tests passed in each of five repeats |
| Additional selected-source TSan stress | 4,800 pending-reset lifecycles passed |
| Python | 11 dummy + five Classic Control tests passed |
| Rollout parity | 5,499 arrays exact; zero mismatches |
| CPU XLA/JAX | Eight case records exact across CartPole and HalfCheetah |
| Deterministic CPU PPO | 101 checkpoints, 223 arrays, 4,925 tensors exact; zero difference |
| PPO comparator negative controls | All three injected differences detected |
| Standalone admission-boundary gate | Passed ASan/UBSan and TSan |

The permanent queue regressions cover eight consumers with a two-slot ring,
unchanged cursors/payloads and output on stop, and normal ordered delivery through
ring wraparound. Pool regressions cover idle and pending-reset destruction for
`(environments, workers)` of `(1, 2)`, `(1, 8)`, and `(2, 3)`, repeated 32 times
per shape. A gated reset verifies destruction joins in-flight work before
releasing its environment. An [additional TSan stress run](results/one-check-pending-stress.txt)
repeated the selected-source pending-reset test 50 times: 32 lifecycles per
shape across three shapes, totaling 4,800. This count is separate from any
alternative design's original-reproducer run.

The additional [admission-boundary test](admission_boundary_test.cc) pauses
exactly after successful stop checking and before any slot access, then stops
seven idle consumers and releases the admitted consumer. It verifies the real
action survives, is consumed exactly once, and all idle/next dequeues stop
without phantom actions or overwritten slots. Its
[preparation helper](prepare_admission_boundary.py) inserts a single hook in an
isolated copy of the actual header; production clients do not contain that
hook. The parent timeout bounds hangs without detaching consumers. This tests
queue admission; the gated pool test separately checks environment-work joining.

### Actual build and execution method

Local validation used Linux x86-64, GCC 14.2.0, C++17, and standalone native
builds with existing read-only Bazel dependency artifacts. Client flags were:

```text
-O3 -g0 -DNDEBUG -fPIC -fvisibility=hidden -fno-omit-frame-pointer
-ffunction-sections -fdata-sections -fstack-protector
-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1 -pthread
```

The native test build removes `-DNDEBUG`. Sanitizer builds additionally use
`-O1 -g1 -fsanitize=address,undefined` or `-fsanitize=thread` at compile/link time.
The 36-test sanitizer runs omit only the two long existing dummy SinglePlayer
and MultiPlayers stress tests; both passed in the full 38-test native run.
TSan uses `TSAN_OPTIONS=halt_on_error=1`. ASan uses
`ASAN_OPTIONS=detect_leaks=0` because LeakSanitizer cannot inspect threads in this
sandbox. This is not a leak-clean claim; external dependencies were not all
rebuilt with sanitizers.

Validation ran serially. Only Classic Control, MuJoCo Gym, and dummy native
clients were rebuilt; reused modules establish no new coverage for other
families. [The sanitized build manifest](results/native-build-commands.json)
records the native build commands with location placeholders. The Bazel recipes
below are suggested portable reruns, not the commands used for these local
standalone builds.

## Runtime measurements and selection

The final comparison contains 128 samples: four configurations, eight balanced
four-variant Williams blocks, and one fresh process per sample. `baseline_a`
and `baseline_b` use the **identical runtime**; the other labels are the selected
one-check design and the rejected guarded-sentinel alternative. Runs were
serial, default-scheduled on nine available logical CPUs, with 400 untimed
warmup vector calls, seed 42, and at least three measured seconds. No builds or
correctness checks ran concurrently with this measurement window.

Rates count environment responses, including ordinary autoresets, not vector
calls or pure non-reset physics steps. Within each block, the control is the
equal-weight mean of the two baseline log rates. The reported change is the
geometric mean of the eight candidate/control ratios. Intervals resample those
eight blocks; they are exploratory descriptive bootstrap ranges, not reliable
significance guarantees on this shared, nonstationary host.

N = environments, B = batch size, T = workers. Selected one-check effects:

| Environment | N/B/T | Change vs pooled control | Descriptive 95% range |
| --- | --- | ---: | ---: |
| CartPole-v1 | 20/20/1 | +0.40% | -6.83% to +8.47% |
| CartPole-v1 | 256/256/4 | -0.15% | -10.30% to +10.45% |
| CartPole-v1 | 1024/256/8 | -3.22% | -7.31% to +1.11% |
| HalfCheetah-v4 | 256/64/4 | -1.53% | -3.07% to -0.11% |

Do not read this as "all noise" or a speedup. The asynchronous CartPole and
HalfCheetah observations indicate possible costs that need confirmation on a
controlled machine. The identical-runtime A/A control was itself noisy: for
CartPole 256/256/4, baseline B versus A differed by about -19.6% by ratio of
medians and -13.2% by paired block geometric mean. These observations limit
precision and causal attribution, but do not erase negative candidate results.

The selected implementation balances a direct correctness fix with the observed
tradeoffs. Both rejected alternatives passed independent correctness checks:

- The earlier two-check wake-only design retained the post-copy check. In its
  separate 128-sample/eight-case experiment, full-batch CartPole 20/20/1 and
  256/256/4 had paired effects of -6.6% and -5.8%, negative in all four blocks.
- The guarded-sentinel alternative serialized shutdown writes with the existing
  dequeue semaphore, leaving worker/`Dequeue` source unchanged. In the final
  balanced experiment, CartPole 20/20/1 was -6.74% versus pooled control, negative
  in all eight blocks (descriptive range -9.07% to -4.31%). Unchanged source also
  did not imply identical code: GCC inlined its dequeue and grew the inspected
  worker from 1,719 to 2,124 bytes. These body sizes reflect inlining and cold
  paths, not total executed-work estimates, and do not establish timing causes.

The [raw final samples](results/comparison.jsonl),
[complete per-variant statistics](results/comparison-summary.json), and
[pooled-control contrasts](results/comparison-pooled.json) retain all cases.
The [two-check](prototypes/two-check.patch) and
[guarded](prototypes/guarded.patch) source/test patches can each be applied separately
to a fresh published-baseline worktree to reproduce the rejected alternatives.
Use `git apply --index --unidiff-zero PATH_TO_PATCH` in that untouched baseline
worktree; the saved patches omit context lines and their source blobs were
verified against the tested variants. Do not apply either on top of the selected
fix.

No results from separate experiments are pooled. Choosing the one-check design
is not a claim of no regressions, a universal optimum, or general equivalence of
shutdown cancellation timing.

### End-to-end PPO throughput

Eight full-budget CPU CartPole PPO runs were serialized in two ABBA/BAAB blocks,
four samples per runtime. Each measured run used 100 updates, 256,000 environment
responses, and 8,000 optimizer steps. Two untimed priming updates used disposable
training state; the measured policy, optimizer, collector, and environment were
then recreated with fresh seeded state. Setup, hashing, and result writing were
outside the single whole-training interval.

| Runtime | Samples | Median seconds | Observed range, seconds |
| --- | ---: | ---: | ---: |
| Baseline | 4 | 20.479 | 20.013 to 20.729 |
| Selected one-check | 4 | 20.322 | 20.045 to 20.930 |

These are similar measured times with overlapping ranges, **not a demonstrated
training speedup**. All eight semantic fingerprints matched. Those fingerprints
are a timing-harness sanity check; the separate full PPO parity test above is
the detailed correctness evidence. See the
[training summary](results/ppo-throughput-summary.json) and
[eight raw samples](results/ppo-throughput.jsonl).

## Reproduce on a fresh Linux checkout

Install the repository's normal build prerequisites first; see
[the build instructions](../../../docs/content/build.rst). Run the existing
native test targets in the fixed checkout:

```sh
bazel test --test_output=errors \
  //envpool/core/... //envpool/dummy:dummy_envpool_test

bazel test -c opt --copt=-O1 --copt=-g1 \
  --copt=-fno-omit-frame-pointer --copt=-fsanitize=thread \
  --linkopt=-fsanitize=thread \
  --test_env=TSAN_OPTIONS=halt_on_error=1 \
  --test_arg='--gtest_filter=*-DummyEnvPoolTest.SinglePlayer:DummyEnvPoolTest.MultiPlayers' \
  --test_arg=--gtest_repeat=5 --test_output=errors \
  //envpool/core/... //envpool/dummy:dummy_envpool_test
```

For ASan/UBSan, replace both `-fsanitize=thread` flags with
`-fsanitize=address,undefined` and the TSan environment option with
`--test_env=ASAN_OPTIONS=detect_leaks=0`. A compatible host may separately enable
leak detection. A sanitizer-free run does not substitute for a sanitizer run.

To exercise the original minimal reproducer on the baseline, use a **new,
throwaway detached worktree**, with `BASELINE_WORKTREE` set to an unused path.
Ensure the published baseline commit is fetched first. This intentionally
replaces the dummy test source only inside that throwaway worktree:

```sh
FIX_ROOT=$(git rev-parse --show-toplevel)
BASELINE=f1728fec30423f5651584a849a4ea3f3c985674e
git worktree add --detach "$BASELINE_WORKTREE" "$BASELINE"
cp "$FIX_ROOT/benchmark/core_runtime/shutdown/reproduce_pending_reset.cc" \
  "$BASELINE_WORKTREE/envpool/dummy/dummy_envpool_test.cc"
(
  cd "$BASELINE_WORKTREE"
  bazel test -c opt --copt=-O1 --copt=-g1 \
    --copt=-fno-omit-frame-pointer --copt=-fsanitize=thread \
    --linkopt=-fsanitize=thread \
    --test_env=TSAN_OPTIONS=halt_on_error=1 \
    --test_arg=--gtest_repeat=5 --test_output=errors \
    //envpool/dummy:dummy_envpool_test
)
```

The expected baseline failure is the reported enqueue/dequeue race, but detection
is scheduler-dependent. Repeat with a separate new worktree at the fixed commit
to check the same reproducer against the fix. Do not overwrite the fixed
checkout's permanent regression tests. The existing dummy Bazel target supplies
the required dependencies without widening private target visibility.

For behavior parity, build baseline and fixed packages with identical toolchains,
flags, dependencies, and assets. Use distinct package roots and the existing
[rollout recorder/comparator](../README.md#correctness-and-allocation-evidence),
[CPU XLA harness](../xla/README.md), and
[deterministic PPO harness](../ppo/README.md#reproduce). For this shutdown change,
CPU XLA validation selects `CartPole-v1` and `HalfCheetah-v4`; it does not claim
new MiniGrid coverage. Those adjacent documents describe their own earlier
results, not results for this change.

## Reproduce the balanced runtime comparison

Build separate baseline, selected, and guarded packages with identical native
flags and dependencies. Set each root below to the directory **containing** its
`envpool` package; both baseline labels intentionally use the same root. Set
`CASES_JSON` and `OUT` to fresh output paths, and `PYTHON` to an explicit
interpreter sharing the required dependencies. Run without other builds/tests:

```sh
cat > "$CASES_JSON" <<'JSON'
[
  {"env":"CartPole-v1","n":20,"batch":20,"threads":1},
  {"env":"CartPole-v1","n":256,"batch":256,"threads":4},
  {"env":"CartPole-v1","n":1024,"batch":256,"threads":8},
  {"env":"HalfCheetah-v4","n":256,"batch":64,"threads":4}
]
JSON
"$PYTHON" benchmark/core_runtime/experiments/run_blocks.py \
  --python "$PYTHON" --cases "$CASES_JSON" --out "$OUT" \
  --variant "baseline_a=$BASELINE_ROOT" \
  --variant "baseline_b=$BASELINE_ROOT" \
  --variant "one_check=$SELECTED_ROOT" \
  --variant "guarded=$GUARDED_ROOT" \
  --blocks 8 --seconds 3 --warmup 400 --affinities default
"$PYTHON" benchmark/core_runtime/experiments/summarize_blocks.py \
  "$OUT" --baseline baseline_a
```

The summarizer reports contrasts to baseline A, including the A/A check. The
pooled-control table above instead subtracts the equal-weight mean of baseline
A and B log rates within each block before exponentiating the mean difference.
Keep these estimands separate. Changing the affinity, sample duration, hardware,
or dependencies creates a new experiment rather than extending this one.

## Limits

These are bounded Linux x86-64 results. Full all-family/all-platform coverage,
a clean wheel/release build, full release CI, and GPU/CUDA XLA were not run.
CPU XLA covers CartPole and HalfCheetah only. PPO parity covers one matched-seed,
fixed-budget synchronous CPU CartPole setup, not arbitrary training or async
collection order. No general race-freedom, leak-freedom, or throughput-improvement
claim is made. Binaries, training weights, large rollout archives, and private
command logs are excluded.
