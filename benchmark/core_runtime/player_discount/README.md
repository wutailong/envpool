# Initialize discount for every player, without empty-slice writes

The selected repair in `Env::Allocate` preserves scalar assignment when the
actual `player_num` is exactly one and uses `Fill(float(!done))` otherwise.
It preserves the existing default `1 - done`, initializes every allocated
player, and performs no element stores when there are zero players. The check
uses the actual count, not the configured maximum. Public
signatures, native field layout, queues, and completion ordering are unchanged.
Environment-specific overrides can still run after `Allocate` returns.

This is a correctness fix, not a speedup claim. The baseline is the source tree
published as
[`6685e510c1fad428b02152c1a378d6289880c4df`](https://github.com/wutailong/envpool/commit/6685e510c1fad428b02152c1a378d6289880c4df).
The unrelated action-metadata optimization was rejected and is documented
[separately](../action_metadata/README.md); its runtime change is not applied.

## Contract and reproduced defects

`common_state_spec` declares discount as `Spec<float>({-1}, {0.0, 1.0})`.
The `-1` axis is per-player, and the C++ extension documentation explicitly
allows zero players (`docs/content/new_env.rst`). The Python interface documents
the default as `1 - done` (`docs/content/python_interface.rst`).
`TArray::operator=` writes one scalar through its pointer: it does not broadcast
and does not check whether the view has any elements. `TArray::Fill` operates on
the view's full size. Changing the single call site repairs two distinct cases:

1. **Multiple players:** only the first discount previously received `1 - done`.
   Other elements retained backing-buffer contents; fresh backing storage is
   zero-initialized. Ordinary nonterminal Dummy episodes can therefore return
   `[1, 0]` or `[1, 0, 0]`. With `max_num_players=4` and seed 10, the first Step
   has two players and is nonterminal. The new real-pool regression checks all
   players through reset, varying player counts, and the two-player terminal
   transition at step 10.
2. **Zero players:** a zero-length slice does not advance the player offset.
   Its old unconditional scalar store could overwrite the first element of the
   next positive-player slice. The synthetic Env regression pauses the empty
   writer in `IsDone`, after reservation but before metadata initialization;
   completes a one-player nonterminal write; then releases the empty terminal
   writer. The old code deterministically replaces the real player's discount
   1 with 0. Promise synchronization establishes the order, with no sleeps or
   probabilistic scheduling; all explicit writers are joined before assertions.

Ordinary Dummy never allocates zero players; missing input action players are
not the same as a zero-player output state. The zero-player result demonstrates
the documented generic Env extension case. Unsynchronized overlapping writes
can also be a race, but this ordered regression demonstrates corruption without
claiming a captured baseline TSan race report.

The new five selected regressions were compiled against unchanged baseline
headers. Exactly three failed, as expected: nonterminal multi-player defaults,
zero-player adjacent-slot preservation, and the real Dummy regression. The
single-player and terminal controls passed. See the
[negative-control record](results/baseline-negative-control.json),
[complete baseline log](results/baseline-discount-log.txt), and
[baseline XML](results/baseline-discount.xml).

The terminal-only cases also pass against fresh zero-initialized baseline
storage. They check the required result, but alone do not prove that every
terminal element was written; the nonterminal and adjacent-slot negative
controls establish the defect.

## Intentional behavior change and validation

Multi-player discount values after the first player are intentionally corrected.
Zero-player writers stop modifying another environment's player row. Do not
interpret this repair as bitwise preservation of those previously wrong outputs.
Single-player values should remain identical, and the executed single-player
rollout, CPU XLA, and PPO comparisons confirm that within their stated scope.

Both the direct-Fill and selected scalar-preserving sources passed:

- 55 native core/Dummy tests, including all five new regressions
- 53 tests in each of three ASan/UBSan repeats and three TSan repeats
- 11 Dummy and five Classic Control Python tests
- 5,499 single-player rollout arrays exactly equal to the published baseline
- Eight CPU XLA records exactly equal for CartPole and HalfCheetah
- Full synchronous CartPole PPO parity: 256,000 responses, 100 updates, 8,000
  optimizer steps, 101 checkpoints, 223 arrays, 4,925 tensors, and 106,238 scalars;
  maximum absolute difference zero

The two long existing Dummy tests passed natively and were omitted only from
sanitizer runs. Short sync/async/multiplayer tests remained enabled.
LeakSanitizer is disabled because thread inspection is unavailable in this
sandbox; dependencies were not all rebuilt with sanitizers. No leak-clean claim
is made. Only Classic Control, MuJoCo Gym, and Dummy native clients were rebuilt.
Other families, GPU XLA, other platforms, and the complete release build are not
newly validated. The new comparison-summary helper passes two synthetic tests;
all 23 inherited experiment-harness tests also pass in the selected checkout.
One deterministic synchronous PPO configuration is not proof
of general or asynchronous training equivalence.

Native flags and toolchain match the
[preceding initialization validation](../timing_initialization/README.md): Linux
x86-64, GCC 14.2, C++17/O3 clients, Python 3.12.14, and existing read-only Bazel
dependencies. Tests remove `-DNDEBUG`; sanitizer tests add `-O1 -g1` and the
appropriate sanitizer at compile/link time. MuJoCo compiler garbage-collector
parameters limit compilation memory without changing runtime optimization flags.

## Bounded cost check

The saved plan uses four representative sync/async configurations, eight blocks,
400 untimed warmup calls, seed 42, at least three measured seconds per fresh
process, and default scheduling on nine available logical CPUs. Labels `a/b`
share the identical baseline runtime; `c/d` share the identical fixed runtime.
The treatment-balanced orders are `acdb`, `cabd`, `bdca`, `dbac`. The primary
contrast pools each pair's log rates inside each block, then weights blocks
equally. All 128 samples are retained; same-binary contrasts expose host variation.
No builds, native/runtime checks, or other benchmarks run during timing.
Rows contain no measured binary hashes; the separate
[runtime provenance](results/runtime-provenance.json) records identical control
roots and complete frozen-manifest checks before and after measurement.
Rates count environment responses, including ordinary autoresets.

Eight separate full-budget PPO trials use two ABBA/BAAB blocks, with two untimed
priming updates discarded before every freshly seeded run. Final model,
optimizer, RNG, and metric fingerprints are compared outside the timing region.
The shared host is noisy: descriptive block-bootstrap intervals and sample
ranges are evidence about these runs, not guarantees of performance neutrality.

### Initial direct-Fill trial

The complete first trial is retained in [raw samples](results/direct-throughput.jsonl)
and [paired summary](results/direct-paired-summary.json). Direct Fill measured
-6.07% for CartPole 20/20/1, with seven of eight blocks negative and a descriptive
interval [-8.54%, -3.29%]. Other effects were -2.87% for 256/256/4, -1.32% for
1024/256/8, and +1.27% for HalfCheetah 256/64/4; their intervals crossed zero.
Its eight PPO timings had overlapping ranges: baseline median 20.071 s,
direct Fill 19.912 s. All semantic fingerprints matched. This triggered exactly
one alternative, rather than retaining an avoidable generic fill unexamined.

### Code-generation evidence and final same-window comparison

[Static assembly evidence](results/codegen.json) shows that direct Fill leaves
runtime size checks, vector-loop selection, and scalar-remainder handling in
CartPole's WriteState. Its size grows from 2,711 to 2,887 bytes. The actual-count
scalar path restores the baseline's normalized 451-instruction sequence exactly
in this function, including its scalar store. Only absolute target addresses
and RIP-relative displacements were normalized; registers, immediates, stack
offsets, call symbols, and local branch offsets match. This is one function on
this GCC build, not whole-binary identity or proof of a throughput cause.

The [final plan](results/comparison-plan.json) compares all three runtimes in
one window: duplicate baseline labels a/b, direct Fill, and the scalar-preserving
repair. It uses four-label Williams orders, eight blocks, and all
[128 raw samples](results/comparison.jsonl). Control log rates are pooled within
each block. The [complete summary](results/comparison-summary.json) includes
both repairs versus control, the direct between-repair contrast, and A/A.

| Environment | N/B/T | Direct Fill vs control | Selected vs control | Selected descriptive 95% interval |
| --- | --- | ---: | ---: | ---: |
| CartPole | 20/20/1 | +0.72% | +2.85% | -3.36% to +8.68% |
| CartPole | 256/256/4 | +7.98% | -2.08% | -11.14% to +9.19% |
| CartPole | 1024/256/8 | -1.56% | -1.25% | -5.88% to +3.56% |
| HalfCheetah | 256/64/4 | +0.25% | +0.46% | -0.98% to +1.95% |

The original -6.07% does **not reproduce**: unchanged direct Fill is now +0.72%
in that case. Do not say the fast path proved a 6% speed recovery. The selected
repair also has a possible disadvantage versus direct Fill at CartPole 256/4:
-9.32%, with interval [-17.47%, +0.32%] and six negative blocks. The HalfCheetah
same-binary b/a contrast itself is -2.29%, interval [-3.54%, -1.04%], illustrating
why these few-block shared-host intervals are not significance guarantees.

We select the scalar-preserving repair for its proven semantics and restored
original single-player code path. The measurements establish neither universal
speedup nor performance neutrality. No additional redesign or sampling followed
this final comparison.

The final eight PPO runs again have identical semantic fingerprints. Baseline
median is 19.875 s (range 19.607–20.252); selected median is 19.984 s
(range 19.671–20.164). Their ranges overlap. See
[raw PPO samples](results/fast-ppo-throughput.jsonl) and
[summary](results/fast-ppo-throughput-summary.json).


## Reproduce

The new source targets are `//envpool/core:env_discount_test` and
`//envpool/dummy:dummy_envpool_test`. The baseline failure uses the new test files
with the recorded old headers; do not alter the control checkout in place.
For example, filter a baseline test binary to
`EnvDiscountTest.*:DummyEnvPoolTest.DiscountMatchesDoneForEveryPlayer` and expect
three failures. Run the fixed binary with the same filter and expect all five
cases to pass. The full core/Dummy and sanitizer suites are separate gates.

Build separate runtime roots. Use `experiments/run_blocks.py` with
`experiments/shutdown-cases.json`, `--blocks 8 --seconds 3 --warmup 400`,
`--affinities default --paired-replicates`, and labels `a/b` for the unchanged
baseline and `c/d` for the fix. Use `experiments/summarize_paired_blocks.py` with
the saved plan, and `experiments/run_ppo_blocks.py --blocks 2` for training
throughput. Never overwrite a native module while a process imports or uses it.

The initial two-replicate trial uses `--paired-replicates`. The final comparison
uses the runner's default four-label Williams design, without that flag, and
labels `a`, `b`, `direct`, `fast`. Run `summarize_comparison.py INPUT OUTPUT`
for its pooled-control estimates and `test_comparison.py` for its synthetic
checks. The archived `prototypes/direct-fill.patch` recreates the direct-Fill
header from an isolated selected checkout.
