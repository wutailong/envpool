# Reuse the typed Container owner and shape

This one-function change follows the validated
[Container ownership repair](../container_ownership/README.md), published as
[`15f80b0934689f635c49ec8764e6b5211e5fd218`](https://github.com/wutailong/envpool/commit/15f80b0934689f635c49ec8764e6b5211e5fd218).
The baseline remains a separate frozen runtime and branch.

## Mechanism and measured allocation reduction

The repaired factory first allocated a typed `Container<T>[]` with a shared
owner, then captured that owner inside a second shared-pointer control block.
It also computed `Shape()` again while constructing the outer `Array`.
The new factory aliases the existing typed owner into `shared_ptr<char>` and
moves its already-computed shape into the existing protected Array constructor.
This matches the aliasing ownership model already used by primitive Array
storage. A fieldless local derived bridge makes that constructor accessible
without exposing a new public API or changing class layouts.

The deleting control block still owns the typed array, so it destroys the full
original capacity at the last strong reference, including populated payloads
outside truncated metadata. Weak observers keep only the control block alive.
The local bridge's Array base is moved into the return value; the bridge is not
used for polymorphic deletion. C++17's aliasing constructor shares an lvalue
owner; ordinary subsequent shared-pointer/vector moves do not allocate.

The isolated [allocation audit](allocation_probe.cc) ran before the edit and
again against the candidate with matching GCC/C++17/O3 flags, without LTO. Each
of 12 dtype/shape combinations performs eight warmups and 10,000 measured
creations/destructions. Specs and printing are outside the counting region;
Array destruction is inside. A volatile factory pointer and checked checksum
consume the constructed metadata and initialized storage.

Container<int> and Container<float> produce the same results:

| Outer shape | Allocation calls before / after | Requested bytes before / after |
| --- | ---: | ---: |
| scalar `{}` | 3 / 2 | 80 / 40 |
| `{64}` | 5 / 3 | 600 / 552 |
| `{64, 3}` | 5 / 3 | 1,640 / 1,584 |
| `{0, 2}` | 5 / 3 | 104 / 48 |

The four primitive-float controls are unchanged: 2/3/3/2 calls and 44/304/824/56
requested bytes respectively. Checksums agree and outstanding counted blocks are
zero. The non-scalar result removes two allocations per Container field per
backing-buffer creation, not per environment transition or per payload element.
A scalar has no shape-vector heap allocation to remove, so it saves one call.
These are implementation-specific requested bytes, not allocator footprint,
resident memory, or an end-to-end speedup.

A nonallocating 64-slot pointer tracker additionally injects `bad_alloc` at each
observed allocation ordinal plus a non-failing sentinel. All 12 shape/type sweeps
pass in each revision, with 46 injected failures in the baseline and 32 in the
candidate. Every trial verifies propagation and zero outstanding intercepted
blocks; tracking overflow or unexpected exceptions fail the executable.
This covers ordinary global new/new[] and nothrow forms, not aligned allocation,
malloc, other threads or arbitrary shapes. Initially null slots do not prove
populated-payload cleanup; the separate lifetime suite is rerun for that purpose.

The first textual-edit precondition failed harmlessly and left the source
unchanged. Its resulting probe run is preserved as an unchanged-control repeat,
not candidate evidence. A subsequent patch-service transport failure also made
no change; the source was checked before the successful edit.

## Executed correctness gates

The frozen candidate code is local commit `20a12b88e97f9c84bf443d0d26f6ecdaba372909`.
Independent review found no lifetime, exception-safety or API/layout blocker.
The following gates were rerun against the new implementation:

- 64 native core/Dummy tests, including all nine ownership regressions
- 62 tests in each of three ASan/UBSan repeats and three TSan repeats
- 11 Dummy and five Classic Control Python tests
- Two normal NumPy lifetime/partial-conversion tests and 25 sanitizer repeats
  of both cases, including the untouched third payload after a conversion error
- Four Box2D test-mode cases: 260 identical recursively fingerprinted records,
  1,040 inner Container arrays, eight retained payloads after pool/outer deletion,
  and repeated truncation/autoreset coverage
- 5,499 exact rollout arrays for CartPole, Acrobot, Pendulum and HalfCheetah
- Eight exact CPU XLA records for CartPole/HalfCheetah
- Full synchronous PPO parity: 101 checkpoints, 223 arrays, 4,925 tensors,
  106,238 scalars; maximum absolute difference zero

Classic Control, MuJoCo Gym, Dummy and the complete six-unit Box2D test client
were rebuilt. Box2D uses `ENVPOOL_TEST` in both comparison clients because those
dynamic diagnostic outputs are absent from its ordinary release spec. Only the
two long existing Dummy stress tests are excluded from sanitizer runs; they pass
natively. LeakSanitizer remains disabled, dependencies/Python are not all
instrumented, and the negative-shape NumPy probe executes on audited NumPy
1.26.4 with libasan and libstdc++ preloaded. It does not inject capsule-allocation
failure. All-family release builds, other platforms, GPU or Container-valued XLA,
and general asynchronous training are outside this validation.

The prior repair's raw-storage/manual-lifetime boundaries remain unchanged.
No queue ordering, worker scheduling, cancellation or publication logic changes.

## Predeclared runtime check

The comparison uses the same five general/Box2D configurations and two Dummy
player configurations as the preceding repair: eight balanced blocks, labels
`a/b` on one identical control and `c/d` on one identical candidate, orders
`acdb`, `cabd`, `bdca`, `dbac`, 400 untimed warmup calls, seed 42 and at least
three measured seconds per fresh process. All 224 samples are retained. Eight
separate full-budget PPO runs use two ABBA/BAAB blocks and discarded priming
updates. No build, other benchmark, sanitizer or desktop sampling runs during
measurement. Rates count environment responses, including autoresets; vector
calls are separate. Dummy measures real normal Python Container conversion;
BipedalWalker uses the diagnostic test-mode spec.

All seven primary throughput intervals include zero. Percentages below are
paired block log-rate contrasts, positive as faster; 10,000 complete-block
bootstrap resamples provide descriptive intervals, not significance guarantees.

| Configuration (environments / batch / threads) | Rate effect | Exploratory 95% interval |
| --- | ---: | ---: |
| CartPole 20 / 20 / 1 | -0.06% | -4.34% to +4.85% |
| CartPole 256 / 256 / 4 | -1.19% | -7.28% to +5.41% |
| CartPole 1024 / 256 / 8 | -3.11% | -9.44% to +3.26% |
| HalfCheetah 256 / 64 / 4 | -0.18% | -1.18% to +0.83% |
| BipedalWalker test mode 64 / 16 / 4 | +0.43% | -0.72% to +1.73% |
| Dummy, max players 1, 64 / 64 / 4 | +6.96% | -1.31% to +14.94% |
| Dummy, max players 4, 64 / 64 / 4 | +5.13% | -1.62% to +11.75% |

The eight-thread CartPole estimate is negative in six of eight blocks. Positive
Dummy points do not establish a speedup: same-binary control b/a effects are
+5.61% and +4.77%, and CartPole 256 b/a moves +9.70%. These diagnostics expose
substantial variation; they do not prove that every difference is noise.

### Initial PPO caution and one fixed confirmation

The first eight PPO timings were adverse: control median 20.734 s, range
20.538–20.778 s; candidate median 20.916 s, range 20.870–21.237 s. Every candidate
was slower than every control in this small trial, with approximately +0.88%
median elapsed time. This result is preserved and was not pooled away.

Because PPO uses a primitive-only Classic Control client, a code-generation
check and exactly one predeclared confirmation followed, with no implementation
or runtime-path changes. Both Classic extensions have 3,232,623 bytes of .text
at the same address. Their raw code/data bytes differ. Among 755,137 decoded
instructions, stripping comments and only RIP-relative data displacements leaves
three differences: diagnostic source-line immediates 85 versus 88 for the
factory/spec-size CHECK. CartPole Step and WriteState match under that limited
normalization. Classic Control state specs contain no Container, and neither
extension contains an out-of-line specialized Container factory. This narrows
the suspected mechanism; it is not full binary identity and cannot rule out
address-layout or other runtime effects.

The additional confirmation runs four fixed blocks of a/b identical controls
and c/d identical candidates using acdb/cabd/bdca/dbac. All 16 samples use the
unchanged full budget, fresh processes and original frozen paths. The paired
rate effect is +1.37%, exploratory interval -0.11% to +2.86%. Control median is
20.746 s, range 20.608–21.324 s; candidate median 20.483 s, range 20.234–21.045 s.
Same-binary b/a and d/c rate effects are -0.56% and +0.74%. Elapsed ranges now
overlap, and all 24 initial/confirmation semantic fingerprints agree.

The initial fixed slowdown did not reproduce, but four confirmation blocks
also do not prove a gain or zero cost. No further timing loops were used to seek
a favorable result. The two windows stay separate in the evidence.

## Decision and reproduction

Retain the narrow change for its directly measured allocation reduction and
single standard shared-owner chain. It adds no persistent cache, public API,
object fields, or scheduling changes. End-to-end speed remains unproven;
possible regressions and both PPO windows remain visible. This is an isolated
experimental performance branch, not a recommendation to replace every runtime
or a universal speed claim.

Use the preceding [ownership validation procedures](../container_ownership/README.md)
with the published ownership runtime as control. All current raw performance
samples, native/Python/sanitizer records, provenance and normalized compiler
commands are in results. The allocation audit is a standalone executable built
against each checkout with matching options; it is not linked into EnvPool.
An unchanged-control repeat caused by the harmless patch-precondition mismatch
is labeled separately from the actual aliasing candidate.

The extra PPO confirmation and its independent summary are reproducible as:

```sh
D=benchmark/core_runtime/container_storage
python "$D/run_paired_ppo.py" --python "$PPO_PYTHON" --baseline-root "$CONTROL_ROOT" --candidate-root "$CANDIDATE_ROOT" --output ppo-confirmation
python "$D/summarize_paired_ppo.py" ppo-confirmation --output ppo-confirmation/summary.json
```

The summary validates all 16 row identities, paired roots/native hashes, positive
elapsed times, matching settings/budgets/semantic fingerprints and fixed orders.
It does not pool the initial eight trials. Source and 84 native manifest entries
were verified before and after both timing windows. Full prior comparators,
main and earlier branches remain unchanged. No binaries, weights, assets,
credentials or private machine paths are published.

Key records: [allocation comparison](results/allocation-comparison.json),
[allocation control](results/allocation-control.jsonl), [allocation candidate](results/allocation-alias-candidate.jsonl),
[validation scope](results/validation.json), [native](results/native.xml),
[ASan/UBSan](results/address.log), [TSan](results/thread.log),
[general throughput](results/performance-summary.json), [Dummy throughput](results/container-performance-summary.json),
[initial adverse PPO window](results/ppo-initial/summary.json),
[fixed confirmation](results/ppo-confirmation/summary-recomputed.json),
[code-generation check](results/classic-codegen.json), and
[build commands](results/build-commands.json). Raw samples accompany each summary.
