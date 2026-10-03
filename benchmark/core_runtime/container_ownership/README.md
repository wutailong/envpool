# Own dynamic Container outputs through their actual storage lifetime

This repair fixes missing destruction of `Container<T>` payloads when a queued
output is abandoned or a C++ `Recv()` result loses its last owning `Array`.
It also makes the Python conversion handoff exception-safe. This is a resource
lifetime correction; it does not assume an end-to-end speedup.

The comparator is the published player-discount repair
[`3b00a18695cbd7b7c28a1c081c5d7c6769818d4b`](https://github.com/wutailong/envpool/commit/3b00a18695cbd7b7c28a1c081c5d7c6769818d4b).
Main, the original runtime, and all earlier experimental branches are retained.

## Defect and minimal ownership change

`Container<T>` is a `unique_ptr<TArray<T>>`. Previously, `AsyncEnvPool` erased
state element types to `ShapeSpec` before allocation. Its backing `vector<char>`
did not know that placement-constructed Container elements needed destruction.
The successful Python conversion path moved each payload into a capsule and
explicitly destroyed its source slot, so ordinary consumed Python output was
already released. Unreceived output at pool shutdown and ordinary C++ output
destruction bypassed that converter and lost the nested owner.

The three original regression cases are preserved in
[`negative_control.cc`](negative_control.cc). Compiled against unchanged
comparator headers, queued full/partial output and received output both fail:
the counted payload is created once and destroyed zero times after its last
owner disappears. Explicitly extracting and releasing it passes. The test's
negative-control cleanup runs only after the failed assertions and after all
pool/output owners are gone; it cannot mask the failure. The same three cases
pass with this repair.

The production change preserves dtype before erasure and passes matching
allocation recipes into the state queue. Container fields own an actual array
of live, initially null `unique_ptr` elements. Initial buffers, background stock,
and synchronous fallback all use the same recipe. An owning `Array` keeps the
entire original capacity alive, independently of its truncated metadata. Its
last strong reference reclaims that capacity even while weak references retain
the outer control block.

Python conversion now moves each source payload but leaves the source object
live and null. Scoped holders guard both capsule creation and subsequent NumPy
array construction. A partially failing conversion is consuming, not a rollback:
already moved slots remain null; untouched slots remain owned by native storage.

Primitive-only pools receive an empty recipe and retain their existing
`MakeArray` storage path. Queue ordering, completion, and shutdown protocols are
unchanged. The queue stores a new recipe vector, so C++ clients must be rebuilt;
this is not an ABI-compatibility claim. `Array` and `StateBuffer` layouts are
unchanged.

## Boundaries and migration

- Owning `Array` copies extend storage lifetime. `Slice`, indexing, and the
  temporary producer state views remain non-owning.
- C++ consumers may move a payload out or reset it. They must not explicitly
  end a typed source slot's lifetime before the owning backing array is deleted.
- The original ShapeSpec-only buffer/queue constructors, direct raw
  `Array`/`TArray<Container>` construction, and `MakeArray(tuple)` remain manual
  lifetime paths. The new opt-in recipes must match their fields and element
  sizes; arbitrary mismatched recipes are not supported.
- Existing Container placement initialization is retained for raw-storage
  callers. For typed backing it replaces fresh null objects in disjoint,
  once-reserved slots. Reinitializing populated slots is not supported.
- The change supports the existing Container alias, not arbitrary nontrivial
  dtypes. Bytewise `Assign`/`Zero` are still inappropriate for live Containers.
- Existing background allocation failure and partial thread-startup exception
  behavior are outside this repair.

## Executed correctness checks

The candidate passed 64 native core/Dummy tests, including nine ownership cases:
abandoned full/partial batches, last owning output, extracted payloads, original
capacity after both truncation APIs, retained weak observers, mixed shared/player
multidimensional and zero-sized fields, and deterministic stock/fallback reuse.
The replacement tests encode the current six-entry ring for `(batch=1, envs=1)`.
Their public allocation callback gates are released before queue destruction or
fatal assertions. No sleep-based scheduling or private production hooks are used.

ASan/UBSan and TSan each passed 62 tests in three repeats. Only the two long
existing Dummy stress tests are excluded from sanitizer runs; they passed
natively. LeakSanitizer is disabled in this sandbox, so leak evidence here is
the exact counted-deleter regression, not a global leak-clean result.

The candidate-only [`conversion_probe.cc`](conversion_probe.cc) exercises the
actual native pool and production converter. Its two Python cases check retained
inner views after every native owner and outer array is gone, and failure on the
second of three elements: the first was converted, the second rejected, and the
third remained untouched until native cleanup. All three payloads are reclaimed
exactly once. The two cases passed normally and in 25 ASan/UBSan repeats.

The malformed-shape failure case is restricted to the audited NumPy versions
1.26.4/2.1.0; it was executed with 1.26.4. It owns one real integer, performs no
access using the invalid metadata, and relies on NumPy rejecting a negative
dimension before using that external buffer. It does not inject capsule-allocation
failure. The first sanitizer run failed in ASan's `__cxa_throw` interceptor because
the unsanitized Python process loaded libstdc++ only later. The same frozen probe
passed after both libasan and libstdc++ were preloaded. Both logs are retained.

Additional matched-runtime checks passed:

- 11 Dummy and five Classic Control Python tests
- 5,499 rollout arrays exactly equal for CartPole, Acrobot, Pendulum, HalfCheetah
- Eight CPU XLA canonical records exactly equal for CartPole/HalfCheetah
- Full synchronous CartPole PPO: 256,000 responses, 100 updates, 8,000 optimizer
  steps, 101 checkpoints, 223 arrays, 4,925 tensors and 106,238 scalars; maximum
  absolute difference zero
- Four Box2D cases, CarRacing/BipedalWalker with seeds 0/42, two environments and
  threads, 64 steps plus reset, and a 13-step limit. All 260 output records have
  identical recursive byte fingerprints, including 1,040 inner Container arrays.
  Eight retained inner arrays remain valid after pool deletion and outer-array
  collection; each environment covers four truncations and autoresets.

Box2D dynamic diagnostic fields exist only with `ENVPOOL_TEST`. Both complete
six-unit Box2D clients were rebuilt with that flag. This is not ordinary release
Box2D state coverage. Classic Control, MuJoCo Gym and Dummy were separately
rebuilt; other families, GPU XLA, all-platform behavior and a complete release
build are not newly validated. Dependencies and the Python interpreter are not
fully sanitizer-instrumented. One synchronous PPO configuration does not establish
general or asynchronous training equivalence.

## Bounded cost comparison

The predeclared trial uses eight treatment-balanced blocks, seed 42, 400 untimed
warmup calls, at least three measured seconds per fresh process, and default
scheduling on nine available logical CPUs. Labels `a/b` share one frozen control;
`c/d` share one frozen candidate. Orders are `acdb`, `cabd`, `bdca`, `dbac`.
Five general/Box2D configurations contribute 160 samples; two Dummy player
configurations contribute 64 more. Eight separate full-budget PPO runs use two
ABBA/BAAB blocks and discard two untimed priming updates before each fresh run.
No builds, sanitizer tests, parallel benchmarks, or desktop sampling run during
these measurements. All samples, including regressions, are retained.

Rates count environment responses, including ordinary autoresets, not player
responses. Vector calls are recorded separately. Dummy explicitly exercises
normal `_send`/`_recv` Container-to-NumPy conversion, with `N=B=64`, four threads,
and configured maximum players one/four. It is not an RSS or leak benchmark.
BipedalWalker measurements use the diagnostic test-mode output described above.

The primary effect averages the paired log-rate difference within each block,
then weights blocks equally. The intervals below resample complete blocks 10,000
times; with only eight blocks on a shared host they are descriptive uncertainty,
not a significance guarantee.

| Configuration (environments / batch / threads) | Candidate vs control | Exploratory 95% interval |
| --- | ---: | ---: |
| CartPole 20 / 20 / 1 | +1.79% | -2.94% to +6.30% |
| CartPole 256 / 256 / 4 | +3.22% | -2.44% to +8.27% |
| CartPole 1024 / 256 / 8 | +1.94% | -4.39% to +8.68% |
| HalfCheetah 256 / 64 / 4 | +0.81% | -0.23% to +1.93% |
| BipedalWalker test mode 64 / 16 / 4 | -0.54% | -1.49% to +0.64% |
| Dummy, max players 1, 64 / 64 / 4 | -1.38% | -6.22% to +3.89% |
| Dummy, max players 4, 64 / 64 / 4 | +1.93% | -2.12% to +6.06% |

Every primary interval includes zero. This establishes neither a speedup nor
zero cost. BipedalWalker is negative in six of eight blocks, so a small cost
remains plausible. Same-binary comparisons also move substantially: Dummy
players-one control `b/a` is +9.33% (interval +1.46% to +18.28%), and CartPole
256 treatment `d/c` is -8.69%. Neither is a code change. Do not attribute all
variation to the repair, or conclude that host variation explains every effect.

PPO control elapsed time has median 20.595 s, range 20.109–21.337 s; candidate
median 20.359 s, range 20.069–20.581 s. All eight semantic fingerprints agree,
and elapsed ranges overlap. Four samples per variant cannot establish a training
speedup. The repair is retained for demonstrated reclamation correctness, with
these mixed cost measurements disclosed.

The generic runner has no row-level binary hashes; its identical-root label
mapping and complete frozen manifests were verified before and after. Dummy
also hashes its imported extension before/after each sample. Source and 84 native
manifest entries across the comparator/candidate runtimes remained unchanged.
Four short harness smoke calls preceded the declared trial and are not pooled
into performance results.

## Reproduction and evidence

Use separate checkouts/runtimes for the published comparator and this repair;
never overwrite an imported extension. Keep Python/dependencies, compiler flags,
seeds, CPU allocation and runtime selection matched. The recorded Linux x86-64
build uses GCC 14.2, C++17/O3 and Python 3.12.14. Test binaries omit `-DNDEBUG`;
sanitizers add `-O1 -g1` and `-fsanitize=address,undefined` or `thread` at compile
and link. MuJoCo uses the same compiler garbage-collector memory limits as the
comparator. Normal parity/performance uses NumPy 2.5.3; PPO and the audited
negative-shape probe use NumPy 1.26.4. CPU XLA uses JAX/JAXLIB 0.11.1.

The checked-in native regression target is
`//envpool/core:container_output_test`. For example, the repository's existing
Bazel test configuration can run it with:

```sh
bazel test --config=test //envpool/core:container_output_test
```

The recorded execution used direct compilation against the existing read-only
Bazel dependency tree, not a new full Bazel release build. Exact compiler/link
argument arrays are in [build-commands.json](results/build-commands.json), with
machine roots replaced by named variables. `native-baseline` uses the separately
preserved three-case negative source; its original temporary filename is
relocated to `negative_control.cc`. The baseline is expected to fail two cases.
Use the complete current native test for the candidate. The standalone Python
probe must be compiled as `container_conversion_probe` for the same Python ABI;
its recorded normal and sanitizer link commands are included. It is candidate
only because inspecting baseline slots after their explicit destruction would
be invalid.

With `CONTROL_ROOT` and `CANDIDATE_ROOT` pointing to the parent of each envpool
package, these commands reproduce the bounded comparisons from the repo root:

```sh
B=benchmark/core_runtime
python "$B/container_ownership/check_box2d.py" record --variant control --envpool-root "$CONTROL_ROOT" --out control-box.json
python "$B/container_ownership/check_box2d.py" record --variant candidate --envpool-root "$CANDIDATE_ROOT" --out candidate-box.json
python "$B/container_ownership/check_box2d.py" compare control-box.json candidate-box.json
python "$B/experiments/run_blocks.py" --python "$PYTHON" --variant a="$CONTROL_ROOT" --variant b="$CONTROL_ROOT" --variant c="$CANDIDATE_ROOT" --variant d="$CANDIDATE_ROOT" --paired-replicates --cases "$B/container_ownership/results/cases.json" --blocks 8 --seconds 3 --warmup 400 --affinities default --out general.jsonl
python "$B/container_ownership/run_container_blocks.py" --python "$PYTHON" --baseline-root "$CONTROL_ROOT" --candidate-root "$CANDIDATE_ROOT" --out dummy.jsonl
python "$B/experiments/summarize_paired_blocks.py" general.jsonl --plan "$B/container_ownership/results/plan.json" --json-out general-summary.json
python "$B/experiments/summarize_paired_blocks.py" dummy.jsonl --plan dummy.jsonl.plan.json --binary-hash-field binary_sha256 --json-out dummy-summary.json
python "$B/experiments/run_ppo_blocks.py" --python "$PPO_PYTHON" --variant control="$CONTROL_ROOT" --variant candidate="$CANDIDATE_ROOT" --blocks 2 --output ppo-times
```

Box2D commands require both `ENVPOOL_TEST` builds; they fail explicitly without
the diagnostic Container fields. Use the existing [PPO configuration and parity
harness](../ppo/README.md) for the full checkpoint comparison. The new benchmark
helper passes two synthetic routing/schema tests; all 23 inherited experiment
helper tests also passed. The exact executed validation scripts are included as
normalized text records.

Key evidence:
- [Validation scope](results/validation.json), [negative-control log](results/baseline.log),
  [native XML](results/native.xml), [ASan/UBSan repeats](results/address.log),
  [TSan repeats](results/thread.log)
- [NumPy normal probe](results/conversion-final.log),
  [initial sanitizer setup failure](results/conversion-asan.log),
  [corrected sanitizer launch](results/conversion-sanitizer-launch.json),
  [50 passing sanitizer cases](results/conversion-asan-preloaded-cxx.log)
- [Box2D comparison](results/box2d-comparison.json) and
  [per-case counts/fingerprints](results/box2d-record-summary.json),
  [rollout comparison](results/rollout-comparison.json),
  [CPU XLA](results/xla-comparison.json), [PPO parity](results/ppo-parity.json)
- All [160 general/Box2D samples](results/performance.jsonl) and
  [summary](results/performance-summary.json); all [64 Dummy samples](results/container-performance.jsonl)
  and [summary](results/container-performance-summary.json)
- All [eight PPO timing samples](results/ppo-throughput-samples.jsonl) and
  [summary](results/ppo-throughput-summary.json), [frozen-runtime provenance](results/runtime-provenance.json)

No native binaries, weights, assets, credentials or private machine paths are
included. Full Box2D per-element traces and PPO checkpoints remain local; the
repository contains the deterministic recorder, aggregate records/fingerprints,
and all raw performance samples.
