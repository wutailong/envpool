# Observation-only reference PPO phase profile

These tools observe the unchanged reference workload. This document describes
the instrumentation; completed real-runtime validation and all 32 fixed-trial
results are in [the study report](README.md). No runtime or algorithm was changed.

## Files and source fidelity

- `profile_ppo.py`: reuses the untouched `experiments/bench_ppo.py` parser and
  stdlib helpers; copies its `train` and `_train` with narrowly inserted observers.
- `phase_clock.py`: dependency-free aggregate timing/count helper and EnvPool proxy.
- `test_phase_clock.py`: deterministic fake-clock, forwarding and source-AST tests.

The observation-only changes to `_train` are exactly:

1. Create one initially inactive `PhaseClock` and wrap each fresh synchronous
   environment after the original async check, before constructing `Collector`.
2. Route the existing `policy.train`, `collector.collect`, `policy.update` and
   `collector.reset_buffer` calls through the observer, without changing arguments,
   order, returned objects, buffer-length inspection or result-reference retention.
3. Activate immediately after the original whole-training start clock; deactivate
   in `finally` immediately after the measured updates, before the original stop
   clock. Initial setup/reset, disposable warmup, reseeding and final validation
   are uncounted. Measured episode/subset resets are counted.
4. After the original native hash check, validate aggregate phase counts/timing;
   add separate phase/instrumentation fields and hashes of the profile script,
   helper and reference script. Original semantic fingerprints remain unchanged.

An AST test removes only these listed insertions and asserts that both training
functions exactly equal the reference. This is a static observation-diff check,
not a substitute for actual profiled/plain semantic comparisons or the full
step-level parity harness. The original harness and configuration are untouched.

## Preserved workload and checks

The reference configuration is CartPole-v1, 20 synchronous environments, 2 EnvPool
threads, one Torch/intra-op and inter-op thread, CPU deterministic PPO, 100 updates,
2,560 responses per collect, 2 repeats and minibatches of 64. This yields 256,000
environment responses, 12,800 EnvPool `step` calls, and 8,000 Adam steps. The script
retains all original configuration, budget, version/import-origin, native-binary,
optimizer, metrics, final/initial state, and RNG fingerprint checks. No policy
weights, checkpoints or rollout arrays are written.

The helper validates all four top-level call counts against measured updates and
`env_step` against responses divided by 20; a shorter `--iterations 2` run expects
2 of each top phase and 256 `env_step` calls. Reset calls vary with episode groups,
and setup resets do not contribute. All measured env calls must be nested inside
collection. The helper's count-only mode (`clock=None`) exists for light tests;
the executable always uses `time.perf_counter_ns`.

## Interpretation

`phases` contains call/exception counts, inclusive ns, direct-child ns and
exclusive ns. `disjoint_partition_ns` splits training into policy.train, remaining
collection wall time, env.step inside collect, env.reset inside collect, PPO
update, buffer reset, and loop/other residual. Do not add inclusive collection to
its already-included env call times. The partition sums to the rounded original
whole-training wall interval, except a separately reported <=1 microsecond
float-clock rounding adjustment. Nested integer-clock durations and residuals
must be exactly nonnegative; larger wall-envelope discrepancies fail validation.

Env call time includes the Python adapter, conversions and native waits, plus some
observer call/stack overhead. It is not pure C++ time. Collection-minus-env is
remaining collection wall time, not pure Python or non-core time. Native/background
workers can affect other phases. Any later Amdahl estimate must be conditional on
explicit attribution assumptions; this profile does not establish an unconditional
core bound. All phase fractions and wall totals include instrumentation overhead.
Measure that overhead with separate runs of the original untouched harness; never
claim an observed profile/plain difference is a runtime speedup or subtract total
overhead mechanically from individual phases.

The proxy preserves the Collector-facing argument/return aliasing, normal attribute
reads/writes/deletes, length and exception identity; it consumes no RNG. Like the
parity harness's `RecordedEnv`, it changes Python object identity/type and does not
implement every special protocol. Aggregate observers are single-threaded; the
reference's driver calls are serial. The completed study compares matching native/runtime profiled/plain semantic
fingerprints; its scope does not replace full step-level parity for every workload.

## Single-run invocation

Use the same pinned runtime/dependencies, process environment and CPU affinity as
the untouched harness. All argument names/defaults are reused; `--help` describes
the reference argument interface. For example, replace the placeholders before use:

```sh
python -B benchmark/core_runtime/ppo_phase/profile_ppo.py \
  --package /path/to/envpool --label production-profile \
  --native-sha256 VERIFIED_NATIVE_SHA256 --output /new/path/profile.json
```

Never omit independently verified native hashes from formal trials. JSON files are
created exclusively; existing evidence is not overwritten. Only sanitized hashes
and flags are reported, not supplied package/config/output paths. The semantic
hash excludes instrumentation fields; script/provenance hashes identify which
observer and reference produced a result.

## Light checks (no training or scientific imports)

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s benchmark/core_runtime/ppo_phase -p 'test_*.py' -v
```

The initial ten instrumentation tests cover deterministic nesting/partitioning, inactive setup/warmup, count-only
mode, alias-preserving forwarding, unchanged RNG, original exceptions, nesting and
budget failures, negative-clock failures, explicit tolerance, and exact AST fidelity.
