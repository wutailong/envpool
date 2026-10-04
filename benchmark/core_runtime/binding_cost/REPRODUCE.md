# CartPole binding-cost probe

`probe.py` is standalone Linux diagnostic tooling (Python 3.12+, NumPy,
Gymnasium, and an already-built EnvPool runtime). It uses `/proc`, CPU affinity,
and POSIX resource counters. It does not rebuild or modify EnvPool.

## Select and verify a runtime

Use a fresh interpreter for every invocation. Set `PROBE` to this repository's
`benchmark/core_runtime/binding_cost/probe.py`, and run outside any EnvPool source
checkout with `PYTHONPATH` cleared. `--source` defaults to this repository and
selects only `benchmark/core_runtime/runtime.py`, the shared loader.

- Omit `--envpool-root` for the selected interpreter's installed EnvPool.
- Or pass a package root containing `envpool/__init__.py`; the loader verifies
  that the imported package belongs to that root.
- Supply `--native-sha256` with the independently recorded SHA-256 of the intended
  `classic_control_envpool` native module. The probe resolves the actual module
  from the pool's class hierarchy and checks its bytes before reset, warmup, or
  measurement. A label never substitutes for this identity check.
- `--variant` and `--label` are arbitrary descriptive labels. JSON records actual
  package, wrapper, native-module, loader, and driver paths and hashes, the
  expected native hash, interpreter/dependency versions, and thread settings.

For example, with `PROBE`, `EXPECTED_SHA256`, and optionally `RUNTIME_ROOT` set
for the runtime you intend to inspect:

```bash
# Installed runtime, using its interpreter:
PYTHONPATH= python "$PROBE" --variant installed \
  --native-sha256 "$EXPECTED_SHA256" --validate-only

# Explicit built runtime, using an interpreter with its dependencies:
PYTHONPATH= python "$PROBE" --variant candidate \
  --envpool-root "$RUNTIME_ROOT" \
  --native-sha256 "$EXPECTED_SHA256" --validate-only
```

Untimed validation covers both cases (`small`: 20 envs/1 worker; `medium`:
256 envs/4 workers), each with four fresh, sequential, seed-42 pools. It compares
32-step positive ordinary-input rollouts across modes within that runtime:
public output leaves, native receive arrays, dtypes, shapes, bytes, key order,
and metadata. It checks termination/autoreset, unchanged input buffers, hook
cleanup, pool release, and worker-thread cleanup. It does not establish
cross-runtime parity or audit empty arrays or other negative/edge inputs.

## Diagnostic modes and timing

- `public`: actual public `env.step(action)` with preallocated zero actions.
- `raw`: convert/check once, then repeatedly call native `_send`/`_recv`; excludes
  output wrapping and public receive bookkeeping. This bypass is diagnostic,
  not a behavior-preserving production change.
- `profile`: public stepping with instance-local, non-nested `_from`,
  `_check_action`, `_send`, `_recv`, and `_to` hooks; each records wall, process
  CPU, and calling-thread CPU clocks.
- `profile-no-clock`: the same public hooks and counters, with phase clocks off.

Only run timings after validation and explicit coordination with other work:

```bash
PYTHONPATH= python "$PROBE" --variant candidate \
  --envpool-root "$RUNTIME_ROOT" --native-sha256 "$EXPECTED_SHA256" \
  --case small --mode public --calls 10000 --warmup 400 \
  --label diagnostic --timing-authorized
```

Each invocation emits one JSON object to stdout and creates no result files.
Select the mode explicitly and use a fresh process for each sample; serialize
runs. `--pin` requests caller CPU 0 and workers starting at CPU 1, requires those
CPUs in the inherited affinity, and verifies actual thread masks. It does not
establish physical-core isolation. `--system-diagnostics` adds advisory host
snapshots outside the timed loop. Neither option is required for validation.

Clock/hook costs remain in the measured totals; no overhead is subtracted.
Process CPU can include overlapping workers and is not exclusive phase
attribution. Scheduler snapshots bracket a slightly wider interval than the
clocks. Do not treat diagnostic bypass or instrumented timings as production
speedups. Keep raw outputs outside the repository.

## Portability boundary

This is a packaged copy of the completed study's original
`binding_phase_probe.py`, whose measured SHA-256 was
`93dbbf2764de99cec26f7b1bd62f0447172d40bae05658f676b204cd84ddbff7`.
The measured loops, phase hooks, raw runner, and positive validation logic are
unchanged apart from formatting. Portability changes replace workspace-specific
runtime/label restrictions with explicit roots or interpreter selection, add
required native-hash verification, derive the default source root, and remove
one host-specific nine-logical-CPU allocator-count annotation. Local lint
exemptions preserve the original operations rather than refactor them.

The fixed study results belong to that original helper and its recorded runtime
identities. They are not retroactively attributed to this portable copy. Port
validation and any tiny smoke checks provide no new performance evidence.
