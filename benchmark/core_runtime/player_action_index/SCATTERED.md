# Scattered player-action throughput

Raw run files have been removed from the current tree. This report retains the
conclusions, adverse findings, method and limitations; evidence links below point
to the immutable pre-cleanup commit. See [archive and replay instructions](../ARCHIVE.md).

`bench_interleaved.py` adds one real Dummy case, `DummyInterleavedPlayers4`,
with N=B=64, T=4, max_num_players=4, scalar seed=42, state_num=10,
action_num=6, default process affinity, and worker affinity offset=-1.
It uses ordinary native `_send` / `_recv` and normal Python Container
conversion. It does not retain old observations, bypass conversion, force
collection, or make a leak/RSS claim.

## Targeted workload

The driver sets **env_seed=[42] * 64** at native spec construction. This is a
necessary, explicit difference from the inherited Container workload: Dummy
ends an episode at its per-environment seed, so the normal 42+env_id seeds
would desynchronize player counts after the first terminal environment.
Equal seeds keep all environments at the same state counter across auto-reset.
This is a synchronized equal-player stress workload; do not pool its results
with normal-seed or grouped-player Dummy samples. The explicit seed vector and
`player_order="interleaved-by-player-index"` appear in each row and plan case.

For each possible p=1..4, the driver precomputes
`arange(64*p).reshape(64, p).T.ravel()` before warmup and timing. Dummy actually
cycles p=1..3 when max_num_players=4. The p=4 permutation is still prepared and
unit-tested. Every step applies the same cached permutation to
`players.env_id`, `players.id`, and `players.action`. The environment-level
`env_id` array and `list_action` buffer keep their existing order and identity.
For p>1, each environment's player rows are 64 positions apart, exercising the
native ParseAction gather path. For p=1 the permutation is naturally identity.

Full-batch unique env IDs, equal player counts, grouped input rows, normal
Container conversion, and final layout are checked outside timing. Every
warmup input and its actual outgoing interleaving are checked; the driver
rejects a warmup that never sends p>1 actions. The 400-step runner also requires
that p=1,2,3 were all observed. There is no silent mixed-count fallback.
The timed path includes the same three NumPy indexing/copy operations for both
control and candidate, including p=1. All permutations are reused. This is an
end-to-end stress measurement, not an isolated C++ ParseAction microbenchmark.

## Harness reuse

The sample driver subclasses `container_ownership/bench_container.py`'s
`DummyLoop`. Scoped adapters replace its loop factory and expose a spec factory
that substitutes only env_seed; its real native pool and module path remain
unchanged. The inherited setup, hash checks, scheduler/rusage measurements,
3-second timing loop, response units, and existing JSON fields are retained.
Only the case name and additional workload/coverage metadata are added to the
emitted row. Warmup switches to a separate timed step method so layout checks
and coverage bookkeeping are outside the measured loop.

`run_interleaved_blocks.py` wraps `container_ownership/run_container_blocks.py`
with a one-case plan, the new child script path, and extra row validation. Its
CLI, serial fresh subprocesses, clean subprocess directory and environment,
90-second timeout, exit-code/config/count checks, exclusive output/plan file
creation, and per-root binary-hash consistency checks remain inherited.
Outputs must be distinct new files; a failed run leaves its plan/partial rows
for diagnosis, which the paired summarizer rejects as incomplete.

## Run and analyze

Run in an otherwise idle, isolated measurement window:

```sh
PYTHONDONTWRITEBYTECODE=1 python benchmark/core_runtime/player_action_index/run_interleaved_blocks.py \
  --python /path/to/venv/bin/python \
  --baseline-root /path/to/control \
  --candidate-root /path/to/candidate \
  --out /path/to/new/interleaved.jsonl
```

The fixed plan is 8 blocks × 4 slots = **32 fresh-process samples**, each with
400 warmup steps and at least 3 measured seconds. Labels a/b use the same
control root; c/d use the same candidate root. Block orders are
acdb / cabd / bdca / dbac, repeated twice, with each label in each slot twice.
`--dry-run` prints the plan and all commands without importing an EnvPool
runtime, creating result files, or launching benchmark subprocesses. The
interpreter and both runtime package roots must exist even in dry-run mode.

Use the unchanged paired-block analysis:

```sh
python benchmark/core_runtime/experiments/summarize_paired_blocks.py \
  /path/to/new/interleaved.jsonl \
  --plan /path/to/new/interleaved.jsonl.plan.json \
  --json-out /path/to/new/interleaved-summary.json
```

Check that a/b and c/d measured binary hashes match within their pairs, and
interpret treatment/control effects alongside the b/a and d/c noise controls.
No throughput benefit is claimed by adding this driver.

## Helper validation (no native workload)

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover \
  -s benchmark/core_runtime/player_action_index -p 'test_interleaved.py' -v
```

The tests cover p=1..4 permutations, arbitrary environment order, all three
player fields, untouched environment fields, cached permutation reuse,
validation-free timed stepping, rejected layout/count violations, required
p>1 warmup coverage, actual synchronized native-spec arguments (using a fake
spec), preserved measured-row fields, scoped adapter restoration, exact
32-sample pairing, and the existing paired-summary schema/hash validator.
They use NumPy arrays and fake pools/specs only; no EnvPool runtime is loaded.

Validation on 2026-10-04: all 10 helper tests passed (0.021 s); Ruff 0.16.10
check and format-check passed for the three new Python files; `git diff --check`
passed. An executable `--dry-run` produced exactly one plan and 32 commands,
with no output/plan files created. These checks did not load native EnvPool,
run throughput measurements, build code, commit, or publish anything.
