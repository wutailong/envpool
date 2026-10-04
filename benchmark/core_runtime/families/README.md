# Additional ToyText / MiniGrid core validation

This is historical **first-round** validation of
[7de9691f](https://github.com/wutailong/envpool/commit/7de9691f724e5739f73c83d52a0ab57bf166d37b)
against the original 9c31c547 runtime. Their recorded success must not be
attributed to later core revisions. A separate [current-core report](../current_families/README.md)
now records newly rebuilt ce1c47f2 ToyText/MiniGrid modules and new positive
regression results; see also the [current scope](../REVIEW.md).

All work is isolated from the original checkout, installed wheel, Bazel outputs,
and the primary candidate runtime. No production-source files were edited here.
The candidate family modules were rebuilt against the five modified core headers.

## Results

- ToyText: one translation unit rebuilt.
- MiniGrid: all 17 translation units rebuilt, including every one of the 15
  static-library members. Every original dependency file included changed
  `envpool/core/array.h`; reusing the old MiniGrid static archive would have left
  stale core code. The final archive consists entirely of newly compiled objects.
- Only unchanged external native dependency libraries were reused, read-only.
- Existing ToyText test suite: 17 tests passed.
- Existing MiniGrid determinism suite: 2 tests passed, covering all 82 registered
  MiniGrid IDs for same-seed full-length rollouts and different-seed behavior.
- Original/candidate rollout comparison: all 59 cases passed, with 6,575 exact
  dtype/shape/byte comparisons covering 65,304,189 bytes. Full counts and coverage
  are in `results/summary.json` and `results/parity/comparison.json`.

The final build discovered the installed OpenCV include version (`opencv5`), after
an initial include-path attempt used `opencv4`. This was a build-script issue;
no EnvPool implementation change was needed.

## Portable comparison

`compare_families.py` accepts paths through the command line and imports the
baseline and candidate in separate Python processes:

```sh
python benchmark/core_runtime/families/compare_families.py \
  --candidate-runtime "$CANDIDATE_RUNTIME" \
  --output "$RESULTS" --steps 192
```

The current Python's installed EnvPool wheel is the baseline. An explicit
`--original-runtime "$BASELINE_RUNTIME"` is also supported. A runtime is a folder
containing an `envpool/` package, with its native modules already built. Keep the
candidate runtime private: copy the installed package into a fresh folder and
replace only that copy's `toy_text/toy_text_envpool.so` and
`minigrid/minigrid_envpool.so` with newly built modules. All normal runtime
requirements must be available. Source native dependency environment settings
when those libraries are not installed system-wide.

```sh
python benchmark/core_runtime/families/compare_families.py --compare-only --output "$RESULTS"
```

This rechecks saved original/candidate NPZ files, requiring identical array keys,
dtypes, shapes and bytes. It does not claim a numerical tolerance pass.

Coverage consists of 19 task/configuration variants at each of:

- Sync: 16 environments, 16 batch, one thread
- Sync: 32 environments, 32 batch, four threads
- Async: 32 environments, eight batch, four threads

Each full case records 192 outputs per environment, seed 71, episode horizon 43.
Async arrivals are normalized by environment ID and per-ID output index; actions
depend only on those counters and seed. Observations (including MiniGrid image,
direction and mission arrays), rewards, terminated/truncated flags and every
nested info array are all checked. Eight early batches per full case stay alive
across later calls and environment closure/deletion, with byte checks throughout.

Additional Taxi-v4 and MiniGrid-DoorKey-8x8-v0 cases use seed 91, horizon 31, and
three subsets: `[7,1,15,3]`, `[8]`, and descending IDs `[15,...,0]`. Each subset is
explicitly reset and stepped 96 times, checking ID order. MiniGrid RGB frames at
steps 0, 31, 63 and 95 are compared byte-for-byte. All six reset-result batches
remain unchanged after subsequent subset resets/steps and environment deletion.

## Portable isolated native rebuild

```sh
python benchmark/core_runtime/families/build_families.py \
  --source "$CANDIDATE_SOURCE" --baseline-source "$ORIGINAL_SOURCE" \
  --bazel-execroot "$EXISTING_BAZEL_EXECROOT" \
  --output "$PRIVATE_BUILD/toy_text" --family toy_text
python benchmark/core_runtime/families/build_families.py \
  --source "$CANDIDATE_SOURCE" --baseline-source "$ORIGINAL_SOURCE" \
  --bazel-execroot "$EXISTING_BAZEL_EXECROOT" \
  --output "$PRIVATE_BUILD/minigrid" --family minigrid
```

This helper targets the existing Linux x86_64 optimized Bazel outputs and Python
3.12 ABI used for these runs. It is portable across workspace locations, not a
replacement for the project's cross-platform build. It records source/header,
dependency-file, library and output hashes, plus exact serial compiler/linker
commands. The original `.d` files determine every affected translation unit;
fresh `.d` files must reference the candidate core headers.

## Publication and limits

Publish the two scripts, this README and sanitized JSON summaries. The concise
`results/summary.json`, `results/build-and-test-summary.json`, and
`results/parity/comparison.json` contain no machine paths. Do not upload native
binaries, NPZ rollout evidence, object files, or raw path-containing build logs.

This is Linux CPU regression coverage. It is not a clean all-family build,
cross-platform release validation, a performance measurement, or a MiniGrid
upstream-Python oracle run. The complete existing MiniGrid determinism suite is
candidate-versus-candidate; the separate 59-case comparison checks the original
wheel against the candidate exactly. The upstream MiniGrid package was not
installed, so MiniGrid/BabyAI Python-oracle suites were not run here.
