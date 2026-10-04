# Current-core ToyText and MiniGrid coverage

The retained public core
[ce1c47f2](https://github.com/wutailong/envpool/commit/ce1c47f238a069454f732a70089d1b9857dcc6de)
now has fresh ToyText and MiniGrid builds and ordinary positive regression
coverage. All **59 original-versus-current cases match exactly: 6,575 arrays,
65,304,189 bytes, zero mismatches**. Seventeen ToyText tests and both MiniGrid
determinism tests also pass. No production code or benchmark algorithm changed.

This closes a version-attribution gap: the older [family report](../families/README.md)
tested first-round 7de9691f. Its evidence stays historical; the records here were
newly generated against the current seven-header cumulative change.

## Build and runtime identity

The source is the same clean public GitHub clone used in the
[focused rebuild check](../published_rebuild/README.md): exact ce1c47f2 commit,
tree `7521d1b10ca2f53d6098d0c8d24667a6c5f59358`. The original control remains
9c31c547 and its installed Python 3.12 wheel. Source and selected native files
remain unchanged after the tests.

The existing [build helper](../families/build_families.py) ran serially with
GCC 14.2.0, C++17 and the recorded release flags:

- ToyText: all one translation unit rebuilt; 45 external archives reused
- MiniGrid: all 17 translation units rebuilt, including both binding objects
  and every one of the 15 library members; 50 external archives reused
- Every unit depends on changed core headers. New dependency files point to
  the public checkout, and final links contain no old EnvPool objects/archives.
- Each MiniGrid archive member matches its freshly compiled object byte-for-byte.
  Reused external-library hashes match their recorded values.

| Client | Original native SHA256 | Current native SHA256 |
| --- | --- | --- |
| ToyText | ba077ec2331a017c5d280628b5b5da0bdf388f854964c5f958622539fe3ac21d | 9e786ac141ac53a437842a44a7973ceac4b022a3520f4fd3d97b536a0a9338b7 |
| MiniGrid | 92a8e4d2c8ff47217f9d7b1b7636c7d661bd587863986215dc89c12c5932bc04 | 0c8e835a2023203fc288c99921584da162580fc4dca20157417ce9246311147d |

The new runtime copies the published Python sources and adds these two freshly
built modules. It retains two previously rebuilt ce1 clients (Classic Control
and Dummy) and 17 original native support modules for eager family imports.
It is a mixed support package; only ToyText/MiniGrid are newly built and tested
in this cycle. Explicit worker import paths and hashes match the frozen runtime.

## Exact coverage

The unchanged [comparison harness](../families/compare_families.py) uses
19 task/configuration variants at each of 16/16/1 sync, 32/32/4 sync, and
32/8/4 async, plus two partial/shuffled-ID cases: 59 cases total. Full streams
record 192 responses per environment with seed 71 and horizon 43; partial
Taxi-v4 and DoorKey cases use seed 91, horizon 31 and 96 steps per subset.
Each runtime records 296,826 environment outputs including resets.

Observation, reward, termination/truncation and nested info arrays compare by
dtype, shape and exact bytes. Async arrivals are normalized by environment ID
and per-ID sequence, not required to arrive in identical order. Previously
returned batches remain checked across subsequent calls and environment closure.
DoorKey render parity covers 12 arrays / 84 selected-environment RGB frames at
four post-step checkpoints across three subsets. It is not all-task render or
reset-frame coverage.

The 17 ToyText tests are the existing suite. The two MiniGrid determinism tests
cover all **82 MiniGrid-prefixed IDs**, comparing candidate versus candidate.
Same-seed runs use complete registered horizons; different-seed checks may stop
after observing a difference. They exclude BabyAI. Cross-runtime parity covers
**one of 96 registered BabyAI IDs**, BabyAI-GoToRedBallGrey-v0; it does not certify
all BabyAI tasks. The upstream MiniGrid package is unavailable here, so no
MiniGrid/BabyAI Python-oracle suite was run.

These are single-player paths: family state writers use the default one-player
allocation, whose discount assignment was unchanged. The Gymnasium comparison
does not expose dm_env discount. No intentional output change was observed, but
these results do not validate the earlier multi-player discount repair.

No new sanitizer, XLA/GPU, full PPO, all-family release, cross-platform or
performance result is claimed. Existing external dependencies are reused rather
than rebuilt from scratch. The machine had roughly 285 MiB free afterward;
further substantial builds need a capacity plan. No original/control data was
deleted to make this test fit.

## Reproduce

Use the pinned public source, the original 9c31 checkout, and the existing
Linux/Python 3.12 optimized native dependency tree. Set `SOURCE`,
`ORIGINAL_SOURCE`, `BAZEL_EXECROOT`, `PYTHON`, and fresh `BUILD`, `RUNTIME`, `OUT`
paths. `ORIGINAL_RUNTIME` contains the original `envpool/` package.
`SUPPORT_RUNTIME` is the frozen runtime from the preceding focused rebuild;
it supplies the other family modules required by eager imports. Apply the
same native include/library environment described in that report. These are
focused cached-dependency commands, not a clean-install recipe.

```sh
set -e
unset PYTHONPATH ENVPOOL_ASSETS_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
test ! -e "$BUILD" && test ! -e "$RUNTIME" && test ! -e "$OUT"
for FAMILY in toy_text minigrid; do
  "$PYTHON" -B "$SOURCE/benchmark/core_runtime/families/build_families.py" --source "$SOURCE" --baseline-source "$ORIGINAL_SOURCE" --bazel-execroot "$BAZEL_EXECROOT" --output "$BUILD/$FAMILY" --family "$FAMILY"
done
"$PYTHON" -B - "$SOURCE" "$SUPPORT_RUNTIME" "$BUILD" "$RUNTIME" <<'PY'
import shutil, sys
from pathlib import Path
source, support, build, runtime = [Path(p).resolve() for p in sys.argv[1:]]
if runtime.exists():
    raise FileExistsError(runtime)
if runtime.is_relative_to(source) or runtime.is_relative_to(support):
    raise ValueError("runtime must be separate from source and support inputs")
shutil.copytree(source / "envpool", runtime / "envpool",
                ignore=shutil.ignore_patterns("__pycache__"))
for native in (support / "envpool").rglob("*.so"):
    relative = native.relative_to(support)
    target = runtime / relative
    family = relative.parts[1]
    if family in ("toy_text", "minigrid"):
        shutil.copy2(build / family / native.name, target)
    else:
        target.symlink_to(native.resolve())
PY
"$PYTHON" -B "$SOURCE/benchmark/core_runtime/families/compare_families.py" --original-runtime "$ORIGINAL_RUNTIME" --candidate-runtime "$RUNTIME" --output "$OUT" --steps 192
PYTHONPATH="$RUNTIME" "$PYTHON" -B "$SOURCE/envpool/toy_text/toy_text_test.py"
PYTHONPATH="$RUNTIME" "$PYTHON" -B "$SOURCE/envpool/minigrid/minigrid_deterministic_test.py"
```

Check emitted native paths/hashes and freeze files before testing. Do not replace
modules during a running process. Both helpers can reuse existing output paths,
so the fresh-path checks above are essential; keep distinct revisions separate.
Use `--compare-only --output "$OUT"` to recheck newly saved NPZ records without
recording more rollouts. Large NPZ arrays are kept locally, not published.

Evidence: [validation](results/validation.json), [plan](results/plan.json),
[compiled closure](results/build-closure.json),
[ToyText build commands](results/build-toy_text.json),
[MiniGrid build commands](results/build-minigrid.json),
[runtime freeze](results/runtime-freeze.json),
[59-case comparison](results/comparison.json),
[original imports/cases](results/original-report.json),
[current imports/cases](results/candidate-report.json),
[registry scope](results/registry-scope.json),
[ToyText tests](results/toy-text-tests.log),
[MiniGrid tests](results/minigrid-deterministic-tests.log),
[parity log](results/parity.log).
Build diagnostic logs retain warnings; duplicated command lines are represented
once in the build JSON. Private roots are normalized. No binaries, assets,
credentials or model artifacts are uploaded.
