# Fresh public-source client rebuild

A new clone from the public fork at
[ce1c47f2](https://github.com/wutailong/envpool/commit/ce1c47f238a069454f732a70089d1b9857dcc6de)
successfully rebuilt Classic Control and Dummy, passed six existing positive
Python/API/render methods, and matched the retained runtime's short PPO semantic
fingerprint. This verifies a focused reproduction using existing dependencies.
There is no new production change, full clean-release build, or speed claim.

## What was independently rebuilt

The clean GitHub clone has exact commit
`ce1c47f238a069454f732a70089d1b9857dcc6de` and tree
`7521d1b10ca2f53d6098d0c8d24667a6c5f59358`. It stayed clean after all checks.
The [helper](build_clients.py) intentionally requires this pinned revision.
It runs GCC 14.2.0 serially with the recorded C++17/release flags and recompiles:

- `envpool/classic_control/classic_control.cc`
- `envpool/classic_control/render_utils.cc`
- `envpool/dummy/dummy_envpool.cc`

The earlier recorded recipe reused `libclassic_control_env.pic.a`, whose only
member is the renderer. Its source/header matched this revision and did not
include core headers; that reuse was not evidence of stale core. This fresh
exercise nevertheless rebuilds it so every EnvPool object in the two links is
from the new checkout. The inherited Dummy link also lists this archive, even
though its BUILD target does not need it; the fresh archive replaces it there
without otherwise redesigning the link recipe.

Compiler dependency files contain 611 entries for Classic Control, 606 for
Dummy and 394 for render_utils,
deduplicating to 693 files, including 27 project source/header files. Every
project input resolves inside the new checkout. Both link maps list the new
archive as a link input; only Classic extracts its renderer member. Neither map
references an old EnvPool archive. Fifty existing external archives
(four OpenCV, one zlib, 45 Abseil) are reused and fingerprinted before/after.
External headers, system/Python libraries and the toolchain are also reused.
This does not establish a clean build of that dependency stack.

The runtime contains all 181 Python files copied byte-for-byte from the public
checkout, the two fresh extensions, and 19 read-only original native support
modules. EnvPool's entry module eagerly imports other families, so retaining
those support modules avoids modifying published Python sources. This is a
**mixed support package with two freshly rebuilt clients**, not an entirely
fresh all-family installation. Actual Classic/Dummy imports and all staged
Python/native fingerprints are checked before/after the positive methods.

Fresh Classic Control SHA256:
`666da472b20cad0ca9f26fdfffd01339a624a4903b4e6058a80e8d10f18493ef`.
Fresh Dummy SHA256:
`98574e44ffba32bcc68ad25c92c33e1a732869f127ed234904e710d965363f7e`.
Different source paths/build inputs can change bytes; matching a prior binary's
hash is not required. Record your own hashes and verify their actual use.

## Checks that ran

- 14 synthetic build-plan/provenance unit tests; Ruff check/format pass
- Six existing positive test methods: CartPole v0/v1 same/different-seed loops
  with 5,000 steps each, Gymnasium vector-wrapper behavior, batch-consistent
  rendering across five Classic Control tasks, and Dummy configuration, spec
  and environment-seed override checks
- Two fresh PPO processes, one per fresh/retained client: two discarded priming
  updates followed by two checked updates, 5,120 responses and 160 optimizer
  steps each. Configuration, budget and semantic fingerprint agree exactly:
  `98e7735aabc2de0b5d7e2fd54695646e56deb39d5588b0cea1a2a58f717ab55e`.

The PPO comparison uses the existing retained Classic extension
`accfaebe22982718d595f45a2c7c4ba67515e22f4d8084bf656cd9768c1b164c`, also built
from ce1c47f2 production code. It checks reproduction of that build's behavior,
not a new comparison against original main. Two-update fingerprints do not
replace the earlier full-budget checkpoint/step-journal parity tests. Any elapsed
times in the smoke JSON are incidental and must not be interpreted as performance.

This cycle did not rerun native sanitizer, XLA, MuJoCo, Box2D, full-budget PPO,
other platforms or all-family release tests. The earlier evidence retains its
original scope and binary identities. Inherited OpenXLA enum-stream functions
still emit three `-Wreturn-type` warnings per binding compile; those logs are
preserved. This is not a warning-free build or a hosted-CI pass.

## Reproduce with existing native dependencies

Set absolute paths first: `TOOLS` is this directory in the report/tool checkout;
`SOURCE` is a new clone directory; `BUILD`, `RUNTIME` and the JSON outputs must
not exist. `BAZEL_EXECROOT` is an existing Linux x86_64/Python 3.12 optimized
Bazel dependency tree with the recorded layout, including OpenCV 5 headers.
`ORIGINAL_PACKAGE` and `RETAINED_PACKAGE` are actual `envpool/` directories.
`PYTHON` supplies the general Python dependencies; `PPO_PYTHON` supplies the
separate [recorded CPU RL environment](../ppo/README.md).

Supply any native include/library environment required by your existing build.
The recorded local setup used `CPATH`, `LIBRARY_PATH` and `LD_LIBRARY_PATH` for
the extracted native dependencies. The helper's reference to `deps/env.sh` is
that local prerequisite, not a file supplied by this repository. The helper
does not download dependencies, install packages or configure Bazel. A cached
dependency tree is required; this is not a standalone clean-install recipe.

```sh
git clone --depth 1 --branch perf/core-state-tuple https://github.com/wutailong/envpool.git "$SOURCE"
git -C "$SOURCE" switch --detach ce1c47f238a069454f732a70089d1b9857dcc6de
unset PYTHONPATH ENVPOOL_ASSETS_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$PYTHON" -B "$TOOLS/build_clients.py" --source "$SOURCE" --bazel-execroot "$BAZEL_EXECROOT" --output "$BUILD"
"$PYTHON" -B "$TOOLS/stage_runtime.py" --source "$SOURCE" --original-package "$ORIGINAL_PACKAGE" --classic-binary "$BUILD/build/classic/classic_control_envpool.so" --dummy-binary "$BUILD/build/candidate-dummy/dummy_envpool.so" --output "$RUNTIME"
"$PYTHON" -B "$TOOLS/positive_smoke.py" --runtime "$RUNTIME" --output fresh-positive-smoke.json
"$PPO_PYTHON" -B "$SOURCE/benchmark/core_runtime/experiments/bench_ppo.py" --package "$RUNTIME/envpool" --label fresh --iterations 2 --warmup 2 --output fresh-ppo.json
"$PPO_PYTHON" -B "$SOURCE/benchmark/core_runtime/experiments/bench_ppo.py" --package "$RETAINED_PACKAGE" --label retained --iterations 2 --warmup 2 --output retained-ppo.json
"$PYTHON" -B -c 'import json; a=json.load(open("fresh-ppo.json")); b=json.load(open("retained-ppo.json")); assert a["config"]==b["config"] and a["budget"]==b["budget"] and a["semantic_sha256"]==b["semantic_sha256"]; print("PPO smoke fingerprints match")'
```

Run outside either runtime/source package and keep files frozen throughout.
For stricter independent native identity checks, pass each freshly recorded
SHA256 to the PPO driver's `--native-sha256`; the executed commands did so.
Do not benchmark concurrently with compilation or tests.

## Evidence

[Validation scope](results/validation.json),
[exact commands/toolchain/project dependencies/link-map audit](results/build-provenance.json),
[deduplicated header fingerprints](results/header-inputs.json),
[post-smoke source/header verification](results/final-source-verification.json),
[mixed runtime and Python manifest](results/runtime-manifest.json),
[positive methods](results/positive-smoke.json),
[PPO comparison](results/ppo-smoke-comparison.json),
[fresh PPO record](results/ppo-fresh.json),
[retained PPO record](results/ppo-retained.json),
[helper tests](results/helper-tests.log),
[positive test log](results/positive-smoke.log),
[Classic warnings](results/compile-classic.log),
[Dummy warnings](results/compile-dummy.log).

Machine roots are normalized placeholders. No binaries, checkpoints, assets or
credentials are uploaded. Existing branch controls and native files are unchanged.
