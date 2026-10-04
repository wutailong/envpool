# Conclusions first; raw runs in Git history

The current branch keeps the [review and conclusions](REVIEW.md), detailed
per-study reports, regression tests, benchmark/reproduction helpers, dependency
notes and the input configurations needed for fresh runs. Production EnvPool
source and native regression tests are unchanged by this cleanup.

## What changed

- Removed 497 generated record paths (6,527,716 bytes, about 6.23 MiB): repeated
  sample JSON/JSONL, build/runtime manifests, stdout logs, sanitizer/native XML,
  and machine-readable summaries already explained in the reports.
- Three of those paths are input specifications, so their unchanged contents
  moved to `container_ownership/inputs/` and `timing_initialization/inputs/`.
  Other existing experiment case lists and `ppo/config.json` remain in place.
- Raw records were not selectively filtered: unsuccessful trials, adverse
  timings, rejected candidates and setup failures remain in the same immutable
  history as positive outcomes. Their conclusions and caveats remain in reports.
- Evidence links point to
  [the complete pre-cleanup tree at a1e56dde](https://github.com/wutailong/envpool/tree/a1e56dde2923c22b28b6bdf7f123ab36d22c4751/benchmark/core_runtime).
  They describe historical runs, not tests performed during this cleanup.
- Reporting unit tests now generate small deterministic fixtures and check
  medians, ranges, CV, grouping, speed ratios, formatting and invalid inputs.
  They no longer require hundreds of historical timing records.

This is an ordinary deletion commit on the existing cumulative branch. It does
not rewrite history or shrink a full clone's existing Git object database.
The current checkout becomes smaller and easier to browse. No new performance,
training, sanitizer, GPU, release or all-family validation is implied.

## Short conclusion map

- The later [NumPy input-owner repair](numpy_input_owner/README.md) is applied
  on top of ce1. It fixes leaked ownership on rejected conversions and keeps
  accepted alias/copy semantics. Its new regression checks and results belong
  to this repair, not to the older a1e56dde raw-record archive.

- The later [generated ActionSlice batch experiment](generated_enqueue/README.md)
  is also unapplied: less allocation and a favorable CartPole signal, but
  adverse HalfCheetah results. Its fresh raw data stays local; it is not part
  of the historical a1e56dde evidence archive.

- The later [direct Python receive-list experiment](recv_list/README.md) is
  unapplied: one allocation saved, no reliable net speed gain, and adverse
  multi-player timings. Its current report contains conclusions and methods;
  its newly generated raw runs are local, not in the older a1e56dde archive.

- At cleanup, the retained runtime was the cumulative **ce1c47f2** core. Completion
  lifetime, shutdown, timing initialization, player discount and Container
  ownership repairs are documented in [REVIEW.md](REVIEW.md).
- Direct tuple construction removes one 576-byte temporary allocation per
  CartPole state. Window-specific vector-step gains do not establish an
  end-to-end PPO speedup; prior adverse timings and A/A noise remain relevant.
- Action metadata, cursor and player-index prototypes remain unapplied. Their
  reports explain the inconclusive or adverse performance findings.
- The ce1 ToyText/MiniGrid comparison at cleanup: 59 cases, 6,575 arrays and 65,304,189 bytes
  match exactly. Seventeen ToyText tests and two MiniGrid determinism tests
  passed, covering 82 MiniGrid-prefixed IDs. BabyAI parity covers only one of
  96 tasks; render parity covers DoorKey. No new XLA/oracle/all-family claim.
- Earlier family/XLA and PPO results retain their original revision attribution.
  Local evidence does not establish a hosted CI pass.

## Replay archived measurements, only when needed

No archive download is needed to read conclusions or run the synthetic helper
tests. To recompute an old summary, export the fixed records to a separate fresh
directory. From a full clone containing the cleanup commit and its parent:

```sh
ARCHIVE_COMMIT=a1e56dde2923c22b28b6bdf7f123ab36d22c4751
ARCHIVE_ROOT=/absolute/path/to/new-envpool-evidence
export ARCHIVE_ROOT
test ! -e "$ARCHIVE_ROOT"
git cat-file -e "$ARCHIVE_COMMIT^{commit}"
mkdir -p "$ARCHIVE_ROOT"
git archive "$ARCHIVE_COMMIT" benchmark/core_runtime | tar -x -C "$ARCHIVE_ROOT"
```

For a shallow clone, retrieve that exact commit first using the repository's
normal authenticated Git workflow. Keep `ARCHIVE_ROOT` outside the source tree.
The relevant report's historical-recomputation commands use that exported root;
write summaries into new output paths. Alternatively, the original historical
helpers and commands remain together under the exported `benchmark/core_runtime`.
Do not mistake recomputing old measurements for measuring a new build.

For fresh measurements, use the retained scripts and fresh output paths. Keep
separate frozen runtimes, verify source/native identity, match dependencies and
preserve unsuccessful runs outside the source tree. The focused
`published_rebuild/build_clients.py` intentionally reads command templates from
its exact, separately checked-out **ce1c47f2** source; its source-identity gate
rejects the current docs/cleanup checkout. Its historical template path therefore
remains valid and must not be redirected to a different source revision.

## Cleanup validation

All 78 lightweight diagnostic/helper unit tests passed after the cleanup,
including nine reporting tests using synthetic data. Ruff checks/formatting and
`git diff --check` passed. All 173 immutable evidence-file links resolve to paths
in the pre-cleanup tree and all 91 local Markdown links resolve in the cleaned
checkout. Production source and native regression-test blobs were unchanged.
Native environment tests, sanitizer builds, full PPO and performance trials were
not rerun for this documentation/artifact-only change. The separate NumPy native
conversion probe and full PPO comparator require their native/run inputs and
are not included in the 78 lightweight tests.
