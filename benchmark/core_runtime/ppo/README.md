# Deterministic synchronous PPO parity

The `report.json` beside this page is the historical **first-round**
[7de9691f](https://github.com/wutailong/envpool/commit/7de9691f724e5739f73c83d52a0ab57bf166d37b)
comparison against original 9c31c547. Later cumulative revisions reused this
harness and recorded their own baseline/candidate reports, including the
[retained direct-tuple revision](../state_tuple/README.md). Do not relabel the
old report as a new run. The separate [32-run PPO phase study](../ppo_phase/README.md)
compares original against retained runtime using fingerprints; it is not a new
full checkpoint/step-journal comparison.

The original and optimized native runtimes match exactly in one CPU CartPole-v1
PPO experiment, with identical initialization, seeds, hyperparameters and budgets.

- 256,000 training transitions, 100 collection/update rounds, 8,000 Adam steps
- 969 completed training episodes
- Identical actions, observations, rewards, termination/truncation/info and reset outputs
- Identical processed batches, advantages/returns/values and 32,000 loss values
- All 101 model/optimizer/normalization/Python/NumPy/Torch RNG checkpoints match
- 223 arrays and 4,925 tensors compared with zero numeric difference
- Both held-out evaluations: 100/100 episodes at return 500; full traces match
- Negative controls detect metric drift and one-ULP observation/optimizer changes

This is a matched synchronous experiment, not proof for every PPO setup, async
collection order, another platform, or training throughput. Other verification
work ran concurrently, so instrumented timings are not performance evidence.

## Reproduce

Set `PYTHON` to the Python 3.12 interpreter sharing the RL dependencies.
Use the recorded dependencies in `requirements.txt`. This is an
observed environment inventory, not a complete cross-platform lockfile or an
independently validated clean-install recipe. To install the recorded Linux CPU
Torch variant, use the [official PyTorch CPU wheel index](https://pytorch.org/get-started/previous-versions/#v251):

```sh
"$PYTHON" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Check that `torch.__version__` is `2.5.1+cpu` in the resulting Linux environment,
and match the remaining recorded dependencies before comparing runs. This
documentation audit did not install packages or validate a fresh resolver run.
Build the
original and candidate EnvPool packages separately. Set `ORIGINAL_PACKAGE` and
`CANDIDATE_PACKAGE` to their `envpool` package directories (not site-packages):

```sh
"$PYTHON" benchmark/core_runtime/ppo/verify_ppo_parity.py run \
  --original "$ORIGINAL_PACKAGE" \
  --candidate "$CANDIDATE_PACKAGE" \
  --config benchmark/core_runtime/ppo/config.json \
  --output "$OUTPUT_DIR"

"$PYTHON" benchmark/core_runtime/ppo/test_comparator.py \
  --results "$OUTPUT_DIR"
```

The harness refuses to overwrite existing original/candidate run directories.
It verifies distinct native imports, requires deterministic single-threaded CPU
Torch, selects EnvPool without replacing shared Python dependencies, and records
binary/source hashes. It collects fixed 2,560-transition batches with 20 slots,
runs two passes of 64-example minibatches, and repeats 100 times. There is no
score-triggered stop, episode quota or model selection.

Outputs include complete rollout/processed-batch NPZ arrays, model and optimizer
checkpoints, every metric, import/version provenance, and an exact comparator
report. Comparisons use scalar values, tensor dtype/shape and contiguous bytes;
wall times, runtime paths and archive serialization bytes are excluded.

`report.json` is the sanitized numeric result. Large local checkpoints and trace
arrays are intentionally excluded from Git. Rerunning the harness recreates them.
