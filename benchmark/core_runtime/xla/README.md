# CPU XLA / JAX correctness

Tested JAX and jaxlib 0.11.1 on CPU with x64 enabled. Three environments cover
small numeric states (CartPole), floating-point dynamics (HalfCheetah), and
image/dictionary states (MiniGrid-Empty-5x5).

For each environment, the harness checks 128 calls through jitted step,
separately jitted send/recv, and a compiled lax.scan against ordinary NumPy
stepping, including short episodes and autoresets. It also records the first
128 outputs per environment in a 32-env / 8-batch / 4-worker asynchronous pool.
Async streams are canonicalized by environment ID and local sequence number;
arrival order is explicitly not required to match.

Run the same command in original and candidate installations, or pass
`--envpool-root` pointing to the directory containing the desired `envpool`
package. Use identical Python/JAX dependencies:

```sh
python benchmark/core_runtime/xla/check_xla.py \
  --env CartPole-v1 --env HalfCheetah-v4 --env MiniGrid-Empty-5x5-v0 \
  --output original-xla.json

python benchmark/core_runtime/xla/check_xla.py \
  --envpool-root "$CANDIDATE_PACKAGE_PARENT" \
  --env CartPole-v1 --env HalfCheetah-v4 --env MiniGrid-Empty-5x5-v0 \
  --output candidate-xla.json
```

Both JSON `cases` arrays must match exactly. Synchronous modes additionally
perform byte equality checks within the script against NumPy stepping. The
sanitized `comparison.json` records all cases, hashes and package versions.

This validates CPU FFI only. CUDA/GPU XLA, cross-platform behavior, and all
registered environment families are not claimed.
