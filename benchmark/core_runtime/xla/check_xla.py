# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check CPU EnvPool FFI against NumPy stepping and canonical async streams."""

import argparse
import gc
import hashlib
import importlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--envpool-root", type=Path)
parser.add_argument("--env", action="append", dest="envs")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.envpool_root:
    sys.path.insert(0, str(args.envpool_root.resolve()))
import jax
import jax.numpy as jnp
import numpy as np

import envpool

jax.config.update("jax_enable_x64", True)


def flat(tree, prefix=""):
    """Flatten public outputs into named NumPy arrays."""
    if isinstance(tree, dict):
        return {
            p: v
            for k in sorted(tree)
            for p, v in flat(tree[k], prefix + "." + k).items()
        }
    if isinstance(tree, tuple):
        return {
            p: v
            for k, x in enumerate(tree)
            for p, v in flat(x, prefix + "." + str(k)).items()
        }
    return {prefix: np.asarray(tree)}


def action(env, ids, step):
    """Generate actions from environment IDs and local step counters."""
    if hasattr(env.action_space, "n"):
        return ((ids + step) % env.action_space.n).astype(np.int32)
    return (
        np.sin(
            ids[:, None] * 0.7
            + np.asarray(step)[..., None] * 0.3
            + np.arange(env.action_space.shape[0]) * 0.2
        )
        * 0.4
    ).astype(env.action_space.dtype)


def equal(actual, expected, tag):
    """Require identical output keys, dtypes, shapes and bytes."""
    a, b = flat(actual), flat(expected)
    assert a.keys() == b.keys(), tag
    for k in a:
        assert a[k].dtype == b[k].dtype, (tag, k, a[k].dtype, b[k].dtype)
        assert a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes(), (
            tag,
            k,
        )
    return a


def digest(rows):
    """Hash canonical output streams without pointer-valued handles."""
    h = hashlib.sha256()
    for row in rows:
        for k, v in sorted(row.items()):
            h.update(k.encode())
            h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode())
            h.update(v.tobytes())
    return h.hexdigest()


def scan_function(step_fn):
    """Bind one pool's side-effecting step before compiling its scan."""

    def scan(handle, actions):
        return jax.lax.scan(step_fn, handle, actions)

    return jax.jit(scan)


cases = []
for name in args.envs or ["CartPole-v1", "HalfCheetah-v4"]:
    for mode in ["jit_step", "jit_send_recv", "scan"]:
        config = dict(
            num_envs=8,
            batch_size=8,
            num_threads=2,
            seed=65,
            max_episode_steps=37,
        )
        reference = envpool.make_gymnasium(name, **config)
        compiled = envpool.make_gymnasium(name, **config)
        equal(compiled.reset(), reference.reset(), (name, mode, "reset"))
        handle, recv, send, step = compiled.xla()
        fstep, frecv, fsend = jax.jit(step), jax.jit(recv), jax.jit(send)
        actions = np.stack([
            action(reference, np.arange(8), i) for i in range(128)
        ])
        rows = []
        if mode == "scan":
            handle, outputs = scan_function(step)(handle, jnp.asarray(actions))
            jax.block_until_ready(outputs)
            for i, act in enumerate(actions):
                row = jax.tree.map(lambda x, index=i: x[index], outputs)
                rows.append(equal(row, reference.step(act), (name, mode, i)))
        else:
            for i, act in enumerate(actions):
                if mode == "jit_step":
                    handle, outputs = fstep(handle, jnp.asarray(act))
                else:
                    handle = fsend(handle, jnp.asarray(act))
                    handle, outputs = frecv(handle)
                jax.block_until_ready(outputs)
                rows.append(
                    equal(outputs, reference.step(act), (name, mode, i))
                )
        cases.append(
            dict(
                env=name,
                mode=mode,
                config=config,
                calls=128,
                env_responses=1024,
                arrays=sum(map(len, rows)),
                sha256=digest(rows),
            )
        )
        print(name, mode, "bitwise NumPy parity PASS", flush=True)
        del (
            reference,
            compiled,
            outputs,
            rows,
            fstep,
            frecv,
            fsend,
            step,
            recv,
            send,
        )
        jax.clear_caches()
        gc.collect()
    # Async arrival order is intentionally NOT treated as deterministic.
    # Canonicalize the first 128 outputs of each env, with actions based on its local counter.
    config = dict(
        num_envs=32, batch_size=8, num_threads=4, seed=65, max_episode_steps=37
    )
    compiled = envpool.make_gymnasium(name, **config)
    compiled.async_reset()
    handle, recv, send, _ = compiled.xla()
    frecv, fsend = jax.jit(recv), jax.jit(send)
    counts = np.zeros(32, dtype=np.int32)
    streams = [[] for _ in range(32)]
    while True:
        handle, outputs = frecv(handle)
        jax.block_until_ready(outputs)
        data = flat(outputs)
        ids = np.asarray(outputs[-1]["env_id"])
        for j, eid in enumerate(ids):
            if counts[eid] < 128:
                streams[eid].append({k: v[j].copy() for k, v in data.items()})
            counts[eid] += 1
        if counts.min() >= 128:
            break
        handle = fsend(
            handle,
            jnp.asarray(action(compiled, ids, counts[ids])),
            jnp.asarray(ids),
        )
    rows = [row for stream in streams for row in stream]
    cases.append(
        dict(
            env=name,
            mode="async_send_recv",
            config=config,
            env_responses=len(rows),
            arrays=sum(map(len, rows)),
            sha256=digest(rows),
        )
    )
    print(name, "async canonical stream captured", flush=True)
    del compiled, outputs, rows, recv, send, frecv, fsend
    jax.clear_caches()
    gc.collect()
result = dict(
    envpool_module=envpool.__file__,
    jax=jax.__version__,
    jaxlib=importlib.import_module("jaxlib").__version__,
    backend=jax.default_backend(),
    numpy=np.__version__,
    cases=cases,
)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
