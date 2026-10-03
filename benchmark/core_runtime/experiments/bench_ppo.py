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
"""Time real synchronous CartPole collection plus Tianshou PPO optimization.

Run each runtime in a fresh process, with dependencies from ../ppo/requirements.txt.
The default is the reference 100 updates, 256,000 transitions, 8,000 Adam steps.
--iterations changes the measured update count; --warmup changes untimed updates.
Warmup uses a disposable policy, optimizer, collector and pool. All are recreated
after reseeding Python, NumPy and Torch before the fixed-budget measured run.
Framework/JIT caches remain warm. Initial setup/reset, hashing, validation and JSON
writing are outside timing. No rollout journals or checkpoints are produced.

Only one whole-training wall-clock interval is timed. Normal collect/update result
objects are retained once per update for post-timing fingerprints. These catch
observed semantic drift but do not replace ../ppo/verify_ppo_parity.py's complete
step-level equivalence test. Run serialized, interleaved baseline/candidate trials
on the same otherwise-idle CPUs; a single timing is not a speedup conclusion.
Reports contain hashes and verified import flags, never supplied filesystem paths.
"""

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import re
import struct
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "ppo/config.json"
NATIVE_FILE = Path("classic_control/classic_control_envpool.so")
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def positive_int(value):
    """Parse a nonzero positive integer for fixed work budgets."""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a positive integer"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def digest_arg(value):
    """Validate an optional independently recorded native binary digest."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("expected a 64-digit SHA-256 digest")
    return value.lower()


def label_arg(value):
    """Allow short sample labels, preventing accidental path disclosure."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise argparse.ArgumentTypeError(
            "use a short alphanumeric sample label"
        )
    return value


def parse_args(argv=None):
    """Parse options without importing any training dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="EnvPool package directory containing __init__.py",
    )
    parser.add_argument("--label", type=label_arg, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON file; an existing file is never overwritten",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=100,
        help="Measured collection/update rounds (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=2,
        help="Disposable untimed collection/update rounds (default: 2)",
    )
    parser.add_argument(
        "--native-sha256",
        type=digest_arg,
        help="Fail if the selected native binary does not match",
    )
    return parser.parse_args(argv)


def load_config(path, iterations):
    """Reuse the parity configuration and require the intended workload shape."""
    cfg = json.loads(Path(path).read_text())
    required = {
        "task",
        "seed",
        "train_seed",
        "baseline_seed",
        "eval_seed",
        "hidden_sizes",
        "lr",
        "gamma",
        "training_num",
        "eval_num",
        "num_threads",
        "torch_threads",
        "torch_interop_threads",
        "updates",
        "steps_per_collect",
        "repeat_per_collect",
        "batch_size",
        "vf_coef",
        "ent_coef",
        "eps_clip",
        "max_grad_norm",
        "gae_lambda",
        "reward_normalization",
        "value_clip",
        "advantage_normalization",
        "recompute_advantage",
        "max_episode_steps",
        "device",
    }
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError("config must have exactly the reference PPO fields")
    fixed = {
        "task": "CartPole-v1",
        "device": "cpu",
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "training_num": 20,
        "steps_per_collect": 2560,
        "repeat_per_collect": 2,
        "batch_size": 64,
        "hidden_sizes": [64, 64],
        "max_episode_steps": 500,
    }
    for key, value in fixed.items():
        if type(cfg[key]) is not type(value) or cfg[key] != value:
            raise ValueError(f"config requires {key}={value!r}")
    for key in ("seed", "train_seed", "baseline_seed", "eval_seed"):
        if type(cfg[key]) is not int or not 0 <= cfg[key] < 2**31:
            raise ValueError(f"config requires a seed in [0, 2**31): {key}")
    for key in ("num_threads", "eval_num", "updates"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"config requires a positive integer: {key}")
    for key in (
        "reward_normalization",
        "value_clip",
        "advantage_normalization",
        "recompute_advantage",
    ):
        if type(cfg[key]) is not bool:
            raise ValueError(f"config requires a boolean: {key}")
    for key in (
        "lr",
        "gamma",
        "vf_coef",
        "ent_coef",
        "eps_clip",
        "max_grad_norm",
        "gae_lambda",
    ):
        value = cfg[key]
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"config requires a finite nonnegative number: {key}"
            )
    if not (
        0 < cfg["lr"]
        and 0 < cfg["max_grad_norm"]
        and cfg["gamma"] <= 1
        and cfg["gae_lambda"] <= 1
    ):
        raise ValueError(
            "invalid learning rate, gradient limit, gamma or lambda"
        )
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    cfg["updates"] = iterations
    return cfg


def sha256(path):
    """Hash a file outside the measured interval."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value, np=None, torch=None):
    """Hash typed, finite values and exact array bytes with explicit framing.

    Dictionary insertion order and file serialization are irrelevant. Dtype,
    shape, signed zero, sequence type and scalar type remain significant.
    Scientific dependencies are passed explicitly so helper tests stay light.
    """
    digest = hashlib.sha256()

    def token(tag, payload=b""):
        digest.update(tag + struct.pack("!Q", len(payload)) + payload)

    def array(tag, item):
        item = np.asarray(item)
        if item.dtype.kind not in "buifc":
            raise TypeError("fingerprints require numeric arrays")
        if item.dtype.kind in "fc" and not np.isfinite(item).all():
            raise ValueError("nonfinite array in semantic fingerprint")
        token(tag)
        visit(item.dtype.str)
        visit(item.shape)
        token(b"data", item.tobytes(order="C"))

    def visit(item):
        if torch is not None and isinstance(item, torch.Tensor):
            array(b"tensor", item.detach().cpu().numpy())
        elif np is not None and isinstance(item, (np.ndarray, np.generic)):
            array(b"array" if isinstance(item, np.ndarray) else b"scalar", item)
        elif item is None:
            token(b"none")
        elif type(item) is bool:
            token(b"bool", bytes([item]))
        elif type(item) is int:
            token(b"int", str(item).encode())
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite scalar in semantic fingerprint")
            token(b"float", struct.pack("!d", item))
        elif isinstance(item, str):
            token(b"str", item.encode())
        elif isinstance(item, bytes):
            token(b"bytes", item)
        elif isinstance(item, dict):
            if any(type(key) not in (str, int) for key in item):
                raise TypeError(
                    "fingerprint mapping keys must be strings or ints"
                )
            token(b"dict", str(len(item)).encode())
            for key in sorted(
                item, key=lambda key: (type(key).__name__, str(key))
            ):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            token(
                b"list" if isinstance(item, list) else b"tuple",
                str(len(item)).encode(),
            )
            for child in item:
                visit(child)
        else:
            raise TypeError(
                f"unsupported fingerprint type: {type(item).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def write_result(path, result):
    """Write one finite, sanitized JSON result without replacing earlier evidence."""
    encoded = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with Path(path).open("x") as stream:
        stream.write(encoded)


def load_package(package, expected_sha256=None):
    """Select EnvPool alone and verify both Python and native module origins."""
    package = Path(package).expanduser().resolve()
    init = package / "__init__.py"
    native_path = (package / NATIVE_FILE).resolve()
    if not init.is_file() or not native_path.is_file():
        raise ValueError(
            "--package must contain EnvPool and its classic-control binary"
        )
    if any(
        name == "envpool" or name.startswith("envpool.") for name in sys.modules
    ):
        raise RuntimeError("EnvPool is already imported; use a fresh process")
    binary_hash = sha256(native_path)
    if expected_sha256 is not None and binary_hash != expected_sha256:
        raise ValueError(
            "selected native binary does not match --native-sha256"
        )
    spec = importlib.util.spec_from_file_location(
        "envpool",
        init,
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["envpool"] = module
    spec.loader.exec_module(module)
    native = importlib.import_module(
        "envpool.classic_control.classic_control_envpool"
    )
    if Path(module.__file__).resolve() != init.resolve():
        raise RuntimeError(
            "EnvPool Python import did not use the selected package"
        )
    if Path(native.__file__).resolve() != native_path:
        raise RuntimeError("native import did not use the selected package")
    return module, native_path, binary_hash


def train(args):
    """Prime a disposable training run, then measure fresh fixed-budget training."""
    cfg = load_config(args.config, args.iterations)
    if args.output.exists():
        raise FileExistsError("output already exists; choose a new result file")
    if not args.output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    if any(
        name in sys.modules for name in ("torch", "numpy", "numba", "tianshou")
    ):
        raise RuntimeError(
            "training dependencies are already imported; use a fresh process"
        )
    for key in THREAD_ENV:
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    with tempfile.TemporaryDirectory(prefix="envpool-ppo-numba-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        return _train(args, cfg)


def _train(args, cfg):
    envpool, native_path, binary_hash = load_package(
        args.package, args.native_sha256
    )
    import gymnasium as gym
    import numba
    import numpy as np
    import tianshou
    import torch
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.policy import PPOPolicy
    from tianshou.utils.net.common import ActorCritic, Net
    from tianshou.utils.net.discrete import Actor, Critic

    if tianshou.__version__ != "0.5.1":
        raise RuntimeError(
            "use the reference dependencies: Tianshou 0.5.1 required"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    def fresh_run():
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        net = Net((4,), hidden_sizes=cfg["hidden_sizes"], device="cpu")
        actor = Actor(net, 2, device="cpu")
        critic = Critic(net, device="cpu")
        actor_critic = ActorCritic(actor, critic)
        for layer in actor_critic.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=cfg["lr"])
        policy = PPOPolicy(
            actor,
            critic,
            optim,
            torch.distributions.Categorical,
            discount_factor=cfg["gamma"],
            max_grad_norm=cfg["max_grad_norm"],
            eps_clip=cfg["eps_clip"],
            vf_coef=cfg["vf_coef"],
            ent_coef=cfg["ent_coef"],
            gae_lambda=cfg["gae_lambda"],
            reward_normalization=cfg["reward_normalization"],
            dual_clip=None,
            value_clip=cfg["value_clip"],
            action_space=gym.spaces.Discrete(2),
            advantage_normalization=cfg["advantage_normalization"],
            recompute_advantage=cfg["recompute_advantage"],
            deterministic_eval=True,
        )
        env = envpool.make_gymnasium(
            cfg["task"],
            num_envs=cfg["training_num"],
            batch_size=cfg["training_num"],
            num_threads=cfg["num_threads"],
            seed=cfg["train_seed"],
            max_episode_steps=cfg["max_episode_steps"],
        )
        if env.is_async:
            env.close()
            raise RuntimeError(
                "only synchronous EnvPool collection is supported"
            )
        buffer = VectorReplayBuffer(
            cfg["steps_per_collect"], cfg["training_num"]
        )
        collector = Collector(policy, env, buffer)
        return policy, optim, env, buffer, collector

    def updates(policy, buffer, collector, count):
        rows = []
        for _ in range(count):
            policy.train()
            collected = collector.collect(n_step=cfg["steps_per_collect"])
            buffered = len(buffer)
            losses = policy.update(
                0,
                buffer,
                batch_size=cfg["batch_size"],
                repeat=cfg["repeat_per_collect"],
            )
            collector.reset_buffer(keep_statistics=True)
            rows.append((collected, losses, buffered))
        return rows

    def state_fingerprints(policy, optim):
        values = {
            "policy": policy.state_dict(),
            "optimizer": optim.state_dict(),
            "return_rms": vars(policy.ret_rms),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        return {
            key: fingerprint(value, np, torch) for key, value in values.items()
        }

    policy, optim, env, buffer, collector = fresh_run()
    try:
        updates(policy, buffer, collector, args.warmup)
    finally:
        env.close()
    del policy, optim, env, buffer, collector
    gc.collect()

    policy, optim, env, buffer, collector = fresh_run()
    try:
        initial = state_fingerprints(policy, optim)
        gc.collect()
        cpu_started = time.process_time()
        started = time.perf_counter()
        rows = updates(policy, buffer, collector, cfg["updates"])
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        # All validation, hashing, formatting and filesystem writes are untimed.
        steps_per_update = (
            cfg["repeat_per_collect"]
            * cfg["steps_per_collect"]
            // cfg["batch_size"]
        )
        optimizer_steps = steps_per_update * cfg["updates"]
        parameter_ids = {
            item for group in optim.param_groups for item in group["params"]
        }
        if set(optim.state) != parameter_ids or any(
            state["step"].item() != optimizer_steps
            for state in optim.state.values()
        ):
            raise RuntimeError(
                "Adam did not take the expected steps for every parameter"
            )
        for collected, losses, buffered in rows:
            if (
                collected["n/st"] != cfg["steps_per_collect"]
                or buffered != cfg["steps_per_collect"]
            ):
                raise RuntimeError(
                    "collection did not satisfy the fixed transition budget"
                )
            if set(losses) != {
                "loss",
                "loss/clip",
                "loss/vf",
                "loss/ent",
            } or any(
                len(values) != steps_per_update for values in losses.values()
            ):
                raise RuntimeError(
                    "PPO did not return the expected minibatch metrics"
                )
        env_steps = cfg["updates"] * cfg["steps_per_collect"]
        if collector.collect_step != env_steps:
            raise RuntimeError(
                "collector total differs from the fixed transition budget"
            )
        semantics = {
            "config": fingerprint(cfg),
            "initial": initial,
            "final": state_fingerprints(policy, optim),
            "metrics": fingerprint(rows, np, torch),
        }
        if sha256(native_path) != binary_hash:
            raise RuntimeError("native binary changed during training")
        result = {
            "schema_version": 1,
            "label": args.label,
            "config": cfg,
            "budget": {
                "updates": cfg["updates"],
                "env_steps": env_steps,
                "optimizer_steps": optimizer_steps,
                "train_episodes": int(collector.collect_episode),
            },
            "timing": {
                "training_seconds": elapsed,
                "process_cpu_seconds": cpu_elapsed,
                "env_steps_per_second": env_steps / elapsed,
                "optimizer_steps_per_second": optimizer_steps / elapsed,
            },
            "measurement": {
                "warmup_updates": args.warmup,
                "fresh_seeded_run_after_warmup": True,
                "timed_region": "serial policy.train, collect, PPO update, buffer reset; one result reference per update",
                "excluded": "imports, JIT priming, new policy/pool, initial reset, semantic checks, fingerprints, JSON output",
                "evaluation": "not performed; fixed work budget, no score-triggered stop",
            },
            "semantic_fingerprints": semantics,
            "semantic_sha256": fingerprint(semantics),
            "last_update": {
                "mean_train_return": float(rows[-1][0]["rew"]),
                "mean_losses": {
                    key: float(np.mean(value))
                    for key, value in rows[-1][1].items()
                },
            },
            "provenance": {
                "native_sha256": binary_hash,
                "native_module": "envpool.classic_control.classic_control_envpool",
                "package_path_verified": True,
                "native_path_verified": True,
                "expected_native_sha256_verified": args.native_sha256
                is not None,
                "script_sha256": sha256(__file__),
                "reference_config_sha256": sha256(args.config),
                "python_version": sys.version.split()[0],
                "versions": {
                    "envpool": envpool.__version__,
                    "torch": torch.__version__,
                    "tianshou": tianshou.__version__,
                    "numpy": np.__version__,
                    "gymnasium": gym.__version__,
                    "numba": numba.__version__,
                },
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
                "thread_environment": {
                    key: os.environ[key] for key in THREAD_ENV
                },
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
            },
        }
    finally:
        env.close()
    write_result(args.output, result)
    return result


def main():
    """Emit the same sanitized evidence to a new JSON file and standard output."""
    result = train(parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
