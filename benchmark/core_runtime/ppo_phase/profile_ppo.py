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
"""Observe the reference PPO phases; never interpret this timing as a speedup.

Arguments and training semantics are those of ../experiments/bench_ppo.py.
Disposable warmup, initial reset and validation remain outside measurement.
EnvPool call timing includes Python adaptation, conversion and native waiting;
it is not pure C++ time. Run the untouched harness separately for plain timing.
"""

import gc
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from phase_clock import PhaseClock, ProfiledEnv

REFERENCE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments/bench_ppo.py"
)
_spec = importlib.util.spec_from_file_location(
    "_ppo_phase_reference", REFERENCE_SCRIPT
)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)

# Exact stdlib-only helpers and parser from the untouched reference harness.
THREAD_ENV = _reference.THREAD_ENV
parse_args = _reference.parse_args
load_config = _reference.load_config
load_package = _reference.load_package
fingerprint = _reference.fingerprint
sha256 = _reference.sha256
write_result = _reference.write_result


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
    observer = PhaseClock()
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
        env = ProfiledEnv(env, observer)
        buffer = VectorReplayBuffer(
            cfg["steps_per_collect"], cfg["training_num"]
        )
        collector = Collector(policy, env, buffer)
        return policy, optim, env, buffer, collector

    def updates(policy, buffer, collector, count):
        rows = []
        for _ in range(count):
            observer.call("policy_train", policy.train)
            collected = observer.call(
                "collect", collector.collect, n_step=cfg["steps_per_collect"]
            )
            buffered = len(buffer)
            losses = observer.call(
                "ppo_update",
                policy.update,
                0,
                buffer,
                batch_size=cfg["batch_size"],
                repeat=cfg["repeat_per_collect"],
            )
            observer.call(
                "buffer_reset", collector.reset_buffer, keep_statistics=True
            )
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
        observer.active = True
        try:
            rows = updates(policy, buffer, collector, cfg["updates"])
        finally:
            observer.active = False
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
        phase_profile = observer.report(
            elapsed, cfg["updates"], env_steps // cfg["training_num"]
        )
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
            "phase_profile": phase_profile,
            "instrumentation": {
                "schema_version": 1,
                "kind": "observation-only Python phase/call wall-clock profiling",
                "clock": "time.perf_counter_ns",
                "clock_resolution_seconds": time.get_clock_info(
                    "perf_counter"
                ).resolution,
                "active_region": "same measured updates as reference; no setup/initial reset or disposable warmup",
                "env_call_boundary": "Python EnvPool step/reset including adapter, conversion, and native wait; not pure C++",
                "overhead": "present in all measured totals, including some stack/call overhead within env timings; compare to separate untouched bench_ppo.py plain runs",
                "residual_interpretation": "collect excluding env calls is remaining collection wall time, not pure Python or non-core time; background workers may affect other phases",
                "semantic_hashes_include_phase_data": False,
                "saves_weights_or_rollout_arrays": False,
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
                "reference_script_sha256": sha256(REFERENCE_SCRIPT),
                "phase_helper_sha256": sha256(
                    Path(__file__).with_name("phase_clock.py")
                ),
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
