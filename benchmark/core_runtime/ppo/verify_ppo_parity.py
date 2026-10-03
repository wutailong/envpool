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
"""Independent-process, fixed-budget synchronous CartPole PPO comparison.

Uses the same Tianshou PPO implementation/network as the reference trainer.
Does not modify either supplied EnvPool package. Checkpoint contents, arrays,
and metrics are compared by exact values and array bytes, not archive bytes.
No claim about asynchronous scheduling or general PPO equivalence is implied.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def sha256(path):
    """Return the SHA-256 of a local artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write stable, finite JSON evidence."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_package(package):
    """Select only EnvPool; do not put another venv's dependencies on sys.path."""
    package = Path(package).resolve()
    assert "envpool" not in sys.modules
    spec = importlib.util.spec_from_file_location(
        "envpool",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["envpool"] = module
    spec.loader.exec_module(module)
    return module


def train(args):
    """Run one instrumented fixed-budget PPO process."""
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    os.environ["NUMBA_CACHE_DIR"] = str(out / "numba-cache")
    envpool = load_package(args.package)
    import envpool.classic_control.classic_control_envpool as native
    import gymnasium as gym
    import numpy as np
    import tianshou
    import torch
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.policy import PPOPolicy
    from tianshou.utils.net.common import ActorCritic, Net
    from tianshou.utils.net.discrete import Actor, Critic

    cfg = json.loads(args.config.read_text())
    assert cfg["device"] == "cpu"
    assert cfg["steps_per_collect"] % cfg["training_num"] == 0
    assert cfg["steps_per_collect"] % cfg["batch_size"] == 0
    torch.set_num_threads(cfg["torch_threads"])
    torch.set_num_interop_threads(cfg["torch_interop_threads"])
    torch.use_deterministic_algorithms(True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    write_json(out / "config.json", cfg)
    native_path = Path(native.__file__).resolve()
    assert (
        native_path
        == args.package.resolve() / "classic_control/classic_control_envpool.so"
    )
    provenance = {
        "python": sys.executable,
        "envpool_package": envpool.__file__,
        "native_binary": str(native_path),
        "native_sha256": sha256(native_path),
        "versions": {
            "envpool": envpool.__version__,
            "torch": torch.__version__,
            "tianshou": tianshou.__version__,
            "numpy": np.__version__,
            "gymnasium": gym.__version__,
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
        "script_sha256": sha256(__file__),
    }
    write_json(out / "provenance.json", provenance)
    print(json.dumps(provenance), flush=True)

    class Journal:
        def __init__(self):
            self.fields = {}
            self.events = []
            self.counts = {}

        def record(self, name, values):
            index = self.counts.get(name, 0)
            self.counts[name] = index + 1
            self.events.append((name, index))

            def visit(prefix, value):
                if isinstance(value, dict):
                    for key, child in sorted(value.items()):
                        visit(prefix + "/" + key, child)
                else:
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    value = np.asarray(value)
                    if value.ndim == 0:
                        value = value.reshape(1)
                    assert value.dtype != object, prefix
                    self.fields.setdefault(prefix, []).append(value.copy())

            visit(name, values)

        def save(self, path):
            arrays = {}
            for key, values in sorted(self.fields.items()):
                arrays[key] = np.concatenate(values, axis=0)
                arrays[key + "__offsets"] = np.cumsum(
                    [0] + [len(value) for value in values], dtype=np.int64
                )
            # Full event ordering covers interleaved subset resets and steps.
            names = sorted(self.counts)
            arrays["event_kind"] = np.array(
                [names.index(n) for n, _ in self.events], dtype=np.int32
            )
            arrays["event_index"] = np.array(
                [i for _, i in self.events], dtype=np.int64
            )
            np.savez_compressed(path, **arrays)
            return {
                "events": self.counts,
                "event_kind_names": names,
                "arrays": len(arrays),
                "elements": sum(a.size for a in arrays.values()),
            }

    class RecordedEnv:
        def __init__(self, env, journal):
            self.env = env
            self.journal = journal
            assert not env.is_async

        def __len__(self):
            return len(self.env)

        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self, env_id=None, **kwargs):
            result = self.env.reset(env_id, **kwargs)
            obs, info = result
            ids = (
                np.arange(len(self.env), dtype=np.int32)
                if env_id is None
                else env_id
            )
            self.journal.record(
                "reset", {"requested_ids": ids, "obs": obs, "info": info}
            )
            return result

        def step(self, action, env_id=None):
            result = self.env.step(action, env_id)
            obs, rew, terminated, truncated, info = result
            ids = (
                np.arange(len(self.env), dtype=np.int32)
                if env_id is None
                else env_id
            )
            self.journal.record(
                "step",
                {
                    "requested_ids": ids,
                    "action": action,
                    "obs_next": obs,
                    "rew": rew,
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": terminated | truncated,
                    "info": info,
                },
            )
            return result

    processed_journal = Journal()

    class RecordedPPO(PPOPolicy):
        def process_fn(self, batch, buffer, indices):
            batch = super().process_fn(batch, buffer, indices)
            fields = {
                name: batch[name]
                for name in (
                    "obs",
                    "obs_next",
                    "act",
                    "rew",
                    "terminated",
                    "truncated",
                    "done",
                    "adv",
                    "returns",
                    "v_s",
                    "logp_old",
                )
            }
            fields["indices"] = indices
            processed_journal.record("update", fields)
            return batch

    net = Net((4,), hidden_sizes=cfg["hidden_sizes"], device="cpu")
    actor = Actor(net, 2, device="cpu")
    critic = Critic(net, device="cpu")
    actor_critic = ActorCritic(actor, critic)
    for layer in actor_critic.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=cfg["lr"])
    policy = RecordedPPO(
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
    optimizer_steps = 0

    def count_optimizer_step(*unused):
        nonlocal optimizer_steps
        optimizer_steps += 1

    optim.register_step_post_hook(count_optimizer_step)

    def checkpoint(update):
        torch.save(
            {
                "policy": policy.state_dict(),
                "optim": optim.state_dict(),
                "return_rms": vars(policy.ret_rms),
                "config": cfg,
                "update": update,
                "env_steps": update * cfg["steps_per_collect"],
                "optimizer_steps": optimizer_steps,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            out / "checkpoints" / f"update-{update:03d}.pt",
        )

    def evaluate(seed, name):
        journal = Journal()
        env = RecordedEnv(
            envpool.make_gymnasium(
                cfg["task"],
                num_envs=cfg["eval_num"],
                batch_size=cfg["eval_num"],
                num_threads=cfg["num_threads"],
                seed=seed,
            ),
            journal,
        )
        obs, _ = env.reset()
        active = np.ones(cfg["eval_num"], dtype=bool)
        returns = np.zeros(cfg["eval_num"])
        lengths = np.zeros(cfg["eval_num"], dtype=np.int32)
        terminated_count = truncated_count = 0
        policy.eval()
        for _ in range(cfg["max_episode_steps"]):
            with torch.no_grad():
                probs, _ = policy.actor(obs)
                actions = probs.argmax(dim=-1).cpu().numpy().astype(np.int32)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            returns[active] += rewards[active]
            lengths[active] += 1
            terminated_count += int((active & terminated).sum())
            truncated_count += int((active & truncated).sum())
            active &= ~(terminated | truncated)
            if not active.any():
                break
        assert not active.any()
        env.close()
        result = {
            "seed": seed,
            "episodes": cfg["eval_num"],
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "returns": returns.tolist(),
            "lengths": lengths.tolist(),
            "terminated": terminated_count,
            "truncated": truncated_count,
            "trace": journal.save(out / (name + "-trace.npz")),
        }
        write_json(out / (name + ".json"), result)
        print(
            f"{name}: mean={result['mean_return']}, truncated={truncated_count}",
            flush=True,
        )
        return result

    checkpoint(0)
    baseline = evaluate(cfg["baseline_seed"], "baseline")
    journal = Journal()
    env = RecordedEnv(
        envpool.make_gymnasium(
            cfg["task"],
            num_envs=cfg["training_num"],
            batch_size=cfg["training_num"],
            num_threads=cfg["num_threads"],
            seed=cfg["train_seed"],
        ),
        journal,
    )
    buffer = VectorReplayBuffer(cfg["steps_per_collect"], cfg["training_num"])
    collector = Collector(policy, env, buffer)
    started = time.monotonic()
    metrics = []
    with (out / "metrics.jsonl").open("w") as metric_file:
        for update in range(1, cfg["updates"] + 1):
            policy.train()
            collected = collector.collect(n_step=cfg["steps_per_collect"])
            assert collected["n/st"] == cfg["steps_per_collect"]
            assert len(buffer) == cfg["steps_per_collect"]
            losses = policy.update(
                0,
                buffer,
                batch_size=cfg["batch_size"],
                repeat=cfg["repeat_per_collect"],
            )
            expected_steps = (
                update
                * cfg["repeat_per_collect"]
                * cfg["steps_per_collect"]
                // cfg["batch_size"]
            )
            assert optimizer_steps == expected_steps
            assert all(
                len(values) == expected_steps // update
                for values in losses.values()
            )
            collector.reset_buffer(keep_statistics=True)
            checkpoint(update)
            row = {
                "update": update,
                "env_steps": update * cfg["steps_per_collect"],
                "optimizer_steps": optimizer_steps,
                "collection": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in collected.items()
                },
                "losses": losses,
            }
            metric_file.write(json.dumps(row, allow_nan=False) + "\n")
            metric_file.flush()
            metrics.append(row)
            if update == 1 or update % 10 == 0:
                print(
                    f"update={update} steps={row['env_steps']} optimizer_steps={optimizer_steps} "
                    f"train_return={collected['rew']:.3f} elapsed={time.monotonic() - started:.2f}s",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    env.close()
    final = evaluate(cfg["eval_seed"], "final")
    summary = {
        "config": cfg,
        "env_steps": collector.collect_step,
        "train_episodes": collector.collect_episode,
        "optimizer_steps": optimizer_steps,
        "updates": cfg["updates"],
        "baseline": baseline,
        "final": final,
        "train_trace": journal.save(out / "train-trace.npz"),
        "processed_batches": processed_journal.save(
            out / "processed-batches.npz"
        ),
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "timing.json",
        {
            "training_seconds_including_instrumentation": elapsed,
            "note": "Uncontrolled concurrent workloads; not a PPO speed benchmark.",
        },
    )
    assert summary["env_steps"] == cfg["updates"] * cfg["steps_per_collect"]
    assert sha256(native_path) == provenance["native_sha256"]
    print(
        json.dumps({
            "complete": True,
            "steps": summary["env_steps"],
            "optimizer_steps": optimizer_steps,
            "final_mean": final["mean_return"],
        }),
        flush=True,
    )


def compare(args):
    """Compare both experiment outputs with zero tolerance."""
    import numpy as np
    import torch

    left, right = args.original.resolve(), args.candidate.resolve()
    failures = []
    counts = {
        "arrays": 0,
        "array_elements": 0,
        "tensors": 0,
        "tensor_elements": 0,
        "scalars": 0,
        "checkpoint_files": 0,
    }
    per_file = {}

    def equal(a, b, path):
        if isinstance(a, torch.Tensor):
            counts["tensors"] += 1
            counts["tensor_elements"] += a.numel()
            if (
                not isinstance(b, torch.Tensor)
                or a.dtype != b.dtype
                or a.shape != b.shape
            ):
                failures.append({
                    "path": path,
                    "reason": "tensor type/dtype/shape mismatch",
                })
                return
            array_equal(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), path
            )
        elif isinstance(a, np.ndarray):
            counts["arrays"] += 1
            counts["array_elements"] += a.size
            array_equal(a, b, path)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                failures.append({
                    "path": path,
                    "reason": "mapping key mismatch",
                })
                return
            for key in a:
                equal(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                failures.append({
                    "path": path,
                    "reason": "sequence type/length mismatch",
                })
                return
            for index, (av, bv) in enumerate(zip(a, b, strict=False)):
                equal(av, bv, path + "/" + str(index))
        else:
            counts["scalars"] += 1
            if type(a) is not type(b) or a != b:
                failures.append({
                    "path": path,
                    "reason": "scalar mismatch",
                    "original": repr(a),
                    "candidate": repr(b),
                })

    def array_equal(a, b, path):
        if (
            not isinstance(b, np.ndarray)
            or a.dtype != b.dtype
            or a.shape != b.shape
        ):
            failures.append({
                "path": path,
                "reason": "array type/dtype/shape mismatch",
            })
            return
        if a.dtype.kind in "fc" and not (
            np.isfinite(a).all() and np.isfinite(b).all()
        ):
            failures.append({"path": path, "reason": "nonfinite numeric array"})
            return
        if a.tobytes(order="C") != b.tobytes(order="C"):
            failure = {
                "path": path,
                "reason": "array bytes differ",
                "differing_values": int(np.count_nonzero(a != b)),
            }
            if a.size and a.dtype.kind in "fciu":
                failure["max_abs_difference"] = float(
                    np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                )
            failures.append(failure)

    for name in ("config.json", "summary.json", "baseline.json", "final.json"):
        equal(
            json.loads((left / name).read_text()),
            json.loads((right / name).read_text()),
            name,
        )
    equal(
        [
            json.loads(line)
            for line in (left / "metrics.jsonl").read_text().splitlines()
        ],
        [
            json.loads(line)
            for line in (right / "metrics.jsonl").read_text().splitlines()
        ],
        "metrics.jsonl",
    )
    for name in (
        "baseline-trace.npz",
        "train-trace.npz",
        "processed-batches.npz",
        "final-trace.npz",
    ):
        before = counts.copy()
        before_failures = len(failures)
        with (
            np.load(left / name, allow_pickle=False) as a,
            np.load(right / name, allow_pickle=False) as b,
        ):
            equal(dict(a), dict(b), name)
        per_file[name] = {
            "equal": before_failures == len(failures),
            **{key: counts[key] - before[key] for key in counts},
        }
    lfiles = sorted((left / "checkpoints").glob("*.pt"))
    rfiles = sorted((right / "checkpoints").glob("*.pt"))
    equal(
        [p.name for p in lfiles],
        [p.name for p in rfiles],
        "checkpoint_filenames",
    )
    for path in lfiles:
        # Both files are trusted artifacts freshly produced by this script.
        equal(
            torch.load(path, map_location="cpu", weights_only=False),
            torch.load(
                right / "checkpoints" / path.name,
                map_location="cpu",
                weights_only=False,
            ),
            "checkpoints/" + path.name,
        )
        counts["checkpoint_files"] += 1
    lp = json.loads((left / "provenance.json").read_text())
    rp = json.loads((right / "provenance.json").read_text())
    equal(lp["versions"], rp["versions"], "versions")
    for name in (
        "torch_threads",
        "torch_interop_threads",
        "torch_deterministic",
        "environment",
        "script_sha256",
    ):
        equal(lp[name], rp[name], "provenance/" + name)
    assert lp["native_binary"] != rp["native_binary"]
    assert lp["native_sha256"] != rp["native_sha256"], (
        "Both runs used the same native binary!"
    )
    result = {
        "passed": not failures,
        "comparison": "Exact values and contiguous array bytes; zero tolerance",
        "scope": "One matched-seed, fixed-budget, synchronous CartPole-v1 PPO experiment on this CPU/runtime",
        "counts": counts,
        "per_file": per_file,
        "failures": failures,
        "max_abs_difference": 0 if not failures else None,
        "original_native_binary": lp["native_binary"],
        "original_native_sha256": lp["native_sha256"],
        "candidate_native_binary": rp["native_binary"],
        "candidate_native_sha256": rp["native_sha256"],
        "excluded": [
            "wall-clock timings",
            "archive serialization bytes",
            "runtime import paths",
        ],
        "limitations": [
            "Not proof of general PPO equivalence",
            "No asynchronous-ordering claim",
            "Not a training-throughput benchmark",
            "No cross-platform or full-family claim",
        ],
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


def run(args):
    """Launch isolated original and candidate processes and compare."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    })
    commands = []
    for name, package in (
        ("original", args.original),
        ("candidate", args.candidate),
    ):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "train",
            "--package",
            str(package.resolve()),
            "--config",
            str(args.config.resolve()),
            "--output",
            str(output / name),
        ]
        commands.append(command)
        print("RUN", " ".join(command), flush=True)
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print((output / (name + ".log")).read_text(), flush=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "compare",
        "--original",
        str(output / "original"),
        "--candidate",
        str(output / "candidate"),
        "--output",
        str(output / "comparison.json"),
    ]
    commands.append(command)
    write_json(
        output / "commands.json",
        {
            "commands": commands,
            "environment": {
                k: env[k]
                for k in (
                    "PYTHONHASHSEED",
                    "PYTHONDONTWRITEBYTECODE",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
        },
    )
    subprocess.run(command, env=env, check=True)


def main():
    """Parse the experiment command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "compare", "run"):
        current = sub.add_parser(mode)
        current.add_argument("--output", type=Path, required=True)
        if mode == "train":
            current.add_argument("--package", type=Path, required=True)
        else:
            current.add_argument("--original", type=Path, required=True)
            current.add_argument("--candidate", type=Path, required=True)
        if mode != "compare":
            current.add_argument(
                "--config",
                type=Path,
                default=Path(__file__).with_name("config.json"),
            )
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
