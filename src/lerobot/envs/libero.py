#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.types import RobotObservation

from .utils import _LazyAsyncVectorEnv, parse_camera_names


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


# LIBERO-plus perturbation variants encode the perturbation in the filename
# but on disk only the base `.pruned_init` exists — strip the suffix to match
# LIBERO-plus's own suite.get_task_init_states() (we reimplement it here so we
# can pass weights_only=False for PyTorch 2.6+ numpy pickles).
_LIBERO_PERTURBATION_SUFFIX_RE = re.compile(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")


def get_task_init_states(task_suite: Any, i: int, is_libero_plus: bool = False) -> np.ndarray:
    task = task_suite.tasks[i]
    filename = Path(task.init_states_file)
    root = Path(get_libero_path("init_states"))

    if not is_libero_plus:
        init_states_path = root / task.problem_folder / filename.name
        return torch.load(init_states_path, weights_only=False)  # nosec B614

    # LIBERO-plus: `_add_` / `_level` variants store extra-object layouts under
    # libero_newobj/ as a flat array that must be reshaped to (1, -1).
    if "_add_" in filename.name or "_level" in filename.name:
        init_states_path = root / "libero_newobj" / task.problem_folder / filename.name
        init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
        return init_states.reshape(1, -1)

    # LIBERO-plus perturbation variants encode the perturbation in the filename
    # but on disk only the base `.pruned_init` exists — strip the suffix to match.
    stripped = _LIBERO_PERTURBATION_SUFFIX_RE.sub("", filename.stem) + filename.suffix
    init_states_path = root / task.problem_folder / stripped
    return torch.load(init_states_path, weights_only=False)  # nosec B614


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
        is_libero_plus: bool = False,
    ):
        super().__init__()
        self.task_id = task_id
        self.is_libero_plus = is_libero_plus
        self.obs_type = obs_type
        self.render_mode = render_mode

        # Depth dump 配置（通过环境变量控制）
        self._depth_dump_enabled = os.environ.get("LIBERO_DUMP_DEPTH", "0") == "1"
        self._depth_dump_dir = Path(os.environ.get("LIBERO_DEPTH_DIR", "./depth_dumps"))
        self._depth_frame_counter = 0
        if self._depth_dump_enabled:
            self._depth_dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DepthDump] Enabled for task_id={task_id}. Dir: {self._depth_dump_dir}")
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self.episode_length = episode_length
        # Load once and keep
        self._init_states = (
            get_task_init_states(task_suite, self.task_id, is_libero_plus=self.is_libero_plus)
            if self.init_states
            else None
        )
        self._reset_stride = n_envs  # when performing a reset, append `_reset_stride` to `init_state_id`.

        self.init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        # Extract task metadata without allocating GPU resources (safe before fork).
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        self._task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self._env: OffScreenRenderEnv | None = (
            None  # deferred — created on first reset() inside the worker subprocess
        )

        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        images = {}
        depth_spaces = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            # depth: 单通道 float32, 值域 [0, 1]
            depth_spaces[self.camera_name_mapping[cam]] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.observation_height, self.observation_width),
                dtype=np.float32,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "depths": spaces.Dict(depth_spaces),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "depths": spaces.Dict(depth_spaces),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        """Create the underlying OffScreenRenderEnv on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own clean EGL context rather than inheriting a stale one from the
        parent process (which causes EGL_BAD_CONTEXT crashes with AsyncVectorEnv).
        """
        if self._env is not None and hasattr(self._env, 'env') and self._env.env is not None:
            return
        print(f"[DEBUG] Rebuilding env (self._env={self._env is not None})")
        env = OffScreenRenderEnv(
            bddl_file_name=self._task_bddl_file,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            camera_depths=True,  # 启用深度渲染，obs 中会多出 {camera_name}_depth 键
        )
        extent = env.sim.model.stat.extent
        znear = env.sim.model.vis.map.znear * extent     # ← 修复 stat.extent
        zfar = env.sim.model.vis.map.zfar * extent
        import os
        os.environ['LIBERO_ZNEAR'] = str(znear)
        os.environ['LIBERO_ZFAR'] = str(zfar)
        print(f"[LIBERO env] depth params (from MuJoCo): znear={znear}, zfar={zfar} → env var set")
        env.reset()
        self._env = env
      

    def render(self):
        self._ensure_env()
        raw_obs = self._env.env._get_observations()
        formatted = self._format_raw_obs(raw_obs)

        panels = []

        # 1. Agentview RGB
        if "image" in formatted["pixels"]:
            img = formatted["pixels"]["image"]
            img = img[::-1, ::-1]  # flip H and W
            panels.append(img)

        # 2. Wrist RGB
        if "image2" in formatted["pixels"]:
            img2 = formatted["pixels"]["image2"]
            img2 = img2[::-1, ::-1]
            panels.append(img2)

        # 3. Agentview Depth（灰度转 RGB 以便拼接）
        if "depths" in formatted and "image" in formatted["depths"]:
            depth = formatted["depths"]["image"]
            depth = depth[::-1, ::-1]
            # 归一化到 0-255，转成 3 通道灰度图
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
            depth_rgb = np.stack([depth_norm] * 3, axis=-1)
            # resize 到和 RGB 同尺寸（depth 可能尺寸不同）
            if depth_rgb.shape[:2] != panels[0].shape[:2]:
                from PIL import Image
                depth_rgb = np.array(Image.fromarray(depth_rgb).resize(
                    (panels[0].shape[1], panels[0].shape[0]),
                    resample=Image.NEAREST,
                ))
            panels.append(depth_rgb)
        # 4. Wrist Depth（灰度转 RGB 以便拼接）
        if "depths" in formatted and "image2" in formatted["depths"]:
            depth2 = formatted["depths"]["image2"]
            depth2 = depth2[::-1, ::-1]
            d_min, d_max = depth2.min(), depth2.max()
            if d_max - d_min > 1e-6:
                depth_norm2 = ((depth2 - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm2 = np.zeros_like(depth2, dtype=np.uint8)
            depth_rgb2 = np.stack([depth_norm2] * 3, axis=-1)
            if depth_rgb2.shape[:2] != panels[0].shape[:2]:
                from PIL import Image
                depth_rgb2 = np.array(Image.fromarray(depth_rgb2).resize(
                    (panels[0].shape[1], panels[0].shape[0]),
                    resample=Image.NEAREST,
                ))
            panels.append(depth_rgb2)
        # 左右拼接: [agentview | wrist | depth]
        if len(panels) > 1:
            return np.concatenate(panels, axis=1)
        return panels[0]

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        assert self._env is not None, "_format_raw_obs called before _ensure_env()"
        images = {}
        depths = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

            # 提取 depth: "agentview_image" → "agentview_depth"
            depth_key = camera_name.replace("_image", "_depth")
            if depth_key in raw_obs:
                d = raw_obs[depth_key].astype(np.float32).squeeze()  # (raw_H, raw_W)
                # resize 到和 RGB 同尺寸（depth 用最近邻插值保持值不失真）
                if d.shape != (self.observation_height, self.observation_width):
                    from PIL import Image
                    d = np.array(Image.fromarray(d).resize(
                        (self.observation_width, self.observation_height),
                        resample=Image.NEAREST,
                    ))
                depths[self.camera_name_mapping[camera_name]] = d

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "depths": depths,  # 新增：深度图（不会传给 policy）
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy(), "depths": depths.copy()}

        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        super().reset(seed=seed)
        #print(seed)
        if seed is not None:
            self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            self.init_state_id += self._reset_stride  # Change init_state_id when reset

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            self.reset()
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
            is_libero_plus=is_libero_plus,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

    camera_names = parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    is_async = env_cls is gym.vector.AsyncVectorEnv

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        # All tasks in a suite share identical observation/action spaces.
        # Probe once and reuse to avoid creating a temp env per task.
        cached_obs_space: spaces.Space | None = None
        cached_act_space: spaces.Space | None = None
        cached_metadata: dict[str, Any] | None = None

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
                camera_name_mapping=camera_name_mapping,
                is_libero_plus=is_libero_plus,
            )
            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[suite_name][tid] = lazy
            else:
                out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}