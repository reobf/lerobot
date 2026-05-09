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
import os
from dataclasses import dataclass, field
import torch
from pathlib import Path
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE, OBS_STR
import numpy as np
from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="libero_processor")
class LiberoProcessorStep(ObservationProcessorStep):
    """
    Processes LIBERO observations into the LeRobot format.

    This step handles the specific observation structure from LIBERO environments,
    which includes nested robot_state dictionaries and image observations.

    **State Processing:**
    -   Processes the `robot_state` dictionary which contains nested end-effector,
        gripper, and joint information.
    -   Extracts and concatenates:
        - End-effector position (3D)
        - End-effector quaternion converted to axis-angle (3D)
        - Gripper joint positions (2D)
    -   Maps the concatenated state to `"observation.state"`.

    **Image Processing:**
    -   Rotates images by 180 degrees by flipping both height and width dimensions.
    -   This accounts for the HuggingFaceVLA/libero camera orientation convention.
    """
# depth dump 配置（通过环境变量控制）
    depth_dump_dir: str = field(default_factory=lambda: os.environ.get("LIBERO_DEPTH_DIR", "./depth_dumps"))
    depth_dump_enabled: bool = field(default_factory=lambda: os.environ.get("LIBERO_DUMP_DEPTH", "0") == "1")
    _depth_frame_counter: int = field(default=0, init=False, repr=False)
    _znear: float = field(default=0.01, init=False, repr=False)
    _zfar: float = field(default=500.0, init=False, repr=False)
 
    def __post_init__(self):
        if self.depth_dump_enabled:
            Path(self.depth_dump_dir).mkdir(parents=True, exist_ok=True)
            print(f"[DepthDump] Enabled. Saving depth to: {self.depth_dump_dir}")

    def set_depth_params(self, znear, zfar):
        """从环境获取 near/far 后调用。"""
        self._znear = znear
        self._zfar = zfar
        print(f"[DepthEncoder] znear={znear}, zfar={zfar}")

    def _encode_depth_16bit(self, depth_zbuffer):
        """z-buffer tensor → 16-bit RGB tensor (和数据集格式一致)。

        输入: (B, 1, H, W) 或 (B, H, W), 值域 [0, 1] z-buffer
        输出: (B, 3, H, W), R=高8位/255 G=低8位/255 B=0, 值域 [0, 1] float
        """
        if depth_zbuffer.ndim == 4:
            d = depth_zbuffer[:, 0, :, :]
        else:
            d = depth_zbuffer

        near, far = self._znear, self._zfar
        d_real = near / (1.0 - d * (1.0 - near / far))

        d_mm = (d_real * 10000).clamp(0, 65535)
        d_int = d_mm.to(torch.int32)
        high = ((d_int >> 8) & 0xFF).float() / 255.0
        low = (d_int & 0xFF).float() / 255.0
        zero = torch.zeros_like(high)

        return torch.stack([high, low, zero], dim=1)
 
    def _dump_depths(self, depths: dict):
        """保存 depth 到磁盘。depths 是 {camera_name: depth_array} 的 dict。"""
        #print(depths)
        if not self.depth_dump_enabled:
            return
        save_dict = {}
        for cam_name, depth in depths.items():
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()
            save_dict[cam_name] = depth.astype(np.float32)
        fname = Path(self.depth_dump_dir) / f"frame_{self._depth_frame_counter:06d}.npz"
        np.savez_compressed(fname, **save_dict)
        self._depth_frame_counter += 1
    def _process_observation(self, observation):
    # === Lazy init: 从 env var 读 znear/zfar (由 LIBERO env 在 _ensure_env 中写) ===
        if not getattr(self, '_depth_params_initialized', False):
            znear_env = os.environ.get('LIBERO_ZNEAR')
            zfar_env = os.environ.get('LIBERO_ZFAR')
            if znear_env and zfar_env:
                self._znear = float(znear_env)
                self._zfar = float(zfar_env)
                print(f"[DepthEncoder] lazy init from env: znear={self._znear}, zfar={self._zfar}")
            else:
                print(f"[DepthEncoder] WARN: LIBERO_ZNEAR/ZFAR env var not set, "
                    f"using fallback znear={self._znear}, zfar={self._zfar} "
                    f"(可能跟 hdf.py 不一致, 数据会错!)")
            self._depth_params_initialized = True        
        """
        Processes both image and robot_state observations from LIBERO.
        """
        #print(f"[DEBUG processor] keys: {list(observation.keys())}")
        processed_obs = observation.copy()
        depth_key = OBS_PREFIX + "depths"
        #print(processed_obs)
        if depth_key in processed_obs:
            depths = processed_obs[depth_key]
            self._dump_depths(depths)
            # 展开成和 images 一样的 key 格式，并做 180° 翻转 + 16-bit 编码
            # observation.depths → observation.depths.image, observation.depths.image2
            if isinstance(depths, dict):
                for cam_name, depth_tensor in depths.items():
                    full_key = f"{depth_key}.{cam_name}"
                    if isinstance(depth_tensor, torch.Tensor):
                        if depth_tensor.ndim == 4:
                            depth_tensor = torch.flip(depth_tensor, dims=[2, 3])  # (B, C, H, W)
                        elif depth_tensor.ndim == 3:
                            depth_tensor = torch.flip(depth_tensor, dims=[1, 2])  # (B, H, W)
                        # z-buffer → 真实距离 → 16-bit RGB 编码（和数据集一致）
                        depth_tensor = self._encode_depth_16bit(depth_tensor)
                    processed_obs[full_key] = depth_tensor
            del processed_obs[depth_key]  # 删掉原始的 dict 形式，保留展开后的

        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]

                # Flip both H and W
                img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # Process robot_state into a flat state vector
        observation_robot_state_str = OBS_PREFIX + "robot_state"
        if observation_robot_state_str in processed_obs:
            robot_state = processed_obs.pop(observation_robot_state_str)

            # Extract components
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_quat = robot_state["eef"]["quat"]  # (B, 4,)
            gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,)

            # Convert quaternion to axis-angle
            eef_axisangle = self._quat2axisangle(eef_quat)  # (B, 3)
            # Concatenate into a single state vector
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)

            # ensure float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transforms feature keys from the LIBERO format to the LeRobot standard.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # copy over non-STATE features
        for ft, feats in features.items():
            if ft != FeatureType.STATE:
                new_features[ft] = feats.copy()

        # rebuild STATE features
        state_feats = {}

        # add our new flattened state
        state_feats[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(8,),  # [eef_pos(3), axis_angle(3), gripper(2)]
        )

        new_features[FeatureType.STATE] = state_feats

        return new_features

    def observation(self, observation):
        return self._process_observation(observation)

    def _quat2axisangle(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert batched quaternions to axis-angle format.
        Only accepts torch tensors of shape (B, 4).

        Args:
            quat (Tensor): (B, 4) tensor of quaternions in (x, y, z, w) format

        Returns:
            Tensor: (B, 3) axis-angle vectors

        Raises:
            TypeError: if input is not a torch tensor
            ValueError: if shape is not (B, 4)
        """

        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")

        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]

        w = quat[:, 3].clamp(-1.0, 1.0)

        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

        result = torch.zeros((batch_size, 3), device=device)

        mask = den > 1e-10

        if mask.any():
            angle = 2.0 * torch.acos(w[mask])  # (M,)
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)

        return result


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_arena_processor")
class IsaaclabArenaProcessorStep(ObservationProcessorStep):
    """
    Processes IsaacLab Arena observations into LeRobot format.

    **State Processing:**
    - Extracts state components from obs["policy"] based on `state_keys`.
    - Concatenates into a flat vector mapped to "observation.state".

    **Image Processing:**
    - Extracts images from obs["camera_obs"] based on `camera_keys`.
    - Converts from (B, H, W, C) uint8 to (B, C, H, W) float32 [0, 1].
    - Maps to "observation.images.<camera_name>".
    """

    # Configurable from IsaacLabEnv config / cli args: --env.state_keys="robot_joint_pos,left_eef_pos"
    state_keys: tuple[str, ...]

    # Configurable from IsaacLabEnv config / cli args: --env.camera_keys="robot_pov_cam_rgb"
    camera_keys: tuple[str, ...]

    def _process_observation(self, observation):
        """
        Processes both image and policy state observations from IsaacLab Arena.
        """
        processed_obs = {}

        if f"{OBS_STR}.camera_obs" in observation:
            camera_obs = observation[f"{OBS_STR}.camera_obs"]

            for cam_name, img in camera_obs.items():
                if cam_name not in self.camera_keys:
                    continue

                img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.dtype != torch.float32:
                    img = img.float()

                processed_obs[f"{OBS_IMAGES}.{cam_name}"] = img

        # Process policy state -> observation.state
        if f"{OBS_STR}.policy" in observation:
            policy_obs = observation[f"{OBS_STR}.policy"]

            # Collect state components in order
            state_components = []
            for key in self.state_keys:
                if key in policy_obs:
                    component = policy_obs[key]
                    # Flatten extra dims: (B, N, M) -> (B, N*M)
                    if component.dim() > 2:
                        batch_size = component.shape[0]
                        component = component.view(batch_size, -1)
                    state_components.append(component)

            if state_components:
                state = torch.cat(state_components, dim=-1)
                state = state.float()
                processed_obs[OBS_STATE] = state

        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Not used for policy evaluation."""
        return features

    def observation(self, observation):
        return self._process_observation(observation)