#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""
from lerobot.policies.smolvla.pcd_encoder import PCDDepthEncoder
import os
import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel

_DECODE_DEPTH_CALL_COUNT = 0  # global counter for sampled diagnostic

# ============================================================
# Depth filter ranges (PER CAMERA, in meters)
# ----------------------------------------------------------------
# 控制点云有效深度范围. 比 z_min 近 / 比 z_max 远的点视为 invalid:
#   - 可视化时用红色高亮
#   - PointNet 反投影时丢点 (如果 PCD encoder 用同样阈值)
#
# 索引按 image_features 顺序: 0 = agent (cam0), 1 = wrist (cam1).
# Note: 你 hdf.py stat.extent=10 没乘进去, 单位被压缩 ~10×.
#       当前数据: agent z 实际范围 ~[0.04, 0.29],  wrist z ~[0.003, 0.20]
#
# 改完后: 可视化立即生效, train/eval 都会用新阈值.
# 如果以后修复了 stat.extent bug, 这里改成真实米值即可.
# ============================================================
Z_RANGES = {
    # 修了 stat.extent + decode /10000 后, depth 是真实米单位
    # 根据你 eval 看到的真实值 (decode_depth /10000 修复后):
    #   agent: 真实 z ~ 0.6-2.9 m  (桌面 ~1.5m)
    #   wrist: 真实 z ~ 0.04-0.4 m (夹爪贴近物体)
    0: (0.5, 3.0),       # cam0 = agent: 0.5m 以下=invalid 占位 / 3.0m 以上=远场景
    1: (0.04, 1.5),      # cam1 = wrist: 4cm 以下=夹爪表面 invalid / 1.5m 以上=outlier
}


def get_z_range(cam_idx: int) -> tuple[float, float]:
    """Get (z_min, z_max) for a given camera index. Falls back to permissive default."""
    return Z_RANGES.get(cam_idx, (0.0, 100.0))


def decode_depth(depth_map):
    """16-bit RGB → 真实距离（米）。1 LSB = 0.1mm。

    匹配 hdf.py 和 LiberoProcessorStep 的 ×10000 编码:
      生成端: d_mm_int = depth_real × 10000  (0.1mm 量化)
      解码端: depth_real = d_mm_int / 10000

    Max 编码距离 = 65535 / 10000 = 6.5535 m (LIBERO 桌面 ~1.5m, 余量大)
    """
    if depth_map.ndim == 4:
        if depth_map.shape[1] <= 3:  # (B, C, H, W)
            high = depth_map[:, 0, :, :].float() * 255.0
            low = depth_map[:, 1, :, :].float() * 255.0
        else:  # (B, H, W, C)
            high = depth_map[:, :, :, 0].float() * 255.0
            low = depth_map[:, :, :, 1].float() * 255.0
    else:
        return depth_map.float()
    return (high * 256 + low) / 10000.0  # 0.1mm → m

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()
    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        depths = self.prepare_depths(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, depths=depths, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        #print(batch.keys())
        #exit(0)
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        depths = self.prepare_depths(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        # === PCD Evolution print (每 500 forward) ===
        if (getattr(self.model, "use_pcd", False)
                and self.model.pcd_encoder_agent is not None):
            if not hasattr(self, "_pcd_step_count"):
                self._pcd_step_count = 0
                self._pcd_last_grads = {}
                # 自适应 PointNet (pcd_encoder.py) vs PCT (pcd_encoder_transformer.py):
                # 检测 encoder 实际 attribute 决定挂哪些 hook
                self._pcd_backbone_kind = None
                for _name, _mod in [("agent", self.model.pcd_encoder_agent),
                                    ("wrist", self.model.pcd_encoder_wrist)]:
                    if hasattr(_mod, "pointnet"):
                        # PointNet 版本 (pcd_encoder.py)
                        self._pcd_backbone_kind = "pointnet"
                        hook_targets = [
                            (f"{_name}.proj.w", _mod.proj.weight),
                            (f"{_name}.pn.conv1.w", _mod.pointnet.conv1.weight),
                            (f"{_name}.pn.conv3.w", _mod.pointnet.conv3.weight),
                            (f"{_name}.stn.fc3.w", _mod.pointnet.stn.fc3.weight),
                        ]
                    elif hasattr(_mod, "pct"):
                        # PCT 版本 (pcd_encoder_transformer.py)
                        # 自适应: attn_blocks / concat_proj 可能不存在 (退化版)
                        self._pcd_backbone_kind = "pct"
                        hook_targets = [
                            (f"{_name}.proj.w", _mod.proj.weight),
                        ]
                        if hasattr(_mod.pct, "input_embed"):
                            hook_targets.append(
                                (f"{_name}.pct.input_embed.0.w", _mod.pct.input_embed[0].weight)
                            )
                        if hasattr(_mod.pct, "attn_blocks") and len(_mod.pct.attn_blocks) > 0:
                            hook_targets.append(
                                (f"{_name}.pct.attn0.q.w", _mod.pct.attn_blocks[0].q_proj.weight)
                            )
                        if hasattr(_mod.pct, "concat_proj"):
                            hook_targets.append(
                                (f"{_name}.pct.concat.0.w", _mod.pct.concat_proj[0].weight)
                            )
                    else:
                        # Unknown encoder kind, skip hooks
                        hook_targets = []
                    for _key, _p in hook_targets:
                        if not _p.requires_grad:
                            # frozen param (e.g. FREEZE_PCT=1) 不能注册 hook, 跳过
                            continue
                        def _make_hook(_n):
                            def _hook(grad):
                                self._pcd_last_grads[_n] = grad.detach().float().norm().item()
                                return None
                            return _hook
                        _p.register_hook(_make_hook(_key))

            if self._pcd_step_count % 100 == 0:
                a = self.model.pcd_encoder_agent
                w = self.model.pcd_encoder_wrist
                lg = self._pcd_last_grads
                # 直接看 token magnitude (raw output 的量级反映场景几何复杂度)
                with torch.no_grad():
                    a_proj_w_norm = a.proj.weight.float().norm().item()
                    w_proj_w_norm = w.proj.weight.float().norm().item()
                print(
                    f"[PCD-Evol] step={self._pcd_step_count}  "
                    f"agent.proj.w_norm={a_proj_w_norm:.3f}  |  "
                    f"wrist.proj.w_norm={w_proj_w_norm:.3f}"
                )
                # grad keys 也要跟 backbone kind 对应
                if self._pcd_backbone_kind == "pointnet":
                    grad_keys = ("agent.proj.w", "agent.pn.conv1.w", "agent.stn.fc3.w",
                                 "wrist.proj.w", "wrist.pn.conv1.w")
                elif self._pcd_backbone_kind == "pct":
                    grad_keys = ("agent.proj.w", "agent.pct.input_embed.0.w",
                                 "agent.pct.attn0.q.w", "agent.pct.concat.0.w",
                                 "wrist.proj.w", "wrist.pct.input_embed.0.w")
                else:
                    grad_keys = ()
                grad_str = "  ".join(
                    f"{k}.g={lg.get(k, float('nan')):.4f}"
                    for k in grad_keys
                )
                print(f"[PCD-Grad] step={self._pcd_step_count}  {grad_str}")
            self._pcd_step_count += 1

        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, depths=depths)
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_depths(self, batch):
        """Extract depth maps from batch, aligned with prepare_images output order.

        Depth key naming convention:
            observation.images.image  → observation.depths.image
            observation.images.image2 → observation.depths.image2

        Returns:
            List of depth tensors (one per camera), or None if no depth in batch.
        """
        depths = []
        has_any = False
        present_img_keys = [key for key in self.config.image_features if key in batch]

        for key in present_img_keys:
            depth_key = key.replace("observation.images.", "observation.depths.")
            if depth_key in batch:
                depth = batch[depth_key]
                depth = depth[:, -1, :, :, :] if depth.ndim == 5 else depth
                depths.append(depth)
                has_any = True
            else:
                depths.append(None)

        return depths if has_any else None

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # =================================================================
        # PCD Depth Branch (baseline E, 3D-CAVLA inspired)
        # ----------------------------------------------------------------
        # 通过环境变量 USE_PCD=1 启用. 默认关闭, 行为与原版 SmolVLA 完全一致.
        # 设计思路 (跟 baseline B/C/D 的对照):
        #   - B (DeFM prefix): 9 个 image-like depth tokens / cam → 干扰主干 attention
        #   - C (Ego3D PE):    PE 加在 RGB token 上 → 模型主动抵抗
        #   - D (Cross-Attn):  cross-attn 选择性查询 depth → K/V 不学 (gate 卡 0)
        #   - E (PCD/PointNet): 1 个 global geometry token / cam, 显式异质于 RGB
        #
        # 假设: 异质性大反而更好学. PointNet feature 跟 SigLIP feature 完全
        # 不同分布, 模型 unambiguously 识别为新模态.
        # =================================================================
        self.use_pcd = os.environ.get("USE_PCD", "0") == "1"

        # ADD_PCD: 是否在 prefix 里给 PCD 留槽位 (默认 1).
        # - ADD_PCD=1, USE_PCD=1: 每个 image 后面 append 一个真 PCD token (PointNet 算)
        # - ADD_PCD=1, USE_PCD=0: 每个 image 后面 append 一个 zero 占位 token
        #   → prefix 长度跟 USE_PCD=1 完全一样, 主干看到的 token 结构对齐,
        #     消除 "USE_PCD=0 vs USE_PCD=1 prefix 长度不同导致的额外学习成本"
        # - ADD_PCD=0: 啥都不加, 等价 baseline A 的 prefix
        self.add_pcd = os.environ.get("ADD_PCD", "1") == "1"

        # 总是创建 PCD encoder (即使 USE_PCD=0) — 保证 ckpt 始终含 PCD 参数,
        # USE_PCD=0 → USE_PCD=1 resume 时不会 "List length mismatch".
        # 当 USE_PCD=0 时, encoder 创建但 freeze + 不参与 forward.
        vla_hidden = self.vlm_with_expert.config.text_config.hidden_size
        self.pcd_encoder_agent = PCDDepthEncoder(
            vla_hidden=vla_hidden,
            fovy_deg=45.0,             # LIBERO agent view
            img_size=512,
            num_points=4096,
            cam_label="agent",
        )
        self.pcd_encoder_wrist = PCDDepthEncoder(
            vla_hidden=vla_hidden,
            fovy_deg=75.0,             # LIBERO wrist view
            img_size=512,
            num_points=4096,
            cam_label="wrist",
        )

        # =================================================================
        # Placeholder + Modality embeddings
        # ----------------------------------------------------------------
        # 1) PCD placeholder (learnable, per-camera):
        #    USE_PCD=0 时塞这个代替 zero token, 避免 zero 毒化主干 attention.
        #    每个相机一个独立 placeholder, 让主干能学到 "agent slot empty" vs
        #    "wrist slot empty" 的区分.
        #
        # 2) Modality embeddings (learnable):
        #    给 RGB / PCD / lang token 各加一个可学 embedding, 让主干第一层
        #    就能识别 token 模态. ViT/Flamingo/CLIP 标准做法.
        #    init scale = 0.02 跟 BERT/GPT embedding init 一致, 不破坏 token
        #    量级 (现有 RGB / PCD / lang token std≈1).
        #
        # 控制开关 (默认全开):
        #   PCD_LEARNABLE_PLACEHOLDER=1: USE_PCD=0 时用 learnable placeholder
        # =================================================================
        self.use_learnable_placeholder = (
            os.environ.get("PCD_LEARNABLE_PLACEHOLDER", "1") == "1"
        )

        # PCD placeholders — 只在 ADD_PCD=1 + 用 learnable placeholder 时有意义.
        # 但永远创建 (即使 USE_PCD=1 时), 保证 ckpt 结构一致, USE_PCD=0↔1 resume
        # 不报 missing key.
        self.pcd_placeholder_agent = nn.Parameter(
            torch.randn(1, 1, vla_hidden) * 0.02
        )
        self.pcd_placeholder_wrist = nn.Parameter(
            torch.randn(1, 1, vla_hidden) * 0.02
        )

        # 如果 placeholder 关闭, freeze 参数 (不参与训练,
        # 但仍 save 到 ckpt — 后续切换不会 missing key).
        if not self.use_learnable_placeholder:
            self.pcd_placeholder_agent.requires_grad = False
            self.pcd_placeholder_wrist.requires_grad = False

        print(
            f"[Modality] use_learnable_placeholder={int(self.use_learnable_placeholder)}. "
            f"Extra trainable params: "
            f"placeholder={2 * vla_hidden if self.use_learnable_placeholder else 0:,}"
        )

        if self.use_pcd:
            print(
                f"[PCD] Baseline E (3D-CAVLA mimic) ENABLED. ADD_PCD={int(self.add_pcd)}. "
                f"PointNet → Linear(1024, {vla_hidden}) → 1 token / cam, no gate/scale/mod_emb. "
                f"agent: {self.pcd_encoder_agent.num_trainable_params():,} params, "
                f"wrist: {self.pcd_encoder_wrist.num_trainable_params():,} params"
            )
            self._pcd_check_done = False
        else:
            if self.add_pcd:
                print(
                    f"[PCD] USE_PCD=0, ADD_PCD=1 → injecting ZERO placeholder token after each image. "
                    f"PCD encoders FROZEN (kept in ckpt for forward-compat with USE_PCD=1 resume)."
                )
            else:
                print(
                    f"[PCD] USE_PCD=0, ADD_PCD=0 → no PCD slot in prefix (= baseline A). "
                    f"PCD encoders FROZEN (kept in ckpt for forward-compat with USE_PCD=1 resume)."
                )
            for p in self.pcd_encoder_agent.parameters():
                p.requires_grad = False
            for p in self.pcd_encoder_wrist.parameters():
                p.requires_grad = False
            self._pcd_check_done = True

        self.set_requires_grad()

        # PCD diagnostic — set_requires_grad 之后检查
        if self.use_pcd:
            print("=" * 70)
            print("[PCD-DIAG] Trainable params after set_requires_grad():")
            for name, mod in [
                ("pcd_encoder_agent", self.pcd_encoder_agent),
                ("pcd_encoder_wrist", self.pcd_encoder_wrist),
            ]:
                n_train = mod.num_trainable_params()
                proj_rg = mod.proj.weight.requires_grad
                print(f"  {name}: trainable={n_train:,}, "
                      f"proj.requires_grad={proj_rg}")
            print("=" * 70)

        # =================================================================
        # Frozen-state diagnostic (always run, baseline E or not)
        # ----------------------------------------------------------------
        # SmolVLA default config: freeze_vision_encoder=True, train_expert_only=True
        # 等价于 3D-CAVLA 的 "LoRA frozen backbone + train action head" setup
        # 启动时确认参数 frozen 状态, 便于 debug.
        # =================================================================
        print("=" * 70)
        print("[FREEZE-DIAG] Top-level module trainable param breakdown:")
        print(f"  config.freeze_vision_encoder = {self.config.freeze_vision_encoder}")
        print(f"  config.train_expert_only     = {self.config.train_expert_only}")
        print(f"  config.train_state_proj      = {self.config.train_state_proj}")

        total_train = 0
        total_all = 0
        for child_name, child_module in self.named_children():
            train_p = sum(p.numel() for p in child_module.parameters() if p.requires_grad)
            all_p = sum(p.numel() for p in child_module.parameters())
            total_train += train_p
            total_all += all_p
            if all_p > 0:
                pct = 100.0 * train_p / all_p
                print(f"  {child_name}: {train_p:,} / {all_p:,} trainable ({pct:.1f}%)")

        if total_all > 0:
            overall_pct = 100.0 * total_train / total_all
            print(f"  ----------------------------------")
            print(f"  TOTAL: {total_train:,} / {total_all:,} trainable ({overall_pct:.1f}%)")
        print("=" * 70)
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None, depths=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        # 在 embed_prefix 开头
        if depths is not None:
            depths = [decode_depth(d) if d is not None else None for d in depths]
        #merge_type = int(os.environ.get("MERGE_TYPE", "0"))
        embs = []
        pad_masks = []
        att_masks = []

        # Per-segment magnitude log: list of (label, tensor) for first-forward print
        _do_token_mag_log = not hasattr(self, "_token_mag_log_done")
        _token_log = [] if _do_token_mag_log else None
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if not hasattr(self, f"_dbg_cam_{_img_idx}"):
                from torchvision.utils import save_image
                save_image(
                    img[0] * 0.5 + 0.5,  # SmolVLM 输入归一化到 [-1,1]，反归一化回 [0,1]
                    f"/tmp/cam_idx{_img_idx}.png"
                )
                print(f"[CAM] idx={_img_idx} saved to /tmp/cam_idx{_img_idx}.png")
                setattr(self, f"_dbg_cam_{_img_idx}", True)      
                
                if False and depths is not None and depths[_img_idx] is not None:
                    d_raw = depths[_img_idx][0].float()  # 第一个样本，可能是 (3,H,W) 或 (H,W)
        
                    # 判断是否已解码：如果是 3 通道就还是 RG16 编码，需要解码
                    if d_raw.ndim == 3 and d_raw.shape[0] == 3:
                        exit(0)
                    elif d_raw.ndim == 3 and d_raw.shape[0] == 1:
                        d_m = d_raw[0]  # (H, W)
                    else:
                        d_m = d_raw  # already (H, W)
        
                    print(f"[CAM] idx={_img_idx} depth(m): "
                        f"min={d_m.min():.3f} max={d_m.max():.3f} "
                        f"median={d_m.median():.3f} mean={d_m.mean():.3f}")
        
                    # 可视化：min-max 归一化，近物体亮、远物体暗
                    d_vis = 1.0 - (d_m - d_m.min()) / (d_m.max() - d_m.min() + 1e-8)
                    save_image(d_vis.unsqueeze(0), f"/tmp/depth_idx{_img_idx}.png")

                if depths is not None and depths[_img_idx] is not None:
                    d_raw = depths[_img_idx][0].float()  # (H, W) 已解码，米
                    
                    if d_raw.ndim == 3 and d_raw.shape[0] == 3:
                        exit(0)
                    elif d_raw.ndim == 3 and d_raw.shape[0] == 1:
                        d_m = d_raw[0]
                    else:
                        d_m = d_raw

                    print(f"[CAM] idx={_img_idx} depth(m): "
                        f"min={d_m.min():.3f} max={d_m.max():.3f} "
                        f"median={d_m.median():.3f} mean={d_m.mean():.3f}")

                    # === Vis 1: 标准灰度深度图 (近=亮 远=暗) ===
                    d_vis = 1.0 - (d_m - d_m.min()) / (d_m.max() - d_m.min() + 1e-8)
                    save_image(d_vis.unsqueeze(0), f"/tmp/depth_idx{_img_idx}.png")

                    # === Vis 2: 剔除高亮版本 — invalid 像素染红 ===
                    # 拿到这个相机的 z 阈值
                    z_min, z_max = get_z_range(_img_idx)
                    H, W = d_m.shape
                    invalid_mask = (d_m < z_min) | (d_m > z_max)  # (H, W) bool
                    n_invalid = invalid_mask.sum().item()
                    pct_invalid = 100.0 * n_invalid / (H * W)
                    print(f"[CAM] idx={_img_idx} filter z∈[{z_min:.3f}, {z_max:.3f}] → "
                        f"invalid {n_invalid}/{H*W} ({pct_invalid:.1f}%)")

                    # 构造 RGB: 默认灰度 (从 d_vis 复制 3 通道), invalid 染红
                    rgb_filtered = d_vis.unsqueeze(0).repeat(3, 1, 1).clone()  # (3, H, W) [0,1]
                    rgb_filtered[0][invalid_mask] = 1.0  # R = 1
                    rgb_filtered[1][invalid_mask] = 0.0  # G = 0
                    rgb_filtered[2][invalid_mask] = 0.0  # B = 0
                    save_image(rgb_filtered, f"/tmp/depth_idx{_img_idx}_filtered.png")
                    print(f"[CAM] idx={_img_idx} filtered vis saved to "
                        f"/tmp/depth_idx{_img_idx}_filtered.png "
                        f"(red = z<{z_min:.3f} or z>{z_max:.3f})")

                    # === 3D 点云 PNG ===
 
                setattr(self, f"_dbg_cam_{_img_idx}", True)
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
                if _do_token_mag_log:
                    _token_log.append((f"img_start_cam{_img_idx}", image_start_token))

            # 标准 SigLIP → connector → 64 image tokens
            img_emb = self.vlm_with_expert.embed_image(img)          # (B, 64, 960)

            # ============================================================
            # RGB Token Compression (env var 控制, 默认开)
            # ----------------------------------------------------------------
            # RGB_COMPRESS=N (perfect square 4/9/16/25/36/49) → 用 adaptive
            #   avg pool 把 8×8=64 token 压缩成 √N × √N = N 个 token.
            # 默认 RGB_COMPRESS=9 (8×8 → 3×3, 节省 86% prefix 长度).
            # 设 RGB_COMPRESS=0 关闭压缩 (用 vanilla 64 token, 跟 SmolVLA 论文一致).
            # 设 RGB_COMPRESS=64 也等价不压缩 (8×8 → 8×8, no-op).
            #
            # Why pool (not learn): connector 已经把 raw 特征压缩好了, 我们
            #   再压一次只用 average pool 不引入新参数, 也不破坏 SmolVLA 预训练.
            # Why default 9: 9 token 让 batch size 能拉到 80+ (比 64 token 的
            #   bs=20 有 4× 余量), 适合做 PCD ablation 跑得快. SmolVLA 论文
            #   也有早期实验用 4-9 token 做 token-efficient 配置.
            # ============================================================
            try:
                rgb_compress = int(os.environ.get("RGB_COMPRESS", "9"))
            except ValueError:
                rgb_compress = 9
            if rgb_compress > 0:
                B_rgb, N_rgb, D_rgb = img_emb.shape  # 期望 N_rgb=64
                # 输入必须是 perfect square (8×8=64)
                src_grid = int(N_rgb ** 0.5)
                tgt_grid = int(rgb_compress ** 0.5)
                if (src_grid * src_grid == N_rgb
                        and tgt_grid * tgt_grid == rgb_compress
                        and tgt_grid <= src_grid):
                    # (B, N, D) → (B, D, src_grid, src_grid) → pool → (B, D, tgt, tgt) → (B, tgt², D)
                    img_emb_grid = img_emb.transpose(1, 2).reshape(B_rgb, D_rgb, src_grid, src_grid)
                    img_emb_grid = F.adaptive_avg_pool2d(img_emb_grid, (tgt_grid, tgt_grid))
                    img_emb = img_emb_grid.reshape(B_rgb, D_rgb, rgb_compress).transpose(1, 2)

                    # ---- Magnitude compensation ----
                    # Avg pool 在 src/tgt 比例下让 std 下降 √(src/tgt) 倍.
                    # 不补偿的话, RGB token 在主干 attention 里 magnitude 会变弱,
                    # 影响 fair comparison vs baseline A.
                    # 补偿系数 = √(每个 target cell 平均的 source 像素数)
                    pool_factor = (src_grid / tgt_grid)
                    img_emb = img_emb * pool_factor

                    if not getattr(self, "_rgb_compress_logged", False):
                        print(
                            f"[RGB-Compress] enabled: {N_rgb} tokens → {rgb_compress} tokens "
                            f"({src_grid}×{src_grid} → {tgt_grid}×{tgt_grid} adaptive_avg_pool2d, "
                            f"magnitude × {pool_factor:.2f} compensation)"
                        )
                        self._rgb_compress_logged = True
                else:
                    if not getattr(self, "_rgb_compress_warned", False):
                        print(
                            f"[RGB-Compress] WARN: invalid config "
                            f"(N_rgb={N_rgb}, rgb_compress={rgb_compress}). "
                            f"src_grid={src_grid}, tgt_grid={tgt_grid}. Falling back to no compression."
                        )
                        self._rgb_compress_warned = True

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            if _do_token_mag_log:
                _token_log.append((f"img_emb_cam{_img_idx}", img_emb))

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
                if _do_token_mag_log:
                    _token_log.append((f"img_end_cam{_img_idx}", image_end_token))

            # ===================================================================
            # PCD token injection (per camera, paired with RGB)
            # ----------------------------------------------------------------
            # 在每个 image 的 RGB tokens (+ optional image_end_token) 之后插入
            # 一个 PCD token, 对应同一个相机的 depth. 这样空间语义对齐:
            #   [agent_RGB(64) | agent_PCD(1) | wrist_RGB(64) | wrist_PCD(1) | lang ...]
            #
            # ADD_PCD=0:                  跳过, 不加
            # ADD_PCD=1, USE_PCD=0:       加 placeholder token
            #   - PCD_LEARNABLE_PLACEHOLDER=1 (默认): 加 learnable per-cam placeholder
            #   - PCD_LEARNABLE_PLACEHOLDER=0:        加 zero 占位 (legacy, 主干被毒化)
            # ADD_PCD=1, USE_PCD=1:       加真 PCD token (PointNet 算)
            #
            # 修复 (vs 上版本):
            #   1) 用 learnable placeholder 代替 zero — zero token 会让 attention
            #      budget 被无效消耗, 主干 RGB 间 attention 被稀释, 是个隐性 bug.
            #      learnable placeholder 让主干能学到 "这个槽位是空的" 的语义,
            #      不破坏其他 token 间的 attention 强度.
            #   2) PCD token 加 sqrt(d) magnitude alignment (跟 RGB / lang 一致).
            # ===================================================================
            if self.add_pcd:
                B_cam = img.shape[0]
                vla_hidden = self.vlm_with_expert.config.text_config.hidden_size
                target_dtype = embs[-1].dtype
                target_device = embs[-1].device

                is_real_pcd = (
                    self.use_pcd
                    and depths is not None
                    and _img_idx < len(depths)
                    and depths[_img_idx] is not None
                )

                if is_real_pcd:
                    d = depths[_img_idx]
                    if d.ndim == 4 and d.shape[1] == 1:
                        d = d.squeeze(1)
                    encoder = self.pcd_encoder_agent if _img_idx == 0 else self.pcd_encoder_wrist
                    pcd_token = encoder(d, target_dtype=target_dtype)  # (B, 1, vla_hidden)

                    # 强制 contiguous + dtype 一致, 防止主干 attention 因为 token 内存
                    # 布局或 dtype 异常 fallback 到非 fused path, 显存暴涨.
                    pcd_token = pcd_token.to(dtype=target_dtype).contiguous()

                    # Magnitude alignment: 跟 RGB / lang 一样乘 sqrt(d).
                    # PCD → Linear 输出 std 大约是 1, sqrt(960)≈31 后跟 RGB 一致.
                    pcd_token = pcd_token * torch.tensor(
                        vla_hidden**0.5, dtype=pcd_token.dtype, device=pcd_token.device
                    )
                else:
                    if self.use_learnable_placeholder:
                        # Learnable placeholder, broadcast 到 batch (shape (1, 1, D))
                        ph = (
                            self.pcd_placeholder_agent
                            if _img_idx == 0
                            else self.pcd_placeholder_wrist
                        )
                        # Magnitude alignment (跟 real PCD 保持一致)
                        ph = ph * torch.tensor(
                            vla_hidden**0.5, dtype=ph.dtype, device=ph.device
                        )
                        pcd_token = ph.expand(B_cam, -1, -1).to(
                            dtype=target_dtype, device=target_device
                        )
                    else:
                        # Legacy: zero token (有毒化 attention 的隐性 bug)
                        pcd_token = torch.zeros(
                            B_cam, 1, vla_hidden, dtype=target_dtype, device=target_device
                        )

                pcd_mask = torch.ones(B_cam, 1, dtype=torch.bool, device=target_device)

                # 记录 PCD token 在 prefix 中的精确索引 (用于 attention probe)
                pcd_token_idx = sum(e.shape[1] for e in embs)
                if not hasattr(self, "_last_pcd_indices"):
                    self._last_pcd_indices = []
                if _img_idx == 0:
                    self._last_pcd_indices = []  # 重置 (每次 forward 都重新填)
                self._last_pcd_indices.append(pcd_token_idx)

                embs.append(pcd_token)
                pad_masks.append(pcd_mask)
                att_masks += [0] * 1
                if _do_token_mag_log:
                    pcd_label = f"pcd_REAL_cam{_img_idx}" if is_real_pcd else f"pcd_PLACEHOLDER_cam{_img_idx}"
                    _token_log.append((pcd_label, pcd_token))

                if not self._pcd_check_done and _img_idx == 0:
                    if is_real_pcd:
                        kind = "REAL PCT"
                    elif self.use_learnable_placeholder:
                        kind = "LEARNABLE placeholder"
                    else:
                        kind = "ZERO placeholder (legacy)"
                    print(
                        f"[PCD] First-forward inject: 1 {kind} token after each RGB. "
                        f"(use_pcd={int(self.use_pcd)}, add_pcd={int(self.add_pcd)}, "
                        f"learnable_placeholder={int(self.use_learnable_placeholder)})"
                    )
                    self._pcd_check_done = True

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        if _do_token_mag_log:
            _token_log.append(("lang_emb", lang_emb))

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        if _do_token_mag_log:
            _token_log.append(("state_emb", state_emb))
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        # ============================================================
        # Token magnitude diagnostic (first forward only)
        # 各 segment magnitude/std 是否一致, 直接看出来.
        # 期望: 所有 segment magnitude 接近 (RGB/lang 经 sqrt(d) 缩放, PCD 也乘了 sqrt(d)).
        # state magnitude 通常较小 (没乘 sqrt(d)).
        # ============================================================
        if _do_token_mag_log:
            print("[TokenMag] embed_prefix segments:")
            for label, t in _token_log:
                with torch.no_grad():
                    t_f = t.float()
                    print(
                        f"  {label:30s} shape={str(tuple(t.shape)):20s} "
                        f"std={t_f.std().item():.4f}  "
                        f"abs_mean={t_f.abs().mean().item():.4f}  "
                        f"per_token_norm={t_f.norm(dim=-1).mean().item():.4f}  "
                        f"dtype={t.dtype}"
                    )
            # 一次性 flag, 后续 forward 不再打印
            self._token_mag_log_done = True

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # Token magnitude diagnostic for suffix (first forward only)
        if not hasattr(self, "_suffix_mag_log_done"):
            with torch.no_grad():
                emb_f = embs.float()
                print(
                    f"[TokenMag] embed_suffix (action expert input): "
                    f"shape={str(tuple(embs.shape))} "
                    f"std={emb_f.std().item():.4f}  "
                    f"abs_mean={emb_f.abs().mean().item():.4f}  "
                    f"per_token_norm={emb_f.norm(dim=-1).mean().item():.4f}  "
                    f"dtype={embs.dtype}"
                )
            self._suffix_mag_log_done = True

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, depths=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, depths=depths
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # ============================================================
        # ATTENTION PROBE: 测主干每层 attention 对 PCD token 的 attention weight
        # ----------------------------------------------------------------
        # ATTN_PROBE_FREQ 环境变量控制采样频率 (默认 200 = 每 200 forward 探测一次).
        # 设 0 关闭. 探测时 attention probs 被 detach 缓存到 vlm_with_expert,
        # 不进梯度图, 但占额外显存 (B × num_layers × num_heads × seq² × 4 bytes).
        # ============================================================
        probe_attn_now = False
        try:
            probe_freq = int(os.environ.get("ATTN_PROBE_FREQ", "2000"))
        except ValueError:
            probe_freq = 0
        if probe_freq > 0 and self.add_pcd:
            if not hasattr(self, "_attn_probe_count"):
                self._attn_probe_count = 0
            self._attn_probe_count += 1
            probe_attn_now = (self._attn_probe_count % probe_freq == 0)
        if probe_attn_now:
            self.vlm_with_expert._probe_attn = True
            self.vlm_with_expert._probe_attn_buf = []
        # ============================================================

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        # ============================================================
        # ATTENTION PROBE: 分析缓存的 probs, print 每层对 PCD 列的关注度
        # ----------------------------------------------------------------
        # probs shape per layer: (B, num_heads, total_seq, total_seq)
        # PCD 列在 prefix 中的位置由 self._last_pcd_indices 给出.
        # 我们看 EXPERT (action expert) query → PCD key 的 attention,
        # 因为这才是"主干在生成 action 时多看 PCD 的程度".
        # ============================================================
        if probe_attn_now and hasattr(self.vlm_with_expert, "_probe_attn_buf"):
            self._analyze_pcd_attn(
                self.vlm_with_expert._probe_attn_buf,
                prefix_embs.shape[1],
                suffix_embs.shape[1],
            )
            # 清理: 关探针, 释放 buf, 避免下次默认 forward 还在缓存
            self.vlm_with_expert._probe_attn = False
            self.vlm_with_expert._probe_attn_buf = []
        # ============================================================

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def _analyze_pcd_attn(self, probs_list, prefix_len, suffix_len):
        """
        分析 attention probs, print 每层对 PCD 列的关注度.

        probs_list: list of (B, num_heads, seq, seq), 一个 layer 一个 tensor.
                    注意 seq 可能 < prefix_len + suffix_len (forward_attn_layer
                    在 fill_kv_cache=False 模式下可能切短).
        prefix_len: prefix 长度 (含 RGB / image_end / PCD / lang / state)
        suffix_len: suffix 长度 (action expert tokens, chunk_size 个)
        """
        if not probs_list or not getattr(self, "_last_pcd_indices", None):
            return

        pcd_indices = self._last_pcd_indices
        suffix_start = prefix_len  # expert query 起始位置 (理想情况)

        # ---- 一次性 sanity check: 跳过形状不符或 PCD 越界的层 ----
        # probs[k] 第三/第四维可能比 prefix_len 还小 (early layer w/ truncated mask).
        # 也可能比 prefix_len+suffix_len 大 (kv cache 累积).
        # 我们要求最小 shape 是 max(pcd_indices)+1, 否则跳过该层.
        max_pcd_idx = max(pcd_indices)

        non_pcd_prefix_rows_full = [
            i for i in range(prefix_len) if i not in pcd_indices
        ]

        uniform = 1.0 / prefix_len
        print(f"[AttnProbe] PCD indices in prefix: {pcd_indices} "
              f"(prefix_len={prefix_len}, suffix_len={suffix_len}, uniform={uniform:.5f})")

        for layer_idx, probs in enumerate(probs_list):
            seq_len_k = probs.shape[-1]   # key 维度
            seq_len_q = probs.shape[-2]   # query 维度

            # 跳过 PCD column 越界
            if seq_len_k <= max_pcd_idx:
                print(f"[AttnProbe] L{layer_idx}: SKIP "
                      f"(seq_len_k={seq_len_k} <= max_pcd_idx={max_pcd_idx})")
                continue

            # 限定 prefix rows 在合法范围内
            non_pcd_rows_safe = [
                i for i in non_pcd_prefix_rows_full if i < seq_len_q
            ]

            # 1) Expert query → PCD key
            #    expert 行可能在 probs 里完全不存在 (seq_len_q <= prefix_len, 比如
            #    fill_kv_cache=True path), 也可能存在.
            if seq_len_q > suffix_start:
                expert_q_end = min(suffix_start + suffix_len, seq_len_q)
                expert_block = probs[:, :, suffix_start:expert_q_end, :][
                    :, :, :, pcd_indices
                ]
                # NaN-safe: 用 nanmean 而不是 mean (causal mask 早期行可能全 0/NaN)
                expert_to_pcd = torch.nanmean(expert_block.float())
            else:
                expert_to_pcd = torch.tensor(float("nan"))

            # 2) Prefix query → PCD key
            if non_pcd_rows_safe:
                prefix_block = probs[:, :, non_pcd_rows_safe, :][
                    :, :, :, pcd_indices
                ]
                prefix_to_pcd = torch.nanmean(prefix_block.float())
            else:
                prefix_to_pcd = torch.tensor(float("nan"))

            e_val = expert_to_pcd.item()
            p_val = prefix_to_pcd.item()
            e_str = (
                f"expert→PCD={e_val:.5f} ({e_val / uniform:.2f}× uniform)"
                if e_val == e_val   # not NaN
                else "expert→PCD=N/A (no expert rows in this layer)"
            )
            p_str = (
                f"prefix→PCD={p_val:.5f} ({p_val / uniform:.2f}× uniform)"
                if p_val == p_val
                else "prefix→PCD=N/A"
            )
            print(f"[AttnProbe] L{layer_idx}: {e_str}  {p_str}")

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        depths=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, depths=depths
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t