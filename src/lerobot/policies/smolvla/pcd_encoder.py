"""
PointNet Depth Encoder for SmolVLA — 严格复刻 3D-CAVLA
========================================================
完全跟 3D-CAVLA (Bhat et al., CVPR 2025 Workshop) 设计一致, 不加任何额外模块.

Pipeline (跟 3D-CAVLA 论文 + depth_projectors.py 一一对应):
  depth map (B, H, W)
  → 反投影 (camera intrinsics from fovy)
  → point cloud (B, N=4096, 3)
  → STN3d (T-Net, 3x3 transform, identity init)
  → bmm(point_cloud, transform)
  → Conv1d(3→64) + BN + ReLU       (跟 3D-CAVLA 一致)
  → Conv1d(64→128) + BN + ReLU
  → Conv1d(128→1024) + BN
  → max-pool over points → (B, 1024)
  → Linear(1024 → vla_hidden=960)
  → 1 token (B, 1, vla_hidden) — 直接送 SmolVLM prefix

跟 3D-CAVLA 的唯一差异:
  - LLM hidden: 我 960 (SmolVLM), 他 4096 (LLaMA-2 7B)

为什么用 BN (跟 3D-CAVLA 一致, 推翻之前用 GN 的决定):
  PointNet 张量是 (B, C, N) 形式, N=4096. BN1d 在 (B, N) 维度上算 mean/var,
  effective batch = B × N = 40 × 4096 = 163,840 per channel — 极稳定.
  GN 在每样本独立 norm, 对 LIBERO 这种 input variance 极小的场景反而把
  细微差异锐化, 导致 PointNet weights 量级被迫放大, gradient 爆炸.

  ⚠️ 用 BN 的代价: 训练/推理切换时务必调用 .train() / .eval()
  - train() 时 BN 用 batch 内 stats, 同时更新 running stats
  - eval() 时 BN 用 running stats (frozen), 单样本推理也安全
  - 如果忘记切, 推理时 batch=1 BN 会爆炸

不再有的东西 (跟之前我加的"安全保护"切割):
  ✗ token_norm (LayerNorm)         — 抹平 magnitude, 丢 task-relevant 信号
  ✗ token_scale                    — 多余, raw PointNet output magnitude 自然
  ✗ modality_emb                   — 不对称设计 (只 PCD 加, RGB 不加), 逻辑有缺陷
  ✗ gate (tanh)                    — 复杂化, 引入 freeze timing 等额外超参
  ✗ z_min / z_max filter           — 3D-CAVLA 不过滤; 让 PointNet 自己学

Reference:
  - 3D-CAVLA: arxiv 2505.05800
  - depth_projectors.py: https://github.com/vineet2104/3dcavla/blob/main/prismatic/models/depth_projectors.py
  - PointNet (Qi et al. 2017): arxiv 1612.00593
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Camera intrinsics utility
# ============================================================================
def _fovy_to_fx_fy(fovy_deg: float, img_size: int) -> tuple[float, float]:
    """从 vertical FOV (degrees) 算出 fx/fy (pixels). 假设方形像素."""
    fovy_rad = math.radians(fovy_deg)
    f = (img_size / 2.0) / math.tan(fovy_rad / 2.0)
    return f, f


# ============================================================================
# STN3d: T-Net (跟 3D-CAVLA 一致, 输出 3x3 transform, identity init)
# ============================================================================
class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # fc3 zero-init + identity 残差 → 训练初期 transform=I
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        iden = torch.eye(3, dtype=torch.float32).flatten()
        self.register_buffer("iden", iden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, N) → (B, 3, 3)."""
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x + self.iden.unsqueeze(0).expand(B, -1)
        return x.view(B, 3, 3)


# ============================================================================
# PointNetEncoder: PointNet 主体 (跟 3D-CAVLA PointNetfeat 一致)
# ============================================================================
class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 3) → (B, 1024)."""
        x = points.transpose(1, 2).contiguous()  # (B, 3, N)

        # T-Net align (3D-CAVLA 也有这步)
        trans = self.stn(x)
        x = x.transpose(1, 2)        # (B, N, 3)
        x = torch.bmm(x, trans)
        x = x.transpose(1, 2)        # (B, 3, N)

        # PointNet MLP (跟 3D-CAVLA 完全一致)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # 不 ReLU before max-pool (跟 3D-CAVLA 一致)
        x = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)
        return x


# ============================================================================
# PCDDepthEncoder: 完整 wrapper (3D-CAVLA mimic)
# ============================================================================
class PCDDepthEncoder(nn.Module):
    """
    Pipeline: depth (B, H, W) → point cloud → PointNet → token (B, 1, vla_hidden)

    严格复刻 3D-CAVLA. 唯一外部参数:
        vla_hidden: SmolVLM hidden dim (默认 960, LIBERO 上 SmolVLA 的值)
        fovy_deg:   vertical FOV in degrees (LIBERO: agent=45, wrist=75)
        img_size:   反投影虚拟图像尺寸 (默认 512)
        num_points: PointNet 输入点数 (默认 4096)
        cam_label:  "agent" 或 "wrist" — 仅用于 debug print 区分
    """

    def __init__(
        self,
        vla_hidden: int = 960,
        fovy_deg: float = 45.0,
        img_size: int = 512,
        num_points: int = 4096,
        cam_label: str = "cam",
    ):
        super().__init__()
        self.vla_hidden = vla_hidden
        self.fovy_deg = fovy_deg
        self.img_size = img_size
        self.num_points = num_points
        self.cam_label = cam_label

        # ---- 反投影 grid ----
        self.grid_size = int(math.sqrt(num_points))
        assert self.grid_size * self.grid_size == num_points, \
            f"num_points={num_points} must be a perfect square"

        fx, fy = _fovy_to_fx_fy(fovy_deg, img_size)
        cx, cy = img_size / 2.0, img_size / 2.0
        self.register_buffer("fx", torch.tensor(fx, dtype=torch.float32))
        self.register_buffer("fy", torch.tensor(fy, dtype=torch.float32))
        self.register_buffer("cx", torch.tensor(cx, dtype=torch.float32))
        self.register_buffer("cy", torch.tensor(cy, dtype=torch.float32))

        u_idx = torch.arange(self.grid_size, dtype=torch.float32)
        v_idx = torch.arange(self.grid_size, dtype=torch.float32)
        u_centers = (u_idx + 0.5) * (img_size / self.grid_size)
        v_centers = (v_idx + 0.5) * (img_size / self.grid_size)
        vv, uu = torch.meshgrid(v_centers, u_centers, indexing="ij")
        self.register_buffer("u_grid", uu.flatten())
        self.register_buffer("v_grid", vv.flatten())

        # ---- PointNet body (跟 3D-CAVLA 一致, BatchNorm1d) ----
        self.pointnet = PointNetEncoder()

        # ---- Linear projection 1024 → vla_hidden (跟 3D-CAVLA 一致) ----
        # 3D-CAVLA 用 nn.Linear(1024, 4096), 我们 nn.Linear(1024, 960)
        self.proj = nn.Linear(1024, vla_hidden)

        self._dbg_done = False

        print(
            f"[PCDEncoder.{cam_label}] 3D-CAVLA mimic config (BatchNorm1d): "
            f"fovy={fovy_deg}° img_size={img_size} num_points={num_points} → "
            f"PointNet → Linear(1024, {vla_hidden}) → 1 token"
        )

    def _backproject(self, depth_m: torch.Tensor) -> torch.Tensor:
        """depth_m: (B, H, W) → (B, num_points, 3) in camera frame.

        不做 z 过滤 — 跟 3D-CAVLA 一致, 让 PointNet 自己学.
        """
        B, H, W = depth_m.shape
        device = depth_m.device

        if H != self.img_size or W != self.img_size:
            depth_m = F.interpolate(
                depth_m.unsqueeze(1).float(),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        u = self.u_grid.to(device)
        v = self.v_grid.to(device)
        u_norm = (u / (self.img_size - 1)) * 2 - 1
        v_norm = (v / (self.img_size - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, self.num_points, 2).contiguous()

        depth_input = depth_m.unsqueeze(1).float()
        z_per_point = F.grid_sample(
            depth_input, grid, mode="bilinear", align_corners=True
        ).squeeze(1).squeeze(1)  # (B, num_points)

        u_b = u.unsqueeze(0).expand(B, self.num_points)
        v_b = v.unsqueeze(0).expand(B, self.num_points)
        x = (u_b - self.cx) * z_per_point / self.fx
        y = (v_b - self.cy) * z_per_point / self.fy
        z = z_per_point
        xyz = torch.stack([x, y, z], dim=-1)  # (B, num_points, 3)

        return xyz

    def forward(self, depth_m: torch.Tensor, target_dtype: torch.dtype = None) -> torch.Tensor:
        """
        depth_m: (B, H, W) depth in meters
        return:  (B, 1, vla_hidden) — 直接送 prefix, 没 norm/scale/gate/mod_emb
        """
        depth_f32 = depth_m.float()

        points = self._backproject(depth_f32)         # (B, N, 3)
        feat = self.pointnet(points)                  # (B, 1024)
        token = self.proj(feat).unsqueeze(1)          # (B, 1, vla_hidden)
        if target_dtype is not None:
            token = token.to(target_dtype)

        # 一次性 debug print (验证用, 不影响计算)
        if not self._dbg_done:
            with torch.no_grad():
                token_std = token.float().std().item()
                token_norm = token.float().norm(dim=-1).mean().item()
                points_std = points.std().item()
                feat_std = feat.std().item()
                z = points[..., 2]
                z_min_obs = z.min().item()
                z_max_obs = z.max().item()
                z_mean_obs = z.mean().item()
                print(
                    f"[PCDEncoder.{self.cam_label}] depth_in: shape={tuple(depth_m.shape)}, "
                    f"min={depth_m.float().min().item():.3f} max={depth_m.float().max().item():.3f}"
                )
                print(
                    f"[PCDEncoder.{self.cam_label}] points: shape={tuple(points.shape)} "
                    f"std={points_std:.3f}, z=[{z_min_obs:.3f}, {z_max_obs:.3f}] z_mean={z_mean_obs:.3f}"
                )
                print(
                    f"[PCDEncoder.{self.cam_label}] feat std={feat_std:.3f}  →  "
                    f"token std={token_std:.4f} norm={token_norm:.4f}"
                )
            self._dbg_done = True

        return token

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)