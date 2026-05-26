"""
PCT 退化版 — 只保留 input embedding + max-pool + Linear, 不要 attention.
等效于 PointNet, 但用 PCT 的代码结构.

如果用这个版本 batch=8 显存能跑到 vanilla + ~1 GB (大概 6-8 GB), 说明
是 attention 部分引起的显存爆炸.
如果显存还是 30+ GB, 说明跟 attention 无关, 是别的问题.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fovy_to_fx_fy(fovy_deg: float, img_size: int) -> tuple[float, float]:
    fovy_rad = math.radians(fovy_deg)
    f = (img_size / 2.0) / math.tan(fovy_rad / 2.0)
    return f, f


class PCTEncoder(nn.Module):
    """退化版 PCT, 移除 attention, 只保留 input embedding + max-pool."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.input_embed = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, 1),
            nn.GroupNorm(num_groups=8, num_channels=embed_dim),
            nn.GELU(),
        )

        # 直接从 embed_dim 升到 1024
        self.feat_proj = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.GroupNorm(num_groups=32, num_channels=1024),
            nn.GELU(),
        )

        # MA-Pool projection (跟原 PCT 一致, max + avg)
        self.ma_pool_proj = nn.Linear(2048, 1024)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 3) → (B, 1024)."""
        x = points.transpose(1, 2).contiguous()  # (B, 3, N)
        x = self.input_embed(x)                   # (B, embed_dim, N)
        x = self.feat_proj(x)                     # (B, 1024, N)

        max_pool = torch.max(x, dim=2)[0]
        avg_pool = torch.mean(x, dim=2)
        global_feat = torch.cat([max_pool, avg_pool], dim=1)
        global_feat = self.ma_pool_proj(global_feat)
        return global_feat


class PCDDepthEncoder(nn.Module):
    """Hot-swap, 但内部 PCT 退化成 no-attention 版本."""

    def __init__(
        self,
        vla_hidden: int = 960,
        fovy_deg: float = 45.0,
        img_size: int = 512,
        num_points: int = 1024,
        cam_label: str = "cam",
    ):
        super().__init__()
        self.vla_hidden = vla_hidden
        self.fovy_deg = fovy_deg
        self.img_size = img_size
        self.num_points = num_points
        self.cam_label = cam_label

        self.grid_size = int(math.sqrt(num_points))
        assert self.grid_size * self.grid_size == num_points

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

        self.pct = PCTEncoder()
        self.proj = nn.Linear(1024, vla_hidden)

        self._dbg_done = False

        print(
            f"[PCDEncoder.{cam_label}] PCT-DEGRADED config (no attention): "
            f"fovy={fovy_deg}° img_size={img_size} num_points={num_points} → "
            f"Conv embed + maxpool + Linear → 1 token"
        )

    @property
    def num_tokens(self) -> int:
        return 1

    def _backproject(self, depth_m: torch.Tensor) -> torch.Tensor:
        B, H, W = depth_m.shape
        if H != self.img_size or W != self.img_size:
            depth_m = F.interpolate(
                depth_m.unsqueeze(1).float(),
                size=(self.img_size, self.img_size),
                mode="nearest-exact",
            ).squeeze(1)

        u_norm = (self.u_grid / (self.img_size - 1)) * 2 - 1
        v_norm = (self.v_grid / (self.img_size - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, self.num_points, 2).contiguous()

        depth_input = depth_m.unsqueeze(1).float()
        z_per_point = F.grid_sample(
            depth_input, grid, mode="nearest", align_corners=True
        ).squeeze(1).squeeze(1)

        u_b = self.u_grid.unsqueeze(0).expand(B, self.num_points)
        v_b = self.v_grid.unsqueeze(0).expand(B, self.num_points)
        x = (u_b - self.cx) * z_per_point / self.fx
        y = (v_b - self.cy) * z_per_point / self.fy
        z = z_per_point
        xyz = torch.stack([x, y, z], dim=-1)
        return xyz

    def forward(self, depth_m: torch.Tensor, target_dtype: torch.dtype = None) -> torch.Tensor:
        depth_f32 = depth_m.float()
        points = self._backproject(depth_f32)

        compute_dtype = target_dtype if target_dtype is not None else torch.bfloat16
        points = points.to(compute_dtype)

        with torch.amp.autocast("cuda", dtype=compute_dtype, enabled=torch.cuda.is_available()):
            feat = self.pct(points)
            token = self.proj(feat).unsqueeze(1)

        if target_dtype is not None:
            token = token.to(target_dtype)

        if not self._dbg_done:
            with torch.no_grad():
                token_std = token.float().std().item()
                token_norm = token.float().norm(dim=-1).mean().item()
                print(
                    f"[PCDEncoder.{self.cam_label}] DEGRADED — out tokens={tuple(token.shape)} "
                    f"std={token_std:.4f} norm={token_norm:.4f}"
                )
            self._dbg_done = True

        return token

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
