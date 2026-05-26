"""
PCT (Point Cloud Transformer) Depth Encoder for SmolVLA
=========================================================
PointNet transformer 风格替代方案. Hot-swap 兼容 pcd_encoder.py (相同输入/输出 shape).

Reference: PCT: Point Cloud Transformer (Guo et al., CVM 2021)
  - https://arxiv.org/abs/2012.09688

跟 pcd_encoder.py (3D-CAVLA mimic) 的核心区别:
  - PointNet (per-point MLP + max-pool, 无远距离点交互)
  - PCT (input embed + self-attention × 4, attention 直接捕获远距离关系)
  - PCT 没用 STN3d (attention 学 spatial alignment)
  - 参数量: PointNet ~1.9M vs PCT ~4.2M / cam

显存提示: PCT 在 Blackwell (sm_120, RTX 5090) 上不可用 — FlashAttention 还没适配,
SDPA fallback math kernel 物化 (B, heads, N, N) attention map, 显存爆炸.
推荐在 Ampere+ (sm_80+, A100/A6000/3090/4090/4080) 上跑.

接口兼容:
  PCDDepthEncoder(vla_hidden=960, fovy_deg=45.0, img_size=512, num_points=1024, cam_label='cam')
  forward(depth, target_dtype) → (B, 1, vla_hidden)
  .num_tokens: 1
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Camera intrinsics utility
# ============================================================================
def _fovy_to_fx_fy(fovy_deg: float, img_size: int) -> tuple[float, float]:
    fovy_rad = math.radians(fovy_deg)
    f = (img_size / 2.0) / math.tan(fovy_rad / 2.0)
    return f, f


# ============================================================================
# Self-Attention Block (PCT-style, simplified)
# ============================================================================
class SelfAttentionBlock(nn.Module):
    """
    Single self-attention block on point features.
    Input/Output: (B, C, N) — N points, C channels.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels={channels} must be divisible by num_heads={num_heads}"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv1d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv1d(channels, channels, 1, bias=False)

        # FFN — point-wise 2-layer MLP
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, 1),
        )

        # GroupNorm 不依赖 batch size, train/eval 数值一致
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N) → (B, C, N)."""
        B, C, N = x.shape

        # --- Self-attention block (pre-norm style) ---
        residual = x
        h = self.norm1(x)
        q = self.q_proj(h).view(B, self.num_heads, self.head_dim, N)
        k = self.k_proj(h).view(B, self.num_heads, self.head_dim, N)
        v = self.v_proj(h).view(B, self.num_heads, self.head_dim, N)

        # SDPA 期望 (B, num_heads, N, head_dim)
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # 强制 q/k/v 用 bf16, GroupNorm 输出可能 fp32, 这里 cast bf16
        # bf16 attention 比 fp32 显存少一半, 也更可能走 FlashAttention 路径
        if q.dtype == torch.float32:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        # PyTorch SDPA 自动选 backend (FlashAttn / MemEfficient / math)
        # 在 Ampere+ 上自动启用 FlashAttention, 显存 O(N)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # cast 回原 dtype (匹配 out_proj 跟下游 GroupNorm)
        out = out.to(x.dtype)

        # out: (B, num_heads, N, head_dim) → (B, C, N)
        out = out.transpose(2, 3).contiguous().view(B, C, N)
        out = self.out_proj(out)
        x = residual + out

        # --- FFN block ---
        residual = x
        h = self.norm2(x)
        x = residual + self.ffn(h)

        return x


# ============================================================================
# PCT Encoder Body (4 stacked attention blocks + multi-scale concat)
# ============================================================================
class PCTEncoder(nn.Module):
    """
    PCT-style point cloud encoder.

    Input: (B, N, 3) point cloud
    Output: (B, 1024) global feature
    """

    def __init__(self, num_attention_blocks: int = 4, embed_dim: int = 128):
        super().__init__()
        self.num_blocks = num_attention_blocks
        self.embed_dim = embed_dim

        # Input embedding: 3 → 64 → 128
        self.input_embed = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, 1),
            nn.GroupNorm(num_groups=8, num_channels=embed_dim),
            nn.GELU(),
        )

        # 4 attention blocks
        self.attn_blocks = nn.ModuleList([
            SelfAttentionBlock(channels=embed_dim, num_heads=4)
            for _ in range(num_attention_blocks)
        ])

        # Concat 4 layer outputs → project to 1024
        self.concat_proj = nn.Sequential(
            nn.Conv1d(embed_dim * num_attention_blocks, 1024, 1),
            nn.GroupNorm(num_groups=32, num_channels=1024),
            nn.GELU(),
        )

        # MA-Pool projection: (max + avg) = 2048 → 1024
        self.ma_pool_proj = nn.Linear(2048, 1024)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 3) → (B, 1024).

        Attention blocks 用 gradient checkpointing: forward 不存中间 activation,
        backward 时重算. 显存 -50%, 训练慢 ~30%. 这是 PCT 在 SmolVLA 训练上下文里
        显存可控的关键 — 4 个 attention block × 双 cam, activation 多, 不 checkpoint
        会让 batch=8 显存爆到 30 GB.
        """
        x = points.transpose(1, 2).contiguous()  # (B, 3, N)

        # Input embedding (不 checkpoint, 它便宜)
        x = self.input_embed(x)  # (B, embed_dim, N)

        # 4 attention blocks 用 gradient checkpointing
        attn_outputs = []
        use_checkpoint = self.training and x.requires_grad
        for block in self.attn_blocks:
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x = block(x)
            attn_outputs.append(x)

        # Concat along channel: (B, embed_dim * 4, N)
        concat = torch.cat(attn_outputs, dim=1)
        concat = self.concat_proj(concat)  # (B, 1024, N)

        # MA-Pool: max + avg
        max_pool = torch.max(concat, dim=2)[0]  # (B, 1024)
        avg_pool = torch.mean(concat, dim=2)     # (B, 1024)
        global_feat = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2048)
        global_feat = self.ma_pool_proj(global_feat)          # (B, 1024)

        return global_feat


# ============================================================================
# PCDDepthEncoder: hot-swap drop-in for pcd_encoder.PCDDepthEncoder
# ============================================================================
class PCDDepthEncoder(nn.Module):
    """
    Hot-swap drop-in for pcd_encoder.PCDDepthEncoder.

    Input/output shape:
      forward(depth_m, target_dtype) → (B, 1, vla_hidden)
      .num_tokens: 1
    """

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

        # ---- PCT encoder ----
        self.pct = PCTEncoder()

        # ---- Projection 1024 → vla_hidden ----
        self.proj = nn.Linear(1024, vla_hidden)
        # PCT 输出会因为 GroupNorm + Linear bias 累积偏离 0-centered,
        # 加 LayerNorm 强制 mean=0, std=1, 让外层 sqrt(d) magnitude alignment 后
        # 跟 RGB token 量级对齐 (per_token_norm ≈ 30, 而非 30000).
        self.out_norm = nn.LayerNorm(vla_hidden)

        self._dbg_done = False

        # FREEZE_PCT=1: 冻结 PCT 主体 (input_embed + attn_blocks + concat_proj + ma_pool_proj),
        # 只保留 proj (Linear 1024→960) + out_norm (LayerNorm) trainable.
        # 用于阶段 2 微调: PCT 几何特征已学好, 固定它, 只让连接层 + 主干专家继续适应.
        # 好处: 不存 PCT backward activation, 显存大降, batch 能开回 32.
        self.freeze_pct = os.environ.get("FREEZE_PCT", "0") == "1"
        if self.freeze_pct:
            for p in self.pct.parameters():
                p.requires_grad = False
            print(
                f"[PCDEncoder.{cam_label}] FREEZE_PCT=1: PCT 主体已冻结 "
                f"(仅 proj + out_norm trainable)"
            )

        print(
            f"[PCDEncoder.{cam_label}] PCT-Transformer config: "
            f"fovy={fovy_deg}° img_size={img_size} num_points={num_points} → "
            f"PCT (4 attn blocks, embed_dim=128) → Linear(1024, {vla_hidden}) → 1 token "
            f"(freeze_pct={int(self.freeze_pct)})"
        )

    @property
    def num_tokens(self) -> int:
        return 1

    def _backproject(self, depth_m: torch.Tensor) -> torch.Tensor:
        """depth_m: (B, H, W) → (B, num_points, 3) in camera frame."""
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
        """
        depth_m: (B, H, W) depth in meters
        return:  (B, 1, vla_hidden)
        """
        # Backproject 必须 fp32 (depth 数值精度要求)
        depth_f32 = depth_m.float()
        points = self._backproject(depth_f32)         # (B, N, 3) fp32

        feat = self.pct(points)                        # (B, 1024)
        token = self.proj(feat).unsqueeze(1)           # (B, 1, vla_hidden)
        token = self.out_norm(token)                    # 强制 mean=0, std=1

        if target_dtype is not None:
            token = token.to(target_dtype)

        # 一次性 debug print
        if not self._dbg_done:
            with torch.no_grad():
                token_std = token.float().std().item()
                token_norm = token.float().norm(dim=-1).mean().item()
                points_std = points.float().std().item()
                feat_std = feat.float().std().item()
                z = points[..., 2]
                z_min_obs = z.float().min().item()
                z_max_obs = z.float().max().item()
                z_mean_obs = z.float().mean().item()
                print(
                    f"[PCDEncoder.{self.cam_label}] depth_in: shape={tuple(depth_m.shape)}, "
                    f"min={depth_m.float().min().item():.3f} max={depth_m.float().max().item():.3f}"
                )
                print(
                    f"[PCDEncoder.{self.cam_label}] points: shape={tuple(points.shape)} "
                    f"std={points_std:.3f}, z=[{z_min_obs:.3f}, {z_max_obs:.3f}] z_mean={z_mean_obs:.3f}"
                )
                print(
                    f"[PCDEncoder.{self.cam_label}] feat std={feat_std:.3f}  dtype={feat.dtype}  →  "
                    f"token std={token_std:.4f} dtype={token.dtype} norm={token_norm:.4f}"
                )
            self._dbg_done = True

        return token

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)