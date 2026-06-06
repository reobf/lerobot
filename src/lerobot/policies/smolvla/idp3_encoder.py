"""
iDP3 PointNet Depth Encoder for SmolVLA
========================================
Drop-in 替换 pcd_encoder.py / pcdpp_encoder.py — 类名/接口/方法完全一致.

跟 pcd_encoder (原版 PointNet) 的区别:
  - 去掉 T-Net (对 manipulation 任务有害, DP3 ablation Table VI)
  - 去掉 BatchNorm
  - 改用 multi-stage PointNet (每层 local + global concat reduce, 多层堆叠)
  - 用 LeakyReLU(0.0) 而非 ReLU
  - **输出 conv_out 直接出 vla_hidden=960**, 不需 Linear projection

参考: Ze et al., IROS 2025, iDP3 (multi_stage_pointnet.py).
PointVLA (arxiv 2503.07778) 也是用 iDP3 encoder 训 3D VLA.

Pipeline:
  depth (B, H, W)
    → 反投影 (camera intrinsics from fovy)
    → point cloud (B, N=4096, 3)
    → transpose → (B, 3, N)
    → Conv1d(3 → h_dim=128), LeakyReLU
    → 4 个 stage, 每 stage:
        Conv1d(h_dim → h_dim), LeakyReLU
        cat([y, y_global.expand], dim=1) → (B, 2*h_dim, N)
        Conv1d(2*h_dim → h_dim), LeakyReLU
        feat_list.append(y)
    → cat(feat_list) → (B, 4*h_dim=512, N)
    → Conv1d(512 → vla_hidden=960)
    → max-pool over points → (B, vla_hidden)
    → unsqueeze(1) → (B, 1, vla_hidden)

参数量估算 (h_dim=128, num_layers=4, out=960, 4096 points):
  conv_in: 3 → 128 = 384
  4 × (layers[i] 128→128 + global_layers[i] 256→128) = 4 × (16K + 32K) = 192K
  conv_out: 512 → 960 = 491K
  total ≈ 700K — 比 pcd_encoder 原版 (~1.9M) 少, 比 pcdpp 轻量 (~750K) 接近
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fovy_to_fx_fy(fovy_deg: float, img_size: int):
    """Convert vertical FOV (deg) to focal lengths fx=fy (assume square sensor)."""
    fovy_rad = math.radians(fovy_deg)
    fy = (img_size / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    return fx, fy


class _STNAlias(nn.Module):
    """Dummy STN wrapper: 让 modeling diag 代码 `pointnet.stn.fc3.weight` 能访问到.
    iDP3 没有 T-Net, 用一个已存在 Conv 层占位 (fc3 指向它的 weight).
    """
    def __init__(self, target_module):
        super().__init__()
        # 不注册为子模块 (避免重复占 state_dict), 只持有引用
        self._target = target_module

    @property
    def fc3(self):
        return self._target


class MultiStagePointNetEncoder(nn.Module):
    """
    iDP3 multi-stage PointNet (copied/adapted from
    YanjieZe/Improved-3D-Diffusion-Policy/diffusion_policy_3d/model/vision_3d/multi_stage_pointnet.py).

    每个 stage: local conv → global pool → cat → reduce, 多层后 concat 全部输出.
    """

    def __init__(self, h_dim: int = 128, out_channels: int = 960, num_layers: int = 4):
        super().__init__()
        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        # 直接出 out_channels (vla_hidden), 不需后续 Linear
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

        # ---- 兼容 modeling diag hook (它假设 pointnet 有 conv1/conv3/stn) ----
        # 用 object.__setattr__ 绕过 nn.Module 的子模块注册逻辑, 这样:
        # - alias 不会出现在 state_dict (不重复占 ckpt 空间)
        # - 不会被 .parameters() 重复枚举
        # - hasattr/getattr 仍能访问 (走 __getattr__ fallback 到 __dict__)
        # modeling forward 中 `_mod.pointnet.conv1.weight` 会读到 conv_in 的 weight.
        object.__setattr__(self, "conv1", self.conv_in)
        object.__setattr__(self, "conv3", self.conv_out)
        object.__setattr__(self, "stn", _STNAlias(self.global_layers[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) — 跟 pcd_encoder/pcdpp 接口一致."""
        x = x.transpose(1, 2)  # (B, 3, N)
        y = self.act(self.conv_in(x))  # (B, h_dim, N)
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))             # (B, h_dim, N)
            y_global = y.max(-1, keepdim=True).values   # (B, h_dim, 1)
            y = torch.cat([y, y_global.expand_as(y)], dim=1)   # (B, 2*h_dim, N)
            y = self.act(self.global_layers[i](y))      # (B, h_dim, N)
            feat_list.append(y)
        x_cat = torch.cat(feat_list, dim=1)             # (B, num_layers*h_dim, N)
        x_out = self.conv_out(x_cat)                    # (B, out_channels, N)
        x_global = x_out.max(-1).values                  # (B, out_channels)
        return x_global


class PCDDepthEncoder(nn.Module):
    """
    iDP3 版 depth encoder. 类名/接口/forward 签名跟 pcd_encoder/pcdpp_encoder 完全一致,
    可以无缝替换 import.

    Pipeline: depth (B, H, W) → point cloud → MultiStagePointNet → token (B, 1, vla_hidden)

    Args (跟 pcd_encoder 一致):
        vla_hidden: SmolVLM hidden dim (默认 960)
        fovy_deg:   vertical FOV in degrees (LIBERO: agent=45, wrist=75)
        img_size:   反投影虚拟图像尺寸 (默认 512)
        num_points: PointNet 输入点数 (默认 4096, 必须 perfect square)
        cam_label:  "agent" 或 "wrist" — debug print
    """

    def __init__(
        self,
        vla_hidden: int = 960,
        fovy_deg: float = 45.0,
        img_size: int = 512,
        num_points: int = 4096,
        cam_label: str = "cam",
        h_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.vla_hidden = vla_hidden
        self.fovy_deg = fovy_deg
        self.img_size = img_size
        self.num_points = num_points
        self.cam_label = cam_label

        # ---- 反投影 grid (跟 pcd_encoder 完全一致) ----
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

        # ---- iDP3 multi-stage PointNet (out_channels 直接 = vla_hidden) ----
        self.pointnet = MultiStagePointNetEncoder(
            h_dim=h_dim,
            out_channels=vla_hidden,
            num_layers=num_layers,
        )

        # ---- self.proj: dummy 占位, 满足 modeling diag 代码访问 .proj.weight ----
        # iDP3 不需要真 proj (pointnet 直接出 vla_hidden), 但 modeling 的
        # [PCD-DIAG] 代码会访问 mod.proj.weight.requires_grad 做 sanity check.
        # 用 Identity Linear (vla_hidden→vla_hidden, weight 不参与训练) 占位.
        # 这个 Linear 实际不进 forward, 不影响计算. weight=eye + bias=0 让"如果误调用"
        # 也是恒等变换 (虽然实际上 forward 里不调用 self.proj).
        self.proj = nn.Linear(vla_hidden, vla_hidden, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(vla_hidden))
            self.proj.bias.zero_()
        # 不参与训练 (forward 也不调用, 这个 freeze 只是双保险)
        self.proj.weight.requires_grad = False
        self.proj.bias.requires_grad = False

        self._dbg_done = False

        print(
            f"[iDP3Encoder.{cam_label}] iDP3 multi-stage PointNet: "
            f"fovy={fovy_deg}° img_size={img_size} num_points={num_points}, "
            f"h_dim={h_dim} num_layers={num_layers} out={vla_hidden} → 1 token"
        )

    def _backproject(self, depth: torch.Tensor) -> torch.Tensor:
        """
        depth: (B, H, W) in meters. 同 pcd_encoder/pcdpp 实现.
        return: (B, N, 3) point cloud in camera frame.
        """
        B, H, W = depth.shape
        # 双线性采样 depth 到 grid_size × grid_size
        # depth 当作 (B, 1, H, W), grid 是归一化 [-1, 1] 范围的 (B, gs, gs, 2)
        u_norm = (self.u_grid / (W - 1)) * 2 - 1  # (N,) 在 [-1, 1]
        v_norm = (self.v_grid / (H - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1)         # (N, 2)
        grid = grid.view(1, self.grid_size, self.grid_size, 2).expand(B, -1, -1, -1)
        z = F.grid_sample(
            depth.unsqueeze(1).float(), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )                                                     # (B, 1, gs, gs)
        z = z.view(B, self.num_points)                        # (B, N)

        # 反投影: u/v 用 pixel 坐标, z 是 depth
        u = self.u_grid.unsqueeze(0).expand(B, -1)
        v = self.v_grid.unsqueeze(0).expand(B, -1)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points = torch.stack([x, y, z], dim=-1)               # (B, N, 3)
        return points

    def forward(self, depth_m: torch.Tensor, target_dtype: torch.dtype = None) -> torch.Tensor:
        """
        depth_m: (B, H, W) depth in meters
        return: (B, 1, vla_hidden) — 1 个 global token
        """
        depth_f32 = depth_m.float()
        points = self._backproject(depth_f32)         # (B, N, 3)
        # iDP3 multi-stage PointNet 直接出 vla_hidden 维度
        feat_global = self.pointnet(points)            # (B, vla_hidden)
        token = feat_global.unsqueeze(1)               # (B, 1, vla_hidden)

        if target_dtype is not None:
            token = token.to(target_dtype)

        if not self._dbg_done:
            with torch.no_grad():
                token_std = token.float().std().item()
                token_norm = token.float().norm(dim=-1).mean().item()
                points_std = points.std().item()
                feat_std = feat_global.std().item()
                z = points[..., 2]
                print(
                    f"[iDP3Encoder.{self.cam_label}] depth_in: shape={tuple(depth_m.shape)}, "
                    f"min={depth_m.float().min().item():.3f} max={depth_m.float().max().item():.3f}"
                )
                print(
                    f"[iDP3Encoder.{self.cam_label}] points: shape={tuple(points.shape)} "
                    f"std={points_std:.3f}, z=[{z.min().item():.3f}, {z.max().item():.3f}] "
                    f"z_mean={z.mean().item():.3f}"
                )
                print(
                    f"[iDP3Encoder.{self.cam_label}] feat_global std={feat_std:.3f}  →  "
                    f"out tokens={tuple(token.shape)} std={token_std:.4f} norm/tok={token_norm:.4f}"
                )
            self._dbg_done = True

        return token

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)