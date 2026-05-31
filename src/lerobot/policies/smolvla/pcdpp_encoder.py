"""
PointNet++ MSG encoder (with Dropout Points 数据增强).

接口跟 pcd_encoder_4c 的 PCDDepthEncoder 一致, 可直接替换:
  - __init__(vla_hidden, fovy_deg, img_size, num_points, cam_label)
  - forward(depth_m, target_dtype=None) → (B, 1, vla_hidden)
  - num_tokens = 1
  - num_trainable_params()

跟 PointNet 原版区别:
  (a) 显式建模局部邻域 (kNN grouping + mini-PointNet per neighborhood)
  (b) 分层抽象 (2 个 SA 层), Multi-Scale Grouping (MSG, 3 个尺度)
  (c) 训练时 Dropout Points (DP): 每个 batch sample 一次 dropout 概率 ∈ [0, 0.9],
      按此概率随机丢弃输入点 (剩下的复制填充回 num_points, 保持 batch shape)
  (d) FPS centroid 选择: 用 random sample 近似 (full FPS 训练时慢, 配合 DP
      已经有充足随机性. inference 时也 random — 因为 PointNet++ 本身对 centroid
      具体位置不敏感, 学的是局部结构)

DP 策略 (论文 PointNet++ Sec 3.3): 训练时每个样本随机选 θ ∈ [0, 0.95], 然后每个点
以 θ 概率被丢弃. 实现上为保持 batch 矩阵形状, 丢弃后从剩余点 random sample
回 num_points (重复采样). 测试时不丢弃.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 工具: depth → 点云反投影 ----------

def _fovy_to_fx_fy(fovy_deg: float, img_size: int) -> tuple[float, float]:
    fovy_rad = math.radians(fovy_deg)
    fy = (img_size / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    return fx, fy


# ---------- PointNet++ 核心算子 ----------

def _square_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (B, N, 3), b: (B, M, 3) → (B, N, M) squared L2 distance."""
    # |a - b|^2 = |a|^2 - 2a·b + |b|^2
    B, N, _ = a.shape
    M = b.shape[1]
    dist = -2 * torch.matmul(a, b.transpose(1, 2))         # (B, N, M)
    dist = dist + (a ** 2).sum(-1, keepdim=True)            # + |a|^2
    dist = dist + (b ** 2).sum(-1, keepdim=True).transpose(1, 2)  # + |b|^2
    return dist.clamp(min=0)


def _knn_group(points: torch.Tensor, centroids: torch.Tensor, k: int) -> torch.Tensor:
    """对每个 centroid 取 k 近邻索引.
    points: (B, N, 3), centroids: (B, M, 3) → idx (B, M, k) 在 points 里的索引.
    用 kNN 替代 PointNet++ 原版的 ball query, 简化实现 (无外部 CUDA op).
    """
    dist = _square_distance(centroids, points)             # (B, M, N)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]         # (B, M, k)
    return idx


def _gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points: (B, N, C), idx: (B, M, k) → (B, M, k, C).
    用 advanced indexing 避免 expand+gather 物化大张量 (B,M,N,C).
    """
    B, N, C = points.shape
    M, k = idx.shape[1], idx.shape[2]
    # batch_idx: (B, 1, 1) broadcast 到 (B, M, k)
    batch_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand(B, M, k)
    # advanced indexing: points[batch_idx, idx] → (B, M, k, C)
    return points[batch_idx, idx]


def _random_centroids(points: torch.Tensor, M: int) -> torch.Tensor:
    """随机采 M 个点作为 centroid (近似 FPS, 训练时配 DP 随机性已足够).
    points: (B, N, 3) → centroids (B, M, 3).
    """
    B, N, _ = points.shape
    idx = torch.randint(0, N, (B, M), device=points.device)
    idx_exp = idx.unsqueeze(-1).expand(B, M, 3)
    return torch.gather(points, 1, idx_exp)


class _SharedMLP(nn.Module):
    """对点级别的 (B, ?, k, C_in) 跑 shared MLP: Conv2d(1×1) 复用,
    每个点独立处理. 末尾 BN + ReLU.
    """
    def __init__(self, c_in: int, channels: list[int]):
        super().__init__()
        layers = []
        prev = c_in
        for c in channels:
            layers.append(nn.Conv2d(prev, c, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            prev = c
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, M, k) — Conv2d 期望的格式
        return self.net(x)


class _MSGSetAbstraction(nn.Module):
    """Multi-Scale Grouping Set Abstraction:
    输入 (B, N, C_in) 点 (C_in=3 时只有 xyz, 否则 xyz + 特征),
    随机选 M centroids, 每个 centroid 用多个尺度 (k_list 不同) 抽局部特征,
    各尺度 mini-PointNet (shared MLP + max-pool over k), concat 多尺度.
    输出: (B, M, C_out), C_out = sum(每尺度 MLP 最后通道数).
    """
    def __init__(
        self,
        c_in: int,           # 输入点特征维度 (xyz=3, 或 3+前层 channels)
        m: int,              # centroid 数
        k_list: list[int],   # 每尺度 kNN 邻居数
        mlp_list: list[list[int]],  # 每尺度的 shared MLP 通道列表
    ):
        super().__init__()
        assert len(k_list) == len(mlp_list)
        self.m = m
        self.k_list = k_list
        # 每尺度: 输入 = (xyz_relative=3) + (c_in-3=前层特征) = c_in
        # 但 mini-PointNet 处理的是 relative_xyz + point_features, 维度 c_in (xyz=3 含在内)
        self.mlps = nn.ModuleList([
            _SharedMLP(c_in, mlp) for mlp in mlp_list
        ])
        self.out_channels = sum(m[-1] for m in mlp_list)

    def forward(self, points_with_feat: torch.Tensor) -> torch.Tensor:
        """points_with_feat: (B, N, C_in), C_in 头 3 维是 xyz, 后面是点特征.
        return: (B, M, C_out) — centroid 在前 3 维是 xyz, 后面是聚合特征.
        """
        xyz = points_with_feat[..., :3]                    # (B, N, 3)
        centroids_xyz = _random_centroids(xyz, self.m)     # (B, M, 3)

        scale_feats = []
        for k, mlp in zip(self.k_list, self.mlps):
            idx = _knn_group(xyz, centroids_xyz, k)        # (B, M, k)
            grouped = _gather_points(points_with_feat, idx)  # (B, M, k, C_in)
            # relative coords: xyz - centroid
            grouped_xyz = grouped[..., :3] - centroids_xyz.unsqueeze(2)
            grouped = torch.cat([grouped_xyz, grouped[..., 3:]], dim=-1)  # (B,M,k,C_in)
            # 转 Conv2d 格式 (B, C, M, k)
            x = grouped.permute(0, 3, 1, 2).contiguous()
            x = mlp(x)                                     # (B, C', M, k)
            x = x.max(dim=-1)[0]                           # max over k → (B, C', M)
            scale_feats.append(x)

        feat = torch.cat(scale_feats, dim=1)               # (B, C_out, M)
        # 拼上 centroid xyz 作为新的 c_in (供下层用)
        feat = feat.transpose(1, 2)                        # (B, M, C_out)
        out = torch.cat([centroids_xyz, feat], dim=-1)     # (B, M, 3 + C_out)
        return out


class _GlobalSetAbstraction(nn.Module):
    """对整个 (B, N, C_in) 跑 mini-PointNet 然后 max-pool over N → (B, C_out).
    跟 PointNet 原版的全局聚合一致, 用在 PointNet++ 最后一层.
    """
    def __init__(self, c_in: int, mlp: list[int]):
        super().__init__()
        self.mlp = _SharedMLP(c_in, mlp)
        self.out_channels = self.mlp.out_channels

    def forward(self, points_with_feat: torch.Tensor) -> torch.Tensor:
        # (B, N, C_in) → (B, C_in, N, 1) → mlp → (B, C_out, N, 1) → max-pool → (B, C_out)
        x = points_with_feat.transpose(1, 2).unsqueeze(-1)  # (B, C_in, N, 1)
        x = self.mlp(x)                                     # (B, C_out, N, 1)
        x = x.squeeze(-1).max(dim=-1)[0]                    # (B, C_out)
        return x


# ---------- 主 encoder ----------

class PCDDepthEncoder(nn.Module):
    """
    Pipeline: depth (B, H, W) → point cloud → DP (训练时) → PointNet++ MSG → token (B, 1, vla_hidden)

    PointNet++ MSG 配置 (改自 ModelNet40 标准):
      SA1: M=512 centroids, k_list=[16, 32, 128] (3 尺度),
           MLPs=[[32,32,64],[64,64,128],[64,96,128]], out=320
      SA2: M=128 centroids, k_list=[32, 64, 128],
           MLPs=[[64,64,128],[128,128,256],[128,128,256]], out=640
      Global: SharedMLP([256, 512, 1024]) + max-pool → 1024
      Linear: 1024 → vla_hidden

    DP (Dropout Points, 训练时): 每个样本随机 θ ∈ [0, 0.9], 按 θ 丢弃点.
    保持 num_points 不变 (用剩余点 resample 回 num_points).
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

        # ---- PointNet++ MSG 主体 (轻量配置, batch 友好) ----
        # 相比标准 ModelNet40 配置 (M=512, k=[16,32,128], MLP=[32,32,64]...) 大幅缩小,
        # 适合 LIBERO 桌面任务 + 大 batch (64×2cam=128) 训练. 内存从 ~13GB 降到 ~0.8GB.
        # SA1: 输入 (B, N, 3) xyz, 输出 (B, 256, 3+160)
        self.sa1 = _MSGSetAbstraction(
            c_in=3,
            m=256,
            k_list=[8, 16, 32],
            mlp_list=[[16, 16, 32], [32, 32, 64], [32, 48, 64]],
        )
        # SA2: 输入 (B, 256, 3+160), 输出 (B, 64, 3+256)
        sa1_out_c = 3 + self.sa1.out_channels   # 3 + 160 = 163
        self.sa2 = _MSGSetAbstraction(
            c_in=sa1_out_c,
            m=64,
            k_list=[16, 32],
            mlp_list=[[64, 64, 128], [64, 96, 128]],
        )
        # Global: 输入 (B, 64, 3+256), max-pool over 64 → (B, 512)
        sa2_out_c = 3 + self.sa2.out_channels   # 3 + 256 = 259
        self.global_sa = _GlobalSetAbstraction(
            c_in=sa2_out_c,
            mlp=[128, 256, 512],
        )
        # ---- Linear projection 512 → vla_hidden ----
        self.proj = nn.Linear(512, vla_hidden)

        # DP 参数
        self.dp_max = 0.9  # 论文 0.95, 这里用 0.9 留点保守

        self._dbg_done = False

        print(
            f"[PCDppEncoder.{cam_label}] PointNet++ MSG (light config): "
            f"fovy={fovy_deg}° img_size={img_size} num_points={num_points} → "
            f"SA1(M=256,k=[8,16,32],out=160) → SA2(M=64,k=[16,32],out=256) → "
            f"Global([128,256,512]) → Linear(512,{vla_hidden}) → 1 token. "
            f"DP enabled (training only, θ ∈ [0, {self.dp_max}])."
        )

    @property
    def num_tokens(self) -> int:
        return 1

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _backproject(self, depth_m: torch.Tensor) -> torch.Tensor:
        """depth_m: (B, H, W) → (B, num_points, 3) in camera frame."""
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

    def _dropout_points(self, points: torch.Tensor) -> torch.Tensor:
        """训练时 DP: 每个样本随机 θ ∈ [0, dp_max], 按 θ 丢弃点.
        保持 num_points 不变 (剩余点 random resample 填回).
        测试 (eval mode) 不丢弃, 直接 return.
        向量化实现 (无 Python 循环, batch 友好).
        """
        if not self.training:
            return points
        B, N, _ = points.shape
        device = points.device

        # 每个 sample 一个 θ ∈ [0, dp_max]
        theta = torch.rand(B, device=device) * self.dp_max          # (B,)
        keep_prob = (1 - theta).unsqueeze(1).expand(B, N)            # (B, N)
        keep_mask = torch.bernoulli(keep_prob).bool()                # (B, N)

        # 极端情况兜底: 任何样本若全丢, 强制保留第 0 个点
        all_dropped = ~keep_mask.any(dim=1)                          # (B,)
        if all_dropped.any():
            keep_mask[all_dropped, 0] = True

        # 向量化 resample: 把每个 sample 的 keep 集合"延展"成 N 个 idx.
        # 思路: 对 keep_mask 做 cumsum 得到 keep idx, 然后对 N 个位置用 mod 取 keep idx.
        # 但 keep 数量每个 sample 不同, 简单做法: 用 multinomial.
        # multinomial 支持 batch, prob (B, N) → idx (B, N) with replacement.
        # keep_prob 太接近 0 会问题, 用 keep_mask.float() + 1e-6 作为采样概率.
        sample_prob = keep_mask.float() + 1e-6                       # (B, N), keep 点概率高, 丢弃点几乎 0
        resample_idx = torch.multinomial(sample_prob, N, replacement=True)  # (B, N)

        # gather: out[b, i] = points[b, resample_idx[b, i]]
        idx_exp = resample_idx.unsqueeze(-1).expand(B, N, 3)
        out = torch.gather(points, 1, idx_exp)
        return out

    def forward(self, depth_m: torch.Tensor, target_dtype: torch.dtype = None) -> torch.Tensor:
        """depth_m: (B, H, W) → token (B, 1, vla_hidden)."""
        depth_f32 = depth_m.float()
        points = self._backproject(depth_f32)               # (B, N, 3)

        # 训练时 DP
        points = self._dropout_points(points)

        # PointNet++ MSG: SA1 → SA2 → Global → proj
        feat_sa1 = self.sa1(points)                         # (B, 512, 3+320)
        feat_sa2 = self.sa2(feat_sa1)                       # (B, 128, 3+640)
        feat_global = self.global_sa(feat_sa2)              # (B, 1024)
        token = self.proj(feat_global).unsqueeze(1)         # (B, 1, vla_hidden)

        if target_dtype is not None:
            token = token.to(target_dtype)

        # 一次性 debug print
        if not self._dbg_done:
            with torch.no_grad():
                tok_norm = token.float().norm(dim=-1).mean().item()
                print(
                    f"[PCDppEncoder.{self.cam_label}] first forward: "
                    f"points={tuple(points.shape)} → token norm={tok_norm:.2f}, "
                    f"training={self.training} (DP active={self.training})"
                )
            self._dbg_done = True

        return token