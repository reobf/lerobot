"""手动 LoRA 实现 (无 peft 依赖, ckpt 兼容设计).

设计要点:
1. LoRALinear 持有原 base.weight 和 base.bias (同名), state_dict 保持 q_proj.weight
   原 key, 只多 q_proj.lora_A / q_proj.lora_B 两个新 key.
2. lora_A: Kaiming 初始化, lora_B: 全 0 初始化 → 启动时 LoRA 输出 = 0, 模型行为 = vanilla
3. 只 wrap VLM (text_model) 的 self_attn 的 q/k/v/o_proj, 不动 expert/vision/state_proj/FFN.
4. 加载兼容:
   - lora→lora: ckpt 有 lora_A/B, 直接加载.
   - 非lora→lora: ckpt 无 lora_A/B, strict=False 走默认初始化 (= vanilla 起点) ✓
   - lora→非lora: 不考虑 (你说的不需要兼容)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear. weight/bias 保留原名, 只多 lora_A/lora_B.

    forward(x) = base(x) + (alpha/r) * (x @ lora_A.T @ lora_B.T)
    其中 lora_A: (r, in_features), lora_B: (out_features, r)
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        # 把原 Linear 的 Parameter 拿过来 (引用, 不重新分配显存)
        # 注意 register Parameter 用 nn.Parameter() 包过的对象, 直接赋值给 self.xx 即可
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        # base 永久 freeze
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA 新参数: lora_A Kaiming, lora_B 0
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # lora_B 已经是 0, 保证启动时 LoRA 输出 = 0 (模型行为 = vanilla)

        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base forward
        base_out = F.linear(x, self.weight, self.bias)
        # LoRA forward: x @ A.T @ B.T, 注意 dtype 对齐 (base weight 可能是 bf16)
        x_dtype = x.dtype
        lora_A = self.lora_A.to(dtype=x_dtype)
        lora_B = self.lora_B.to(dtype=x_dtype)
        lora_out = x @ lora_A.T @ lora_B.T
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


def wrap_vlm_lora(vlm_text_model, rank: int, alpha: float, include_ffn: bool = False, verbose: bool = True):
    """Wrap VLM text_model's self_attn q/k/v/o_proj with LoRALinear.
    可选 include_ffn=True 时也包 FFN 层 (gate_proj/up_proj/down_proj).

    Args:
        vlm_text_model: SmolVLM2 text model (含 .layers list)
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha / rank)
        include_ffn: 是否同时包 MLP/FFN 的 gate/up/down_proj (默认 False, 仅 attention)
        verbose: 打印 wrap 信息
    """
    attn_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ffn_names = ["gate_proj", "up_proj", "down_proj"]
    target_names = attn_names + (ffn_names if include_ffn else [])

    n_wrapped = 0
    n_wrapped_attn = 0
    n_wrapped_ffn = 0
    n_lora_params = 0

    for layer_idx, layer in enumerate(vlm_text_model.layers):
        # attention 投影在 self_attn 下
        sa = layer.self_attn
        for name in attn_names:
            if not hasattr(sa, name):
                continue
            old_linear = getattr(sa, name)
            if not isinstance(old_linear, nn.Linear):
                continue
            new_linear = LoRALinear(old_linear, rank, alpha)
            setattr(sa, name, new_linear)
            n_wrapped += 1
            n_wrapped_attn += 1
            n_lora_params += new_linear.lora_A.numel() + new_linear.lora_B.numel()

        # FFN 投影在 layer.mlp 下 (SmolVLM2 / Llama 风格)
        if include_ffn:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for name in ffn_names:
                    if not hasattr(mlp, name):
                        continue
                    old_linear = getattr(mlp, name)
                    if not isinstance(old_linear, nn.Linear):
                        continue
                    new_linear = LoRALinear(old_linear, rank, alpha)
                    setattr(mlp, name, new_linear)
                    n_wrapped += 1
                    n_wrapped_ffn += 1
                    n_lora_params += new_linear.lora_A.numel() + new_linear.lora_B.numel()

    if verbose:
        n_layers = len(vlm_text_model.layers)
        scope = "q/k/v/o" + (" + gate/up/down (FFN)" if include_ffn else "")
        print(
            f"[LoRA] wrapped {n_wrapped} Linear layers across {n_layers} VLM layers "
            f"(target: {scope}). attn={n_wrapped_attn}, ffn={n_wrapped_ffn}. "
            f"rank={rank}, alpha={alpha}, scaling={alpha/rank:.3f}. "
            f"Trainable LoRA params: {n_lora_params:,} (~{n_lora_params/1e6:.2f}M)"
        )
