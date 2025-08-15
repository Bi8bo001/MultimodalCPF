
## temporarily supports sum / average / concat / gated fusion / cross_attention
## note: different fusion types require compatible input/output sizes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class SumFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, struct_feat, text_feat):
        assert struct_feat.shape == text_feat.shape, \
            f"SumFusion requires same shape, got {struct_feat.shape} vs {text_feat.shape}"
        return struct_feat + text_feat


class ConcatFusion(nn.Module):
    def forward(self, struct_feat, text_feat):
        return torch.cat([struct_feat, text_feat], dim=-1)


class GatedFusion(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.Sigmoid()
        )

    def forward(self, struct_feat, text_feat):
        fusion_input = torch.cat([struct_feat, text_feat], dim=-1)
        gate_val = self.gate(fusion_input)  # shape: (B, d)
        return gate_val * struct_feat + (1 - gate_val) * text_feat


class CrossAttnFusion(nn.Module):
    def __init__(self, model_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(self, struct_token, text_token, struct_batch=None):
        """
        Args:
            struct_feat: Tensor of shape [B, D] (non-cross-attn) or [N_token, D] (cross-attn)
            text_feat: Tensor of shape [B, D] (non-cross-attn) or [B, L_text, D] (cross-attn)
            struct_batch: LongTensor [N_token] only for cross-attn
        Returns:
            fused_feat: Tensor of shape [B, D] or [B, 2D]
        """

        assert struct_batch is not None, "CrossAttnFusion requires struct_batch index tensor."

        # Step 1: pack struct_token to [B, L_struct, D]
        B = text_token.size(0)
        # L_struct = scatter_mean(torch.ones_like(struct_batch, dtype=torch.float), struct_batch, dim=0).long().max().item()
        D = struct_token.size(-1)
        device = struct_token.device
        assert struct_batch.max().item() + 1 == B, f"Mismatch in batch size: struct_batch max={struct_batch.max().item()} vs B={B}"
        
        Ls = torch.bincount(struct_batch, minlength=B).tolist()
        L_struct = max(Ls)

        # Pad struct tokens to batch shape
        max_idx = struct_batch.max().item() + 1
        assert max_idx == B, f"Mismatch in batch size: struct_batch max={max_idx} vs B={B}"

        padded_struct = torch.zeros(B, L_struct, D, device=device)
        mask = torch.zeros(B, L_struct, dtype=torch.bool, device=device)

        start = 0
        for i, l in enumerate(Ls):
            padded_struct[i, :l] = struct_token[start:start+l]
            mask[i, :l] = 1
            start += l

        # Step 2: cross attention: struct_token (query) attends to text_token (key/value)
        Q = padded_struct         # [B, L_struct, D]
        K = V = text_token        # [B, L_text, D]
        fused, _ = self.attn(Q, K, V)  # [B, L_struct, D]
        fused = self.norm(fused)
        fused = self.output_proj(fused)  # optional

        # Step 3: mean pool over struct token
        fused_batch = torch.cat([
            torch.full((l,), i, dtype=torch.long, device=device)
            for i, l in enumerate(Ls)
        ], dim=0)

        fused_repr = scatter_mean(fused[mask], fused_batch, dim=0, dim_size=B)

        return fused_repr



class FusionBlock(nn.Module):
    def __init__(self, fusion_type: str, model_dim: int):
        super().__init__()
        self.fusion_type = fusion_type
        self.model_dim = model_dim

        ##
        self.use_token_level = (fusion_type == "cross_attn")

        if fusion_type == "sum":
            self.fusion = SumFusion()
            self.output_dim = model_dim
        elif fusion_type == "concat":
            self.fusion = ConcatFusion()
            self.output_dim = model_dim * 2
        elif fusion_type == "gated":
            self.fusion = GatedFusion(model_dim)
            self.output_dim = model_dim
        elif fusion_type == "cross_attn":
            self.fusion = CrossAttnFusion(model_dim)
            self.output_dim = model_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, struct_feat: torch.Tensor, text_feat: torch.Tensor, struct_batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            struct_feat: Tensor of shape [B, d]
            text_feat: Tensor of shape [B, d]
        Returns:
            fused_feat: Tensor of shape [B, d] or [B, 2d] depending on fusion type
        """
        if self.fusion_type == "cross_attn":
            assert struct_feat.dim() == 2, "Expected struct_feat shape [N_token, D] for cross-attn"
            assert text_feat.dim() == 3, "Expected text_feat shape [B, L_text, D] for cross-attn"
            assert struct_batch is not None, "CrossAttnFusion requires struct_batch"
            return self.fusion(struct_feat, text_feat, struct_batch)
        else:
            assert struct_feat.shape == text_feat.shape, f"{self.fusion_type} requires same shape [B, D], got {struct_feat.shape} vs {text_feat.shape}"
            return self.fusion(struct_feat, text_feat)
