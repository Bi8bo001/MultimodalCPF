"""
Compute weighted regression loss from multiple branches (fusion + struct + text)

Args:
    pred_fusion (Tensor): (B, T) fused prediction
    pred_struct (Tensor): (B, T) structure-only prediction
    pred_text (Tensor): (B, T) text-only prediction
    data (object): Batch data containing ground truth
    targets (list of str): Names of target properties
    scale (Tensor): Normalization scale per target (T,)
    bias (Tensor): Normalization bias per target (T,)
    loss_type (str): "l1", "mse", or "smooth_l1"
    weights (list of float): [fusion_weight, struct_weight, text_weight]

Returns:
    Tensor: Final loss value for each sample (B,)
"""

# loss used during training is a weighted combination of multiple branches (fusion + struct + text)
# note: evaluation still uses standard metrics like MAE

import torch
import torch.nn.functional as F


def fusion_regression_loss(
    pred_fusion: torch.Tensor,
    pred_struct: torch.Tensor,
    pred_text: torch.Tensor,
    data,
    targets: list,
    scale: torch.Tensor,
    bias: torch.Tensor,
    loss_type: str = "l1",
    weights: list = [1.0, 0.0, 0.0],
):
    loss_type = loss_type.lower()
    assert loss_type in ("l1", "mse", "smooth_l1"), f"Unsupported loss type: {loss_type}"
    assert len(targets) == pred_fusion.shape[1], "Target size mismatch with predictions"
    assert len(weights) == 3, "weights must be a list of 3 floats"

    if loss_type == "l1":
        loss_fn = F.l1_loss
    elif loss_type == "mse":
        loss_fn = F.mse_loss
    else:
        loss_fn = F.smooth_l1_loss

    # def compute_loss(pred: torch.Tensor, data, name: str):
    #     loss = 0
    #     for i, t in enumerate(targets):
    #         if not hasattr(data, t):
    #             raise AttributeError(f"Target {t} not found in data during {name} loss computation")
    #         target = getattr(data, t).to(pred.device)
    #         if target.dim() == 0:
    #             target = target.view(1)
    #         elif target.dim() == 1:
    #             target = target.view(-1)
    #         target = (target - bias[i]) / scale[i]  # normalize
    #         loss += loss_fn(pred[:, i], target, reduction='none')  # (B,)
    #     return loss


    # unify prediction shape for compatibility with cross-attn style; output shape [B], doesn't affect other files
    def compute_loss(pred: torch.Tensor, data, name: str):
        B = pred.shape[0]
        loss = 0
        for i, t in enumerate(targets):
            if not hasattr(data, t):
                raise AttributeError(f"Target {t} not found in data during {name} loss computation")
            target = getattr(data, t).to(pred.device)

            # Ensure target shape is [B, 1] (compatible with all pred forms)
            if target.dim() == 0:
                target = target.view(1,1)
            elif target.dim() == 1:
                target = target.view(-1,1)
            elif target.dim() == 2 and target.shape[1] != 1:
                raise ValueError(f"[Loss] Unexpected target shape {target.shape} for {name}-{t}")
            elif target.dim() != 2:
                raise ValueError(f"[Loss] Too many dims: {target.shape} for {name}-{t}")

            assert target.shape[0] == B, f"[Loss] Batch size mismatch: pred={pred.shape}, target={target.shape}"
            assert target.shape[0] == pred.shape[0], f"[Loss] Batch size mismatch: pred={pred.shape}, target={target.shape}"

            target = (target - bias[i]) / scale[i]  # normalize
            pred_i = pred[:, i]
            if pred_i.dim() == 1:
                pred_i = pred_i.unsqueeze(1)
            loss += loss_fn(pred_i, target, reduction='none').squeeze(1)  # final shape [B]
        return loss


    loss_fusion = compute_loss(pred_fusion, data, "fusion") if pred_fusion is not None else 0
    loss_struct = compute_loss(pred_struct, data, "structure") if pred_struct is not None else 0
    loss_text = compute_loss(pred_text, data, "text") if pred_text is not None else 0

    total_loss = weights[0] * loss_fusion + weights[1] * loss_struct + weights[2] * loss_text
    return total_loss, loss_fusion, loss_struct, loss_text
