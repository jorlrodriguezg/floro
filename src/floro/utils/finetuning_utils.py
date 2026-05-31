import os
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from floro.utils.dist_utils import is_dist_avail_and_initialized, is_main_process, get_rank
import sklearn.metrics as skmetrics

from torch.amp import autocast

import wandb

# ---------------------------
# Labels helpers
# ---------------------------
def _normalize_labels_shape(labels: torch.Tensor) -> torch.Tensor:
    """
    Ensure labels are (B,H,W) long.
    Accepts (B,H,W), (B,1,H,W), or (B,H,W,1).
    """
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0, ...]        # (B,H,W)
    elif labels.ndim == 4 and labels.shape[-1] == 1:
        labels = labels[..., 0]           # (B,H,W)
    return labels.long()


# ---------------------------
#       Loss helpers
# ---------------------------

def normalize_min_max(
    x: torch.Tensor, 
    min_val: torch.Tensor, 
    max_val: torch.Tensor, 
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Normalizes x to [0, 1] based on provided min_val and max_val.
    """
    # Avoid division by zero if max == min
    denom = (max_val - min_val).clamp_min(epsilon)
    return (x - min_val) / denom

def soft_histogram(x: torch.Tensor, bins: int = 50, sigma: float = 0.02) -> torch.Tensor:
    """
    x: 1D tensor of values, ASSUMED to be in [0, 1]
    bins: number of histogram bins
    sigma: controls kernel width (relative to [0,1] range)
    returns: [bins] normalized probability distribution
    """
    x = x.view(-1, 1)  # [N, 1]
    
    # Bin centers fixed in [0, 1]
    centers = torch.linspace(0.0, 1.0, bins, device=x.device).view(1, -1)  # [1, B]
    
    # Squared distances
    d2 = (x - centers) ** 2  # [N, B]
    
    # Soft weights via Gaussian kernel
    weights = torch.exp(-d2 / (2 * sigma ** 2))  # [N, B]
    
    # Sum over samples -> counts per bin
    hist = weights.sum(dim=0)  # [B]
    
    # Normalize to a probability distribution
    hist = hist / hist.sum().clamp_min(1e-6)
    
    return hist

def distribution_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    bins: int = 50,
    sigma: float = 0.02,
) -> torch.Tensor:
    """
    Calculates histogram loss agnostic to the input data range.
    It dynamically normalizes pred and tgt to [0,1] based on tgt's range.
    """
    
    # Flatten valid values
    flat_pred = pred[mask].view(-1)
    flat_tgt  = tgt[mask].view(-1)
    
    if flat_pred.numel() == 0:
        return pred.new_tensor(0.0)

    # 1. Determine the data range from the GROUND TRUTH (tgt)
    # We detach to ensure we don't backprop through the boundaries, 
    # treating them as fixed anchors for this batch.
    min_val = flat_tgt.min().detach()
    max_val = flat_tgt.max().detach()

    # 2. Normalize both to [0, 1] using the target's range
    flat_pred_norm = normalize_min_max(flat_pred, min_val, max_val)
    flat_tgt_norm  = normalize_min_max(flat_tgt,  min_val, max_val)

    # 3. Clamp inputs to [0, 1] 
    # This ensures that outliers in prediction don't break the histogram
    # (or you can leave them unclamped if you want to penalize "out of bounds" 
    # implicitly by having them fall into the first/last bins with low weight).
    # Clamping is usually safer for stability.
    flat_pred_norm = flat_pred_norm.clamp(0.0, 1.0)
    flat_tgt_norm  = flat_tgt_norm.clamp(0.0, 1.0)

    # 4. Compute histograms on the normalized data
    pred_hist = soft_histogram(flat_pred_norm, bins=bins, sigma=sigma)
    tgt_hist  = soft_histogram(flat_tgt_norm,  bins=bins, sigma=sigma)

    # 5. L2 loss between histograms
    hist_loss = torch.mean((pred_hist - tgt_hist) ** 2)
    
    return hist_loss


def dice_loss_from_logits(logits, targets, num_classes, ignore_index=255, eps=1e-6):
    # Robustify target shape → (B,H,W)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0, ...]
    elif targets.ndim == 4 and targets.shape[-1] == 1:
        targets = targets[..., 0]

    if num_classes == 1:
        probs1 = torch.sigmoid(logits)           # (B,1,H,W)
        probs0 = 1.0 - probs1
        probs  = torch.cat([probs0, probs1], dim=1)  # (B,2,H,W)
        C = 2
    else:
        probs = F.softmax(logits, dim=1)         # (B,C,H,W)
        C = num_classes

    # valid mask (ignore_index-aware)
    valid = (targets != ignore_index) & (targets >= 0)
    safe_t = torch.where(valid, targets, torch.zeros_like(targets))  # (B,H,W)

    one_hot = F.one_hot(safe_t.long(), num_classes=C).permute(0, 3, 1, 2).float()  # (B,C,H,W)
    valid_mask = valid.unsqueeze(1)  # (B,1,H,W)

    probs   = probs * valid_mask
    one_hot = one_hot * valid_mask

    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice_per_class = (2 * inter + eps) / (denom + eps)

    present = (one_hot.sum(dims) > 0)
    return (1.0 - dice_per_class[present].mean()) if present.any() else logits.new_tensor(0.0)


def cross_entropy_with_optional_smoothing(
        logits, targets, ignore_index=255, class_weights=None, label_smoothing=0.0
    ):
    """
    Multi-class CE with optional smoothing. For binary (C==1), caller should use BCEWithLogits instead.
    """
    return F.cross_entropy(
        logits, targets.long(),
        weight=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="mean"
    )

class MaskedMSELoss(nn.Module):
    """
    pred, tgt: [B, 1, H, W] or [B, H, W]
    mask: bool mask of valid pixels; if None, use finite mask
    reduction: 'mean', 'sum', or 'none_for_epoch' (sum + count)
    """
    def __init__(self):
        super().__init__()

    def forward(
            self,
            pred: torch.Tensor,
            tgt: torch.Tensor,
            mask: torch.Tensor | None = None,
            reduction: str = "mean"
        ):
        
        pred = pred.float()
        tgt  = tgt.float()

        if mask is None:
            mask = torch.isfinite(pred) & torch.isfinite(tgt)
        else:
            mask = mask.bool() & torch.isfinite(pred) & torch.isfinite(tgt)

        diff2 = (pred - tgt) ** 2
        diff2 = torch.where(mask, diff2, torch.zeros_like(diff2))

        if reduction == "sum":
            return diff2.sum()
        elif reduction == "mean":
            n = mask.sum().clamp_min(1)
            return diff2.sum() / n
        elif reduction == "none_for_epoch":
            n = mask.sum()
            return diff2.sum(), n
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    

class MaskedRMSELoss(nn.Module):
    """
    pred, tgt: [B, 1, H, W] or [B, H, W]
    mask: bool mask of valid pixels (True = include). If None, use finite mask.
    reduction:
    - "global": sqrt(sum(err^2)/sum(mask)) over all valid pixels (recommended for metric consistency)
    - "mean": mean over batch of per-image RMSE
    - "sum": sum over batch of per-image RMSE (rarely used)
    - "none_for_epoch": returns (sum_sqerr, count) for epoch aggregation into global RMSE
    """
    def __init__(self):
        super().__init__()

    def forward(
            self,
            pred: torch.Tensor,
            tgt: torch.Tensor,
            mask: torch.Tensor | None = None,
            reduction: str = "global",  # "global", "mean", "sum", "none_for_epoch"
            eps: float = 1e-12,
    ):
        
        pred = pred.float()
        tgt  = tgt.float()

        if mask is None:
            mask = torch.isfinite(pred) & torch.isfinite(tgt)
        else:
            mask = mask.bool() & torch.isfinite(pred) & torch.isfinite(tgt)

        # squared error, masked
        se = (pred - tgt) ** 2
        se = torch.where(mask, se, torch.zeros_like(se))

        if reduction == "global":
            denom = mask.sum().clamp_min(1)
            mse = se.sum() / denom
            return torch.sqrt(mse + eps)

        # per-image RMSE (reduce over H,W, and channel if present)
        # figure out which dims are spatial (+ optional channel)
        if pred.ndim == 4:  # [B,1,H,W]
            reduce_dims = (-1, -2, -3)  # W,H,C
        elif pred.ndim == 3:  # [B,H,W]
            reduce_dims = (-1, -2)      # W,H
        else:
            raise ValueError(f"Expected pred/tgt to be 3D or 4D, got {pred.shape}")

        denom_b = mask.sum(dim=reduce_dims).clamp_min(1)        # [B]
        mse_b   = se.sum(dim=reduce_dims) / denom_b             # [B]
        rmse_b  = torch.sqrt(mse_b + eps)                        # [B]

        if reduction == "mean":
            return rmse_b.mean()
        elif reduction == "sum":
            return rmse_b.sum()
        elif reduction == "none_for_epoch":
            # return numerator and denominator for global RMSE aggregation
            return se.sum(), mask.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


# ---------------------------
#     Confusion + Metrics
# ---------------------------
def _fast_confusion_matrix(pred, tgt, num_classes, ignore_index=255):
    """
    pred, tgt: (B,H,W) int64 in [0..C-1]; ignored labels marked by ignore_index or <0.
    Returns (C,C) matrix on pred.device
    """
    mask = (tgt != ignore_index) & (tgt >= 0) & (tgt < num_classes)
    if mask.sum() == 0:
        return torch.zeros((num_classes, num_classes), device=pred.device, dtype=torch.long)
    pred = pred[mask]
    tgt  = tgt[mask]
    idx = tgt * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes*num_classes)
    return cm.view(num_classes, num_classes)

def _miou_from_confmat(confmat, eps=1e-6):
    # IoU_c = TP / (TP + FP + FN)
    TP = torch.diag(confmat).float()
    FP = confmat.sum(0).float() - TP
    FN = confmat.sum(1).float() - TP
    denom = TP + FP + FN + eps
    iou = TP / denom
    # average only over classes that appear in either pred or gt
    valid = denom > eps
    if valid.any():
        return iou[valid].mean().item()
    else:
        return 0.0

def _pixel_accuracy_from_confmat(confmat):
    correct = torch.diag(confmat).sum().float()
    total   = confmat.sum().float()
    return (correct / total).item() if total > 0 else 0.0

def tensor_stats(t: torch.Tensor, name: str):
    """Print safe stats without relying on torch.nanmin/nanmean.
    Works across PyTorch versions and dtypes, handles empty/invalid tensors.
    """
    with torch.no_grad():
        # Ensure floating for stats, but don't modify the original
        x = t.detach()
        if not x.is_floating_point():
            x = x.float()

        numel = x.numel()
        if numel == 0:
            print(f"[debug] {name}: empty tensor")
            return

        finite = torch.isfinite(x)
        n_valid = int(finite.sum().item())
        rank = int(os.environ.get("RANK", "0"))

        if n_valid == 0:
            print(f"[debug][r{rank}] {name}: shape={tuple(x.shape)} valid=0/{numel} (all NaN/Inf)")
            return

        x_valid = x[finite]
        mn  = x_valid.min().item()
        mx  = x_valid.max().item()
        mean = x_valid.mean().item()

        print(
            f"[debug][r{rank}] {name}: shape={tuple(x.shape)} "
            f"valid={n_valid}/{numel} min={mn:.6g} max={mx:.6g} mean={mean:.6g}"
        )

# ----------------------------------------------------
#         One-epoch trainer for SEGMENTATION
# ----------------------------------------------------

# ------------------------helpers
def _ddp_all_or_none_skip(local_wants_to_skip: bool, device) -> bool:
    """If ANY rank wants to skip, ALL ranks skip."""
    if not (dist.is_available() and dist.is_initialized()):
        return local_wants_to_skip
    t = torch.tensor([1 if local_wants_to_skip else 0], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() > 0

def _ddp_zero_backward(models, use_amp: bool, scaler, device):
    """
    Backprop a zero that still touches parameters, so DDP finishes reductions.
    This adds ~no compute and leaves weights unchanged.
    """
    zero = torch.zeros((), device=device)
    for m in models:
        if m is None: 
            continue
        for p in m.parameters():
            if p is not None and p.requires_grad:
                # touch graph; grad will be identically zero
                zero = zero + p.view(-1)[0] * 0.0
    if use_amp and scaler is not None:
        scaler.scale(zero).backward()
    else:
        zero.backward()
        
# ------------------------------------------------------------
        
def finetune_one_epoch_geo_seg_dpt(
    model_enc, model_dec, dataloader,
    optimizer_enc, optimizer_dec,   # optimizer_enc may be None in decoder_only mode
    device, args, scaler=None, scheduler_enc=None, scheduler_dec=None
):
    """
    Train one epoch for image segmentation using a DPT-like decoder on top of your encoder.

    Expects each batch to contain:
      - 'image'         : (B, C_ms, H, W)  multispectral (or RGB) tensor
      - 'dsm'           : (B, 1, H, W)     elevation/DSM (float)
      - 'mask' or 'label': (B, H, W)       integer class map (0..C-1), void=ignore_index
      - 'geo_transform' : (...), whatever your encoder expects

    Required args:
      - args.num_channels (int)         : set 1 for binary (foreground vs background), >=2 for multi-class
      - args.ignore_index (int, opt)   : default 0
      - args.label_smoothing (float)   : default 0.0
      - args.class_weights (list/tuple or None): optional per-class weights (multi-class only)
      - args.lambda_dice (float)       : default 0.0 (set 0.3~0.6 to mix with CE)
      - args.train_mode ("decoder_only"|"full")
      - args.use_autocast (bool)
      - args.accumulation_steps (int)
      - args.masking_ms, args.masking_modality (floats), passed to encoder if applicable

    Returns:
      epoch_loss (float),
      metrics dict: {"ce":..., "dice":..., "loss":..., "miou":..., "pixel_acc":...}
    """

    num_classes   = int(getattr(args, "num_channels", 2))
    ignore_index  = int(getattr(args, "ignore_index", 255))
    label_smooth  = float(getattr(args, "label_smoothing", 0.0))
    lambda_dice   = float(getattr(args, "lambda_dice", 0.0))

    # Optional class weights
    w = getattr(args, "class_weights", None)
    class_weights = None
    if w is not None:
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    # Modes / freezing
    # if getattr(args, "train_mode", "finetune_both") == "decoder_only":
    #     print("Freezing encoder parameters and setting encoder to eval.")
    #     for p in model_enc.parameters():
    #         p.requires_grad = False
    #     model_enc.eval()
    #     optimizer_enc = None
    # else:
    #     for p in model_enc.parameters():
    #         p.requires_grad = True
    #     model_enc.train()
    # model_dec.train()

    amp_ctx = torch.amp.autocast("cuda") if getattr(args, "use_autocast", False) else nullcontext()

    # Zero grads to start
    if optimizer_enc is not None:
        optimizer_enc.zero_grad(set_to_none=True)
    optimizer_dec.zero_grad(set_to_none=True)

    mask_ratio_ms  = getattr(args, "masking_ms", 0.0)
    mask_ratio_mod = getattr(args, "masking_modality", 0.0)
    accum_steps    = max(1, int(getattr(args, "accumulation_steps", 1)))
    accum_counter  = 0
    use_amp = bool(getattr(args, "use_autocast", False))

    def step_optimizers():
        if scaler is not None and use_amp:
            if optimizer_enc is not None:
                scaler.step(optimizer_enc)
            scaler.step(optimizer_dec)
            scaler.update()
        else:
            if optimizer_enc is not None:
                optimizer_enc.step()
            optimizer_dec.step()
        if optimizer_enc is not None:
            optimizer_enc.zero_grad(set_to_none=True)
        optimizer_dec.zero_grad(set_to_none=True)

    # Running stats
    running_loss = 0.0
    running_ce   = 0.0
    running_dice = 0.0
    total_pixels = 0

    # Confusion matrix (multi-class); for binary with C=1 we still track 2 classes for IoU/PA
    conf_C = (2 if num_classes == 1 else num_classes)
    confmat = torch.zeros((conf_C, conf_C), device=device, dtype=torch.long)

    for batch in dataloader:
        image_ms = batch['image'].to(device, non_blocking=True)                # (B,C,H,W)
        elevation = batch['elevation'].to(device, non_blocking=True).float()         # (B,1,H,W)

        # labels may be under 'mask' or 'label'
        if 'mask' in batch:
            labels = batch['mask']
        elif 'target' in batch:
            labels = batch['target']
        else:
            raise KeyError("Batch must provide 'mask' or 'target' for segmentation.")
        
        labels =_normalize_labels_shape(labels)
        labels = labels.to(device, non_blocking=True).long()                   # (B,H,W)
        gt = batch.get('geo_transform', None)
        if gt is None:
            raise KeyError("Batch missing 'geo_transform' expected by encoder.")
        gt = gt.to(device, non_blocking=True)

        # === NEW (A): decide early if this batch has ANY valid pixels ===
        valid_mask = (labels != ignore_index)
        local_has_valid = bool(valid_mask.any().item())
        skip_global = _ddp_all_or_none_skip(not local_has_valid, device)

        # === NEW (B): if skipping, still do a tiny "zero backward" so all
        # ranks perform reductions identically; keep accumulation in sync.
        if skip_global:
            _ddp_zero_backward(
                models=[model_enc if optimizer_enc is not None else None, model_dec],
                use_amp=use_amp, scaler=scaler, device=device
            )
            accum_counter += 1
            if accum_counter % accum_steps == 0:
                step_optimizers()
            # Do NOT update loss/metrics with this batch
            continue

        # === NEW (C): (optional) sync RNG so masking/etc is identical on all ranks
        if dist.is_available() and dist.is_initialized():
            # broadcast a per-iter seed from rank 0
            seed_t = torch.randint(0, 2**31 - 1, (1,), device=device) if dist.get_rank() == 0 else torch.empty(1, device=device, dtype=torch.long)
            dist.broadcast(seed_t, src=0)
            torch.manual_seed(int(seed_t.item()))
            torch.cuda.manual_seed(int(seed_t.item()))
            # If your encoder forward accepts a generator, pass it instead of re-seeding the global RNG.

        # Forward
        if optimizer_enc is None:
            with torch.no_grad():
                enc_out = model_enc(
                    image_ms, elevation, gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )
            with amp_ctx:
                logits = model_dec(enc_out)   # (B,C,H',W') or (B,1,H',W')
        else:
            with amp_ctx:
                enc_out = model_enc(
                    image_ms, elevation, gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )
                logits = model_dec(enc_out)

        # Resize logits to target size if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        # ---- Losses ----
        if num_classes == 1:
            # Binary
            targets_bin = torch.clamp(labels, 0, 1).float()                    # (B,H,W)
            valid = (labels != ignore_index)
            if valid.any():
                bcel = F.binary_cross_entropy_with_logits(
                    logits.squeeze(1)[valid], targets_bin[valid], reduction="mean"
                )
            else:
                bcel = logits.new_tensor(0.0)
            dl = dice_loss_from_logits(logits, labels, num_classes=1, ignore_index=ignore_index) if lambda_dice > 0 else logits.new_tensor(0.0)
            loss = bcel + lambda_dice * dl
            ce_val, dice_val = bcel.detach(), dl.detach()
        else:
            # Multi-class
            cel = cross_entropy_with_optional_smoothing(
                logits, labels,
                ignore_index=ignore_index,
                class_weights=class_weights,
                label_smoothing=label_smooth
            )
            dl = dice_loss_from_logits(logits, labels, num_classes=num_classes, ignore_index=ignore_index) if lambda_dice > 0 else logits.new_tensor(0.0)
            loss = cel + lambda_dice * dl
            ce_val, dice_val = cel.detach(), dl.detach()

        # Backward (with accumulation)
        loss_for_back = loss / accum_steps
        if scaler is not None and use_amp:
            scaler.scale(loss_for_back).backward()
        else:
            loss_for_back.backward()

        accum_counter += 1
        if accum_counter % accum_steps == 0:
            step_optimizers()

        # Metrics update
        B = labels.size(0)
        running_loss += float(loss.detach().item()) * B
        running_ce   += float(ce_val.item()) * B
        running_dice += float(dice_val.item()) * B

        # Confusion / pixel-acc (treat binary as 2-class for metrics)
        with torch.no_grad():
            if num_classes == 1:
                pred_lbl = (torch.sigmoid(logits) > 0.5).long().squeeze(1)     # (B,H,W)
                C = 2
                confmat += _fast_confusion_matrix(pred_lbl, labels.clamp_max(1), C, ignore_index=ignore_index)
                total_pixels += (labels != ignore_index).sum().item()
            else:
                pred_lbl = torch.argmax(logits, dim=1)                          # (B,H,W)
                C = num_classes
                confmat += _fast_confusion_matrix(pred_lbl, labels, C, ignore_index=ignore_index)
                total_pixels += (labels != ignore_index).sum().item()

    # Flush leftover grads if accumulation window incomplete
    if accum_counter % accum_steps != 0:
        step_optimizers()

    # Step schedulers once per epoch
    if scheduler_enc is not None:
        scheduler_enc.step()
    if scheduler_dec is not None:
        scheduler_dec.step()

    # DDP reduce confusion/loss tallies
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([running_loss, running_ce, running_dice, total_pixels], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        running_loss, running_ce, running_dice, total_pixels = t.tolist()
        dist.all_reduce(confmat, op=dist.ReduceOp.SUM)

    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / max(1, dataset_size)
    mean_ce    = running_ce / max(1, dataset_size)
    mean_dice  = running_dice / max(1, dataset_size)

    miou = _miou_from_confmat(confmat)
    pix_acc = _pixel_accuracy_from_confmat(confmat)

    print(f"[SEG] loss={epoch_loss:.4f}  CE={mean_ce:.4f}  Dice={mean_dice:.4f}  mIoU={miou:.4f}  PA={pix_acc:.4f}")
    return epoch_loss, {"loss": epoch_loss, "ce": mean_ce, "dice": mean_dice, "miou": miou, "pixel_acc": pix_acc}

###############################################
# Validation

def log_image_wandb(image_tensor, pred_seg_tensor, seg_gt_tensor):
    np_image = image_tensor[0][[3, 2, 1],:,:].permute(1, 2, 0).cpu().numpy()  # if shape was [C,H,W] -> [H,W,3]
    pred_mask_tensor = torch.argmax(pred_seg_tensor, dim=1)
    pred_mask = pred_mask_tensor[0,:,:].squeeze().detach().cpu().numpy()       # if shape was [1,H,W] -> [H,W]   
    ref_gt_mask = seg_gt_tensor[0,:,:].squeeze().detach().cpu().numpy()
    
    # 1) Define your label names and desired colors for each class ID
    class_labels = {
        1: "Shadows",
        2: "Open foliage shrubs",
        3: "Stemmy shrubs",
        4: "Dense foliage shrubs",
        5: "Trees",
        6: "Herbs/Low Shrubs",
        7: "Dry Vegetation",
        8: "Sand",
        9: "Bare soil",
        10: "Rocky surfaces",
        11: "Artificial objects"
    }

    # Here are some rough example hex colors:
    class_colors = {
        1: "#000000",  # black
        2: "#556B2F",  # dark olive green
        3: "#00FF00",  # bright green
        4: "#CCFF00",  # neon yellowish-green
        5: "#006400",  # dark green
        6: "#BEBE60",  # a light olive shade (approx for "herbs/low shrubs")
        7: "#FFA500",  # orange for "dry vegetation"
        8: "#FFFF00",  # bright yellow
        9: "#A0522D",  # sienna (bare soil)
        10: "#9E9E9E", # grayish for rocky surfaces
        11: "#FF0000"  # red
    }

    # 2) Log the overlay to W&B
    wandb.log({
        "segmentation_overlay": wandb.Image(
            np_image,
            masks={
                "predictions": {
                    "mask_data": pred_mask, 
                    "class_labels": class_labels,
                    "class_colors": class_colors
                },
                "ground_truth": {
                    "mask_data": ref_gt_mask, 
                    "class_labels": class_labels,
                    "class_colors": class_colors
                }
            }
        )
    })

def _per_class_iou_list(confmat, eps=1e-6):
    TP = torch.diag(confmat).float()
    FP = confmat.sum(0).float() - TP
    FN = confmat.sum(1).float() - TP
    denom = TP + FP + FN + eps
    iou = TP / denom
    # Mark classes with no support as NaN to avoid skewing averages
    iou[denom <= eps] = float('nan')
    return [float(v) for v in iou.tolist()]

@torch.no_grad()
def validate_one_epoch_geo_seg_dpt(
    model_enc, model_dec, dataloader, device, args, log_preds_wb=False
):
    """
    Validation for DPT-like segmentation.

    Batch requirements:
      - batch['image']  -> (B,C,H,W) float
      - batch['dsm']    -> (B,1,H,W) float
      - batch['mask'] or batch['label'] -> (B,H,W) long in [0..C-1], void = ignore_index
      - batch['geo_transform'] -> whatever your encoder expects

    args:
      - num_classes (int), set 1 for binary (0/1), >=2 for multi-class
      - ignore_index (int, default 255)
      - label_smoothing (float, default 0.0)
      - class_weights (list/tuple or None)
      - lambda_dice (float, default 0.0)
      - use_autocast (bool)
    """
    num_classes  = int(getattr(args, "num_channels", 2))
    ignore_index = int(getattr(args, "ignore_index", 255))
    lambda_dice  = float(getattr(args, "lambda_dice", 0.0))
    label_smooth = float(getattr(args, "label_smoothing", 0.0))
    w = getattr(args, "class_weights", None)
    class_weights = None
    if w is not None:
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    model_enc.eval()
    model_dec.eval()

    amp_ctx = torch.amp.autocast("cuda") if getattr(args, "use_autocast", False) else nullcontext()

    total_loss = 0.0
    total_ce   = 0.0
    total_dice = 0.0
    n_images   = 0

    # Confusion matrix for metrics (treat binary as 2-class for IoU/PA)
    conf_C = 2 if num_classes == 1 else num_classes
    confmat = torch.zeros((conf_C, conf_C), device=device, dtype=torch.long)

    for batch in dataloader:
        image_ms = batch['image'].to(device, non_blocking=True)
        elevation = batch['elevation'].to(device, non_blocking=True).float()
        labels = (batch['mask'] if 'mask' in batch else batch['target']).to(device, non_blocking=True).long()
        labels = _normalize_labels_shape(labels)
        gt = batch['geo_transform'].to(device, non_blocking=True)

        with amp_ctx:
            enc_out = model_enc(
                image_ms, elevation, gt,
                mask_ratio_ms=getattr(args, "masking_ms", 0.0),
                mask_ratio_mods=getattr(args, "masking_modality", 0.0),
                return_intermediate=True
            )
            logits = model_dec(enc_out)

        # Align logits to label size if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        # ---------- Loss ----------
        if num_classes == 1:
            # Binary: BCEWithLogits + optional Dice
            targets_bin = labels.clamp(0, 1).float()
            valid = (labels != ignore_index)
            if valid.any():
                ce = F.binary_cross_entropy_with_logits(
                    logits.squeeze(1)[valid], targets_bin[valid], reduction="mean"
                )
            else:
                ce = logits.new_tensor(0.0)
            if lambda_dice > 0:
                dl = dice_loss_from_logits(logits, labels, num_classes=1, ignore_index=ignore_index)
            else:
                dl = logits.new_tensor(0.0)
            loss = ce + lambda_dice * dl

            # Predictions for metrics
            pred_lbl = (torch.sigmoid(logits) > 0.5).long().squeeze(1)  # (B,H,W)
            confmat += _fast_confusion_matrix(pred_lbl, labels.clamp_max(1), 2, ignore_index=ignore_index)

        else:
            # Multi-class: Cross-Entropy (+ label smoothing/weights) + optional Dice
            # Guard against all-ignored images (rare but can happen)
            valid = (labels != ignore_index)
            if valid.any():
                ce = cross_entropy_with_optional_smoothing(
                    logits, labels,
                    ignore_index=ignore_index,
                    class_weights=class_weights,
                    label_smoothing=label_smooth,
                )
            else:
                ce = logits.new_tensor(0.0)

            if lambda_dice > 0:
                dl = dice_loss_from_logits(logits, labels, num_classes=num_classes, ignore_index=ignore_index)
            else:
                dl = logits.new_tensor(0.0)
            loss = ce + lambda_dice * dl

            pred_lbl = torch.argmax(logits, dim=1)  # (B,H,W)
            confmat += _fast_confusion_matrix(pred_lbl, labels, num_classes, ignore_index=ignore_index)

            # Log image to W&B
            if log_preds_wb == True:
                log_image_wandb(image_ms, logits, labels)

        B = labels.size(0)
        total_loss += float(loss.item()) * B
        total_ce   += float(ce.item())   * B
        total_dice += float(dl.item())   * B
        n_images   += B

    # ---------- DDP reductions ----------
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([total_loss, total_ce, total_dice, n_images], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_ce, total_dice, n_images = t.tolist()
        dist.all_reduce(confmat, op=dist.ReduceOp.SUM)

    # ---------- Metrics ----------
    epoch_loss = total_loss / max(1, n_images)
    mean_ce    = total_ce   / max(1, n_images)
    mean_dice  = total_dice / max(1, n_images)

    miou     = _miou_from_confmat(confmat)
    pix_acc  = _pixel_accuracy_from_confmat(confmat)
    iou_list = _per_class_iou_list(confmat)

    print(f"[VAL] loss={epoch_loss:.4f}  CE={mean_ce:.4f}  Dice={mean_dice:.4f}  mIoU={miou:.4f}  PA={pix_acc:.4f}")
    return epoch_loss, {
        "loss": epoch_loss,
        "ce": mean_ce,
        "dice": mean_dice,
        "miou": miou,
        "pixel_acc": pix_acc,
        "per_class_iou": iou_list,   # length = 2 if binary else num_classes
    }

def finetune_one_epoch_geo_reg_chmbm(
    model_enc, model_dec, dataloader,
    optimizer_enc, optimizer_dec,
    device, args,
    criterion=None,  scaler=None,
    scheduler_enc=None, scheduler_dec=None,
    alpha=1.0, beta=1.0,
):
    # modes / freezing
    if args.train_mode == "decoder_only":
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for p in model_enc.parameters():
            p.requires_grad = False
        model_enc.eval()
        optimizer_enc = None
    else:
        model_enc.train()
    model_dec.train()

    def sanitize(x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt  = tgt.float()
        mask = mask.bool()
        diff = (pred - tgt) ** 2
        diff = torch.where(mask, diff, torch.zeros_like(diff))
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        n = mask.sum().clamp_min(1)
        return diff.sum() / n

    def safe_item(x: torch.Tensor) -> float:
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return float(x.detach().item())

    running_loss = 0.0
    training_chm_loss = 0.0
    training_bm_loss = 0.0

    amp_ctx = torch.amp.autocast("cuda") if args.use_autocast else nullcontext()
    loss_function = criterion if criterion is not None else MaskedMSELoss()

    # Zero grads to start an accumulation window
    if args.train_mode == "decoder_only":
        optimizer_dec.zero_grad(set_to_none=True)
    else:
        optimizer_enc.zero_grad(set_to_none=True)
        optimizer_dec.zero_grad(set_to_none=True)

    mask_ratio_ms  = args.masking_ms
    mask_ratio_mod = args.masking_modality

    accum_steps = max(1, args.accumulation_steps)
    accum_counter = 0

    def step_optimizers():
        if args.train_mode == "decoder_only":
            if args.use_autocast and scaler is not None:
                scaler.step(optimizer_dec); scaler.update()
            else:
                optimizer_dec.step()
            optimizer_dec.zero_grad(set_to_none=True)
        else:
            if args.use_autocast and scaler is not None:
                scaler.step(optimizer_enc)
                scaler.step(optimizer_dec)
                scaler.update()
            else:
                optimizer_enc.step()
                optimizer_dec.step()
            optimizer_enc.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)

    for batch in dataloader:
        image_ms = batch['image'].to(device, non_blocking=True)
        elevation = sanitize(batch['dsm'].to(device, non_blocking=True))

        # Build masks from raw targets (before sanitizing)
        target_chm_raw = batch['chm'].to(device, non_blocking=True)
        target_bm_raw  = batch['bm'].to(device, non_blocking=True)
        valid_chm = torch.isfinite(target_chm_raw) & (target_chm_raw >= 0)
        valid_bm  = torch.isfinite(target_bm_raw)  & (target_bm_raw  >= 0)

        # Sanitized copies for arithmetic
        target_chm = sanitize(target_chm_raw.clone() )
        target_bm  = sanitize(target_bm_raw.clone())
        target_chm = torch.where(valid_chm, target_chm, torch.zeros_like(target_chm))
        target_bm  = torch.where(valid_bm,  target_bm,  torch.zeros_like(target_bm))

        gt = batch['geo_transform'].to(device, non_blocking=True)

        # Forward
        if args.train_mode == "decoder_only":
            with torch.no_grad():
                outputs_enc = model_enc(
                    image_ms, elevation, gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )
            with amp_ctx:
                prediction = model_dec(outputs_enc)
        else:
            with amp_ctx:
                outputs_enc = model_enc(
                    image_ms, elevation, gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )
                prediction = model_dec(outputs_enc)

        # Expect (B, 2, H, W)
        assert prediction.ndim >= 2 and prediction.size(1) == 2, \
            f"Decoder must output 2 channels (CHM,Biomass). Got shape: {tuple(prediction.shape)}"

        pred_chm = prediction[:, 0, ...]
        pred_bm  = prediction[:, 1, ...]

        # Primary masked losses
        loss_chm = loss_function(pred_chm, target_chm, mask=valid_chm, reduction="mean")
        loss_bm  = loss_function(pred_bm,  target_bm,  mask=valid_bm, reduction="mean")
        masked_loss = alpha * loss_chm + beta * loss_bm

        # Background-only auxiliary loss (valid & ~zero targets)
        eps = 1e-6
        bg_chm = valid_chm & (target_chm <= eps)
        bg_bm  = valid_bm  & (target_bm  <= eps)

        # Since targets are ~0, this is just ||pred||^2 on background
        bg_loss_chm = loss_function(pred_chm, torch.zeros_like(target_chm), mask=bg_chm, reduction="mean")
        bg_loss_bm  = loss_function(pred_bm,  torch.zeros_like(target_bm),  mask=bg_bm, reduction="mean")

        lambda_bg = getattr(args, "lambda_bg", 0.1)  # e.g., 0.05–0.2 works well
        raw_loss = masked_loss + lambda_bg * (alpha * bg_loss_chm + beta * bg_loss_bm)

        # Distributed non-finite guard
        finite_flag = torch.isfinite(raw_loss).to(dtype=torch.int32, device=raw_loss.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

        if finite_flag.item() == 0:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                print("Non-finite loss detected; skipping this micro-batch (all ranks).")
            continue

        # Scale for accumulation (do not use the scaled value for logging)
        loss = raw_loss / accum_steps

        # Backward
        if args.use_autocast and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_counter += 1
        if accum_counter % accum_steps == 0:
            step_optimizers()

        # Accounting (use unscaled loss)
        bsz = image_ms.size(0)
        running_loss     += safe_item(raw_loss) * bsz
        training_chm_loss += safe_item(loss_chm) * bsz
        training_bm_loss  += safe_item(loss_bm)  * bsz

    # Flush leftover grads if the last window was partial
    if accum_counter % accum_steps != 0:
        step_optimizers()

    # Step schedulers once per epoch (after optimizer steps)
    if scheduler_enc is not None:
        scheduler_enc.step()
    if scheduler_dec is not None:
        scheduler_dec.step()

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss, {"chm": training_chm_loss, "bm": training_bm_loss}


@torch.no_grad()
def validate_one_epoch_finetuning_geo_reg_chmbm(
    model_enc, model_dec, dataloader, device, args,
    criterion=None, alpha=1.0, beta=1.0, log_preds_wb=False):
    model_enc.eval()
    model_dec.eval()

    # ---- Masked Loss | NaN-safe
    # def masked_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    #     # Accepts shapes [B,H,W] or [B,1,H,W]
    #     pred = pred.float()
    #     tgt  = tgt.float()
    #     valid = torch.isfinite(pred) & torch.isfinite(tgt)
    #     n = valid.sum()
    #     if n == 0:
    #         return torch.zeros((), device=pred.device, dtype=pred.dtype)  # finite zero, OK for backward
    #     diff = (pred - tgt) ** 2
    #     diff = torch.where(valid, diff, torch.zeros_like(diff))
    #     return diff.sum() / n

    # Default constraints
    # CHM_MAX = 40.0  # choose a safe cap for Saudi sites
    # BM_MAX  = 15.0  # choose p95 or a safe cap for biomass

    amp_ctx = torch.amp.autocast("cuda") if args.use_autocast else nullcontext()
    loss_function = criterion if criterion is not None else MaskedMSELoss()

    total, sum_loss, sum_chm, sum_bm = 0, 0.0, 0.0, 0.0

    def log_val_wandb(
            image_tensor,
            elev_tensor,
            pred_chm_tensor,
            pred_bm_tensor,
            gt_chm_tensor,
            gt_bm_tensor,
            ):
        np_image = image_tensor[0][[3, 2, 1], :, :].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
        elev_image = elev_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                 # [H,W]
        pred_chm = pred_chm_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                 # [H,W]
        pred_bm = pred_bm_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                 # [H,W]
        gt_chm = gt_chm_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                     # [H,W]
        gt_bm = gt_bm_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                     # [H,W]

        # Shade from the northwest, with the sun 45 degrees from horizontal
        ls = LightSource(azdeg=315, altdeg=45)
        cmap = plt.cm.gist_earth

        fig, ax = plt.subplots(2, 3, figsize=(8, 3), dpi=72)

        # Original Image
        ax[0,0].imshow(np_image)
        ax[0,0].set_title('MS Image')
        ax[0,0].axis('off')

        # Ground Truth Canopy Height Model
        gt_plot = ax[0,1].imshow(gt_chm, cmap='viridis')
        ax[0,1].set_title('CHM Truth')
        ax[0,1].axis('off')
        plt.colorbar(gt_plot, ax=ax[0,1], fraction=0.046, pad=0.04)

        # Ground Truth Biomass
        gt_plot = ax[0,2].imshow(gt_bm, cmap='viridis')
        ax[0,2].set_title('BM Truth')
        ax[0,2].axis('off')
        plt.colorbar(gt_plot, ax=ax[0,2], fraction=0.046, pad=0.04)

        # Original Image
        ax[1,0].imshow(ls.shade(elev_image, cmap=cmap, blend_mode='hsv', vert_exag=2))
        #dsm_plot = ax[1,0].imshow(elev_image, cmap='viridis')
        ax[1,0].set_title('DSM')
        ax[1,0].axis('off')
        #plt.colorbar(dsm_plot, ax=ax[1,0], fraction=0.046, pad=0.04)

        # Predicted Canopy Height Model
        pred_plot = ax[1,1].imshow(pred_chm, cmap='viridis')
        ax[1,1].set_title('Pred. CHM')
        ax[1,1].axis('off')
        plt.colorbar(pred_plot, ax=ax[1,1], fraction=0.046, pad=0.04)

        # Predicted Biomass
        pred_plot = ax[1,2].imshow(pred_bm, cmap='viridis')
        ax[1,2].set_title('Pred. BM')
        ax[1,2].axis('off')
        plt.colorbar(pred_plot, ax=ax[1,2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Log the figure to W&B
        wandb.log({
            "regression_results": wandb.Image(fig)
        })

        plt.close(fig)

    for batch in dataloader:
        image_ms  = batch['image'].to(device, non_blocking=True)
        elevation = batch.get('dsm', batch.get('elevation')).to(device, non_blocking=True)
        target_chm = torch.nan_to_num(batch['chm'].to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
        target_bm  = torch.nan_to_num(batch['bm'].to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
        gt = batch['geo_transform'].to(device, non_blocking=True)

        bs = image_ms.size(0)

        with amp_ctx:
            outputs_enc = model_enc(image_ms, elevation, gt,
                                    mask_ratio_ms=args.masking_ms,
                                    mask_ratio_mods=args.masking_modality,
                                    return_intermediate=True)
            prediction = model_dec(outputs_enc)

            assert prediction.size(1) == 2, f"Expected 2-channel output, got {tuple(prediction.shape)}"
            pred_chm = prediction[:, 0, ...]
            pred_bm  = prediction[:, 1, ...]

            # Physical clamps tune to truth data
            # target_chm_s = (target_chm / CHM_MAX).clamp_(0, 1)
            # target_bm_s = (target_bm  / BM_MAX ).clamp_(0, 1)

            #pred_chm_s = torch.sigmoid(pred_chm)    # [0..1]
            #pred_bm_s  = torch.sigmoid(pred_bm)    # [0..1]

            # loss_chm = criterion(pred_chm_s, target_chm_s)
            # loss_bm  = criterion(pred_bm_s,  target_bm_s)
            loss_chm = loss_function(pred_chm, target_chm, mask=None,  reduction="none_for_epoch")
            loss_bm  = loss_function(pred_bm,  target_bm, mask=None,  reduction="none_for_epoch")

            loss = (alpha * loss_chm + beta * loss_bm)

        sum_chm  += loss_chm.detach().item() * bs
        sum_bm   += loss_bm.detach().item()  * bs
        sum_loss += loss.detach().item()     * bs
        total    += bs

        if log_preds_wb and getattr(args, "rank", 0) == 0:
            #pred_chm = pred_chm * CHM_MAX
            #pred_bm  = pred_bm  * BM_MAX

            log_val_wandb(
                image_ms,
                elevation,
                pred_chm,
                pred_bm,
                target_chm,
                target_bm,
                )

    # Reducing across ranks, all_reduce sum_* and total here

    mean_chm, mean_bm, mean_total = sum_chm/total, sum_bm/total, sum_loss/total
    print(f"Validation Loss -> CHM: {mean_chm:.4f} | BM: {mean_bm:.4f} | Total: {mean_total:.4f}")

    return mean_total, {"chm": mean_chm, "bm": mean_bm}

def finetune_one_epoch_geo_reg(
    model_enc, model_dec, dataloader,
    optimizer_enc, optimizer_dec, device, args, criterion=None, scaler=None
):
    # ---- Modes & freezing ---------------------------------------------------
    if args.train_mode == "decoder_only":
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for p in model_enc.parameters():
            p.requires_grad = False
        model_enc.eval()
        optimizer_enc = None
    else:
        model_enc.train()

    model_dec.train()

    def sanitize(x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    def step_optimizers():
        if args.train_mode == "decoder_only":
            if args.use_autocast and scaler is not None:
                scaler.step(optimizer_dec); scaler.update()
            else:
                optimizer_dec.step()
            optimizer_dec.zero_grad(set_to_none=True)
        else:
            if args.use_autocast and scaler is not None:
                scaler.step(optimizer_enc)
                scaler.step(optimizer_dec)
                scaler.update()
            else:
                optimizer_enc.step()
                optimizer_dec.step()
            optimizer_enc.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_bg_loss = 0.0
    running_dist_loss = 0.0
    running_count = 0.0
    accum_steps = max(1, args.accumulation_steps)
    accum_counter = 0

    # Choose AMP/null context once
    amp_ctx = torch.amp.autocast("cuda") if args.use_autocast else nullcontext()
    loss_function = criterion if criterion is not None else MaskedMSELoss()

    # Clear grads at the *start* of an accumulation window
    if args.train_mode == "decoder_only":
        optimizer_dec.zero_grad(set_to_none=True)
    else:
        optimizer_enc.zero_grad(set_to_none=True)
        optimizer_dec.zero_grad(set_to_none=True)

    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device, non_blocking=True)
        elevation = batch['modality'].to(device, non_blocking=True)
        target_raw = batch['target'].to(device, non_blocking=True)
        gt = batch['geo_transform'].to(device, non_blocking=True)

        # Build masks from raw targets (before sanitizing)        
        valid = torch.isfinite(target_raw) & (target_raw >= 0)

        # Sanitized copies for arithmetic
        target = sanitize(target_raw.clone() )
        target = torch.where(valid, target, torch.zeros_like(target))

        # sanitize target
        if not torch.isfinite(target).all():
            target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        mask_ratio_ms  = args.masking_ms
        mask_ratio_mod = args.masking_modality

        # ---- Forward pass (compute encoder ONCE) ----------------------------
        if args.train_mode == "decoder_only":
            with torch.no_grad():
                # encoder is frozen; saving memory by disabling autograd is OK
                outputs_enc = model_enc(
                    image_ms, elevation, gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )
            with amp_ctx:
                prediction = model_dec(outputs_enc)
        else:
            with amp_ctx:
                outputs_enc = model_enc(
                    image_ms, elevation, gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )
                prediction = model_dec(outputs_enc)
        #######################################
        # Primary masked losses
        masked_loss = loss_function(prediction, target, mask=valid, reduction="mean")
        

        # Background-only auxiliary loss (valid & ~zero targets)
        eps = 1e-6
        bg_mask = valid & (target <= eps)

        # Since targets are ~0, this is just ||pred||^2 on background
        bg_loss = loss_function(prediction, torch.zeros_like(target), mask=bg_mask, reduction="mean")
        lambda_bg = getattr(args, "lambda_bg", 0.1)  # 0.05–0.2 works well

        # Distribution loss
        dist_loss = distribution_loss(prediction, target, mask=bg_mask, bins=50, sigma=0.02)

        raw_loss = masked_loss + lambda_bg * bg_loss + 0.1* dist_loss

        # Distributed non-finite guard
        finite_flag = torch.isfinite(raw_loss).to(dtype=torch.int32, device=raw_loss.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

        if finite_flag.item() == 0:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                print("Non-finite loss detected; skipping this micro-batch (all ranks).")
            continue

        # Scale for accumulation (do not use the scaled value for logging)
        loss = raw_loss / accum_steps

        # Backward
        if args.use_autocast and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_counter += 1
        if accum_counter % accum_steps == 0:
            step_optimizers()

        # for logging: accumulate *sum of squared errors* and counts
        with torch.no_grad():
            sse, n = loss_function(prediction, target, mask=valid, reduction="none_for_epoch")
            running_loss += float(sse.item())
            running_count    += int(n.item())

        running_bg_loss += bg_loss.item()
        running_dist_loss += dist_loss.item()
        
    # Flush leftover grads if the last window was partial
    if accum_counter % accum_steps != 0:
        step_optimizers()

    epoch_loss = running_loss / max(running_count, 1)
    epoch_bg_loss = running_bg_loss / max(running_count, 1)
    epoch_dist_loss = running_dist_loss / max(running_count, 1)

    if is_main_process():
        print(f'Training Loss: {epoch_loss:.4f}')

    return epoch_loss, {"bg_loss":epoch_bg_loss, "dist_loss":epoch_dist_loss}
        

@torch.no_grad()
def validate_one_epoch_finetuning_geo_reg(
    model_enc, model_dec, dataloader, device, args, criterion=None, log_preds_wb=False
):
    """
    Validation for one epoch (DDP-safe). Uses inference_mode + autocast (optional).
    Computes a true GLOBAL mean loss if distributed; otherwise local mean.
    """

    model_enc.eval()
    model_dec.eval()
    
    def log_val_wandb(
            image_tensor,
            mod_tensor,
            pred_tensor,
            gt_tensor
            ):
        np_image = image_tensor[0][[3, 2, 1], :, :].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
        mod_image = mod_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                 # [H,W]
        pred_im = pred_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                 # [H,W]
        gt_im = gt_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                     # [H,W]

        vmin = np.nanmin([pred_im])
        vmax = np.nanmax([gt_im])

        # Shade from the northwest, with the sun 45 degrees from horizontal
        ls = LightSource(azdeg=315, altdeg=45)
        cmap = plt.cm.gist_earth

        fig, ax = plt.subplots(1, 4, figsize=(8, 3), dpi=72)

        # Multispectral Image
        ax[0].imshow(np_image)
        ax[0].set_title('MS Image')
        ax[0].axis('off')
        
        # Modality Image
        ax[1].imshow(ls.shade(mod_image, cmap=cmap, blend_mode='hsv', vert_exag=2))
        ax[1].set_title('Modality')
        ax[1].axis('off')
        
        # Ground Truth
        gt_plot = ax[2].imshow(gt_im, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[2].set_title('Target Truth')
        ax[2].axis('off')
        plt.colorbar(gt_plot, ax=ax[2], fraction=0.046, pad=0.04)       

        # Predicted
        pred_plot = ax[3].imshow(pred_im, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[3].set_title('Pred. Target')
        ax[3].axis('off')
        plt.colorbar(pred_plot, ax=ax[3], fraction=0.046, pad=0.04)


        plt.tight_layout()

        # Log the figure to W&B
        wandb.log({
            "regression_results": wandb.Image(fig)
        })

        plt.close(fig)

    # AMP context
    amp_ctx = torch.amp.autocast('cuda') if args.use_autocast else nullcontext()
    loss_function = criterion if criterion is not None else MaskedMSELoss()

    # local accumulators
    running_loss = 0.0 # sum of squared errors
    running_count = 0  # number of valid pixels

    # Disable autograd entirely (faster & safer than no_grad for eval)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            # --- inputs ---
            image_ms  = batch['image'].to(device, non_blocking=True)
            #tensor_stats(image_ms, "MS")
            modality = batch['modality'].to(device, non_blocking=True)
            #tensor_stats(modality, "Modality")
            target    = batch['target'].to(device, non_blocking=True)
            #tensor_stats(target, "target")
            gt        = batch['geo_transform'].to(device, non_blocking=True)

            # sanitize target to avoid NaN/Inf poisoning the loss
            if not torch.isfinite(target).all():
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

            # masking policy during val; keep your flags, or set to 0.0 if you prefer full context
            mask_ratio_ms  = args.masking_ms
            mask_ratio_mod = args.masking_modality

            # --- forward (ONE encoder pass) ---
            with amp_ctx:
                outputs_enc = model_enc(
                    image_ms, modality, gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )
                prediction = model_dec(outputs_enc)
                #loss = criterion(prediction.float(), target.float())

                # global SSE + count for this batch
                sse, n_valid = loss_function(
                    prediction.float(),
                    target.float(),
                    mask=None,                      # will be created inside the function
                    reduction="none_for_epoch"
                )

            if torch.isfinite(sse):
                running_loss  += float(sse.item())
                running_count += int(n_valid.item())

            # Optional: log a few samples on main process only
            if log_preds_wb and is_main_process() and (batch_idx % getattr(args, "val_log_every", 100) == 0):
                # assumes you have this util already
                log_val_wandb(image_ms, modality, prediction, target)

    # ---- aggregate across ranks (GLOBAL mean) ----
    if is_dist_avail_and_initialized():
        loss_tensor  = torch.tensor([running_loss], device=device)
        count_tensor = torch.tensor([running_count], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        total_loss  = loss_tensor.item()
        total_count = int(count_tensor.item())
        epoch_loss  = (total_loss / max(total_count, 1)) if total_count > 0 else float('nan')
    else:
        epoch_loss = running_loss / max(running_count, 1)

    if is_main_process():
        print(f'Validation Loss: {epoch_loss:.4f}')

    return epoch_loss


def _normalize_for_display(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize an image to [0, 1] for visualization.
    Works for HxW or HxWxC arrays.
    """
    img = img.astype(np.float32)
    img_min = np.nanmin(img)
    img_max = np.nanmax(img)
    if img_max - img_min < eps:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - img_min) / (img_max - img_min), 0.0, 1.0)


def log_scene_classification_wandb(
    image_tensor,
    elev_tensor,
    pred_logits,
    gt_tensor,
    args,
    cam=None,
    max_items=1,
    log_gt_cam_when_wrong=True,
):
    """
    Log scene classification examples to W&B with optional CAM heatmaps.

    Args:
        image_tensor (torch.Tensor): [B, C, H, W]
        elev_tensor (torch.Tensor | None): [B, 1, H, W] or None
        pred_logits (torch.Tensor): [B, num_classes]
        gt_tensor (torch.Tensor): [B]
        args: must contain class_names
        cam (torch.Tensor | None): [B, num_classes, Hc, Wc]
        max_items (int): number of samples from batch to log
        log_gt_cam_when_wrong (bool): if True, also show GT CAM when prediction is wrong
    """
    if isinstance(args.class_names, str):
        class_names = [name.strip() for name in args.class_names.split(",")]
    else:
        class_names = list(args.class_names)

    batch_size = image_tensor.shape[0]
    n_items = min(batch_size, max_items)

    logs = {}

    for b in range(n_items):
        # --------- image selection for display ---------
        # EuroSAT-MS suggested false color from your current convention: [3,2,1]
        # Adjust if needed depending on your band order.
        rgb = image_tensor[b][[3, 2, 1], :, :].detach().cpu().permute(1, 2, 0).numpy()
        rgb = _normalize_for_display(rgb)

        elev_img = None
        if elev_tensor is not None:
            elev_img = elev_tensor[b, 0].detach().cpu().numpy()
            elev_img = _normalize_for_display(elev_img)

        # --------- decode labels ---------
        pred_class_idx = torch.argmax(pred_logits[b]).item()
        gt_class_idx = gt_tensor[b].view(-1)[0].item() if gt_tensor[b].ndim > 0 else gt_tensor[b].item()

        pred_label = class_names[pred_class_idx]
        gt_label = class_names[gt_class_idx]

        correct = (pred_class_idx == gt_class_idx)

        # --------- build figure layout ---------
        show_cam = cam is not None
        show_gt_cam = show_cam and log_gt_cam_when_wrong and (not correct)

        if elev_img is not None and show_gt_cam:
            ncols = 4
        elif elev_img is not None or show_gt_cam:
            ncols = 3
        else:
            ncols = 2 if show_cam else 1

        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), dpi=100)
        if ncols == 1:
            axes = [axes]
        else:
            axes = np.atleast_1d(axes)

        col = 0

        # --------- panel 1: RGB / false color ---------
        axes[col].imshow(rgb)
        axes[col].axis("off")
        axes[col].set_title("Input")
        col += 1

        # --------- panel 2: elevation ---------
        if elev_img is not None:
            ls = LightSource(azdeg=315, altdeg=45)
            shaded = ls.shade(elev_img, cmap=plt.cm.gist_earth, blend_mode="hsv", vert_exag=2)
            axes[col].imshow(shaded)
            axes[col].axis("off")
            axes[col].set_title("Elevation")
            col += 1

        # --------- panel 3: predicted CAM overlay ---------
        if show_cam:
            pred_cam = cam[b, pred_class_idx].detach().float().cpu()  # [Hc, Wc]
            pred_cam = pred_cam.unsqueeze(0).unsqueeze(0)  # [1,1,Hc,Wc]
            pred_cam = F.interpolate(
                pred_cam,
                size=(rgb.shape[0], rgb.shape[1]),
                mode="bilinear",
                align_corners=False
            ).squeeze().numpy()

            pred_cam = _normalize_for_display(pred_cam)

            axes[col].imshow(rgb)
            axes[col].imshow(pred_cam, alpha=0.45, cmap="jet")
            axes[col].axis("off")
            axes[col].set_title(f"Pred CAM: {pred_label}")
            col += 1

        # --------- panel 4: GT CAM overlay when wrong ---------
        if show_gt_cam:
            gt_cam = cam[b, gt_class_idx].detach().float().cpu()
            gt_cam = gt_cam.unsqueeze(0).unsqueeze(0)
            gt_cam = F.interpolate(
                gt_cam,
                size=(rgb.shape[0], rgb.shape[1]),
                mode="bilinear",
                align_corners=False
            ).squeeze().numpy()

            gt_cam = _normalize_for_display(gt_cam)

            axes[col].imshow(rgb)
            axes[col].imshow(gt_cam, alpha=0.45, cmap="jet")
            axes[col].axis("off")
            axes[col].set_title(f"GT CAM: {gt_label}")
            col += 1

        # --------- figure title ---------
        if correct:
            title_text = f"✓ Pred: {pred_label} | GT: {gt_label}"
            title_color = "green"
        else:
            title_text = f"✗ Pred: {pred_label} | GT: {gt_label}"
            title_color = "red"

        fig.suptitle(title_text, fontsize=12, color=title_color)
        plt.tight_layout()

        logs[f"scene_classification/sample_{b}"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(logs)



def finetune_one_epoch_geo(
    model_enc,
    model_dec,
    dataloader,
    optimizer_enc,
    optimizer_dec,
    criterion,
    device,
    args,
    scaler=None
):
    decoder_only = (args.train_mode == "decoder_only")

    if decoder_only:
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for param in model_enc.parameters():
            param.requires_grad = False
        model_enc.eval()
        optimizer_enc = None
    else:
        model_enc.train()

    model_dec.train()

    if optimizer_enc is not None:
        optimizer_enc.zero_grad(set_to_none=True)
    optimizer_dec.zero_grad(set_to_none=True)

    running_loss = 0.0
    accumulation_steps = args.accumulation_steps

    for i, batch in enumerate(dataloader):
        image_ms = batch["image"].to(device, non_blocking=True)
        modalities = batch["modalities"].to(device, non_blocking=True)

        has_modalities = bool(batch["has_modalities"][0].item())
        #print(has_modalities)
        if not has_modalities:
            #print("No modalities")
            modalities = None

        if args.task == "segmentation":
            target = batch["target"].squeeze(1).to(device, non_blocking=True).long()
        elif args.task == "classification":
            target = batch["target"].to(device, non_blocking=True).view(-1).long()
        else:
            target = batch["target"].to(device, non_blocking=True)
            nan_inf_mask = torch.isnan(target) | torch.isinf(target)
            if nan_inf_mask.any():
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        gt = batch["geo_transform"].to(device, non_blocking=True)

        # Recommended for downstream finetuning:
        mask_ratio_ms = 0.0
        mask_ratio_mod = 0.0

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")) if args.use_autocast else nullcontext():
            outputs_enc = model_enc(
                image_ms,
                modalities,
                gt,
                mask_ratio_ms=mask_ratio_ms,
                mask_ratio_mods=mask_ratio_mod,
                return_intermediate=True
            )

            prediction = model_dec(outputs_enc)

            if args.task in ["segmentation", "classification"]:
                loss = criterion(prediction, target)
            else:
                loss = criterion(prediction.float(), target.float())

        if torch.isfinite(loss):
            loss_to_backprop = loss / accumulation_steps

            if args.use_autocast and scaler is not None:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if (i + 1) % accumulation_steps == 0:
                if decoder_only:
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_dec.step()
                    optimizer_dec.zero_grad(set_to_none=True)
                else:
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_enc)
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_enc.step()
                        optimizer_dec.step()

                    optimizer_enc.zero_grad(set_to_none=True)
                    optimizer_dec.zero_grad(set_to_none=True)

            running_loss += loss.item() * image_ms.size(0)

    # Final partial accumulation step
    remainder = len(dataloader) % accumulation_steps
    if remainder != 0:
        if decoder_only:
            if args.use_autocast and scaler is not None:
                scaler.step(optimizer_dec)
                scaler.update()
            else:
                optimizer_dec.step()
            optimizer_dec.zero_grad(set_to_none=True)
        else:
            if args.use_autocast and scaler is not None:
                scaler.step(optimizer_enc)
                scaler.step(optimizer_dec)
                scaler.update()
            else:
                optimizer_enc.step()
                optimizer_dec.step()

            optimizer_enc.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Training Loss: {epoch_loss:.4f}")

    return epoch_loss

def validate_one_epoch_finetuning_geo(
    model_enc,
    model_dec,
    dataloader,
    criterion,
    device,
    args,
    log_preds_wb: bool = False,
    return_cam: bool = False,
):
    model_enc.eval()
    model_dec.eval()


    running_loss = 0.0
    split_name = getattr(args, "split", "val")
    rank = getattr(args, "rank", 0)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")) if args.use_autocast else nullcontext():
            
            epoch_preds = []
            epoch_targets = []
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, batch in enumerate(dataloader):
                image_ms = batch["image"].to(device, non_blocking=True)
                modalities = batch["modalities"].to(device, non_blocking=True)

                has_modalities = bool(batch["has_modalities"][0].item())
                #print(has_modalities)
                if not has_modalities:
                    #print("No modalities")
                    modalities = None

                if args.task == "segmentation":
                    target = batch["target"].squeeze(1).to(device, non_blocking=True).long()
                elif args.task == "classification":
                    target = batch["target"].to(device, non_blocking=True).view(-1).long()
                else:
                    target = batch["target"].to(device, non_blocking=True)
                    nan_inf_mask = torch.isnan(target) | torch.isinf(target)
                    if nan_inf_mask.any():
                        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)

                gt = batch["geo_transform"].to(device, non_blocking=True)

                # Recommended for downstream validation:
                mask_ratio_ms = 0.0
                mask_ratio_mod = 0.0

                outputs_enc = model_enc(
                    image_ms,
                    modalities,
                    gt,
                    mask_ratio_ms=mask_ratio_ms,
                    mask_ratio_mods=mask_ratio_mod,
                    return_intermediate=True
                )

                prediction = model_dec(outputs_enc, return_cam=return_cam)

                if return_cam:
                    logits, cam = prediction
                else:
                    logits = prediction
                    cam = None

                if args.task in ["segmentation", "classification"]:
                    loss = criterion(logits, target)
                else:
                    loss = criterion(logits.float(), target.float())

                if torch.isfinite(loss):
                    running_loss += loss.item() * image_ms.size(0)

                # Evaluate accuracy
                preds = torch.argmax(logits, dim=1)
                epoch_correct += (preds == target).sum().item()
                epoch_samples += target.numel()
                epoch_preds.append(preds.cpu().numpy())
                epoch_targets.append(target.cpu().numpy())

                if log_preds_wb:
                    if args.task == "segmentation":
                        log_image_wandb(image_ms, logits, target)
                    elif args.task == "classification":
                        log_scene_classification_wandb(
                            image_ms, modalities, logits, target, args, cam=cam
                        )
                    else:
                        log_image_reg_wandb(image_ms, modalities, logits, target)

            epoch_preds = np.concatenate(epoch_preds, axis=0)
            epoch_targets = np.concatenate(epoch_targets, axis=0)
            
            # Overall accuracy.
            accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0

            precision, recall, f1, _ = skmetrics.precision_recall_fscore_support(
                epoch_targets,
                epoch_preds,
                labels=list(range(args.num_classes)),
                average="macro",
                zero_division=0
            )
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
            wandb.log(metrics)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Validation Loss: {epoch_loss:.4f}")

    return epoch_loss