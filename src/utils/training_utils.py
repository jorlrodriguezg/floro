import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from typing import Union, Dict, Optional
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module
from torch.optim import Optimizer
import wandb
from contextlib import nullcontext

# Helper function to resize and apply mask
def apply_mask_loss(image, mask, patch_size=16):
    batch_size, _, height, width = image.shape
    mask = mask.view(batch_size, 1, height // patch_size, width // patch_size)  # (B, 1, H/patch_size, W/patch_size)
    mask = F.interpolate(mask.float(), size=(height, width), mode='nearest')  # Upsample the mask
    mask = mask.repeat(1, image.shape[1], 1, 1).bool()  # (B, C, H, W)
    return mask

def compute_loss_with_fallback(prediction, target, mask, criterion):
    """
    Computes the loss using the given criterion.
    If mask is completely False (i.e. no unmasked pixels), computes the loss on the entire prediction and target.
    Otherwise, computes the loss only on the unmasked regions.
    """
    # Check if any pixel is unmasked
    if not mask.any():
        return criterion(prediction, target)
    else:
        return criterion(prediction[mask], target[mask])

def train_one_epoch(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion1, criterion2, device, scaler, args):
    model_enc.train()  # Set the encoder to training mode
    model_dec.train()  # Set the decoder to training mode

    running_loss = 0.0
    accumulation_steps = args.accumulation_steps  # Number of steps to accumulate gradients

    optimizer_enc.zero_grad()
    optimizer_dec.zero_grad()

    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        target_ms = batch['target'].to(device)
        target_elev = batch['target_elev'].to(device)
        target_labels = batch['clusters'].to(device).long()
        
        with autocast():
            outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, mask_ratio=args.masking_amount)
            reconstructed_multispectral, reconstructed_modality, pred_labels = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality)
        
            # Calculate loss based on the decoder's output and the true labels
            # loss_ms = criterion(reconstructed_multispectral, target_ms)
            # loss_elev = criterion(reconstructed_modality, target_elev)
            # total_loss = loss_ms + loss_elev
            # Interpolate masks to the original image size
            mask_multi_resized = apply_mask_loss(image_ms, mask_multi, patch_size= args.patch_size)
            mask_modality_resized = apply_mask_loss(elevation, mask_modality, patch_size= args.patch_size)

            # Invert the masks to calculate loss on unmasked patches
            mask_multi_resized = mask_multi_resized
            mask_modality_resized = mask_modality_resized
    
            # Compute losses on unmasked patches
            ms_loss = criterion1(reconstructed_multispectral[mask_multi_resized], target_ms[mask_multi_resized])
            elev_loss = criterion1(reconstructed_modality[mask_modality_resized], target_elev[mask_modality_resized])
            cls_loss = criterion2(pred_labels, target_labels.squeeze(1))
            total_loss = ms_loss + elev_loss + cls_loss
        
        # Backward pass
        if not torch.isnan(total_loss):
            scaler.scale(total_loss).backward() # This will compute gradients for both dec and enc parts of the model

            # Update weights
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer_enc)
                scaler.step(optimizer_dec)
                scaler.update()

                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
            
            running_loss += total_loss.item() * image_ms.size(0)   
        

    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Training Loss: {epoch_loss:.4f}')

    return epoch_loss

def train_one_epoch_geo(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion, device, scaler, args):
    model_enc.train()  # Set the encoder to training mode
    model_dec.train()  # Set the decoder to training mode

    running_loss = 0.0
    accumulation_steps = args.accumulation_steps  # Number of steps to accumulate gradients

    optimizer_enc.zero_grad()
    optimizer_dec.zero_grad()
    
    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        target_ms = batch['target'].to(device)
        target_elev = batch['target_elev'].to(device)
        #target_labels = batch['clusters'].to(device).long()
        gt = batch['geo_transform'].to(device)
        # Total unmasked ratio (16% of the patches will be unmasked, 84% will be masked)
        unmasked_ratio = 0.16
        ratio_ms_keep = np.random.uniform(low=0, high=unmasked_ratio)
        ratio_elev_keep = unmasked_ratio - ratio_ms_keep
        ratio_ms = 1 - ratio_ms_keep
        ratio_elev = 1 - ratio_elev_keep
        
        with autocast():
            outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, gt, mask_ratio_ms=ratio_ms, mask_ratio_elev=ratio_elev)
            reconstructed_multispectral, reconstructed_modality = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt)
            # total_loss = loss_ms + loss_elev
            # Interpolate masks to the original image size
            mask_multi_resized = apply_mask_loss(image_ms, mask_multi, patch_size= args.patch_size)
            mask_modality_resized = apply_mask_loss(elevation, mask_modality, patch_size= args.patch_size)

            # Invert the masks to calculate loss on unmasked patches
            mask_multi_resized = mask_multi_resized
            mask_modality_resized = mask_modality_resized
    
            # Compute losses on unmasked patches
            ms_loss = criterion(reconstructed_multispectral[mask_multi_resized], target_ms[mask_multi_resized])
            elev_loss = criterion(reconstructed_modality[mask_modality_resized], target_elev[mask_modality_resized])
            total_loss = ms_loss + elev_loss
        
        # Backward pass
        if not torch.isnan(total_loss):
            scaler.scale(total_loss).backward() # This will compute gradients for both dec and enc parts of the model
            
            # Update weights
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer_enc)
                scaler.step(optimizer_dec)
                scaler.update()

                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
            
            running_loss += total_loss.item() * image_ms.size(0)   
        

    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Training Loss: {epoch_loss:.4f}')

    return epoch_loss

def train_one_epoch_geo_sc(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion, device, scaler, args):
    model_enc.train()  # Set the encoder to training mode
    model_dec.train()  # Set the decoder to training mode

    running_loss = 0.0
    accumulation_steps = args.accumulation_steps  # Number of steps to accumulate gradients

    optimizer_enc.zero_grad()
    optimizer_dec.zero_grad()
    
    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        target_ms = batch['target'].to(device)
        target_elev = batch['target_elev'].to(device)
        #target_labels = batch['clusters'].to(device).long()
        gt = batch['geo_transform'].to(device)
        # Total unmasked ratio (16% of the patches will be unmasked, 84% will be masked)
        unmasked_ratio = 0.16
        ratio_ms_keep = np.random.uniform(low=0, high=unmasked_ratio)
        ratio_elev_keep = unmasked_ratio - ratio_ms_keep
        ratio_ms = 1 - ratio_ms_keep
        ratio_elev = 1 - ratio_elev_keep
        
        with autocast():
            """
            outputs_enc = model_enc(image_ms, elevation, gt, mask_ratio_ms=ratio_ms, mask_ratio_elev=ratio_elev)
            reconstructed_multispectral, reconstructed_modality = model_dec(outputs_enc, gt)
            # total_loss = loss_ms + loss_elev
            # Interpolate masks to the original image size
            mask_multi_resized = apply_mask_loss(image_ms, outputs_enc['mask_ms'], patch_size= args.patch_size)
            mask_modality_resized = apply_mask_loss(elevation, outputs_enc['mask_modality'], patch_size= args.patch_size)

            # Invert the masks to calculate loss on unmasked patches
            mask_multi_resized = mask_multi_resized
            mask_modality_resized = mask_modality_resized
    
            # Compute losses on unmasked patches
            ms_loss = criterion(reconstructed_multispectral[mask_multi_resized], target_ms[mask_multi_resized])
            elev_loss = criterion(reconstructed_modality[mask_modality_resized], target_elev[mask_modality_resized])
            total_loss = ms_loss + elev_loss
            """
            # Encode and decode the representations
            outputs_enc = model_enc(image_ms, elevation, gt, mask_ratio_ms=ratio_ms, mask_ratio_elev=ratio_elev)
            reconstructed_multispectral, reconstructed_modality = model_dec(outputs_enc, gt)
            
            # Interpolate the masks to the original image size
            mask_multi_resized = apply_mask_loss(image_ms, outputs_enc['mask_ms'], patch_size=args.patch_size)
            mask_modality_resized = apply_mask_loss(elevation, outputs_enc['mask_modality'], patch_size=args.patch_size)
            
            # Process the batch on a per-sample basis
            batch_size = image_ms.shape[0]
            ms_losses = []
            elev_losses = []
            
            for i in range(batch_size):
                ms_loss_sample = compute_loss_with_fallback(
                    reconstructed_multispectral[i],
                    target_ms[i],
                    mask_multi_resized[i],
                    criterion
                )
                elev_loss_sample = compute_loss_with_fallback(
                    reconstructed_modality[i],
                    target_elev[i],
                    mask_modality_resized[i],
                    criterion
                )
                ms_losses.append(ms_loss_sample)
                elev_losses.append(elev_loss_sample)
            
            # Average the losses over the batch
            ms_loss = sum(ms_losses) / batch_size
            elev_loss = sum(elev_losses) / batch_size
            total_loss = ms_loss + elev_loss
        
        # Backward pass
        if not torch.isnan(total_loss):
            scaler.scale(total_loss).backward() # This will compute gradients for both dec and enc parts of the model

            # Update weights
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer_enc)
                scaler.step(optimizer_dec)
                scaler.update()

                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
            
            running_loss += total_loss.item() * image_ms.size(0)   
        

    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Training Loss: {epoch_loss:.4f}')

    return epoch_loss

def validate_one_epoch(model_enc, model_dec, dataloader, criterion1, criterion2, device, args):
    model_enc.eval()  # Set the encoder to evaluation mode
    model_dec.eval()  # Set the decoder to evaluation mode
    
    running_loss = 0.0
        
    # No gradient computation for validation
    with torch.no_grad():
        for batch in dataloader:
            image_ms = batch['image'].to(device)
            elevation = batch['elevation'].to(device)
            target_ms = batch['target'].to(device)
            target_elev = batch['target_elev'].to(device)
            target_labels = batch['clusters'].to(device).long()
            
            with autocast():
                outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, mask_ratio=args.masking_amount)
                reconstructed_multispectral, reconstructed_modality, pred_labels = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality)
            
                # Calculate loss based on the decoder's output and the true labels
                # loss_ms = criterion(reconstructed_multispectral, target_ms)
                # loss_elev = criterion(reconstructed_modality, target_elev)
                # total_loss = loss_ms + loss_elev

                # Interpolate masks to the original image size
                mask_multi_resized = apply_mask_loss(image_ms, mask_multi, patch_size= args.patch_size)
                mask_modality_resized = apply_mask_loss(elevation, mask_modality, patch_size= args.patch_size)
    
                # Invert the masks to calculate loss on unmasked patches
                mask_multi_resized = mask_multi_resized
                mask_modality_resized = mask_modality_resized
        
                # Compute losses on unmasked patches
                ms_loss = criterion1(reconstructed_multispectral[mask_multi_resized], target_ms[mask_multi_resized])
                elev_loss = criterion1(reconstructed_modality[mask_modality_resized], target_elev[mask_modality_resized])
                cls_loss = criterion2(pred_labels, target_labels.squeeze(1))
                total_loss = ms_loss + elev_loss + cls_loss
            
            
                if not torch.isnan(total_loss):
                    running_loss += total_loss.item() * image_ms.size(0)
            
    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f}')

    return epoch_loss

def validate_one_epoch_geo(model_enc, model_dec, dataloader, criterion, device, args):
    model_enc.eval()  # Set the encoder to evaluation mode
    model_dec.eval()  # Set the decoder to evaluation mode
    
    running_loss = 0.0
        
    # No gradient computation for validation
    with torch.no_grad():
        for batch in dataloader:
            image_ms = batch['image'].to(device)
            elevation = batch['elevation'].to(device)
            target_ms = batch['target'].to(device)
            target_elev = batch['target_elev'].to(device)
            #target_labels = batch['clusters'].to(device).long()
            gt = batch['geo_transform'].to(device)
            # Total unmasked ratio (16% of the patches will be unmasked, 84% will be masked)
            unmasked_ratio = 0.16
            ratio_ms_keep = np.random.uniform(low=0, high=unmasked_ratio)
            ratio_elev_keep = unmasked_ratio - ratio_ms_keep
            ratio_ms = 1 - ratio_ms_keep
            ratio_elev = 1 - ratio_elev_keep
            
            with autocast():
                outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, gt, mask_ratio_ms=ratio_ms, mask_ratio_elev=ratio_elev)
                reconstructed_multispectral, reconstructed_modality = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt)
            
                # Calculate loss based on the decoder's output and the true labels
                # loss_ms = criterion(reconstructed_multispectral, target_ms)
                # loss_elev = criterion(reconstructed_modality, target_elev)
                # total_loss = loss_ms + loss_elev

                # Interpolate masks to the original image size
                mask_multi_resized = apply_mask_loss(image_ms, mask_multi, patch_size= args.patch_size)
                mask_modality_resized = apply_mask_loss(elevation, mask_modality, patch_size= args.patch_size)
    
                # Invert the masks to calculate loss on unmasked patches
                mask_multi_resized = mask_multi_resized
                mask_modality_resized = mask_modality_resized
        
                # Compute losses on unmasked patches
                ms_loss = criterion(reconstructed_multispectral[mask_multi_resized], target_ms[mask_multi_resized])
                elev_loss = criterion(reconstructed_modality[mask_modality_resized], target_elev[mask_modality_resized])
                total_loss = ms_loss + elev_loss
            
            
                if not torch.isnan(total_loss):
                    running_loss += total_loss.item() * image_ms.size(0)
            
    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f}')

    return epoch_loss

def validate_one_epoch_geo_sc(model_enc, model_dec, dataloader, criterion, device, args):
    model_enc.eval()  # Set the encoder to evaluation mode
    model_dec.eval()  # Set the decoder to evaluation mode
    
    running_loss = 0.0
        
    # No gradient computation for validation
    with torch.no_grad():
        for batch in dataloader:
            image_ms = batch['image'].to(device)
            elevation = batch['elevation'].to(device)
            target_ms = batch['target'].to(device)
            target_elev = batch['target_elev'].to(device)
            #target_labels = batch['clusters'].to(device).long()
            gt = batch['geo_transform'].to(device)
            # Total unmasked ratio (16% of the patches will be unmasked, 84% will be masked)
            unmasked_ratio = 0.16
            ratio_ms_keep = np.random.uniform(low=0, high=unmasked_ratio)
            ratio_elev_keep = unmasked_ratio - ratio_ms_keep
            ratio_ms = 1 - ratio_ms_keep
            ratio_elev = 1 - ratio_elev_keep
            
            with autocast():
                outputs_enc = model_enc(image_ms, elevation, gt, mask_ratio_ms=ratio_ms, mask_ratio_elev=ratio_elev)
                reconstructed_multispectral, reconstructed_modality = model_dec(outputs_enc, gt)
            
                # Calculate loss based on the decoder's output and the true labels
                # loss_ms = criterion(reconstructed_multispectral, target_ms)
                # loss_elev = criterion(reconstructed_modality, target_elev)
                # total_loss = loss_ms + loss_elev

                # Interpolate masks to the original image size
                mask_multi_resized = apply_mask_loss(image_ms, outputs_enc['mask_ms'], patch_size= args.patch_size)
                mask_modality_resized = apply_mask_loss(elevation, outputs_enc['mask_modality'], patch_size= args.patch_size)
    
                # Invert the masks to calculate loss on unmasked patches
                mask_multi_resized = mask_multi_resized
                mask_modality_resized = mask_modality_resized
        
                # Compute losses on unmasked patches
                ms_loss = criterion(reconstructed_multispectral[mask_multi_resized], target_ms[mask_multi_resized])
                elev_loss = criterion(reconstructed_modality[mask_modality_resized], target_elev[mask_modality_resized])
                total_loss = ms_loss + elev_loss
            
            
                if not torch.isnan(total_loss):
                    running_loss += total_loss.item() * image_ms.size(0)
            
    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f}')

    return epoch_loss

def save_model(model, optimizer, epoch, loss, save_dir, model_name="model", suffix="", dt=""):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Construct the file name
    file_name = f"{model_name}_{suffix}_dt{dt}.pth.jar"
    save_path = os.path.join(save_dir, file_name)

    # Save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

    print(f"Model {file_name} saved to {save_path}.")
    return save_path

def save_model_from_checkpoint(model, optimizer, epoch, loss, file_name):
    # Save the model
    save_path = file_name.replace(".pth.jar",f"_{epoch}.pth.jar")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

    print(f"Model {save_path} saved.")
    return save_path

def plot_ref_pred(image, pred, ref, group=None):
    pred = torch.argmax(pred, dim=1)
    im_np = image[0][[3, 2, 1],:,:].permute(1, 2, 0).cpu().numpy()
    pred_np = pred[0,:,:].squeeze().detach().cpu().numpy()
    ref_np = ref[0,:,:].squeeze().detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 5), ncols=3)

    ax1.imshow(im_np)
    ax1.set_title('Image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.imshow(pred_np, cmap='viridis', interpolation='none')
    ax2.set_title('Predicted Classes')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(ref_np, cmap='viridis', interpolation='none')
    ax3.set_title('Reference Classes')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    if group is not None:
        plt.title(group)

    plt.show()

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


def log_scene_classification_wandb(image_tensor, elev_tensor, pred_logits, gt_tensor, args):
    """
    Logs a multispectral image and elevation with predicted vs. ground truth class.

    Args:
        image_tensor (torch.Tensor): [1, C, H, W]
        elev_tensor (torch.Tensor): [1, 1, H, W]
        pred_logits (torch.Tensor): [B, num_classes]
        gt_tensor (torch.Tensor): [B]
        args: Argument object with class_names list
    """
    # Extract RGB (NIR, R, G) from channels
    np_image = image_tensor[0][[3, 2, 1], :, :].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    elev_image = elev_tensor[0, 0].detach().cpu().numpy()  # [H, W]

    # Decode predictions
    pred_class_idx = torch.argmax(pred_logits, dim=1)[0].item()
    gt_class_idx = gt_tensor[0].item()

    class_names = args.class_names
    # Ensure class_names is parsed as a list, even if passed as a comma-separated string
    if isinstance(args.class_names, str):
        args.class_names = [name.strip() for name in args.class_names.split(",")]
    
    pred_label = class_names[pred_class_idx]
    gt_label = class_names[gt_class_idx]
    

    # Visualization
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.gist_earth
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=72)

    ax[0].imshow(np_image)
    ax[0].axis('off')
    ax[0].set_title("Multispectral")

    ax[1].imshow(ls.shade(elev_image, cmap=cmap, blend_mode='hsv', vert_exag=2))
    ax[1].axis('off')
    ax[1].set_title("Elevation")

    # Title
    if pred_class_idx == gt_class_idx:
        title_text = f"✓ {pred_label}"
        title_color = "green"
    else:
        title_text = f"✗ Pred: {pred_label} | GT: {gt_label}"
        title_color = "red"

    fig.suptitle(title_text, fontsize=12, color=title_color)
    plt.tight_layout()

    wandb.log({
        "scene_classification": wandb.Image(fig)
    })
    plt.close(fig)




def log_image_reg_wandb(image_tensor, elev_tensor, pred_reg_tensor, gt_reg_tensor):
    np_image = image_tensor[0][[3, 2, 1], :, :].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    elev_image = elev_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                 # [H,W]
    pred_reg = pred_reg_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                 # [H,W]
    gt_reg = gt_reg_tensor[0, :, :].squeeze(0).detach().cpu().numpy()                     # [H,W]

    # Shade from the northwest, with the sun 45 degrees from horizontal
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.gist_earth

    fig, ax = plt.subplots(1, 4, figsize=(7, 3), dpi=72)

    # Original Image
    ax[0].imshow(np_image)
    ax[0].set_title('MS Image')
    ax[0].axis('off')

    # Original Image
    ax[1].imshow(ls.shade(elev_image, cmap=cmap, blend_mode='hsv', vert_exag=2))
    ax[1].set_title('Modality')
    ax[1].axis('off')

    # Predicted Regression Heatmap
    pred_plot = ax[2].imshow(pred_reg, cmap='viridis')
    ax[2].set_title('Predicted')
    ax[2].axis('off')
    plt.colorbar(pred_plot, ax=ax[2], fraction=0.046, pad=0.04)

    # Ground Truth Regression Heatmap
    gt_plot = ax[3].imshow(gt_reg, cmap='viridis')
    ax[3].set_title('Ground Truth')
    ax[3].axis('off')
    plt.colorbar(gt_plot, ax=ax[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Log the figure to W&B
    wandb.log({
        "regression_results": wandb.Image(fig)
    })

    plt.close(fig)


def finetune_one_epoch_geo(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion, device, args, scaler=None):
    if args.train_mode == "decoder_only":
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for param in model_enc.parameters():
            param.requires_grad = False
        model_enc.eval()  # Set encoder to evaluation mode.
        optimizer_dec.zero_grad()
        optimizer_enc = None
    else:
        model_enc.train()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

    model_dec.train()  # Set the decoder to training mode

    running_loss = 0.0
    accumulation_steps = args.accumulation_steps  # Number of steps to accumulate gradients

    #optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        if args.task == 'segmentation':  
            target = batch['target'].squeeze(1).to(device).long()
            #print(target.shape)
            #print("Target unique vals:", target.unique())
        elif args.task == 'classification':
            target = batch['target'].to(device)
            target = target.view(-1).long()  # in case it has shape [B, 1]
        else:
            target = batch['target'].to(device)
            # Check and handle NaN or Inf explicitly
            nan_inf_mask = torch.isnan(target) | torch.isinf(target)    
            if nan_inf_mask.any():
                #print(f"Detected NaN or inf in targets at batch index {i}. Handling...")
                target = torch.nan_to_num(target, nan=0.0, posinf=0, neginf=0)
        
        #target_labels = batch['clusters'].to(device).long()
        gt = batch['geo_transform'].to(device)
        # We want the model to see as much of the image context as possible
        
        mask_ratio_ms = args.masking_ms
        mask_ratio_mod = args.masking_modality
        
        context = autocast() if args.use_autocast else torch.no_grad()  # no_grad won't compute grads, so we need just normal context if no autocast
        with context if args.use_autocast else nullcontext():
            # Encode and decode the representations
            outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod)
            prediction = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt)

            if args.task == 'segmentation':
                loss = criterion(prediction, target)
            elif args.task == 'classification':
                loss = criterion(prediction, target)
            else:
                loss = criterion(prediction.float(), target.float())
            
        # Backward pass
        if not torch.isnan(loss) and not torch.isinf(loss):
            if args.use_autocast and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % args.accumulation_steps == 0:
                if args.train_mode == "decoder_only":
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_dec.step()
                    optimizer_dec.zero_grad()
                else:
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_enc)
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_enc.step()
                        optimizer_dec.step()
                    optimizer_enc.zero_grad()
                    optimizer_dec.zero_grad()
        
            running_loss += loss.item() * image_ms.size(0)   
        
    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Training Loss: {epoch_loss:.4f}')

    return epoch_loss

def validate_one_epoch_finetuning_geo(model_enc, model_dec, dataloader, criterion, device, args, log_preds_wb = False):
    """
    Validate the model for one epoch.
    
    This function sets the encoder and decoder to evaluation mode, and it uses
    torch.no_grad() (with optional autocast) to perform inference. It computes the 
    loss over the validation dataset without updating any model parameters.
    
    Parameters:
      model_enc: The encoder model.
      model_dec: The decoder model.
      dataloader: DataLoader for the validation dataset.
      criterion: Loss function (e.g., nn.CrossEntropyLoss).
      device: The device (CPU or GPU) for computations.
      args: Additional arguments (e.g., task type, fine-tune mode).
      
    Returns:
      epoch_loss: The average loss for the validation epoch.
    """
    
    # for param in model_enc.parameters():
    #     param.requires_grad = False
    # model_enc.eval()
    # for param in model_dec.parameters():
    #     param.requires_grad = False
    # model_dec.eval()
    
    running_loss = 0.0

    # Disable gradient computation for validation.
    with torch.no_grad():
        # Optionally use autocast for mixed precision inference.
        with autocast():
            for batch_idx, batch in enumerate(dataloader):
                # Get the input data and send to device.
                image_ms = batch['image'].to(device)
                elevation = batch['elevation'].to(device)
                
                # For segmentation tasks using CrossEntropyLoss, targets should be long.
                if args.task == 'segmentation':  
                    target = batch['target'].squeeze(1).to(device).long()
                elif args.task == 'classification':
                    target = batch['target'].to(device)
                    target = target.view(-1).long()  # Handle [B, 1] or scalar
                else:
                    target = batch['target'].to(device)
                    # Check and handle NaN or Inf explicitly
                    nan_inf_mask = torch.isnan(target) | torch.isinf(target)    
                    if nan_inf_mask.any():
                        #print(f"Detected NaN or inf in targets at batch index {i}. Handling...")
                        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
                
                gt = batch['geo_transform'].to(device)
                
                # During validation, we usually let the model see the full input.
                mask_ratio_ms = args.masking_ms
                mask_ratio_mod = args.masking_modality
                
                # Forward pass through encoder and decoder.
                outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod)
                prediction = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt)
                
                # Process the batch on a per-sample basis
                # batch_size = image_ms.shape[0]
                # losses = []
                
                # for i in range(batch_size):
                #     loss_sample = criterion(prediction[i], target[i])
                #     losses.append(loss_sample)    
                # # Average the losses over the batch
                # loss = sum(losses) / batch_size
                loss = criterion(prediction, target)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    running_loss += loss.item() * image_ms.size(0)
                # Log image to W&B
                if log_preds_wb == True:
                    if args.task == 'segmentation': 
                        log_image_wandb(image_ms, prediction, target)
                    elif args.task == 'classification':
                        log_scene_classification_wandb(image_ms, elevation, prediction, target, args)
                    else:
                        log_image_reg_wandb(image_ms, elevation, prediction, target)
                
    
    # Compute the epoch's average loss.
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    
    return epoch_loss

def finetune_one_epoch_hybrid(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion, device, args, scaler=None):
    if args.train_mode == "decoder_only":
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for param in model_enc.parameters():
            param.requires_grad = False
        model_enc.eval()
        optimizer_dec.zero_grad()
        optimizer_enc = None
    else:
        model_enc.train()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

    model_dec.train()
    running_loss = 0.0
    #accumulation_steps = args.accumulation_steps

    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        gt = batch['geo_transform'].to(device)

        if args.task == 'segmentation':
            target = batch['target'].squeeze(1).to(device).long()
        else:
            target = batch['target'].to(device)
            nan_inf_mask = torch.isnan(target) | torch.isinf(target)
            if nan_inf_mask.any():
                target = torch.nan_to_num(target, nan=0.0, posinf=0, neginf=0)

        mask_ratio_ms = args.masking_ms
        mask_ratio_mod = args.masking_modality

        context = autocast() if args.use_autocast else torch.no_grad()  # no_grad won't compute grads, so we need just normal context if no autocast
        with context if args.use_autocast else nullcontext():
            outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(
                image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod
            )
            prediction, prediction_vit = model_dec(
                outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt
            )

            loss_fuse = criterion(prediction, target) if args.task == 'segmentation' else criterion(prediction.float(), target.float())
            loss_vit = criterion(prediction_vit, target) if args.task == 'segmentation' else criterion(prediction_vit.float(), target.float())
            loss = args.alpha_vit * loss_vit + args.beta_fuse * loss_fuse

        if not torch.isnan(loss) and not torch.isinf(loss):
            if args.use_autocast and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                if args.train_mode == "decoder_only":
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_dec.step()
                    optimizer_dec.zero_grad()
                else:
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_enc)
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_enc.step()
                        optimizer_dec.step()
                    optimizer_enc.zero_grad()
                    optimizer_dec.zero_grad()

        running_loss += loss.item() * image_ms.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Training Loss: {epoch_loss:.4f}')
    return epoch_loss


def validate_one_epoch_finetuning_hybrid(model_enc, model_dec, dataloader, criterion, device, args, log_preds_wb = False):
    """
    Validate the model for one epoch.
    
    This function sets the encoder and decoder to evaluation mode, and it uses
    torch.no_grad() (with optional autocast) to perform inference. It computes the 
    loss over the validation dataset without updating any model parameters.
    
    Parameters:
      model_enc: The encoder model.
      model_dec: The decoder model.
      dataloader: DataLoader for the validation dataset.
      criterion: Loss function (e.g., nn.CrossEntropyLoss).
      device: The device (CPU or GPU) for computations.
      args: Additional arguments (e.g., task type, fine-tune mode).
      
    Returns:
      epoch_loss: The average loss for the validation epoch.
    """
    
    # for param in model_enc.parameters():
    #     param.requires_grad = False
    # model_enc.eval()
    # for param in model_dec.parameters():
    #     param.requires_grad = False
    # model_dec.eval()
    
    running_loss = 0.0

    # Disable gradient computation for validation.
    with torch.no_grad():
        # Optionally use autocast for mixed precision inference.
        with autocast():
            for batch_idx, batch in enumerate(dataloader):
                # Get the input data and send to device.
                image_ms = batch['image'].to(device)
                elevation = batch['elevation'].to(device)
                
                # For segmentation tasks using CrossEntropyLoss, targets should be long.
                if args.task == 'segmentation':  
                    target = batch['target'].squeeze(1).to(device).long()
                else:
                    target = batch['target'].to(device)
                    # Check and handle NaN or Inf explicitly
                    nan_inf_mask = torch.isnan(target) | torch.isinf(target)    
                    if nan_inf_mask.any():
                        #print(f"Detected NaN or inf in targets at batch index {i}. Handling...")
                        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
                
                gt = batch['geo_transform'].to(device)
                
                # During validation, we usually let the model see the full input.
                mask_ratio_ms = args.masking_ms
                mask_ratio_mod = args.masking_modality
                
                # Forward pass through encoder and decoder.
                outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod)
                prediction, prediction_vit = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt)
                
                # Process the batch on a per-sample basis
                # batch_size = image_ms.shape[0]
                # losses = []
                
                # for i in range(batch_size):
                #     loss_sample = criterion(prediction[i], target[i])
                #     losses.append(loss_sample)    
                # # Average the losses over the batch
                # loss = sum(losses) / batch_size
                if args.task == 'segmentation':
                    loss_fuse = criterion(prediction, target)
                    loss_vit = criterion(prediction_vit, target)
                else:
                    loss_fuse = criterion(prediction.float(), target.float())
                    loss_vit = criterion(prediction_vit.float(), target.float())

                #loss = args.alpha_vit * loss_vit + args.beta_fuse * loss_fuse
                loss = args.alpha_vit * loss_vit + args.beta_fuse * loss_fuse
                if not torch.isnan(loss) and not torch.isinf(loss):
                    running_loss += loss.item() * image_ms.size(0)
                # Log image to W&B
                if log_preds_wb == True:
                    if args.task == 'segmentation':
                        if args.beta_fuse > 0: 
                            log_image_wandb(image_ms, prediction, target)
                        else:
                            log_image_wandb(image_ms, prediction_vit, target)
                    else:
                        if args.beta_fuse > 0:
                            log_image_reg_wandb(image_ms, elevation, prediction, target)
                        else:
                            log_image_reg_wandb(image_ms, elevation, prediction_vit, target)
                
    
    # Compute the epoch's average loss.
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    
    return epoch_loss

def finetune_one_epoch_dualpath(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion, device, args, scaler=None):
    if args.train_mode == "decoder_only":
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for param in model_enc.parameters():
            param.requires_grad = False
        model_enc.eval()
        optimizer_dec.zero_grad()
        optimizer_enc = None
    else:
        model_enc.train()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

    model_dec.train()
    running_loss = 0.0
    #accumulation_steps = args.accumulation_steps

    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        gt = batch['geo_transform'].to(device)

        if args.task == 'segmentation':
            target = batch['target'].squeeze(1).to(device).long()
        else:
            target = batch['target'].to(device)
            nan_inf_mask = torch.isnan(target) | torch.isinf(target)
            if nan_inf_mask.any():
                target = torch.nan_to_num(target, nan=0.0, posinf=0, neginf=0)

        mask_ratio_ms = args.masking_ms
        mask_ratio_mod = args.masking_modality

        context = autocast() if args.use_autocast else torch.no_grad()  # no_grad won't compute grads, so we need just normal context if no autocast
        with context if args.use_autocast else nullcontext():
            outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(
                image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod
            )
            prediction, prediction_vit = model_dec(
                outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt
            )

            if args.task == 'segmentation':
                loss_fuse = criterion(prediction, target)
                loss_vit = criterion(prediction_vit, target)
            else:
                loss_fuse = criterion(prediction.float(), target.float())
                loss_vit = criterion(prediction_vit.float(), target.float())

            loss = args.alpha_vit * loss_vit + args.beta_fuse * loss_fuse

        if not torch.isnan(loss) and not torch.isinf(loss):
            if args.use_autocast and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                if args.train_mode == "decoder_only":
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_dec.step()
                    optimizer_dec.zero_grad()
                else:
                    if args.use_autocast and scaler is not None:
                        scaler.step(optimizer_enc)
                        scaler.step(optimizer_dec)
                        scaler.update()
                    else:
                        optimizer_enc.step()
                        optimizer_dec.step()
                    optimizer_enc.zero_grad()
                    optimizer_dec.zero_grad()

        running_loss += loss.item() * image_ms.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Training Loss: {epoch_loss:.4f}')
    return epoch_loss


def validate_one_epoch_finetuning_dualpath(model_enc, model_dec, dataloader, criterion, device, args, log_preds_wb = False):
    """
    Validate the model for one epoch.
    
    This function sets the encoder and decoder to evaluation mode, and it uses
    torch.no_grad() (with optional autocast) to perform inference. It computes the 
    loss over the validation dataset without updating any model parameters.
    
    Parameters:
      model_enc: The encoder model.
      model_dec: The decoder model.
      dataloader: DataLoader for the validation dataset.
      criterion: Loss function (e.g., nn.CrossEntropyLoss).
      device: The device (CPU or GPU) for computations.
      args: Additional arguments (e.g., task type, fine-tune mode).
      
    Returns:
      epoch_loss: The average loss for the validation epoch.
    """
    
    # for param in model_enc.parameters():
    #     param.requires_grad = False
    # model_enc.eval()
    # for param in model_dec.parameters():
    #     param.requires_grad = False
    # model_dec.eval()
    
    running_loss = 0.0

    # Disable gradient computation for validation.
    with torch.no_grad():
        # Optionally use autocast for mixed precision inference.
        with autocast():
            for batch_idx, batch in enumerate(dataloader):
                # Get the input data and send to device.
                image_ms = batch['image'].to(device)
                elevation = batch['elevation'].to(device)
                
                # For segmentation tasks using CrossEntropyLoss, targets should be long.
                if args.task == 'segmentation':  
                    target = batch['target'].squeeze(1).to(device).long()
                else:
                    target = batch['target'].to(device)
                    # Check and handle NaN or Inf explicitly
                    nan_inf_mask = torch.isnan(target) | torch.isinf(target)    
                    if nan_inf_mask.any():
                        #print(f"Detected NaN or inf in targets at batch index {i}. Handling...")
                        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
                
                gt = batch['geo_transform'].to(device)
                
                # During validation, we usually let the model see the full input.
                mask_ratio_ms = args.masking_ms
                mask_ratio_mod = args.masking_modality
                
                # Forward pass through encoder and decoder.
                outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod)
                prediction, prediction_vit = model_dec(outputs_enc, mask_multi, mask_modality, ids_restore_multi, ids_restore_modality, gt)
                
                # Process the batch on a per-sample basis
                # batch_size = image_ms.shape[0]
                # losses = []
                
                # for i in range(batch_size):
                #     loss_sample = criterion(prediction[i], target[i])
                #     losses.append(loss_sample)    
                # # Average the losses over the batch
                # loss = sum(losses) / batch_size
                if args.task == 'segmentation':
                    loss_fuse = criterion(prediction, target)
                    loss_vit = criterion(prediction_vit, target)
                else:
                    loss_fuse = criterion(prediction.float(), target.float())
                    loss_vit = criterion(prediction_vit.float(), target.float())

                #loss = args.alpha_vit * loss_vit + args.beta_fuse * loss_fuse
                loss = args.alpha_vit * loss_vit + args.beta_fuse * loss_fuse
                if not torch.isnan(loss) and not torch.isinf(loss):
                    running_loss += loss.item() * image_ms.size(0)
                # Log image to W&B
                if log_preds_wb == True:
                    if args.task == 'segmentation':
                        if args.beta_fuse > 0: 
                            log_image_wandb(image_ms, prediction, target)
                        else:
                            log_image_wandb(image_ms, prediction_vit, target)
                    else:
                        if args.beta_fuse > 0:
                            log_image_reg_wandb(image_ms, elevation, prediction, target)
                        else:
                            log_image_reg_wandb(image_ms, elevation, prediction_vit, target)
                
    
    # Compute the epoch's average loss.
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    
    return epoch_loss



def finetune_one_epoch_geo_reg(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion, device, args):
    if args.train_mode == "decoder_only":
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for param in model_enc.parameters():
            param.requires_grad = False
        model_enc.eval()  # Set encoder to evaluation mode.
        optimizer_dec.zero_grad()
        optimizer_enc = None
    else:
        model_enc.train()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

    model_dec.train()  # Set the decoder to training mode

    running_loss = 0.0
        
    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        
        target = batch['target'].to(device)
        # Check and handle NaN or Inf explicitly
        nan_inf_mask = torch.isnan(target) | torch.isinf(target)    
        if nan_inf_mask.any():
            #print(f"Detected NaN or inf in targets at batch index {i}. Handling...")
            target = torch.nan_to_num(target, nan=0.0, posinf=0, neginf=0)
        
        #target_labels = batch['clusters'].to(device).long()
        gt = batch['geo_transform'].to(device)
        # We want the model to see as much of the image context as possible
        mask_ratio_ms = args.masking_ms
        mask_ratio_mod = args.masking_modality

        # Encode and decode the representations
        outputs_enc = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod, return_intermediate=True)
        prediction = model_dec(outputs_enc, gt)
        
        loss = criterion(prediction.float(), target.float())
            
        # Backward pass
        if not torch.isnan(loss) and not torch.isinf(loss):
            if args.train_mode == "decoder_only":
                loss.backward()
                optimizer_dec.step()
                optimizer_dec.zero_grad()
            else:        
                loss.backward()
                optimizer_enc.step()
                optimizer_dec.step()
                
                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
        
            running_loss += loss.item() * image_ms.size(0)   
        

    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Training Loss: {epoch_loss:.4f}')

    return epoch_loss

def validate_one_epoch_finetuning_geo_reg(model_enc, model_dec, dataloader, criterion, device, args, log_preds_wb = False):
    """
    Validate the model for one epoch.
    
    This function sets the encoder and decoder to evaluation mode, and it uses
    torch.no_grad() (with optional autocast) to perform inference. It computes the 
    loss over the validation dataset without updating any model parameters.
    
    Parameters:
      model_enc: The encoder model.
      model_dec: The decoder model.
      dataloader: DataLoader for the validation dataset.
      criterion: Loss function (e.g., nn.CrossEntropyLoss).
      device: The device (CPU or GPU) for computations.
      args: Additional arguments (e.g., task type, fine-tune mode).
      
    Returns:
      epoch_loss: The average loss for the validation epoch.
    """
        
    running_loss = 0.0

    # Disable gradient computation for validation.
    with torch.no_grad():
        # Optionally use autocast for mixed precision inference.
        
        for batch_idx, batch in enumerate(dataloader):
            # Get the input data and send to device.
            image_ms = batch['image'].to(device)
            elevation = batch['elevation'].to(device)
            
            
            target = batch['target'].to(device)
            # Check and handle NaN or Inf explicitly
            nan_inf_mask = torch.isnan(target) | torch.isinf(target)    
            if nan_inf_mask.any():
                #print(f"Detected NaN or inf in targets at batch index {i}. Handling...")
                target = torch.nan_to_num(target, nan=0.0, posinf=0, neginf=0)
            
            gt = batch['geo_transform'].to(device)
            
            # During validation, we usually let the model see the full input.
            mask_ratio_ms = args.masking_ms
            mask_ratio_mod = args.masking_modality
            
            # Forward pass through encoder and decoder.
            outputs_enc = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod, return_intermediate=True)
            prediction = model_dec(outputs_enc, gt)
            
            loss = criterion(prediction, target)
            if not torch.isnan(loss) and not torch.isinf(loss):
                running_loss += loss.item() * image_ms.size(0)
            # Log image to W&B
            if log_preds_wb == True:
                log_image_reg_wandb(image_ms, elevation, prediction, target)
                
    
    # Compute the epoch's average loss.
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    
    return epoch_loss

def finetune_one_epoch_geo_sc(model_enc, model_dec, dataloader, optimizer_enc, optimizer_dec, criterion, device, scaler, args):
    if args.train_mode == "decoder_only":
        print("Freezing encoder parameters and setting encoder to evaluation mode.")
        for param in model_enc.parameters():
            param.requires_grad = False
        model_enc.eval()  # Set encoder to evaluation mode.
        optimizer_dec.zero_grad()
        optimizer_enc = None
    else:
        model_enc.train()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

    model_dec.train()  # Set the decoder to training mode

    running_loss = 0.0
    accumulation_steps = args.accumulation_steps  # Number of steps to accumulate gradients
    
    for i, batch in enumerate(dataloader):
        image_ms = batch['image'].to(device)
        elevation = batch['elevation'].to(device)
        if args.task == 'segmentation':  
            target = batch['target'].squeeze(1).to(device).long()
        else:
            target = batch['target'].to(device)
        
        #target_labels = batch['clusters'].to(device).long()
        gt = batch['geo_transform'].to(device)
        # We want the model to see as much of the image context as possible
        mask_ratio_ms = args.masking_ms
        mask_ratio_mod = args.masking_modality
        
        with autocast():
            # Encode and decode the representations
            outputs_enc = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod)
            prediction = model_dec(outputs_enc, gt)
            loss = criterion(prediction, target)
            
        # Backward pass
        if not torch.isnan(loss):
            scaler.scale(loss).backward() # This will compute gradients for both dec and enc parts of the model
            if args.train_mode == "decoder_only":
        
                # Update weights
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer_dec)
                    scaler.update()

                    optimizer_dec.zero_grad()
            else:        
                # Update weights
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer_enc)
                    scaler.step(optimizer_dec)
                    scaler.update()

                    optimizer_enc.zero_grad()
                    optimizer_dec.zero_grad()
            
            running_loss += loss.item() * image_ms.size(0) 
        

    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Training Loss: {epoch_loss:.4f}')

    return epoch_loss

def validate_one_epoch_finetuning_geo_sc(model_enc, model_dec, dataloader, criterion, device, args, log_preds_wb = False):
    """
    Validate the model for one epoch.
    
    This function sets the encoder and decoder to evaluation mode, and it uses
    torch.no_grad() (with optional autocast) to perform inference. It computes the 
    loss over the validation dataset without updating any model parameters.
    
    Parameters:
      model_enc: The encoder model.
      model_dec: The decoder model.
      dataloader: DataLoader for the validation dataset.
      criterion: Loss function (e.g., nn.CrossEntropyLoss).
      device: The device (CPU or GPU) for computations.
      args: Additional arguments (e.g., task type, fine-tune mode).
      
    Returns:
      epoch_loss: The average loss for the validation epoch.
    """
    
    running_loss = 0.0

    # Disable gradient computation for validation.
    with torch.no_grad():
        # Optionally use autocast for mixed precision inference.
        with autocast():
            for batch_idx, batch in enumerate(dataloader):
                # Get the input data and send to device.
                image_ms = batch['image'].to(device)
                elevation = batch['elevation'].to(device)
                
                # For segmentation tasks using CrossEntropyLoss, targets should be long.
                if args.task == 'segmentation':  
                    target = batch['target'].squeeze(1).to(device).long()
                else:
                    target = batch['target'].to(device)
                
                gt = batch['geo_transform'].to(device)
                
                # During validation, we usually let the model see the full input.
                mask_ratio_ms = args.masking_ms
                mask_ratio_mod = args.masking_modality
                
                # Forward pass through encoder and decoder.
                outputs_enc = model_enc(image_ms, elevation, gt, mask_ratio_ms=mask_ratio_ms, mask_ratio_elev=mask_ratio_mod)
                prediction = model_dec(outputs_enc, gt)
                loss = criterion(prediction, target)
                running_loss += loss.item() * image_ms.size(0)

                # Log image to W&B
                if log_preds_wb == True:
                    if args.task == 'segmentation': 
                        log_image_wandb(image_ms, prediction, target)
                    else:
                        log_image_reg_wandb(image_ms, prediction, target)
                
    
    # Compute the epoch's average loss.
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    
    return epoch_loss

def save_checkpoint(
    models: Union[Module, Dict[str, Module]],
    optimizers: Union[Optimizer, Dict[str, Optimizer]],
    epoch: int,
    loss: float,
    save_dir: str,
    prefix: str = "model",
    dt: str = "",
    checkpoint_path: Optional[str] = None,
    create_subdir: bool = False
) -> str:
    """
    Saves a checkpoint for one or multiple models and optimizers.

    Args:
        models (Module or Dict[str, Module]):
            Either a single PyTorch model (nn.Module) or a dictionary mapping
            a string key to multiple models (e.g., {'encoder': enc_model, 'decoder': dec_model}).
        optimizers (Optimizer or Dict[str, Optimizer]):
            Either a single PyTorch optimizer or a dictionary mapping
            a string key to multiple optimizers.
        epoch (int):
            The current training epoch.
        loss (float):
            The training (or validation) loss at this epoch.
        save_dir (str):
            The directory where you want to save the checkpoint (used only if `checkpoint_path` is not specified).
        prefix (str, optional):
            A prefix for the filename (default: "model").
        dt (str, optional):
            A date/time string or unique identifier for the checkpoint file (default: "").
        checkpoint_path (str, optional):
            If provided, saves the checkpoint to this path (overwriting if it exists).
            If None, a filename is constructed using `save_dir`, `prefix`, `dt`, and `epoch`.
            Default: None
        create_subdir (bool, optional):
            If True, a new subdirectory named with the prefix or dt can be created
            to organize checkpoints. Defaults to False.

    Returns:
        str: The final file path of the saved checkpoint.
    """

    # 1. If a direct path is provided, we save there. Otherwise, build the path.
    if checkpoint_path is None:
        if create_subdir and dt:
            save_dir = os.path.join(save_dir, f"{prefix}_{dt}")

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Construct filename
        file_name = f"{prefix}"
        if dt:
            file_name += f"_{dt}"
        file_name += f"_epoch{epoch}.pth"
        save_path = os.path.join(save_dir, file_name)
    else:
        # If checkpoint_path is specified, just use it directly
        #save_path = checkpoint_path
        save_path = checkpoint_path.replace(".pth.jar",f"_FT_{epoch}.pth.jar")
        # Ensure its parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. Create the checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "datetime": dt  # for reference if needed
    }

    # 3. Handle single vs multiple models
    if isinstance(models, dict):
        for model_name, model_obj in models.items():
            checkpoint[f"{model_name}_state_dict"] = model_obj.state_dict()
    else:
        checkpoint["model_state_dict"] = models.state_dict()

    # 4. Handle single vs multiple optimizers
    if isinstance(optimizers, dict):
        for opt_name, opt_obj in optimizers.items():
            checkpoint[f"{opt_name}_state_dict"] = opt_obj.state_dict()
    else:
        checkpoint["optimizer_state_dict"] = optimizers.state_dict()

    # 5. Save the checkpoint
    torch.save(checkpoint, save_path)
    #print(f"Checkpoint saved to: {save_path}")
    return save_path

