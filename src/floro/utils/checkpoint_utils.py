import os
import torch
import torch.distributed as dist
from collections import OrderedDict
from typing import Union, Dict, Optional
from torch.nn import Module
from torch.optim import Optimizer

# ---------- helpers for checkpoint load ----------

def _extract_state_dict(ckpt: dict, which: str | None = None) -> dict | None:
    """
    Try to find the right state_dict inside a checkpoint.
    which: None | 'encoder' | 'decoder'
    """
    if not isinstance(ckpt, dict):
        return None

    # Explicit combined keys
    if which == "encoder" and "encoder_state_dict" in ckpt:
        return ckpt["encoder_state_dict"]
    if which == "decoder" and "decoder_state_dict" in ckpt:
        return ckpt["decoder_state_dict"]

    # Some runs store both nets under a dict
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        msd = ckpt["model_state_dict"]
        if which in ("encoder", "decoder") and which in msd and isinstance(msd[which], dict):
            return msd[which]
        # else: assume the whole dict is the model's state_dict for single-net checkpoints
        if all(isinstance(v, torch.Tensor) for v in msd.values()):
            return msd

    # Common single-net keys
    for k in ("model", "state_dict"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]

    # Maybe the checkpoint itself *is* a state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    return None

def load_weights_strip_module(model, ckpt_path: str, device, which: str | None = None, strict: bool = True) -> bool:
    """
    Load weights into `model` from ckpt_path.
    - Strips DataParallel/DDP 'module.' prefix.
    - Supports combined checkpoints via `which` ('encoder'|'decoder'|None).
    Returns True if something was loaded.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = _extract_state_dict(ckpt, which=which)
    if sd is None:
        return False

    sd = strip_module_prefix(sd)

    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        print(f"[WARN] Strict load failed ({e}). Retrying with strict=False…")
        missing_unexp = model.load_state_dict(sd, strict=False)
        # PyTorch ≥2 returns _IncompatibleKeys, print for visibility
        try:
            print("[INFO] Missing keys:", missing_unexp.missing_keys)
            print("[INFO] Unexpected keys:", missing_unexp.unexpected_keys)
        except Exception:
            pass

    print(f"[✓] Loaded weights into {model.__class__.__name__} from {ckpt_path}" + (f" ({which})" if which else ""))
    return True

# -----------Saving helpers --------------------

def is_main_process() -> bool:
    """True if this process should write checkpoints."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def unwrap_model(model):
    """Return the underlying model if wrapped by DDP/DataParallel."""
    # DDP has .module; DataParallel too.
    return model.module if hasattr(model, "module") else model


def strip_module_prefix(state_dict):
    # Only strip if it actually exists
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[len("module."):]] = v if k.startswith("module.") else v
    return new_sd

def to_cpu_state_dict(state_dict):
    """Ensure all tensors are on CPU for safe/portable serialization."""
    cpu_sd = OrderedDict()
    for k, v in state_dict.items():
        cpu_sd[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return cpu_sd

def save_model(model, optimizer, epoch, loss, save_dir, model_name="model", suffix="", dt="", only_master=True):
    # Save only from master in DDP (rank 0)
    if only_master and not is_main_process():
        return None

    os.makedirs(save_dir, exist_ok=True)

    file_name = f"{model_name}_{suffix}_dt{dt}.pth.jar"
    save_path = os.path.join(save_dir, file_name)

    # Unwrap DDP/DataParallel and remove "module." prefix
    base_model = unwrap_model(model)
    state_dict = strip_module_prefix(base_model.state_dict())

    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss": float(loss) if isinstance(loss, (float, int)) else loss,
    }

    # Atomic save (avoids partial files if job preempts mid-write)
    tmp_path = save_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, save_path)

    print(f"Model {file_name} saved to {save_path}.")
    return save_path

def save_model_from_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss,
    file_name: str,
    extra: dict | None = None,
    only_main: bool = True,
):
    """
    Save a checkpoint (model + optimizer + metadata) safely in DDP.

    - Saves only on rank 0 by default.
    - Unwraps DDP/DataParallel.
    - Strips 'module.' prefix so the checkpoint loads in non-DDP too.
    - Writes atomically (tmp -> final).
    """
    if only_main and not is_main_process():
        return None

    # Build save path
    save_path = file_name.replace(".pth.jar", f"_{epoch}.pth.jar")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    base_model = unwrap_model(model)
    sd = base_model.state_dict()
    sd = strip_module_prefix(sd)
    sd = to_cpu_state_dict(sd)

    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": sd,
        "loss": float(loss) if isinstance(loss, (int, float)) else loss,
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if extra:
        ckpt.update(extra)

    # Atomic save: write temp then rename
    tmp_path = save_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, save_path)

    print(f"Model {save_path} saved.")
    return save_path
# ---------- end helpers ----------


def save_checkpoint(
    models: Union[Module, Dict[str, Module]],
    optimizers: Union[Optimizer, Dict[str, Optimizer]],
    epoch: int,
    loss: float,
    save_dir: str,
    prefix: str = "model",
    dt: str = "",
    checkpoint_path: Optional[str] = None,
    create_subdir: bool = False,
    only_master=True
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

    if only_master and not is_main_process():
        return None

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
        save_path = checkpoint_path.replace(".pth.jar", f"_FT_{epoch}.pth.jar")
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
            # Check if the model is wrapped with DDP and access module.state_dict
            if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
                checkpoint[f"{model_name}_state_dict"] = model_obj.module.state_dict()
            else:
                checkpoint[f"{model_name}_state_dict"] = model_obj.state_dict()
    else:
        # Check if the single model is wrapped with DDP and access module.state_dict
        if isinstance(models, torch.nn.parallel.DistributedDataParallel):
            checkpoint["model_state_dict"] = models.module.state_dict()
        else:
            checkpoint["model_state_dict"] = models.state_dict()

    # 4. Handle single vs multiple optimizers
    if isinstance(optimizers, dict):
        for opt_name, opt_obj in optimizers.items():
            checkpoint[f"{opt_name}_state_dict"] = opt_obj.state_dict()
    else:
        checkpoint["optimizer_state_dict"] = optimizers.state_dict()

    # 5. Save the checkpoint only on rank 0
    if torch.distributed.get_rank() == 0:
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to: {save_path}")
    
    return save_path