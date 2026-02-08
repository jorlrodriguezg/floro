import torch
from collections import OrderedDict

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


def _strip_module_prefix(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
    return state_dict


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

    sd = _strip_module_prefix(sd)

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
# ---------- end helpers ----------
