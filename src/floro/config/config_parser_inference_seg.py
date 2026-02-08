import os
import argparse
import json
import math

def _try_load_yaml(path):
    try:
        import yaml  # optional
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise RuntimeError(
            f"YAML colormap requested but PyYAML is not installed: {path}. "
            "Install PyYAML or provide JSON/inline colormap."
        )

def _parse_rgb_triplet(token):
    # token like "0,128,0"
    parts = token.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Bad RGB triplet '{token}' (want r,g,b)")
    rgb = tuple(int(x) for x in parts)
    if any((v < 0 or v > 255) for v in rgb):
        raise argparse.ArgumentTypeError(f"RGB out of range 0..255 in '{token}'")
    return rgb

def _parse_colormap_inline(s):
    """
    Inline format:
      "0:0,0,0;1:0,128,0;2:0,0,128"
    Spaces are ignored. Keys must be ints; values are r,g,b (0..255).
    """
    cmap = {}
    s = s.strip()
    if not s:
        return cmap
    for kv in s.split(";"):
        kv = kv.strip()
        if not kv:
            continue
        try:
            k, v = kv.split(":")
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Bad colormap entry '{kv}' (want 'class:r,g,b')"
            )
        k = int(k.strip())
        rgb = _parse_rgb_triplet(v.strip())
        cmap[k] = rgb
    return cmap

def _auto_hsv_colormap(n):
    """
    Generate n visually distinct colors (no external deps).
    Simple HSV wheel with fixed saturation/value.
    """
    def hsv_to_rgb(h, s, v):
        h = float(h) % 1.0
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if   i == 0: r,g,b = v, t, p
        elif i == 1: r,g,b = q, v, p
        elif i == 2: r,g,b = p, v, t
        elif i == 3: r,g,b = p, q, v
        elif i == 4: r,g,b = t, p, v
        else:        r,g,b = v, p, q
        return (int(round(r*255)), int(round(g*255)), int(round(b*255)))
    s, v = 0.75, 0.95
    return {i: hsv_to_rgb(i / max(1, n), s, v) for i in range(n)}

def parse_colormap_arg(s):
    """
    Accepts:
      1) path to JSON/YAML file with {int: [r,g,b]} or {int: (r,g,b)}
      2) inline string "0:0,0,0;1:0,128,0;..."
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # File?
    if os.path.exists(s) and os.path.isfile(s):
        ext = os.path.splitext(s)[1].lower()
        if ext in (".json",):
            with open(s, "r") as f:
                raw = json.load(f)
        elif ext in (".yml", ".yaml"):
            raw = _try_load_yaml(s)
        else:
            raise argparse.ArgumentTypeError(
                f"Unsupported colormap file extension '{ext}'. Use .json or .yaml/.yml."
            )
        cmap = {}
        for k, v in raw.items():
            ki = int(k)
            if (isinstance(v, (list, tuple)) and len(v) == 3):
                rgb = tuple(int(c) for c in v)
            else:
                raise argparse.ArgumentTypeError(
                    f"Bad value for key {k} in {s}: expected [r,g,b]"
                )
            if any((c < 0 or c > 255) for c in rgb):
                raise argparse.ArgumentTypeError(
                    f"RGB out of range 0..255 for key {k} in {s}"
                )
            cmap[ki] = rgb
        return cmap
    # Inline
    return _parse_colormap_inline(s)

def finalize_colormap(args):
    """
    Ensure args.colormap is a complete mapping 0..num_classes-1 -> (r,g,b).
    If not provided, use a sensible default for up to 12 classes,
    otherwise auto-generate.
    """
    default_12 = {
        0:  (0, 0, 0),       # background
        1:  (0, 128, 0),
        2:  (0, 0, 128),
        3:  (128, 0, 0),
        4:  (128, 128, 0),
        5:  (0, 128, 128),
        6:  (128, 0, 128),
        7:  (64, 128, 0),
        8:  (0, 64, 128),
        9:  (128, 64, 0),
        10: (64, 0, 128),
        11: (0, 128, 64),
    }
    if args.colormap is None:
        if args.num_classes <= 12:
            cmap = {k: v for k, v in default_12.items() if k < args.num_classes}
        else:
            cmap = _auto_hsv_colormap(args.num_classes)
    else:
        cmap = dict(args.colormap)  # already parsed into {int: (r,g,b)}

    # Fill any missing classes
    missing = [k for k in range(args.num_classes) if k not in cmap]
    if missing:
        # extend with HSV colors for the missing keys
        auto = _auto_hsv_colormap(len(missing))
        for i, k in enumerate(missing):
            cmap[k] = auto[i]

    # Validate final
    for k in range(args.num_classes):
        if k not in cmap:
            raise ValueError(f"Colormap missing class {k}")
        rgb = cmap[k]
        if not (isinstance(rgb, (tuple, list)) and len(rgb) == 3):
            raise ValueError(f"Colormap value for class {k} must be a 3-tuple")
        if any((c < 0 or c > 255) for c in rgb):
            raise ValueError(f"Colormap rgb for class {k} must be in 0..255")
    return cmap

def get_args_parser():
    parser = argparse.ArgumentParser(description='Inference CHM and Biomass DPT decoders', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wb_project', default="InferenceDPT", type=str, help="Project name for Wandb")
    parser.add_argument('--run_name', default="InferenceDPT", type=str, help="Run name for Wandb")
    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true', help="Enable distributed training")
    parser.add_argument('--rank', default=os.getenv('LOCAL_RANK', 0), type=int, help='Rank of the process in distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Total number of processes to run')
    parser.add_argument('--node_rank', default=0, type=int, help='Rank of the node for multi-node distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU to use for training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Dataloaidng
    parser.add_argument('--workers', default=2, type=int, help='Number of workers for dataloading')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Batch | Single image inference
    parser.add_argument('--mode', default='single', choices=['single', 'batch'], help='Run mode: single image or batch')

    # Paths and input parameters
    parser.add_argument('--image_directory', default='', type=str, help='Path to directory with input images (for batch mode)')
    parser.add_argument('--modality_directory', default='', type=str, help='Path to directory with modality files (for batch mode)')
    parser.add_argument('--image_path', default='', type=str, help='Path to the image data')
    parser.add_argument('--modality_path', default='', type=str, help='Path to the modality data')
    parser.add_argument('--bands', nargs='+', default=[1, 2, 3, 4], type=int, help='Bands to be used as input, indexing is 1-based (i.e. --bands 1 2 5 8) | FLORO supports 4 spectral bands (Blue, Green, Red, NIR)')

    # Sliding window inference parameters
    parser.add_argument('--tile_size', default=256, type=int, help='Tile size for inference | Currently FLORO only receives 256 x 256')
    parser.add_argument('--stride', default=200, type=int, help='Stride for sliding window | Determines the overlap between windows (overlap = image_size - stride)')

    # Basic Inference Parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for batched inference')
    parser.add_argument('--masking_ms', default=0.0, type=float, help='Percentage of MS patches to be masked')
    parser.add_argument('--masking_modality', default=0.0, type=float, help='Percentage of modality patches to be masked')    
    parser.add_argument('--lambda_bg', default=0.1, type=float, help='Contribution of background loss to combined loss')
    parser.add_argument('--ignore_index', default=0, type=int, help='Label value to ignore in loss/metrics (void/unknown pixels). Default: 0')
    try:
        bool_action = argparse.BooleanOptionalAction
    except AttributeError:
        bool_action = None
    if bool_action:
        parser.add_argument("--save-confidence", action=bool_action, default=True,
                            help="Save a companion top-1 probability GeoTIFF (default: True).")
    else:
        parser.add_argument("--save-confidence", action="store_true",
                            help="Save a companion top-1 probability GeoTIFF.")
        parser.add_argument("--no-save-confidence", dest="save_confidence",
                            action="store_false", help="Do not save confidence map.")
        parser.set_defaults(save_confidence=True)
    parser.add_argument(
        "--colormap",
        type=parse_colormap_arg,
        default=None,
        help=(
            "Class color map. Either a path to JSON/YAML with {class_id: [r,g,b]}, "
            "or inline '0:0,0,0;1:0,128,0;2:0,0,128'. If omitted, a default/auto palette is used."
        ),
    )
    
    # Model Specific Parameters
    parser.add_argument('--images_size', default=256, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=16, type=int, help='Size of the patches')
    parser.add_argument('--input_channels', default=4, type=int, help='Number of input channels (e.g., RGB+NIR)')
    parser.add_argument('--elev_channels', default=1, type=int, help='Number of elevation channels (e.g., DSM+DTM)')
    parser.add_argument('--d_model', default=1024, type=int, help='Dimension of the model')
    parser.add_argument('--depth', default=24, type=int, help='Depth of the encoder')
    parser.add_argument('--num_heads', default=16, type=int, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='Ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--pos_embed_type', default='absolute', type=str, help='Position embedding type: absolute | geo')

    # Decoder Specific Parameters
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes')
    parser.add_argument('--dec_d_model', default=768, type=int, help='Dimension of the decoder model')
    parser.add_argument('--dpt_channels', nargs='+', default=[256, 512, 1024, 1024], type=int, help='DPT channels')
    
    # Saving
    parser.add_argument('--output_dir', default='', type=str, help='Directory to save models')
    parser.add_argument('--log_name', default='', type=str, help='Log name to store metrics')

    # Train from checkpoints
    parser.add_argument('--encoder_checkpoint', default='', type=str, help="Encoder's checkpoint | Or single file Enc-Dec checkpoint")
    parser.add_argument('--decoder_checkpoint', default='', type=str, help="Decoder's checkpoint")
    
    # Device Configuration
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')

    return parser
