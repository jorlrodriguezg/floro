import os
import sys
from glob import glob
import numpy as np
import torch
import datetime
import wandb
import warnings
from collections import OrderedDict

import json, time
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from pyproj import Transformer

import os, re, hashlib, subprocess
from typing import Dict, Optional, Tuple, List

from floro.models.FLORO_Encoder import MultiMAE_Encoder
from floro.models.FLORO_DPT_Decoder import FLORODPTDecoder
from floro.config.config_parser_inference_seg import get_args_parser

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')


# --------------------------------------
# File Reading Utilities
# --------------------------------------

def grids_match(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader, atol=1e-6) -> bool:
    return (
        (a.crs == b.crs) and
        (a.width == b.width) and
        (a.height == b.height) and
        np.allclose(a.transform, b.transform, atol=atol)
    )

# def read_ms_and_mod_tile(
#     src_img: rasterio.io.DatasetReader,
#     src_mod: rasterio.io.DatasetReader | None,
#     win: Window,
#     ms_indexes,                          # e.g. args.bands, 1-based list like [2,3,4,8]
#     tile_h: int, tile_w: int,
#     elev_indexes=None,                   # e.g. [1] (default first band)
#     resampling=Resampling.bilinear
# ):
#     # MS tile
#     ms_tile = src_img.read(
#         indexes=ms_indexes,
#         window=win,
#         out_shape=(len(ms_indexes), tile_h, tile_w),
#         resampling=resampling
#     ).astype(np.float32)

#     # Modality tile
#     if src_mod is None:
#         # zero modality
#         mod_tile = np.zeros((1, tile_h, tile_w), dtype=np.float32)
#     else:
#         elev_indexes = elev_indexes or [1]
#         if grids_match(src_img, src_mod):
#             # Perfectly aligned → read directly with same window
#             mod_tile = src_mod.read(
#                 indexes=elev_indexes,
#                 window=win,
#                 out_shape=(len(elev_indexes), tile_h, tile_w),
#                 resampling=resampling
#             ).astype(np.float32)
#         else:
#             # Present modality in MS grid via WarpedVRT (no temp files)
#             nodata = src_mod.nodata
#             with WarpedVRT(
#                 src_mod,
#                 crs=src_img.crs,
#                 transform=src_img.transform,
#                 width=src_img.width,
#                 height=src_img.height,
#                 resampling=resampling,
#                 nodata=nodata
#             ) as vrt:
#                 mod_tile = vrt.read(
#                     indexes=elev_indexes,
#                     window=win,
#                     out_shape=(len(elev_indexes), tile_h, tile_w),
#                     resampling=resampling
#                 ).astype(np.float32)

#     return ms_tile, mod_tile

# --------------------------------------
# File Check Utilities
# --------------------------------------

def verify_checkpoint_file(file_path, description="Checkpoint"):
    if not file_path.endswith('.pth.jar'):
        raise ValueError(f"{description} must end with '.pth.jar', got: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found at path: {file_path}")
    print(f"[✓] {description} found: {file_path}")

def verify_file_exists(file_path, description="File"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found at path: {file_path}")
    print(f"[✓] {description} found: {file_path}")

# -----


def _short_sha1_of_file(path: str, n: int = 8) -> str:
    """8-char SHA1 of file contents (robust). Falls back to basename if file unreadable."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except Exception:
        return hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:n]

_ckpt_dt_re = re.compile(
    r"dt(?P<date>\d{8})_(?P<time>\d{6})_(?P<step>\d+)", re.IGNORECASE
)

def _parse_dt_step(basename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = _ckpt_dt_re.search(basename)
    if not m:
        return None, None, None
    return m.group("date"), m.group("time"), m.group("step")

def _parse_encoder_tag(ckpt_path: str) -> str:
    """
    Extract something readable from encoder ckpt, e.g.:
    FLORO_Geo_Encoder_MSGV2_dt20241020_155742_350.pth.jar
      -> MSGV2-20241020-155742-350
    """
    base = os.path.basename(ckpt_path)
    # try to pull 'Encoder_<VARIANT>_' piece
    m = re.search(r"Encoder[_-](?P<var>[A-Za-z0-9]+)", base)
    var = m.group("var") if m else "ENC"
    d, t, s = _parse_dt_step(base)
    parts = [var]
    if d and t and s:
        parts += [d, t, s]
    return "-".join(parts)

def _parse_decoder_tag(ckpt_path: str) -> str:
    """
    Extract task & FT epoch if present, e.g.:
    FLORO_Geo_Encoder_MSGV2_dt20241020_155742_350_CHM_FT_200.pth.jar
      -> CHM-FT200-20241020-155742-350
    """
    base = os.path.basename(ckpt_path)
    # task usually appears as '_CHM_' / '_BM_' / etc.
    m_task = re.search(r"_(CHM|BM|SEG|SCENE|NITROGEN|CARBON|BIOMASS)\b", base, re.IGNORECASE)
    task = (m_task.group(1).upper() if m_task else "DEC")
    m_ft = re.search(r"_FT[_-](\d+)", base, re.IGNORECASE)
    ft = f"FT{m_ft.group(1)}" if m_ft else None
    m_bm = re.search(r"_Bm(\d+)", base, re.IGNORECASE)
    bm = f"Bm{m_bm.group(1)}" if m_bm else None
    m_bg = re.search(r"_Bg(\d+)", base, re.IGNORECASE)
    bg = f"Bg{m_bg.group(1)}" if m_bg else None
    
    d, t, s = _parse_dt_step(base)
    parts = [task]
    if ft:
        parts.append(ft)
    if bg:
        parts.append(bg)
    if bm:
        parts.append(bm)
    if d and t and s:
        parts += [d, t, s]
    return "-".join(parts)

def _bands_tag(bands: List[int]) -> str:
    return "-".join(str(b) for b in bands)

def _git_short_rev() -> Optional[str]:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return rev.decode().strip()
    except Exception:
        return None

def build_out_path(
    image_path: str,
    output_dir: str,
    *,
    bands: List[int],
    tile_size: int,
    stride: int,
    task: Optional[str],
    encoder_ckpt: Optional[str],
    decoder_ckpt: Optional[str],
    add_git_rev: bool = False,
) -> str:
    """Return a reproducible, informative output filename."""
    basename = os.path.splitext(os.path.basename(image_path))[0]

    enc_tag = enc_hash = dec_tag = dec_hash = None
    if encoder_ckpt:
        enc_tag  = _parse_encoder_tag(encoder_ckpt)
        enc_hash = _short_sha1_of_file(encoder_ckpt)
    if decoder_ckpt:
        dec_tag  = _parse_decoder_tag(decoder_ckpt)
        dec_hash = _short_sha1_of_file(decoder_ckpt)

    task_part = (task or (dec_tag.split("-")[0] if dec_tag else "PRED")).upper()
    parts = [
        f"{basename}",
        f"FLORO_{task_part}",
        f"b{_bands_tag(bands)}",
        f"ts{tile_size}",
        f"st{stride}",
    ]
    if enc_tag and enc_hash:
        parts.append(f"enc-{enc_tag}-{enc_hash}")
    if dec_tag and dec_hash:
        parts.append(f"dec-{dec_tag}-{dec_hash}")
    if add_git_rev:
        rev = _git_short_rev()
        if rev:
            parts.append(f"git-{rev}")

    stem = "__".join(parts)
    fname = f"{stem}.tif"
    return os.path.join(output_dir, fname)


def create_feather_mask(height, width, overlap):
    """
    Example: creates a linear feather mask from 1.0 in the interior
    to 0.0 at the boundary within 'overlap' pixels.
    """
    mask = np.ones((height, width), dtype=np.float32)

    # Top edge
    for i in range(overlap):
        alpha = i / overlap
        mask[i, :] = np.minimum(mask[i, :], alpha)

    # Bottom edge
    for i in range(height - overlap, height):
        alpha = (height - i - 1) / overlap
        mask[i, :] = np.minimum(mask[i, :], alpha)

    # Left edge
    for j in range(overlap):
        alpha = j / overlap
        mask[:, j] = np.minimum(mask[:, j], alpha)

    # Right edge
    for j in range(width - overlap, width):
        alpha = (width - j - 1) / overlap
        mask[:, j] = np.minimum(mask[:, j], alpha)

    return mask

def predict_batch(ms_batch, mod_batch, gt_batch, model_enc, model_dec, args):
    """
    Inference function adapted for FLORO encoder-decoder.
    ms_batch: (B, C, H, W)
    mod_batch: (B, C, H, W)
    Returns: (B, num_channels, H, W)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ms_tensor = torch.from_numpy(ms_batch).float().to(device)
    mod_tensor = torch.from_numpy(mod_batch).float().to(device)
    gt_tensor = torch.from_numpy(gt_batch).float().to(device)

    with torch.no_grad():
        model_enc.eval()
        model_dec.eval()
        features = model_enc(
            ms_tensor, mod_tensor, gt_tensor,
            mask_ratio_ms=args.masking_ms,
            mask_ratio_elev=args.masking_modality,
            return_intermediate=True
        )
        outputs = model_dec(features) #logits
        print(outputs.shape)

    B, C, Ht, Wt = outputs.shape
    expected = args.tile_size
    assert (Ht, Wt) == (expected, expected), \
        f"Decoder produced {(Ht, Wt)}, expected {(expected, expected)}."
    assert C == args.num_classes, \
        f"Decoder produced C={C}, expected num_classes={args.num_classes}."


    return outputs.detach().cpu().numpy()

def window_geotransform_3857(src, row_off: int, col_off: int):
    """
    Return a 6-element geotransform (EPSG:3857) for the given window:
    [origin_x, pixel_width, 0, origin_y, 0, pixel_height]
    - origin_x/origin_y refer to the UL *corner* of the window in meters.
    - pixel_width > 0
    - pixel_height < 0 (north-up images)
    """
    # Native-CRS UL corner for the window (pixel corner, not center)
    x0_native, y0_native = (src.transform * (col_off, row_off))
    # One pixel to the right (col+1, row)
    x1_native, y1_native = (src.transform * (col_off + 1, row_off))
    # One pixel down (col, row+1)
    x0_native_d, y0_native_d = (src.transform * (col_off, row_off + 1))

    # Transform those three points to EPSG:3857
    transformer = Transformer.from_crs(src.crs, "EPSG:3857", always_xy=True)
    x0_m, y0_m = transformer.transform(x0_native, y0_native)
    x1_m, y1_m = transformer.transform(x1_native, y1_native)
    xd_m, yd_m = transformer.transform(x0_native_d, y0_native_d)

    # Local per-pixel deltas in meters
    pixel_width_m  = x1_m - x0_m                  # > 0
    pixel_height_m = yd_m - y0_m                  # typically < 0 (rows increase downward)

    # Return GDAL-style geotransform (no rotation terms here)
    return np.array([x0_m, pixel_width_m, 0.0, y0_m, 0.0, pixel_height_m], dtype=np.float32)

def run_floro_inference_with_blending(
    image_path, modality_path, output_dir,
    model_enc, model_dec, args
):
    import os, json, time
    import numpy as np
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling

    # --- Optional deps used for resize and simple edge detection
    try:
        from skimage.transform import resize as sk_resize
    except Exception:
        sk_resize = None

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 0) Read inputs and normalize
    # -----------------------------
    with rasterio.open(image_path) as src_img:
        ms_data = src_img.read(indexes=args.bands).astype(np.float32)   # (C, H, W)
        profile = src_img.profile
        H, W = src_img.height, src_img.width

        # Load optional modality (e.g., DSM/CHM); warp if needed
        if modality_path and os.path.exists(modality_path):
            with rasterio.open(modality_path) as src_mod:
                if grids_match(src_img, src_mod):
                    mod_data = src_mod.read(1).astype(np.float32)[None, ...]  # (1, H, W)
                else:
                    with WarpedVRT(
                        src_mod, crs=src_img.crs, transform=src_img.transform,
                        width=W, height=H, resampling=Resampling.bilinear,
                        nodata=src_mod.nodata
                    ) as vrt:
                        mod_data = vrt.read(1).astype(np.float32)[None, ...]
        else:
            mod_data = np.zeros((1, H, W), dtype=np.float32)

    # Scale MS to ~[0,1]
    vmax = np.nanmax(ms_data)
    if vmax > 1.0:
        ms_data /= (10000.0 if vmax > 1000 else 255.0)
    # Scale modality to ~[0,1] (example)
    mod_data = mod_data / 8849.0

    # To (H, W, C)
    ms_data_np  = np.transpose(ms_data, (1, 2, 0))
    mod_data_np = np.transpose(mod_data, (1, 2, 0))

    # -----------------------------
    # Helpers
    # -----------------------------
    def softmax_np(logits, axis=0):
        m = np.max(logits, axis=axis, keepdims=True)
        ex = np.exp(logits - m)
        return ex / np.clip(ex.sum(axis=axis, keepdims=True), 1e-12, None)

    def hann2d(h, w, halo):
        """Tapered 2D Hann window with flat center and halo fade to 0."""
        if halo <= 0:  # no blending
            return np.ones((h, w), dtype=np.float32)
        y = np.ones(h, dtype=np.float32)
        x = np.ones(w, dtype=np.float32)
        # 1D hann on left/right halos
        t = np.arange(halo, dtype=np.float32)
        hann = 0.5 - 0.5 * np.cos(np.pi * (t + 1) / (halo + 1))
        y[:halo] = hann
        y[-halo:] = hann[::-1]
        x[:halo] = hann
        x[-halo:] = hann[::-1]
        return np.outer(y, x)

    def _resize_im(x, out_hw, anti_aliasing=True):
        """
        Resize array
        Supports (C,H,W) or (H,W).
        """
        if sk_resize is None:
            raise ImportError(
                "skimage is required for resizing. "
                "Install with: pip install scikit-image"
            )

        oh, ow = out_hw

        if x.ndim == 3:  # (C, H, W)
            C, _, _ = x.shape
            out = np.empty((C, oh, ow), dtype=np.float32)
            for c in range(C):
                out[c] = sk_resize(
                    x[c],
                    (oh, ow),
                    order=3,               # bicubic
                    mode="reflect",
                    anti_aliasing=anti_aliasing,
                    preserve_range=True
                ).astype(np.float32)
            return out

        elif x.ndim == 2:  # (H, W)
            return sk_resize(
                x,
                (oh, ow),
                order=3,
                mode="reflect",
                anti_aliasing=anti_aliasing,
                preserve_range=True
            ).astype(np.float32)

        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

    def batch_wrapper(batch, args):
        # batch: {'ms': (B,H,W,C), 'mod': (B,H,W,C), 'gt': ...}
        ms_batch  = np.transpose(batch['ms'],  (0, 3, 1, 2))  # (B,C,H,W)
        mod_batch = np.transpose(batch['mod'], (0, 3, 1, 2))  # (B,C,H,W)
        gt_batch  = batch['gt']
        return predict_batch(ms_batch, mod_batch, gt_batch, model_enc, model_dec, args)

    def build_geotransform_batch(src_img, coords_list):
        if src_img:
            return [window_geotransform_3857(src_img, rr, cc) for (rr, cc) in coords_list]
        else:
            return np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)

    # ---------------------------------------
    # 1) Coarse pass: 512→256, stride 512
    # ---------------------------------------
    coarse_win     = getattr(args, "coarse_window", 512)
    coarse_down    = getattr(args, "coarse_down",   256)
    refine_stride  = getattr(args, "refine_stride", 256)  # fine stride (no overlap)
    halo           = getattr(args, "halo", 16)
    alpha_sigmoid  = getattr(args, "sigmoid_alpha", 6.0)
    conf_thresh    = getattr(args, "coarse_conf_thresh", 0.7)
    edge_percent   = getattr(args, "edge_percentile", 80)  # percentile on edge magnitude
    mask_min_frac  = getattr(args, "mask_min_frac", 0.02)  # min fraction to trigger a fine tile

    num_classes = args.num_classes

    # Accumulate coarse logits over the whole image
    coarse_logits = np.zeros((num_classes, H, W), dtype=np.float32)

    # Iterate 512x512, stride 512
    for r in range(0, H, coarse_win):
        r0 = min(r, H - coarse_win)
        for c in range(0, W, coarse_win):
            c0 = min(c, W - coarse_win)
            ms_big  = ms_data_np[r0:r0+coarse_win, c0:c0+coarse_win, :]
            mod_big = mod_data_np[r0:r0+coarse_win, c0:c0+coarse_win, :]

            # Downsample to 256 with antialiasing (area-equivalent)
            if sk_resize is not None:
                ms_small  = sk_resize(ms_big,  (coarse_down, coarse_down, ms_big.shape[-1]),
                                      order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)
                mod_small = sk_resize(mod_big, (coarse_down, coarse_down, mod_big.shape[-1]),
                                      order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)
            else:
                # Fallback: simple stride slicing (not ideal but safe)
                rs = coarse_win // coarse_down
                ms_small  = ms_big[::rs, ::rs, :]
                mod_small = mod_big[::rs, ::rs, :]

            # Predict at 256
            geos = build_geotransform_batch(None, [(0, 0)])  # dummy; your predict only needs shape
            batch = {'ms': ms_small[None, ...], 'mod': mod_small[None, ...], 'gt': np.array(geos)}
            pred_small = batch_wrapper(batch, args)[0]  # (C, 256, 256) logits

            # Upsample logits back to 512 (logit domain, bicubic)
            pred_big = _resize_im(pred_small, (coarse_win, coarse_win), anti_aliasing=False)

            coarse_logits[:, r0:r0+coarse_win, c0:c0+coarse_win] = pred_big

    # ---------------------------------------
    # 2) Build refinement mask from coarse
    # ---------------------------------------
    coarse_probs = softmax_np(coarse_logits, axis=0)                # (C,H,W)
    coarse_conf  = coarse_probs.max(axis=0).astype(np.float32)      # (H,W)

    # Simple edge proxy from confidence map (gradient magnitude)
    gy, gx = np.gradient(coarse_conf)
    edge_mag = np.sqrt(gx*gx + gy*gy).astype(np.float32)
    edge_thr = np.percentile(edge_mag, edge_percent)

    refine_mask = (coarse_conf < conf_thresh) | (edge_mag > edge_thr)

    # ---------------------------------------
    # 3) Selective fine pass on masked tiles
    # ---------------------------------------
    out_logits = coarse_logits.copy()  # start from coarse everywhere
    HANN = hann2d(refine_stride, refine_stride, halo)               # (256,256) taper

    def tile_needs_refine(r0, c0):
        sub = refine_mask[r0:r0+refine_stride, c0:c0+refine_stride]
        if sub.size == 0:  # edge-case safety
            return False
        frac = np.count_nonzero(sub) / float(sub.size)
        return frac >= mask_min_frac

    # Iterate 256x256 tiles; only refine where needed
    for r in range(0, H, refine_stride):
        r0 = min(r, H - refine_stride)
        for c in range(0, W, refine_stride):
            c0 = min(c, W - refine_stride)
            if not tile_needs_refine(r0, c0):
                continue

            # --- crop tile with halo in native resolution ---
            r_in0 = max(r0 - halo, 0)
            r_in1 = min(r0 + refine_stride + halo, H)
            c_in0 = max(c0 - halo, 0)
            c_in1 = min(c0 + refine_stride + halo, W)

            ms_crop_halo  = ms_data_np[r_in0:r_in1, c_in0:c_in1, :]
            mod_crop_halo = mod_data_np[r_in0:r_in1, c_in0:c_in1, :]

            # Actual halo-crop spatial size
            h_window_halo, w_window_halo = ms_crop_halo.shape[:2]

            # --- downsample halo crop to model input size (256x256) ---
            if sk_resize is None:
                raise ImportError("scikit-image is required for halo->256 resizing (recommended).")

            ms_crop = sk_resize(
                ms_crop_halo,
                (coarse_down, coarse_down, ms_crop_halo.shape[-1]),
                order=3, mode="reflect", anti_aliasing=True, preserve_range=True
            ).astype(np.float32)

            mod_crop = sk_resize(
                mod_crop_halo,
                (coarse_down, coarse_down, mod_crop_halo.shape[-1]),
                order=3, mode="reflect", anti_aliasing=True, preserve_range=True
            ).astype(np.float32)

            # Predict on 256x256
            geos = build_geotransform_batch(None, [(0, 0)])
            batch = {'ms': ms_crop[None, ...], 'mod': mod_crop[None, ...], 'gt': np.array(geos)}
            pred_256 = batch_wrapper(batch, args)[0]  # expected: (C, 256, 256)

            # --- upsample logits back to halo-crop size ---
            pred_full_halo = _resize_im(pred_256, (h_window_halo, w_window_halo), anti_aliasing=False)
            # pred_full_halo: (C, h_window_halo, w_window_halo)

            # --- extract center region corresponding to the original tile (256x256) ---
            # Compute where tile lies inside the halo crop:
            # tile region in halo coords starts at (r0 - r_in0, c0 - c_in0)
            rr0 = r0 - r_in0
            cc0 = c0 - c_in0
            pred_fine = pred_full_halo[:, rr0:rr0+refine_stride, cc0:cc0+refine_stride]

            # Sanity check
            assert pred_fine.shape[1:] == (refine_stride, refine_stride), pred_fine.shape

            # Coarse logits for this tile
            pred_coarse_tile = out_logits[:, r0:r0+refine_stride, c0:c0+refine_stride]
            assert pred_coarse_tile.shape == pred_fine.shape, (pred_coarse_tile.shape, pred_fine.shape)

            # Confidence-based fusion
            Cf = softmax_np(pred_fine, axis=0).max(axis=0)        # (256,256)
            Cc = softmax_np(pred_coarse_tile, axis=0).max(axis=0) # (256,256)

            w  = 1.0 / (1.0 + np.exp(-alpha_sigmoid * (Cf - Cc))) # (256,256)

            L_fused = w[None, ...] * pred_fine + (1.0 - w[None, ...]) * pred_coarse_tile

            # Hann blend into canvas
            Wwin = HANN[None, ...]
            region = out_logits[:, r0:r0+refine_stride, c0:c0+refine_stride]
            out_logits[:, r0:r0+refine_stride, c0:c0+refine_stride] = (1.0 - Wwin) * region + Wwin * L_fused


    # ---------------------------------------
    # 4) Post-process & save
    # ---------------------------------------
    # Robust nodata mask (first band <=0 or NaN)
    mask_nodata = (~np.isfinite(ms_data_np[..., 0])) | (ms_data_np[..., 0] <= 0)

    # Final class map from fused logits
    class_map = np.argmax(out_logits, axis=0).astype(np.uint16)
    class_map[class_map < 0] = 0
    class_map[mask_nodata] = args.ignore_index

    # Optional confidence map (top-1 prob from fused logits)
    if getattr(args, "save_confidence", False):
        probs = softmax_np(out_logits, axis=0)
        conf  = probs.max(axis=0).astype(np.float32)
        conf[mask_nodata] = np.nan
    else:
        conf = None

    # Save class GeoTIFF
    out_profile = profile.copy()
    out_profile.update({
        "count": 1,
        "dtype": "int16",
        "compress": "deflate",
        "tiled": True,
        "no-data": 0,
    })

    out_path = build_out_path(
        image_path=image_path,
        output_dir=args.output_dir,
        bands=args.bands,
        tile_size=getattr(args, "tile_size", refine_stride),
        stride=getattr(args, "stride", refine_stride),
        task=getattr(args, "task", "segmentation"),
        encoder_ckpt=args.encoder_checkpoint,
        decoder_ckpt=args.decoder_checkpoint,
        add_git_rev=True,
    )

    class_map_to_write = class_map.astype(np.int16)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(class_map_to_write, 1)
        if hasattr(args, "colormap") and isinstance(args.colormap, dict) and out_profile["dtype"] == "int16":
            dst.write_colormap(1, args.colormap)
        dst.set_band_description(1, "class_id")
        dst.update_tags(
            ignore_index=str(args.ignore_index),
            num_classes=str(args.num_classes),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    # Optional confidence TIFF
    if getattr(args, "save_confidence", False) and conf is not None:
        conf_profile = profile.copy()
        conf_profile.update({"count": 1, "dtype": "float32", "compress": "deflate", "tiled": True, "no-data": 0})
        conf_path = os.path.splitext(out_path)[0] + "_confidence.tif"
        with rasterio.open(conf_path, "w", **conf_profile) as dst:
            dst.write(conf.astype(np.float32), 1)
            dst.set_band_description(1, "top1_prob")

    # Sidecar JSON
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image": os.path.abspath(image_path),
        "bands": args.bands,
        "tile_size": getattr(args, "tile_size", refine_stride),
        "stride": getattr(args, "stride", refine_stride),
        "encoder_ckpt": os.path.abspath(args.encoder_checkpoint),
        "encoder_sha1_8": _short_sha1_of_file(args.encoder_checkpoint),
        "decoder_ckpt": os.path.abspath(args.decoder_checkpoint),
        "decoder_sha1_8": _short_sha1_of_file(args.decoder_checkpoint),
        "git_rev": _git_short_rev(),
        "decoder_type": "DPT",
        "task": "segmentation",
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        # debug knobs
        "coarse_window": coarse_win,
        "coarse_down": coarse_down,
        "refine_stride": refine_stride,
        "halo": halo,
        "sigmoid_alpha": alpha_sigmoid,
        "coarse_conf_thresh": conf_thresh,
        "edge_percentile": edge_percent,
        "mask_min_frac": mask_min_frac,
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[✓] Saved class map to: {out_path}")
    if getattr(args, "save_confidence", False) and conf is not None:
        print(f"[✓] Saved confidence map to: {conf_path}")

    return out_path


    
# --------------------------------------
# Batch inference
# --------------------------------------

def run_floro_batch_inference(
    image_dir, modality_dir, output_dir,
    model_enc, model_dec, args
):
    """
    Runs FLORO inference across a folder of images and their corresponding modality rasters.
    
    Parameters
    ----------
    image_dir : str
        Directory with multispectral images (e.g., RGBN .tif)
    modality_dir : str
        Directory with modality files (e.g., DSM/DTM)
    output_dir : str
        Directory where outputs will be saved
    model_enc, model_dec : torch.nn.Module
        FLORO encoder and decoder
    args : argparse.Namespace
        Inference arguments (must include .tile_size, .stride, .num_channels, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob(os.path.join(image_dir, "*.tif")))

    if len(image_files) == 0:
        raise ValueError(f"No image files found in: {image_dir}")

    print(f"🔍 Found {len(image_files)} images in: {image_dir}")
    print(f"🔗 Trying to match modality files from: {modality_dir}")

    for image_path in image_files:
        base = os.path.basename(image_path)
        prefix = base.replace("_Orthomosaic.tif", "_DSM.tif").replace(".tif", "")  # customize as needed

        # Try to find a matching modality file
        modality_path = None
        possible = glob(os.path.join(modality_dir, f"{prefix}*.tif"))
        if len(possible) > 0:
            modality_path = possible[0]  # use first match
        else:
            print(f"⚠️ No modality found for {prefix}. Proceeding with masked modality.")

        try:
            result_path = run_floro_inference_with_blending(
                image_path, modality_path, output_dir,
                model_enc, model_dec, args
            )
            print(f"[✓] Inference complete for: {prefix}")
            print(f"File saved in: {result_path}")
        except Exception as e:
            print(f"[✗] Error processing {image_path}: {e}")

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
    ckpt = torch.load(ckpt_path, map_location=device)
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


# --------------------------------------
# Main Inference Routine
# --------------------------------------

def main(args):
    # -----------------------
    # File Validation
    # -----------------------
    verify_checkpoint_file(args.encoder_checkpoint, "Encoder checkpoint")
    
    if args.decoder_checkpoint !="":
        verify_checkpoint_file(args.decoder_checkpoint, "Decoder checkpoint")

    if args.mode == "single":
        verify_file_exists(args.image_path, "Input image")
        verify_file_exists(args.modality_path, "Modality file")

    # -----------------------
    # Set Device and Time
    # -----------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # -----------------------
    # Init Weights & Biases
    # -----------------------
    wandb.init(
        project=args.wb_project,
        config=vars(args)
    )
    if args.run_name:
        wandb.run.name = args.run_name

    # -----------------------
    # Load Models
    # -----------------------
    model_enc = MultiMAE_Encoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        multispectral_channels=args.input_channels,
        srtm_channels=args.elev_channels,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        pos_embed_type=args.pos_embed_type,
    ).to(device)

    model_dec = FLORODPTDecoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        dec_d_model=args.dec_d_model,
        dpt_channels=args.dpt_channels,
        out_channels=args.num_classes
    ).to(device)

    # -----------------------
    # Load Checkpoints
    # -----------------------
    torch.cuda.empty_cache()
    # Load model states (handle both single and multi-model formats)
    try:
        # 1) Encoder: load from its own ckpt (or from a combined ckpt if that’s what you passed)
        ok_enc = load_weights_strip_module(model_enc, args.encoder_checkpoint, device, which="encoder")

        # 2) Decoder: prefer an explicit decoder ckpt if provided
        if args.decoder_checkpoint:
            ok_dec = load_weights_strip_module(model_dec, args.decoder_checkpoint, device, which="decoder")
        else:
            # try to find decoder weights inside the same (combined) encoder checkpoint
            # ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=True)
            # dec_sd = _extract_state_dict(ckpt, which="decoder")
            # if dec_sd is not None:
            #     dec_sd = _strip_module_prefix(dec_sd)
            #     try:
            #         model_dec.load_state_dict(dec_sd, strict=True)
            #     except RuntimeError as e:
            #         print(f"[WARN] Strict load (decoder from combined ckpt) failed: {e}. Retrying with strict=False…")
            #         model_dec.load_state_dict(dec_sd, strict=False)
            #     print("[✓] Decoder weights loaded (from combined checkpoint).")
            #     ok_dec = True
            # else:
            #     print("[INFO] No decoder weights found in combined checkpoint and no --decoder_checkpoint provided. "
            #         "Decoder will remain randomly initialized.")
            #     ok_dec = False
            try:
                ok_dec = load_weights_strip_module(model_dec, args.encoder_checkpoint, device, which="decoder")
            except:
                print(f"[WARN] Strict load (decoder from combined ckpt) failed.")
                ok_dec = False
                

        if not ok_enc:
            raise RuntimeError("Could not locate encoder state_dict in the provided checkpoint.")
        if not ok_dec:
            raise RuntimeError("Could not locate decoder state_dict in the provided checkpoint.")
        # (ok_dec can be False by design if you intentionally run encoder-only)

    except Exception as e:
        print(f"❌ Failed to load encoder/decoder checkpoints: {e}")
        raise


    if args.mode == "single":
        run_floro_inference_with_blending(
            args.image_path, args.modality_path, args.output_dir,
            model_enc, model_dec, args
        )

    elif args.mode == "batch":
        run_floro_batch_inference(
            args.image_directory, args.modality_directory, args.output_dir,
            model_enc, model_dec, args
        )

# --------------------------------------
# CLI Entry Point
# --------------------------------------

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
