import os
import json
import glob
import math
import random as rn
import numpy as np
import torch
import rasterio
from pyproj import Transformer
from skimage import transform as sktf, filters, util
from torchvision import transforms
from torch.utils.data import Dataset, Subset

# --------------------------
# Fixed layouts
# --------------------------
C_IMAGE = 13  # 8 data + 5 validity
# data: 0:B 1:G 2:R 3:RE 4:NIR 5:NIR2 6:SWIR1 7:SWIR2
# valid: 8:v_BGR 9:v_RE 10:v_NIR 11:v_NIR2 12:v_SWIR

C_MOD = 5     # 0:Elev 1:VV 2:VH 3:v_ELEV 4:v_SAR

np.random.seed(0)
torch.manual_seed(0)

# --------------------------
# Helpers packing
# --------------------------
OPT_RANGE  = (0.0, 1.0)
ELEV_RANGE = (-500.0, 9000.0)
SAR_RANGE  = (-60.0, 20.0)
NODATA_SENTINELS = (-32768.0, -32670.0, -9999.0)

def _ensure_chw(a, expected_c=None, name="arr"):
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D (C,H,W); got {a.shape}")
    if expected_c is not None and a.shape[0] != expected_c:
        raise ValueError(f"{name} must have C={expected_c}; got {a.shape[0]}")
    return a.astype(np.float32)

def _clean_with_range(x_chw, lo, hi, nodata_values=None):
    """
    x_chw: (C,H,W) float32
    returns: x_clean (C,H,W), valid_mask (H,W) float32 with 1=valid, 0=invalid
    """
    x = x_chw
    finite = np.isfinite(x)  # (C,H,W)

    inrange = (x >= lo) & (x <= hi)  # (C,H,W)

    valid_c = finite & inrange

    # optional explicit nodata sentinels (exact match)
    if nodata_values is not None:
        # nodata_values can be list/tuple of floats
        for nd in nodata_values:
            valid_c &= (x != np.float32(nd))

    # group validity: pixel valid only if all channels valid
    valid_hw = valid_c.all(axis=0).astype(np.float32)  # (H,W)

    # fill invalid pixels with 0
    invalid_c = ~valid_c
    if invalid_c.any():
        x = x.copy()
        x[invalid_c] = 0.0

    return x, valid_hw

# --------------------------
# Packing
# --------------------------
def pack_image(H, W, BGR=None, RE=None, NIR=None, NIR2=None, SWIR=None,
               vBGR=0.0, vRE=0.0, vNIR=0.0, vNIR2=0.0, vSWIR=0.0):
    img = np.zeros((C_IMAGE, H, W), dtype=np.float32)

    def _ensure_1(a):
        if a is None:
            return None
        if a.ndim == 2:
            return a[None, ...]
        return a

    if BGR is not None:
        BGR = _ensure_chw(BGR, expected_c=3, name="BGR")
        BGR, mBGR = _clean_with_range(BGR, *OPT_RANGE, nodata_values=NODATA_SENTINELS)
        img[0:3] = BGR.astype(np.float32)
        img[8] = np.float32(vBGR) * mBGR.astype(np.float32)

    RE = _ensure_1(RE)
    if RE is not None:
        RE = _ensure_chw(RE, expected_c=1, name="RE")
        RE, mRE = _clean_with_range(RE, *OPT_RANGE, nodata_values=NODATA_SENTINELS)
        img[3:4] = RE.astype(np.float32)
        img[9] = np.float32(vRE) * mRE.astype(np.float32)

    NIR = _ensure_1(NIR)
    if NIR is not None:
        NIR = _ensure_chw(NIR, expected_c=1, name="NIR")
        NIR, mNIR = _clean_with_range(NIR, *OPT_RANGE, nodata_values=NODATA_SENTINELS)
        img[4:5] = NIR.astype(np.float32)
        img[10] = np.float32(vNIR) * np.float32(mNIR)

    NIR2 = _ensure_1(NIR2)
    if NIR2 is not None:
        NIR2 = _ensure_chw(NIR2, expected_c=1, name="NIR2")
        NIR2, mNIR2 = _clean_with_range(NIR2, *OPT_RANGE, nodata_values=NODATA_SENTINELS)
        img[5:6] = NIR2.astype(np.float32)
        img[11] = np.float32(vNIR2) * np.float32(mNIR2)    

    if SWIR is not None:
        SWIR = _ensure_chw(SWIR, name="SWIR")
        
        if SWIR.shape[0] == 1:
            SWIR, mSWIR = _clean_with_range(SWIR, *OPT_RANGE, nodata_values=NODATA_SENTINELS)
            img[6:7] = SWIR.astype(np.float32)
        else:
            SWIR, mSWIR = _clean_with_range(SWIR, *OPT_RANGE, nodata_values=NODATA_SENTINELS)
            img[6:8] = SWIR[:2].astype(np.float32)
        img[12] = np.float32(vSWIR) * np.float32(mSWIR)

    return img

def pack_modalities(H, W, elev=None, sar=None, vE=0.0, vS=0.0):
    mod = np.zeros((C_MOD, H, W), dtype=np.float32)

    if elev is not None:
        if elev.ndim == 2:
            elev = elev[None, ...]
        elev = _ensure_chw(elev, expected_c=1, name="elev")
        elev, mE = _clean_with_range(elev, *ELEV_RANGE, nodata_values=NODATA_SENTINELS)
        mod[0:1] = elev.astype(np.float32)
        mod[3] = np.float32(vE) * np.float32(mE)

    if sar is not None:
        if sar.ndim == 2:
            sar = sar[None, ...]
        if sar.shape[0] != 2:
            raise ValueError("SAR must be (2,H,W) (VV,VH).")
        sar = _ensure_chw(sar, expected_c=2, name="sar")  # (2,H,W)
        sar, mS = _clean_with_range(sar, *SAR_RANGE, nodata_values=NODATA_SENTINELS)
        mod[1:3] = sar.astype(np.float32)
        mod[4] = np.float32(vS) * np.float32(mS)

    return mod

# --------------------------
# Augmentation (CHW-safe)
# --------------------------
class Augmentation(object):
    def __init__(self, rotation_type='fixed90', blur_sigma=1.0, noise_max=0.02, type=None):
        self.rotation_type = rotation_type
        self.blur_sigma = blur_sigma
        self.noise_max = noise_max
        self.type = type

    def __call__(self, sample):
        x = sample["image"]       # (C_IMAGE,H,W)
        m = sample["modalities"]  # (C_MOD,H,W)

        # Convert to HWC for skimage
        x_hwc = np.moveaxis(x, 0, -1)
        m_hwc = np.moveaxis(m, 0, -1)

        # rotation angle
        if self.rotation_type == "free":
            angle = float(np.random.uniform(0, 360))
        elif self.rotation_type == "fixed90":
            angle = float(np.random.choice([0, 90, 180, 270]))
        else:
            angle = 0.0

        if angle != 0.0:
            x_hwc = sktf.rotate(x_hwc, angle, resize=False, preserve_range=True)
            m_hwc = sktf.rotate(m_hwc, angle, resize=False, preserve_range=True)

        if self.type == "training":
            # blur only optical data channels (0..7) to avoid blurring validity masks
            sigma = rn.uniform(0, self.blur_sigma)
            x_data = x_hwc[..., :8]
            try:
                x_data = filters.gaussian(x_data, sigma=sigma, preserve_range=True, channel_axis=-1)
            except TypeError:
                x_data = filters.gaussian(x_data, sigma=(sigma, sigma, 0), preserve_range=True)

            # noise only on optical data channels
            x_data = util.random_noise(x_data, mode="gaussian", var=rn.uniform(0, self.noise_max)**2)

            x_hwc[..., :8] = x_data

        # Back to CHW
        sample["image"] = np.moveaxis(x_hwc, -1, 0).astype(np.float32)
        sample["modalities"] = np.moveaxis(m_hwc, -1, 0).astype(np.float32)
        return sample

# --------------------------
# Validity-aware normalization
# --------------------------
class NormalizeWithValidity(object):
    """
    Expects:
      image: (12,H,W)  data[0..7], validity[8..11]
      modalities: (5,H,W) data[0..2], validity[3..4]
    img_mean/std: length 8
    mod_mean/std: length 3
    """
    def __init__(self, img_mean, img_std, mod_mean, mod_std, eps=1e-6):
        self.img_mean = np.asarray(img_mean, dtype=np.float32)  # (8,)
        self.img_std  = np.asarray(img_std, dtype=np.float32)   # (8,)
        self.mod_mean = np.asarray(mod_mean, dtype=np.float32)  # (3,)
        self.mod_std  = np.asarray(mod_std, dtype=np.float32)   # (3,)
        self.eps = eps

    def __call__(self, sample):
        img = sample["image"]
        mod = sample["modalities"]
        H, W = img.shape[1], img.shape[2]

        # Build per-channel validity for the 8 optical channels
        m_img = np.zeros((8, H, W), dtype=np.float32)
        m_img[0:3] = img[8]     # v_BGR
        m_img[3]   = img[9]     # v_RE
        m_img[4]   = img[10]    # v_NIR -> NIR
        m_img[5]   = img[11]    # v_NIR2 -> NIR2
        m_img[6:8] = img[12]    # v_SWIR

        for c in range(8):
            if self.img_std[c] < self.eps:
                continue
            x = img[c]
            img[c] = np.where(
                m_img[c] > 0.5,
                (x - self.img_mean[c]) / (self.img_std[c] + self.eps),
                0.0
            )

        # Modalities masks
        m_mod = np.zeros((3, H, W), dtype=np.float32)
        m_mod[0] = mod[3]  # v_ELEV
        m_mod[1] = mod[4]  # v_SAR
        m_mod[2] = mod[4]  # v_SAR

        for c in range(3):
            if self.mod_std[c] < self.eps:
                continue
            x = mod[c]
            mod[c] = np.where(
                m_mod[c] > 0.5,
                (x - self.mod_mean[c]) / (self.mod_std[c] + self.eps),
                0.0
            )

        sample["image"] = img
        sample["modalities"] = mod
        return sample

import numpy as np

class BandDropping(object):
    """
    Randomly drops band/modalities groups and updates validity channels.

    Expects packed tensors:
      image: (13,H,W)
        data: 0:B 1:G 2:R 3:RE 4:NIR 5:NIR2 6:SWIR1 7:SWIR2
        valid: 8:v_BGR 9:v_RE 10:v_NIR 11:v_NIR2 12:v_SWIR

      modalities: (5,H,W)
        data: 0:Elev 1:VV 2:VH
        valid: 3:v_ELEV 4:v_SAR

    Notes:
      - Drops are applied only with probability p_apply.
      - If sample["source"] indicates UAV RGB, ONLY elevation can be dropped.
      - Dropping sets the corresponding data channels to 0 and the validity channel to 0.
    """

    def __init__(
        self,
        p_apply=0.7,
        p_drop_bgr=0.05,
        p_drop_re=0.10,
        p_drop_nir=0.10,
        p_drop_nir2=0.10,
        p_drop_swir=0.10,
        p_drop_elev=0.15,
        p_drop_sar=0.10,
        keep_at_least_one_optical_group=True,
        seed=None,
    ):
        self.p_apply = float(p_apply)
        self.p_drop = {
            "bgr": float(p_drop_bgr),
            "re": float(p_drop_re),
            "nir": float(p_drop_nir),
            "nir2": float(p_drop_nir2),
            "swir": float(p_drop_swir),
            "elev": float(p_drop_elev),
            "sar": float(p_drop_sar),
        }
        self.keep_at_least_one_optical_group = bool(keep_at_least_one_optical_group)
        self.rng = np.random.default_rng(seed)

        # Packed indices
        self.idx = {
            "bgr_ch": slice(0, 3),
            "re_ch": slice(3, 4),
            "nir_ch": slice(4, 5),
            "nir2_ch": slice(5, 6),
            "swir_ch": slice(6, 8),
            "v_bgr": 8,
            "v_re": 9,
            "v_nir": 10,
            "v_nir2": 11,
            "v_swir": 12,
            "elev_ch": slice(0, 1),   # modalities
            "sar_ch": slice(1, 3),
            "v_elev": 3,
            "v_sar": 4,
        }

    def _drop_opt(self, img, group):
        if group == "bgr":
            img[self.idx["bgr_ch"], :, :] = 0.0
            img[self.idx["v_bgr"], :, :] = 0.0
        elif group == "re":
            img[self.idx["re_ch"], :, :] = 0.0
            img[self.idx["v_re"], :, :] = 0.0
        elif group == "nir":
            img[self.idx["nir_ch"], :, :] = 0.0
            img[self.idx["v_nir"], :, :] = 0.0
        elif group == "nir2":
            img[self.idx["nir2_ch"], :, :] = 0.0
            img[self.idx["v_nir2"], :, :] = 0.0
        elif group == "swir":
            img[self.idx["swir_ch"], :, :] = 0.0
            img[self.idx["v_swir"], :, :] = 0.0
        else:
            raise ValueError(f"Unknown optical group: {group}")

    def _drop_mod(self, mod, group):
        if group == "elev":
            mod[self.idx["elev_ch"], :, :] = 0.0
            mod[self.idx["v_elev"], :, :] = 0.0
        elif group == "sar":
            mod[self.idx["sar_ch"], :, :] = 0.0
            mod[self.idx["v_sar"], :, :] = 0.0
        else:
            raise ValueError(f"Unknown modality group: {group}")

    def __call__(self, sample):
        if self.rng.random() >= self.p_apply:
            return sample

        img = sample["image"]
        mod = sample["modalities"]

        source = str(sample.get("source", "")).lower()
        is_uav_rgb = ("uav_rgb" in source) or ("rgb" in source and "uav" in source)

        # Determine which groups are currently present (group-level validity)
        present = {
            "bgr":  float(np.mean(img[self.idx["v_bgr"]])) > 0.5,
            "re":   float(np.mean(img[self.idx["v_re"]])) > 0.5,
            "nir":  float(np.mean(img[self.idx["v_nir"]])) > 0.5,
            "nir2": float(np.mean(img[self.idx["v_nir2"]])) > 0.5,
            "swir": float(np.mean(img[self.idx["v_swir"]])) > 0.5,
            "elev": float(np.mean(mod[self.idx["v_elev"]])) > 0.5,
            "sar":  float(np.mean(mod[self.idx["v_sar"]])) > 0.5,
        }

        # Always allow elevation drop (including UAV RGB)
        if present["elev"] and (self.rng.random() < self.p_drop["elev"]):
            self._drop_mod(mod, "elev")

        # UAV RGB rule: do not drop any optical or SAR, only elevation
        if is_uav_rgb:
            sample["modalities"] = mod
            return sample

        # SAR drop
        if present["sar"] and (self.rng.random() < self.p_drop["sar"]):
            self._drop_mod(mod, "sar")

        # Optical drops
        to_drop = []
        for g in ["bgr", "re", "nir", "nir2", "swir"]:
            if present[g] and (self.rng.random() < self.p_drop[g]):
                to_drop.append(g)

        # Ensure at least one optical group remains if requested
        if self.keep_at_least_one_optical_group:
            present_opt = [g for g in ["bgr", "re", "nir", "nir2", "swir"] if present[g]]
            if len(present_opt) > 0:
                remaining = [g for g in present_opt if g not in to_drop]
                if len(remaining) == 0:
                    # keep one present group
                    keep = self.rng.choice(present_opt)
                    to_drop = [g for g in to_drop if g != keep]

        for g in to_drop:
            self._drop_opt(img, g)

        sample["image"] = img
        sample["modalities"] = mod
        return sample

# --------------------------
# ToTensor
# --------------------------
class ToTensor(object):
    def __call__(self, sample):
        image = sample["image"]
        modalities = sample["modalities"]
        gt = sample["geo_transform"]
        return {
            "image": torch.from_numpy(image).float(),
            "modalities": torch.from_numpy(modalities).float(),
            "geo_transform": torch.tensor(gt, dtype=torch.float32),
        }

# --------------------------
# Dataset reading from subfolders
# --------------------------
class ImageRSMultiFolderDataset(Dataset):
    """
    Expects folder structure like:
      root_dir/uav_rgb_dsm/*.tif
      root_dir/uav_ms_dsm/*.tif
      root_dir/france_dtm/*.tif
      root_dir/s2_dem_s1/*.tif
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform

        self.source_rules = {
            "uav_rgb_dsm": dict(bgr=[1,2,3], re=None, nir=None, nir2=None, swir=None, elev=[4], sar=None),
            "uav_ms_dsm":  dict(bgr=[1,2,3], re=[4], nir=[5], nir2=None, swir=None, elev=[6], sar=None),
            "france_dtm":  dict(bgr=[1,2,3], re=None, nir=[4], nir2=None, swir=None, elev=[5], sar=None),
            "s2_dem_s1":   dict(bgr=[1,2,3], re=[4], nir=[5], nir2=[6], swir=[7,8], elev=[9], sar=[10,11]),
        }

        items = []
        for key in self.source_rules.keys():
            fps = sorted(glob.glob(os.path.join(root_dir, key, "*.tif")))
            items += [(fp, key) for fp in fps]

        if len(items) == 0:
            raise FileNotFoundError(f"No .tif files found under {root_dir}/*/*.tif")

        self.items = items

        # Load precomputed geotransforms
        gt_path = os.path.join(root_dir,"geotransforms_3857.json")
        try:
            with open(gt_path, "r") as f:
                self.gt_cache = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {gt_path}. Computing geotransforms on training.")
            self.gt_cache = None
        except json.JSONDecodeError:
            print(f"Invalid JSON in {gt_path}. Computing geotransforms on training.")
            self.gt_cache = None

    def __len__(self):
        return len(self.items)
    
    def _gt_key(self, image_path: str) -> str:
        return os.path.relpath(image_path, self.root_dir).replace("\\", "/")

    def _read(self, src, idxs):
        if idxs is None:
            return None
        arr = src.read(indexes=idxs).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr

    def load_image(self, image_path):
        with rasterio.open(image_path) as src:
            img = src.read().astype(np.float32)  # (C,H,W)
            orig_transform = src.transform
            crs = src.crs

            key = self._gt_key(image_path)
            #print(f"Key: {key}")
            if key not in self.gt_cache:
                raise KeyError(f"GeoTransform cache miss for key='{key}'. "
                            f"Example cache keys: {list(self.gt_cache.keys())[:3]}")

            if self.gt_cache is not None:
                array = np.array(self.gt_cache[key], dtype=np.float32)
                geo_transform = np.array([
                    array[0],
                    array[2],
                    0,
                    array[1],
                    0,
                    -abs(array[3])
                ], dtype=np.float32)
            else:
                # Avoid per-sample reprojection as it will impact training time
                transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
                origin_x, origin_y = orig_transform.c, orig_transform.f
                pixel_width, pixel_height = orig_transform.a, orig_transform.e

                origin_x_3857, origin_y_3857 = transformer.transform(origin_x, origin_y)
                next_x_3857, next_y_3857 = transformer.transform(origin_x + pixel_width, origin_y + pixel_height)

                pixel_width_m = next_x_3857 - origin_x_3857
                pixel_height_m = next_y_3857 - origin_y_3857

                geo_transform = np.array([
                    origin_x_3857,
                    pixel_width_m,
                    0,
                    origin_y_3857,
                    0,
                    -abs(pixel_height_m)
                ], dtype=np.float32)

        # return HWC for slicing logic, or keep CHW and slice there.
        img_hwc = np.moveaxis(img, 0, -1)  # (H,W,C)
        return img_hwc, geo_transform

    def __getitem__(self, idx):
        fp, src_key = self.items[idx]
        #print(f"src_key: {src_key}")
        spec = self.source_rules[src_key]
        #print(f"spec: {spec}")

        #try:
        image_hwc, gt = self.load_image(fp)
        H, W, C = image_hwc.shape

        chw = np.moveaxis(image_hwc, -1, 0)  # (C,H,W)

        # Read groups by spec (1-indexed in spec, but we're slicing numpy 0-indexed here)
        # Because load_image already read ALL bands, we slice from chw directly.
        def pick(idxs):
            if idxs is None:
                return None
            # idxs are 1-indexed band numbers in spec
            z = chw[np.array(idxs) - 1]
            return z

        bgr  = pick(spec["bgr"])
        re   = pick(spec["re"])
        nir  = pick(spec["nir"])
        nir2 = pick(spec["nir2"])
        swir = pick(spec["swir"])
        elev = pick(spec["elev"])
        sar  = pick(spec["sar"])

        # validity flags: group-present / group-missing
        vBGR  = 1.0 if bgr  is not None else 0.0
        vRE   = 1.0 if re   is not None else 0.0
        vNIR  = 1.0 if nir  is not None else 0.0
        vNIR2 = 1.0 if nir2 is not None else 0.0
        vSWIR = 1.0 if swir is not None else 0.0
        vE    = 1.0 if elev is not None else 0.0
        vS    = 1.0 if sar  is not None else 0.0

        img = pack_image(H, W, BGR=bgr, RE=re, NIR=nir, NIR2=nir2, SWIR=swir,
                            vBGR=vBGR, vRE=vRE, vNIR=vNIR, vSWIR=vSWIR)
        mod = pack_modalities(H, W, elev=elev, sar=sar, vE=vE, vS=vS)
        #print(f"Image: {img.shape}")
        #print(f"Modality:{mod.shape}")

        sample = {"image": img, "modalities": mod, "geo_transform": gt}

        if self.transform:
            sample = self.transform(sample)

        return sample

        # except Exception as e:
        #     print(f"Error loading {fp}: {e}")
        #     return None

# --------------------------
# Loader preparation
# --------------------------
def LoadersPreparation(
    root_dir,
    test_size=0.0,
    img_mean=None,
    img_std=None,
    mod_mean=None,
    mod_std=None,
    rotate_type="fixed90",
    blur_sigma=1.0,
    noise_max=0.2,
    type="training",
    seed=0,
):
    train_transform = transforms.Compose([
        Augmentation(rotation_type=rotate_type, blur_sigma=blur_sigma, noise_max=noise_max, type=type),
        NormalizeWithValidity(img_mean=img_mean, img_std=img_std, mod_mean=mod_mean, mod_std=mod_std),
        ToTensor(),
    ])

    test_transform = transforms.Compose([
        NormalizeWithValidity(img_mean=img_mean, img_std=img_std, mod_mean=mod_mean, mod_std=mod_std),
        ToTensor(),
    ])

    train_base = ImageRSMultiFolderDataset(root_dir=root_dir, transform=train_transform)
    test_base  = ImageRSMultiFolderDataset(root_dir=root_dir, transform=test_transform)

    n = len(train_base)
    if test_size == 0 or test_size == 0.0:
        return train_base, None

    if not (0 < test_size < 1):
        raise ValueError("test_size should be a fraction in (0,1).")

    n_test = int(round(test_size * n))
    n_train = n - n_test

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]

    return Subset(train_base, train_idx), Subset(test_base, test_idx)