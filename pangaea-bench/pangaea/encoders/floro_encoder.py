from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
import math
from enum import Enum
from typing import Optional, Type
from itertools import repeat as repeat_tuple
from einops import rearrange

from pangaea.encoders.base import Encoder


class InputScale(str, Enum):
    FLORO_NORM = "floro_norm"     # already in FLORO regime (just finite check)
    UNIT_01    = "unit_01"        # reflectance in [0,1] (range check 0..1)
    DATASET_RAW = "dataset_raw"   # raw dataset numbers -> map to FLORO using ds_stats + floro_stats

class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w, d=embed_dim)
    return pos_emb

def trunc_normal_(tensor, mean=0., std=1.):
    """Truncated normal initialization."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

class DropPath(nn.Module):
    """Stochastic Depth per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma

class FFN(nn.Module):
    """timm/ViT-style MLP, no hidden LayerNorm."""
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_inner, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(d_inner, d_model, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def nchw_to(x: torch.Tensor, fmt: Format):
    """Convert tensor from NCHW format to specified format.

    Args:
        x: Input tensor in NCHW format.
        fmt: Target format.

    Returns:
        Tensor in target format.
    """
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = tuple(repeat_tuple(patch_size,2))
        if img_size is not None:
            self.img_size = tuple(repeat_tuple(img_size,2))
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None
        self.flatten = flatten
        self.output_fmt = 'NCHW'
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
            assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != 'NCHW':
            x = nchw_to(x, self.output_fmt)

        return x

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttentionBlockViT(nn.Module):
    """timm-like pre-norm block (DropPath + optional LayerScale)."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,  # e.g., 1e-5 to enable LayerScale
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_bias: bool = True,
        proj_bias: bool = True,               # if you expose it in Attention/FFN
    ):
        super().__init__()

        self.norm1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.ls1 = LayerScale(d_model, init_values) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(d_model)
        self.mlp = FFN(
            d_model=d_model,
            d_inner=int(d_model * mlp_ratio),
            drop=proj_drop,         
            bias=mlp_bias,
        )
        self.ls2 = LayerScale(d_model, init_values) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def init_patch_embed_as_linear(patch_embed: nn.Module):
    """Initialize PatchEmbed.proj Conv2d weights like a Linear layer."""
    proj = patch_embed.proj
    if isinstance(proj, nn.Conv2d):
        w = proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if proj.bias is not None:
            nn.init.constant_(proj.bias, 0)

class FLOROGeoEncoder(nn.Module):
    def __init__(
        self,
        image_size: int=256, 
        patch_size: int=16, 
        multispectral_channels: int=4,  # Multispectral channels (R, G, B, NIR)
        modalities_channels: int=1,           # modalities channels
        d_model: int=1280,
        depth: int=32,
        num_heads: int=16,
        mlp_ratio: int=4,
        pos_embed_type: str="geo",
        norm_layer=nn.LayerNorm,
        drop_max: float= 0.2 
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.depth = depth
        self.dpr = torch.linspace(0, drop_max, steps=depth).tolist()
        self.pos_embed_type = pos_embed_type
        assert self.d_model % 4 == 0, "d_model must be divisible by 4 for positional encoding."
        
        self.patch_embed_multi = PatchEmbed(image_size,
        patch_size, multispectral_channels, d_model)
        self.patch_embed_modalities = PatchEmbed(image_size,
        patch_size, modalities_channels, d_model)
        
        # Initialize positional embeddings
        self.h_posemb = self.w_posemb = self.image_size // patch_size
        if self.pos_embed_type == 'geo':
            # Hybrid geo encoding: geographic coords will be added dynamically later
            self.pos_emb = nn.Parameter(build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=self.d_model), requires_grad=False)
        elif self.pos_embed_type == 'absolute':
            self.pos_emb = nn.Parameter(build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=self.d_model), requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.h_posemb, self.w_posemb, self.d_model))
            trunc_normal_(self.pos_emb, std=0.02)

        init_patch_embed_as_linear(self.patch_embed_multi)
        init_patch_embed_as_linear(self.patch_embed_modalities) 


        self.transformer_blocks = nn.ModuleList([
            SelfAttentionBlockViT(
                d_model, num_heads, mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=0.0,        # or 0.1; tune
                attn_drop=0.0,        # or small
                drop_path=self.dpr[i],     # <-- schedule
                init_values=1e-5,     # <-- enable LayerScale
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ViT/MAE-style init for Linear + LayerNorm."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_normalized_centroids(self, geotransform, device):
        B = geotransform.shape[0]
        h_posemb, w_posemb = self.h_posemb, self.w_posemb

        # Grid indices
        i_coords = torch.arange(h_posemb, device=device, dtype=torch.float32)
        j_coords = torch.arange(w_posemb, device=device, dtype=torch.float32)
        i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing="ij")  # [H, W] # i=row (y), j=col (x)

        # Absolute coordinates (EPSG:3857 meters)
        origin_x = geotransform[:, 0]  # Easting
        origin_y = geotransform[:, 3]  # Northing
        pixel_width = geotransform[:, 1]
        pixel_height = geotransform[:, 5]

        center_x_abs = origin_x.view(-1,1,1) + (j_grid * self.patch_size + self.patch_size/2) * pixel_width.view(-1,1,1)
        center_y_abs = origin_y.view(-1,1,1) + (i_grid * self.patch_size + self.patch_size/2) * pixel_height.view(-1,1,1)

        # Normalize global coordinates (Web Mercator bounds)
        global_min_x, global_max_x = -20037508.34, 20037508.34
        global_min_y, global_max_y = -20048966.10, 20048966.10

        norm_global_x = (center_x_abs - global_min_x) / (global_max_x - global_min_x)
        norm_global_y = (center_y_abs - global_min_y) / (global_max_y - global_min_y)

        # Stack global coords only → [B, H, W, 2] → reshape to [B, N, 2]
        coords = torch.stack([norm_global_x, norm_global_y], dim=-1).view(B, -1, 2)

        return coords

    def positional_encoding(self, coords):
        """Hybrid encoding: sin-cos global position + static absolute grid embedding"""
        B, N, _ = coords.shape  # coords: [B, N, 2] = [global_x, global_y]

        enc = torch.zeros((B, N, self.d_model), device=coords.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model // 2, 2, device=coords.device) *
            -(math.log(10000.0) / (self.d_model // 2))
        )

        enc[..., 0::4] = torch.sin(coords[..., 0].unsqueeze(-1) * div_term)  # global_x
        enc[..., 1::4] = torch.cos(coords[..., 0].unsqueeze(-1) * div_term)
        enc[..., 2::4] = torch.sin(coords[..., 1].unsqueeze(-1) * div_term)  # global_y
        enc[..., 3::4] = torch.cos(coords[..., 1].unsqueeze(-1) * div_term)

        # Add static absolute 2D positional embedding (flattened)
        pos_emb_flat = self.pos_emb.reshape(1, -1, self.d_model)

        return enc + pos_emb_flat


    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape

        if mask_ratio <= 0.0:
            # no masking, no shuffling
            mask = torch.zeros((N, L), device=x.device)
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)
            return x, mask, ids_restore

        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones((N, L), device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward(
        self,
        multispectral,
        modalities=None,                 
        geotransform=None,
        mask_ratio_ms=0.0,
        mask_ratio_mods=0.0,
        return_intermediate=False,
        output_layers=None,
    ):
        B, _, H, W = multispectral.shape
        device = multispectral.device

        # geotransform safety
        if geotransform is not None:
            assert geotransform.shape[0] >= B, "geotransform batch size must be >= multispectral batch size."
            geotransform = geotransform[:B]

        # Decide whether modalities is actually present/used
        use_modalities = (modalities is not None) and (mask_ratio_mods < 1.0)

        # Patch embed MS always
        patches_multi = self.patch_embed_multi(multispectral)  # (B, N, D)
        N = patches_multi.shape[1]

        # Patch embed modalities only if used
        if use_modalities:
            assert modalities.shape[0] == B and modalities.shape[2] == H and modalities.shape[3] == W, \
                "modalities must match multispectral batch size and spatial dimensions."
            patches_modalities = self.patch_embed_modalities(modalities)          # (B, N, D)
            assert patches_modalities.shape[1] == N, \
                f"Mismatch in number of patches: {patches_modalities.shape[1]} != {N}"
        else:
            patches_modalities = None

        # Positional embedding (shared)
        if geotransform is not None and self.pos_embed_type == 'geo':
            normalized_centroids = self.get_normalized_centroids(geotransform, device)
            pos_embedding = self.positional_encoding(normalized_centroids)  # (B, N, D) or (B, ?, D) depending on your impl
        else:
            pos_embedding = self.pos_emb.reshape(1, -1, self.d_model)        # (1, N, D)

        assert patches_multi.shape[-1] == pos_embedding.shape[-1], \
            f"Patches feature dimension ({patches_multi.shape[-1]}) and positional embedding dimension ({pos_embedding.shape[-1]}) must match."

        # Add pos
        patches_multi = patches_multi + pos_embedding
        if use_modalities:
            patches_modalities = patches_modalities + pos_embedding

        # Masking
        patches_multi, mask_multi, ids_restore_multi = self.random_masking(patches_multi, mask_ratio_ms)

        if use_modalities:
            patches_modalities, mask_modalities, ids_restore_modalities = self.random_masking(patches_modalities, mask_ratio_mods)
            x = torch.cat((patches_multi, patches_modalities), dim=1)
            n_ms = patches_multi.shape[1]
            n_modalities = patches_modalities.shape[1]
        else:
            mask_modalities = None
            ids_restore_modalities = None
            x = patches_multi
            n_ms = patches_multi.shape[1]
            n_modalities = 0

        # Which layers to collect
        if output_layers is None:
            layers_to_collect = {
                self.depth // 4,
                self.depth // 2,
                3 * self.depth // 4,
                self.depth - 1,
            }
        else:
            layers_to_collect = set(output_layers)

        intermediate_features = []
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if return_intermediate and (i in layers_to_collect):
                if i == len(self.transformer_blocks) - 1:
                    intermediate_features.append(self.norm(x).clone())
                else:
                    intermediate_features.append(x.clone())
        x = self.norm(x)

        ##########Debug block
        # import torch.nn.functional as F
        # for i in range(3):
        #     a = intermediate_features[i].flatten(1)
        #     b = intermediate_features[i+1].flatten(1)
        #     print(i, "cos", F.cosine_similarity(a, b, dim=1).mean().item(),
        #             "l2", (a-b).pow(2).mean().item())

        # if mask_ratio_ms == 0:
        #     N = ids_restore_multi.shape[1]
        #     identity = (ids_restore_multi == torch.arange(N, device=ids_restore_multi.device)).all()
        #     print("ids_restore identity:", bool(identity))

        #######################

        if return_intermediate:
            return {
                'encoder_tokens': x,
                'intermediate_features': intermediate_features,
                'mask_multi': mask_multi,
                'mask_modalities': mask_modalities,
                'ids_restore_multi': ids_restore_multi,
                'ids_restore_modalities': ids_restore_modalities,
                'n_ms_tokens': n_ms,         # <-- critical for downstream slicing
                'n_modalities_tokens': n_modalities,
                'use_modalities': use_modalities,
            }
        else:
            return {
                'encoder_tokens': x,
                'mask_multi': mask_multi,
                'mask_modalities': mask_modalities,
                'ids_restore_multi': ids_restore_multi,
                'ids_restore_modalities': ids_restore_modalities,
                'n_ms_tokens': n_ms,         
                'n_modalities_tokens': n_modalities,
                'use_modalities': use_modalities,
            }


class FLORO_Wrapper(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str,
        d_model: int = 1280,
        patch_size: int = 16,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        pos_embed_type: str = "absolute",
        mask_ratio_ms: float = 0.0,
        mask_ratio_mods: float = 0.0,
        drop_max: float = 0.2,
        band_map: dict[str, str | None] | None = None,
        band_map_mod: dict[str, str | None] | None = None,
        use_dem: bool = False,
        use_sar: bool = False,
        nodata_sentinels: tuple[float, ...] = (-32768.0, -32670.0, -9999.0, -3.4028234663852886e+38),
        input_scale: str = "floro_norm",
        ds_stats: dict | None = None,
        floro_stats: dict | None = None,
        z_clip: float = 5.0,
        dataset_cfg: Optional[dict] = None,
    ):
        output_layers = (
            [output_layers] if isinstance(output_layers, int) else list(output_layers)
        )
        super().__init__(
            model_name="floro_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=d_model,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        if "optical" not in input_bands:
            raise ValueError("FLOROWrapper expects 'optical' in input_bands.")

        self.patch_size = patch_size
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.mask_ratio_ms = mask_ratio_ms
        self.mask_ratio_mods = mask_ratio_mods
        self.use_dem = use_dem
        self.use_sar = use_sar
        self.nodata_sentinels = tuple(float(v) for v in nodata_sentinels)

        # Dataset handling
        self.input_scale = InputScale(input_scale)
        self.ds_stats = ds_stats
        self.floro_stats = floro_stats
        self.z_clip = float(z_clip)
        self.dataset_cfg = dataset_cfg

        if self.input_scale == InputScale.DATASET_RAW:
            self.ds_stats = self._build_ds_stats_from_dataset_cfg()
            if self.floro_stats is None:
                raise ValueError("floro_stats must be provided for input_scale='dataset_raw'.")

        self.band_map = band_map or {
            "B": "B2",
            "G": "B3",
            "R": "B4",
            "RE": "B5",
            "NIR": "B8",
            "NIR2": "B8A",
            "SWIR1": "B11",
            "SWIR2": "B12",
        }

        # Map optical band names to channel indices in image["optical"]
        self.optical_band_to_index = {
            band: idx for idx, band in enumerate(input_bands["optical"])
        }
        # Map sar band names to channel indices in image["sar"]
        self.band_map_mod = band_map_mod or {
            "E": "DEM",
            "VV": "VV",
            "VH": "VH"
        }
        self.sar_band_to_index = {
            band: idx for idx, band in enumerate(input_bands.get("sar", []))
        }
    
        # Packed FLORO layouts are fixed
        self.multispectral_channels = 13
        self.modalities_channels = 5

        self.encoder = FLOROGeoEncoder(
            image_size=input_size,
            patch_size=patch_size,
            multispectral_channels=self.multispectral_channels,
            modalities_channels=self.modalities_channels,
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pos_embed_type=pos_embed_type,
            norm_layer=nn.LayerNorm,
            drop_max=drop_max,
        )

    def _get_optical_band(self, optical: torch.Tensor, floro_band: str) -> torch.Tensor | None:
        source_name = self.band_map.get(floro_band)
        if source_name is None:
            return None
        if source_name not in self.optical_band_to_index:
            return None
        idx = self.optical_band_to_index[source_name]
        return optical[:, idx:idx+1]

    def _get_mod_band(self, tensor: torch.Tensor, floro_mod_band: str, kind: str) -> torch.Tensor | None:
        """
        kind: "sar" or "dem"
        floro_mod_band: "E" / "VV" / "VH"
        """
        if kind == "sar":
            src_name = self.band_map_mod.get(floro_mod_band)  # e.g., "VV"
            if src_name is None:
                return None
            if src_name not in self.sar_band_to_index:
                return None
            idx = self.sar_band_to_index[src_name]
            return tensor[:, idx:idx+1]

        if kind == "dem":
            # usually dem is single-band; band_map_mod["E"] may be None meaning "take dem as-is"
            if tensor is None:
                return None
            # If you ever have named DEM bands, you can add a dem_band_to_index like SAR
            return tensor[:, :1]

        raise ValueError(f"Unknown kind={kind}")

    
    def _clean_with_range_torch(
        self,
        x: torch.Tensor,
        lo: float,
        hi: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(x)
        inrange = (x >= lo) & (x <= hi)
        valid_c = finite & inrange

        for nd in self.nodata_sentinels:
            valid_c = valid_c & (x != nd)

        valid_hw = valid_c.all(dim=1).float()  # (B,H,W)

        x = x.clone()
        x[~valid_c] = 0.0
        return x, valid_hw
    
    def _clean_finite_only_torch(self, x):
        finite = torch.isfinite(x)
        valid_c = finite
        for nd in self.nodata_sentinels:
            valid_c = valid_c & (x != nd)
        valid_hw = valid_c.all(dim=1).float()
        x = x.clone()
        x[~valid_c] = 0.0
        return x, valid_hw

    def _build_ds_stats_from_dataset_cfg(self) -> dict:
        """
        Converts dataset_cfg fields like:
        bands.optical, data_mean.optical, data_std.optical, data_min.optical, data_max.optical
        into dict-of-dicts keyed by band name.
        """
        if self.dataset_cfg is None:
            raise ValueError("dataset_cfg is required for input_scale='dataset_raw'.")

        cfg = self.dataset_cfg

        def make_stats(modality: str):
            band_list = list(cfg["bands"][modality]) if isinstance(cfg, dict) else list(getattr(cfg.bands, modality))
            mean_list = list(cfg["data_mean"][modality]) if isinstance(cfg, dict) else list(getattr(cfg.data_mean, modality))
            std_list  = list(cfg["data_std"][modality])  if isinstance(cfg, dict) else list(getattr(cfg.data_std, modality))
            min_list  = list(cfg["data_min"][modality])  if isinstance(cfg, dict) else list(getattr(cfg.data_min, modality))
            max_list  = list(cfg["data_max"][modality])  if isinstance(cfg, dict) else list(getattr(cfg.data_max, modality))

            if not (len(band_list) == len(mean_list) == len(std_list) == len(min_list) == len(max_list)):
                raise ValueError(f"Stats length mismatch for {modality}: "
                                f"bands={len(band_list)} mean={len(mean_list)} std={len(std_list)} "
                                f"min={len(min_list)} max={len(max_list)}")

            return {
                "mean": dict(zip(band_list, mean_list)),
                "std":  dict(zip(band_list, std_list)),
                "min":  dict(zip(band_list, min_list)),
                "max":  dict(zip(band_list, max_list)),
            }

        stats = {}
        if hasattr(cfg, "bands") or (isinstance(cfg, dict) and "bands" in cfg):
            if (isinstance(cfg, dict) and "optical" in cfg["bands"]) or (hasattr(cfg.bands, "optical")):
                stats["optical"] = make_stats("optical")
            if (isinstance(cfg, dict) and "sar" in cfg["bands"]) or (hasattr(cfg.bands, "sar")):
                stats["sar"] = make_stats("sar")

        # DEM stats typically don't exist in these yaml files; handle if present
        if (isinstance(cfg, dict) and "dem" in cfg.get("bands", {})) or (hasattr(cfg, "bands") and hasattr(cfg.bands, "dem")):
            stats["dem"] = make_stats("dem")

        return stats

    def _clean_and_map_ds_to_floro_torch(
        self,
        x: torch.Tensor,              # (B,C,H,W)
        band_names: list[str],        # source band names aligned with x channels
        modality: str,                # "optical" or "sar" or "dem"
        floro_modality: str,          # "optical" or "mods"
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Clean using dataset min/max + nodata sentinels, then map dataset distribution -> FLORO distribution.

        Mapping:
            z = (x - mu_ds) / (std_ds + eps)
            z = clip(z, -self.z_clip, self.z_clip)
            x_out = mu_floro + std_floro * z
        """
        if self.ds_stats is None or self.floro_stats is None:
            raise ValueError("input_is_unnormalized=True requires ds_stats and floro_stats to be provided.")

        device = x.device
        # --- build per-channel tensors from dict stats ---
        ds = self.ds_stats[modality]
        mu_ds = torch.tensor([ds["mean"][b] for b in band_names], device=device, dtype=torch.float32)
        sd_ds = torch.tensor([ds["std"][b]  for b in band_names], device=device, dtype=torch.float32)
        lo    = torch.tensor([ds["min"][b]  for b in band_names], device=device, dtype=torch.float32)
        hi    = torch.tensor([ds["max"][b]  for b in band_names], device=device, dtype=torch.float32)

        fl = self.floro_stats[floro_modality]
        mu_f = torch.tensor([fl["mean"][b] for b in band_names], device=device, dtype=torch.float32)
        sd_f = torch.tensor([fl["std"][b]  for b in band_names], device=device, dtype=torch.float32)

        # reshape for broadcasting
        mu_ds = mu_ds.view(1, -1, 1, 1)
        sd_ds = sd_ds.view(1, -1, 1, 1)
        lo    = lo.view(1, -1, 1, 1)
        hi    = hi.view(1, -1, 1, 1)
        mu_f  = mu_f.view(1, -1, 1, 1)
        sd_f  = sd_f.view(1, -1, 1, 1)

        finite  = torch.isfinite(x)
        inrange = (x >= lo) & (x <= hi)
        valid_c = finite & inrange
        for nd in self.nodata_sentinels:
            valid_c = valid_c & (x != nd)

        valid_hw = valid_c.all(dim=1).float()  # (B,H,W)

        x0 = x.clone()
        x0[~valid_c] = 0.0

        z = (x0 - mu_ds) / (sd_ds + eps)
        z = torch.clamp(z, -self.z_clip, self.z_clip)

        x_out = mu_f + sd_f * z
        x_out[~valid_c] = 0.0
        return x_out, valid_hw

    def _clean_and_normalize(
        self,
        x: torch.Tensor,
        *,
        band_names: list[str] | None = None,
        modality: str | None = None,        # "optical" / "sar" / "dem"
        floro_modality: str | None = None,  # "optical" / "mods"
        lo: float | None = None,
        hi: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x_cleaned_or_mapped, valid_hw).
        """
        if self.input_scale == InputScale.FLORO_NORM:
            return self._clean_finite_only_torch(x)

        if self.input_scale == InputScale.UNIT_01:
            assert lo is not None and hi is not None
            return self._clean_with_range_torch(x, lo, hi)

        # DATASET_RAW
        assert band_names is not None and modality is not None and floro_modality is not None
        return self._clean_and_map_ds_to_floro_torch(
            x, band_names=band_names, modality=modality, floro_modality=floro_modality
        )

    def _pack_optical(self, image: dict[str, torch.Tensor]) -> torch.Tensor:
        optical = self._to_4d(image["optical"])
        b, _, h, w = optical.shape
        packed = torch.zeros((b, 13, h, w), device=optical.device, dtype=optical.dtype)

        # BGR
        B = self._get_optical_band(optical, "B")
        G = self._get_optical_band(optical, "G")
        R = self._get_optical_band(optical, "R")
        if B is not None and G is not None and R is not None:
            BGR = torch.cat([B, G, R], dim=1)
            band_names = [self.band_map["B"], self.band_map["G"], self.band_map["R"]]
            BGR, mBGR = self._clean_and_normalize(
                BGR,
                band_names=band_names,
                modality="optical",
                floro_modality="optical",
                lo=0.0,
                hi=1.0,
            )
            packed[:, 0:3] = BGR
            packed[:, 8] = mBGR

        # RE
        RE = self._get_optical_band(optical, "RE")
        if RE is not None:
            band_names = [self.band_map["RE"]]
            RE, mRE = self._clean_and_normalize(
                RE,
                band_names=band_names,
                modality="optical",
                floro_modality="optical",
                lo=0.0,
                hi=1.0,
            )
            packed[:, 3:4] = RE
            packed[:, 9] = mRE

        # NIR
        NIR = self._get_optical_band(optical, "NIR")
        if NIR is not None:
            band_names = [self.band_map["NIR"]]
            NIR, mNIR = self._clean_and_normalize(
                NIR,
                band_names=band_names,
                modality="optical",
                floro_modality="optical",
                lo=0.0,
                hi=1.0,
            )
            packed[:, 4:5] = NIR
            packed[:, 10] = mNIR

        # NIR2
        NIR2 = self._get_optical_band(optical, "NIR2")
        if NIR2 is not None:
            band_names = [self.band_map["NIR2"]]
            NIR2, mNIR2 = self._clean_and_normalize(
                NIR2,
                band_names=band_names,
                modality="optical",
                floro_modality="optical",
                lo=0.0,
                hi=1.0,
            )
            packed[:, 5:6] = NIR2
            packed[:, 11] = mNIR2

        # SWIR
        SWIR1 = self._get_optical_band(optical, "SWIR1")
        SWIR2 = self._get_optical_band(optical, "SWIR2")
        swir_list = [x for x in [SWIR1, SWIR2] if x is not None]
        if len(swir_list) > 0:
            SWIR = torch.cat(swir_list, dim=1)
            band_names = [self.band_map["SWIR1"], self.band_map["SWIR2"]]
            SWIR, mSWIR = self._clean_and_normalize(
                SWIR,
                band_names=band_names,
                modality="optical",
                floro_modality="optical",
                lo=0.0,
                hi=1.0,
            )
            if SWIR.shape[1] == 1:
                packed[:, 6:7] = SWIR
            else:
                packed[:, 6:8] = SWIR[:, :2]
            packed[:, 12] = mSWIR

        return packed

    def _pack_modalities(self, image: dict[str, torch.Tensor]) -> torch.Tensor | None:
        dem = self._to_4d(image["dem"]) if (self.use_dem and "dem" in image) else None
        sar = self._to_4d(image["sar"]) if (self.use_sar and "sar" in image) else None

        if dem is None and sar is None:
            return None

        ref = dem if dem is not None else sar
        b, _, h, w = ref.shape
        packed = torch.zeros((b, 5, h, w), device=ref.device, dtype=ref.dtype)

        # ---- DEM -> E channel ----
        if dem is not None:
            E = self._get_mod_band(dem, "E", kind="dem")  # (B,1,H,W)
            if E is not None:
                E, mE = self._clean_and_normalize(
                    E,
                    band_names=["E"],
                    modality="dem",
                    floro_modality="mods",
                    lo=-500.0,
                    hi=9000.0,
                )
                packed[:, 0:1] = E
                packed[:, 3] = mE

        # ---- SAR -> VV/VH ----
        if sar is not None:
            VV = self._get_mod_band(sar, "VV", kind="sar")
            VH = self._get_mod_band(sar, "VH", kind="sar")

            if VV is not None and VH is not None:
                S = torch.cat([VV, VH], dim=1)  # (B,2,H,W) in correct order regardless of dataset ordering

                S, mS = self._clean_and_normalize(
                    S,
                    band_names=["VV", "VH"],
                    modality="sar",
                    floro_modality="mods",
                    lo=-60.0,
                    hi=20.0,
                )
                packed[:, 1:3] = S
                packed[:, 4] = mS
            else:
                # If only one polarization is available -> Current behavior: ignore SAR entirely 
                pass

        return packed

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x
        if x.ndim == 5:
            if x.shape[2] == 1:
                return x.squeeze(2)
            return x.mean(dim=2)
        raise ValueError(f"Expected tensor with 4 or 5 dims, got shape {tuple(x.shape)}.")

    def _tokens_to_feature_map(self, tokens: torch.Tensor, modality_present: bool=False) -> torch.Tensor:
        b, l, c = tokens.shape
        #print(f"Num tokens out: {l}")

        #modality_present = self.mask_ratio_mods < 1.0

        # Two streams are concatenated in token space; fuse them back to one grid.
        if l == 2 * self.num_patches:
            ms = tokens[:, : self.num_patches]
            dem = tokens[:, self.num_patches :]

            if not modality_present:          # DEM fully absent
                tokens = ms              # DO NOT average
            else:
                tokens = 0.5 * (ms + dem)
        elif l >= self.num_patches:
            tokens = tokens[:, : self.num_patches]
        else:
            raise ValueError(
                f"Token length {l} is smaller than required spatial patches {self.num_patches}."
            )

        return (
            tokens.transpose(1, 2)
            .reshape(b, c, self.grid_size, self.grid_size)
            .contiguous()
        )

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        multispectral = self._pack_optical(image)
        modalities = self._pack_modalities(image)

        modality_present = modalities is not None

        output = self.encoder(
            multispectral=multispectral,
            modalities=modalities,
            geotransform=None,
            mask_ratio_ms=self.mask_ratio_ms,
            mask_ratio_mods=self.mask_ratio_mods,
            return_intermediate=True,
            output_layers=self.output_layers,
        )
        features = output["intermediate_features"]
        if len(features) == 0:
            features = [output["encoder_tokens"]]
        return [self._tokens_to_feature_map(tokens, modality_present) for tokens in features]

    def load_encoder_weights(self, logger: Logger) -> None:
        if self.encoder_weights is None:
            return

        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        if isinstance(pretrained_model, dict):
            if "model_state_dict" in pretrained_model:
                pretrained_model = pretrained_model["model_state_dict"]
            elif "encoder_state_dict" in pretrained_model:
                pretrained_model = pretrained_model["encoder_state_dict"]

        cleaned_pretrained = {}
        for k, v in pretrained_model.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("encoder."):
                nk = nk[len("encoder.") :]
            cleaned_pretrained[nk] = v

        current_state = self.encoder.state_dict()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}

        for name, param in current_state.items():
            if name not in cleaned_pretrained:
                missing[name] = param.shape
            elif cleaned_pretrained[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, cleaned_pretrained[name].shape)
            else:
                pretrained_encoder[name] = cleaned_pretrained[name]

        self.encoder.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
