import torch
import torch.nn as nn
import math

from floro.models.model_components import SelfAttentionBlockViT, PatchEmbed, build_2d_sincos_posemb, trunc_normal_

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
        image_size=256, 
        patch_size=16, 
        multispectral_channels=4,  # Multispectral channels (R, G, B, NIR)
        modalities_channels=1,            # modalities channels
        d_model=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        pos_embed_type="geo",
        norm_layer=nn.LayerNorm,
        drop_max = 0.2  # starting point 0.2 (tune 0.1–0.3?)
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.depth = depth
        self.dpr = torch.linspace(0, drop_max, steps=depth).tolist()
        self.pos_embed_type = pos_embed_type
        assert self.d_model % 4 == 0, "d_model must be divisible by 4 for positional encoding."
        
        self.patch_embed_multi = PatchEmbed(image_size, patch_size, multispectral_channels, d_model)
        self.patch_embed_modalities = PatchEmbed(image_size, patch_size, modalities_channels, d_model)
        
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

        # Patch embeds
        init_patch_embed_as_linear(self.patch_embed_multi)
        init_patch_embed_as_linear(self.patch_embed_modalities)   

        self.transformer_blocks = nn.ModuleList([
            SelfAttentionBlockViT(
                d_model, num_heads, mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=0.0,        
                attn_drop=0.0,        
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
        modalities=None,                 # optional
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
                intermediate_features.append(self.norm(x).clone())  # clone to avoid aliasing

        x = self.norm(x)

        if return_intermediate:
            return {
                'encoder_tokens': x,
                'intermediate_features': intermediate_features,
                'mask_multi': mask_multi,
                'mask_modalities': mask_modalities,
                'ids_restore_multi': ids_restore_multi,
                'ids_restore_modalities': ids_restore_modalities,
                'n_ms_tokens': n_ms,        
                'n_modalities_tokens': n_modalities,
                'use_modalities': use_modalities,
            }
        else:
            #return x, mask_multi, mask_modalities, ids_restore_multi, ids_restore_modalities
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
