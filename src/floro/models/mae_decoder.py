import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from floro.models.model_components import Attention, CrossAttention, trunc_normal_

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, context):
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        x = x + self.drop_path(self.cross_attn(self.query_norm(x), self.context_norm(context)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Helper function to build 2D sine-cosine positional embeddings
def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
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
    # In the encoder we kept the following reshape and then performed it in the forward
    # here, we simplified the process and just keep [B, (h w), embedding_dim]
    return pos_emb

def restore_tokens(x_vis: torch.Tensor,
                   ids_restore: torch.Tensor,
                   mask_token: torch.Tensor) -> torch.Tensor:
    """
    x_vis: (B, L_keep, D)
    ids_restore: (B, L_full)
    mask_token: (1, 1, D) or (B, 1, D) broadcastable
    returns: (B, L_full, D) restored in original order
    """
    B, L_keep, D = x_vis.shape
    L_full = ids_restore.shape[1]
    n_mask = L_full - L_keep
    if n_mask < 0:
        raise ValueError(f"L_keep ({L_keep}) > L_full ({L_full}).")

    if n_mask > 0:
        mask_tokens = mask_token.expand(B, n_mask, D)
        x_ = torch.cat([x_vis, mask_tokens], dim=1)  # (B, L_full, D) but in shuffled/keep+mask order
    else:
        x_ = x_vis

    index = ids_restore.unsqueeze(-1).expand(-1, -1, D)
    x_full = torch.gather(x_, dim=1, index=index)
    return x_full

class FLOROSSDecoder(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        out_ms_channels=7, # Multispectral channels
        out_mod_channels=3, # modalities channels
        d_model=1280,
        dec_d_model=512,
        num_heads=8,
        mlp_ratio=4,
        dec_depth=4,
        dec_num_heads=16,
        pos_embed_type = "geo",
        norm_layer=nn.LayerNorm,
        task=None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.task = task
        self.out_ms_channels = out_ms_channels
        self.out_mod_channels = out_mod_channels
        self.d_model = d_model
        self.dec_d_model = dec_d_model
        self.pos_embed_type = pos_embed_type

        # Linear projection from encoder to decoder dimension
        self.linear_proj = nn.Linear(d_model, dec_d_model)
        
        # Initialize positional embeddings
        self.h_posemb = self.image_size // patch_size
        self.w_posemb = self.image_size // patch_size
        if self.pos_embed_type == 'geo':
            #Geographic coords emb. will be added dinamically later
            self.pos_emb = nn.Parameter(build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=self.dec_d_model), requires_grad=False)
        elif self.pos_embed_type == 'absolute':
            self.pos_emb = nn.Parameter(build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=self.dec_d_model), requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.h_posemb, self.w_posemb, self.dec_d_model))
            trunc_normal_(self.pos_emb, std=0.02)
            
        self.modality_embed = nn.Parameter(torch.zeros(1, self.num_patches, dec_d_model))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(dim=dec_d_model, num_heads=dec_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for _ in range(dec_depth)
        ])
        self.norm = norm_layer(dec_d_model)

        # Output projections for multispectral and modalities data
        self.ms_mlp = nn.Sequential(
            nn.LayerNorm(dec_d_model),
            nn.Linear(dec_d_model, patch_size ** 2 * out_ms_channels)
        )
        self.modalities_mlp = nn.Sequential(
            nn.LayerNorm(dec_d_model),
            nn.Linear(dec_d_model, patch_size ** 2 * out_mod_channels)
        )

    def generate_context_embeddings(self, bs, device):
        context_embeddings = repeat(self.modality_embed, '() n d -> b n d', b=bs)
        # pos_emb = self.pos_embed.expand(bs, -1, -1, -1)  # Expand to batch size
        # pos_emb = F.interpolate(pos_emb, size=(self.image_size // self.patch_size, self.image_size // self.patch_size), mode='bilinear', align_corners=False)
        # pos_emb = rearrange(pos_emb, 'b d h w -> b (h w) d', h=self.image_size // self.patch_size, w=self.image_size // self.patch_size)
        #context_embeddings = context_embeddings + pos_emb
        return context_embeddings

    def get_queries_and_context(self, context_tokens, ids_restore):
        B = context_tokens.shape[0]
        H, W = self.image_size, self.image_size
        N_H = H // self.patch_size
        N_W = W // self.patch_size

        context_tokens_without_global = context_tokens

        mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=B, n=self.num_patches - context_tokens_without_global.shape[1])
        context_with_mask = torch.cat([context_tokens_without_global, mask_tokens], dim=1)

        context_with_mask = torch.gather(context_with_mask, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        context_emb = self.generate_context_embeddings(bs=B, device=context_tokens.device)
        context_with_mask = context_with_mask + context_emb
        
        queries = repeat(self.mask_token, '() () d -> b n d', b=B, n=N_H * N_W)
        queries = queries + self.pos_emb
        queries = queries + self.modality_embed

        context_tokens_without_global = torch.gather(context_with_mask, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))
        context_tokens = context_tokens_without_global

        return queries, context_tokens

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

        enc = torch.zeros((B, N, self.dec_d_model), device=coords.device)
        div_term = torch.exp(
            torch.arange(0, self.dec_d_model // 2, 2, device=coords.device) *
            -(math.log(10000.0) / (self.dec_d_model // 2))
        )

        enc[..., 0::4] = torch.sin(coords[..., 0].unsqueeze(-1) * div_term)  # global_x
        enc[..., 1::4] = torch.cos(coords[..., 0].unsqueeze(-1) * div_term)
        enc[..., 2::4] = torch.sin(coords[..., 1].unsqueeze(-1) * div_term)  # global_y
        enc[..., 3::4] = torch.cos(coords[..., 1].unsqueeze(-1) * div_term)

        # Add static absolute 2D positional embedding (flattened)
        pos_emb_flat = self.pos_emb.reshape(1, -1, self.dec_d_model)

        return enc + pos_emb_flat

    def reconstruct_image(self, patches, num_channels=1):
        patch_size = self.patch_size
        invert_patch = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                                 h=self.image_size // patch_size,
                                 w=self.image_size // patch_size,
                                 p1=patch_size,
                                 p2=patch_size,
                                 c=num_channels)
        return invert_patch

    def forward(self, encoder_out: dict, geotransform=None):
        x = encoder_out["encoder_tokens"]  # (B, n_ms + n_mods, D_enc) visible tokens concatenated
        # encoder_tokens should be (B, n_ms+n_mods, D_enc)
        assert x.dim() == 3, f"encoder_tokens must be 3D (B,L,D), got {x.shape}"
        B = x.size(0)
        device = x.device

        # project to decoder dim
        context_tokens = self.linear_proj(x)  # (B, n_total, D_dec)
        D = context_tokens.size(-1)

        # token counts (VISIBLE counts, exactly what we need for slicing)
        n_ms = int(encoder_out["n_ms_tokens"])
        n_mods = int(encoder_out["n_modalities_tokens"])
        use_mods = bool(encoder_out["use_modalities"])

        assert n_ms + n_mods == context_tokens.size(1), \
            f"n_ms+n_mods ({n_ms+n_mods}) != context_tokens len ({context_tokens.size(1)})"

        # split visible tokens exactly as encoder concatenated them
        ms_vis = context_tokens[:, :n_ms, :]
        mods_vis = context_tokens[:, n_ms:n_ms+n_mods, :] if use_mods else None

        # restore MS full grid
        ids_restore_ms = encoder_out["ids_restore_multi"]  # (B, L_ms_full)
        # ids_restore should be (B, L_ms_full)
        assert ids_restore_ms.dim() == 2 and ids_restore_ms.size(0) == B
        ms_full = restore_tokens(ms_vis, ids_restore_ms, self.mask_token)

        # restore other modalities full grid (if present)
        if use_mods:
            ids_restore_mods = encoder_out["ids_restore_modalities"]  # (B, L_mods_full)
            assert ids_restore_mods.dim() == 2 and ids_restore_mods.size(0) == B
            mods_full = restore_tokens(mods_vis, ids_restore_mods, self.mask_token)
        else:
            mods_full = None

        # positional embedding (must match full-grid length per stream)
        if geotransform is not None and self.pos_embed_type == "geo":
            normalized_centroids = self.get_normalized_centroids(geotransform, device)
            pos_embedding = self.positional_encoding(normalized_centroids)  # (B, L_full, D_dec) for MS grid
        else:
            pos_embedding = self.pos_emb  # (1, L_full, D_dec) or (B, L_full, D_dec)

        # add pos to each stream
        ms_full = ms_full + pos_embedding
        if mods_full is not None:
            # if modalities share the same grid length as MS, this is fine
            # otherwise you need a separate pos emb sized for that grid
            mods_full = mods_full + pos_embedding

        # build queries/context the way your decoder expects
        q_ms, ctx_ms = self.get_queries_and_context(ms_full, ids_restore_ms)

        x_ms = q_ms
        for blk in self.decoder_blocks:
            x_ms = blk(x_ms, ctx_ms)
        x_ms = self.norm(x_ms)
        x_ms = self.ms_mlp(x_ms)
        recon_ms = self.reconstruct_image(x_ms, num_channels=self.out_ms_channels)

        if mods_full is not None:
            q_mods, ctx_mods = self.get_queries_and_context(mods_full, ids_restore_mods)
            x_mods = q_mods
            for blk in self.decoder_blocks:
                x_mods = blk(x_mods, ctx_mods)
            x_mods = self.norm(x_mods)
            x_mods = self.modalities_mlp(x_mods)  # rename from modalities_mlp if multi-modal
            recon_mods = self.reconstruct_image(x_mods, num_channels=self.out_mod_channels)
        else:
            recon_mods = None

        return recon_ms, recon_mods

    # def forward(self, encoder_tokens, mask_multi, mask_modalities, ids_restore_multi, ids_restore_modalities, geotransform):
    #     B = encoder_tokens.size(0)
    #     device = encoder_tokens.device
    #     #print(encoder_tokens.size())
    #     #expected_tokens = (self.image_size // self.patch_size)**2
    #     #print(expected_tokens)
        
        
    #     # Linear projection
    #     context_tokens = self.linear_proj(encoder_tokens)
    #     #print(context_tokens.size())
        
    #     # Number of patches (fixed, based on image and patch size)
    #     #num_patches = mask_multi.shape[1]
    #     #print(num_patches)

    #     # Apply the same positional embeddings to both
    #     if geotransform is not None and self.pos_embed_type == 'geo':
    #         # Calculate normalized centroids for the entire batch
    #         normalized_centroids = self.get_normalized_centroids(geotransform, device)
    #         # Generate positional embeddings for the entire batch
    #         pos_embedding = self.positional_encoding(normalized_centroids)
        
    #     # Number of visible patches for MS and modalities
    #     num_visible_patches_multi = (mask_multi != 1).sum(dim=1).max().item()
    #     num_visible_patches_modalities = (mask_modalities != 1).sum(dim=1).max().item()
    #     assert num_visible_patches_multi + num_visible_patches_modalities == encoder_tokens.size(1), f"Separated number of tokens {num_visible_patches_multi + num_visible_patches_modalities} and total number of tokens {encoder_tokens.size(1)} do not coincide." 
        
    #     # Separate the encoder inputs for MS and modalities
    #     ms_x = context_tokens[:, :num_visible_patches_multi]
    #     #print(ms_x.size())
    #     modalities_x = context_tokens[:, num_visible_patches_multi:num_visible_patches_multi + num_visible_patches_modalities]
    #     #print(modalities_x.size())
        
    #     # Multispectral decoding
    #     B, N, D = ms_x.shape
    #     ms_mask_tokens = self.mask_token.repeat(B, ids_restore_multi.shape[1] - N, 1)
    #     ms_x_ = torch.cat([ms_x, ms_mask_tokens], dim=1)
    #     ids_restore_multi_expanded = ids_restore_multi.unsqueeze(-1).repeat(1, 1, D)
    #     ms_x = torch.gather(ms_x_, dim=1, index=ids_restore_multi_expanded)
        
    #     # Add positional and modality embeddings
    #     #ms_x = ms_x + self.generate_context_embeddings(B, ms_x.device)
    #     if geotransform is not None and self.pos_embed_type == 'geo':
    #         ms_x = ms_x + pos_embedding
    #     else:
    #         ms_x = ms_x + self.pos_emb
        
    #     # Get queries and context
    #     queries_ms, context_tokens_ms = self.get_queries_and_context(ms_x, ids_restore_multi)
        
    #     for blk in self.decoder_blocks:
    #         ms_x = blk(queries_ms, context_tokens_ms)

    #     ms_x = self.norm(ms_x)
    #     ms_x = self.ms_mlp(ms_x)
    #     reconstructed_multispectral = self.reconstruct_image(ms_x, num_channels=self.ms_channels)

    #     # modalities decoding
    #     B, N, D = modalities_x.shape
    #     modalities_mask_tokens = self.mask_token.repeat(B, ids_restore_modalities.shape[1] - N, 1)
    #     modalities_x_ = torch.cat([modalities_x, modalities_mask_tokens], dim=1)
    #     ids_restore_modalities_expanded = ids_restore_modalities.unsqueeze(-1).repeat(1, 1, D)
    #     modalities_x = torch.gather(modalities_x_, dim=1, index=ids_restore_modalities_expanded)
        
    #     # Add positional and modality embeddings
    #     #modalities_x = modalities_x + self.generate_context_embeddings(B, modalities_x.device)
    #     if geotransform is not None and self.pos_embed_type == 'geo':
    #         modalities_x = modalities_x + pos_embedding
    #     else:
    #         modalities_x = modalities_x + self.pos_emb
        
    #     # Get queries and context
    #     queries_modalities, context_tokens_modalities = self.get_queries_and_context(modalities_x, ids_restore_modalities)

    #     for blk in self.decoder_blocks:
    #         modalities_x = blk(queries_modalities, context_tokens_modalities)

    #     modalities_x = self.norm(modalities_x)
    #     modalities_x = self.modalities_mlp(modalities_x)
    #     reconstructed_modalities = self.reconstruct_image(modalities_x, num_channels=self.modalities_channels)

    #     return reconstructed_multispectral, reconstructed_modalities
