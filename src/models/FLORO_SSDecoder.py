import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from src.models.Model_Components import Attention, CrossAttention, trunc_normal_

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
    #pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w)  # Change the arrangement to NCHW
    return pos_emb

class SatMultiMAEDecoder(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        ms_channels=4,  # Multispectral channels (R, G, B, NIR)
        srtm_channels=1,  # SRTM channels
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
        self.ms_channels = ms_channels
        self.srtm_channels = srtm_channels
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

        # Output projections for multispectral and SRTM data
        self.ms_mlp = nn.Sequential(
            nn.LayerNorm(dec_d_model),
            nn.Linear(dec_d_model, patch_size ** 2 * ms_channels)
        )
        self.srtm_mlp = nn.Sequential(
            nn.LayerNorm(dec_d_model),
            nn.Linear(dec_d_model, patch_size ** 2 * srtm_channels)
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
        i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing="ij")  # [H, W]

        # Absolute coordinates (EPSG:3857 meters)
        origin_x = geotransform[:, 0]  # Easting
        origin_y = geotransform[:, 3]  # Northing
        pixel_width = geotransform[:, 1]
        pixel_height = geotransform[:, 5]

        center_x_abs = origin_x.view(-1, 1, 1) + (i_grid * self.patch_size + self.patch_size / 2) * pixel_width.view(-1, 1, 1)
        center_y_abs = origin_y.view(-1, 1, 1) + (j_grid * self.patch_size + self.patch_size / 2) * pixel_height.view(-1, 1, 1)

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

    def forward(self, encoder_tokens, mask_multi, mask_srtm, ids_restore_multi, ids_restore_srtm, geotransform):
        B = encoder_tokens.size(0)
        device = encoder_tokens.device
        #print(encoder_tokens.size())
        #expected_tokens = (self.image_size // self.patch_size)**2
        #print(expected_tokens)
        
        
        # Linear projection
        context_tokens = self.linear_proj(encoder_tokens)
        #print(context_tokens.size())
        
        # Number of patches (fixed, based on image and patch size)
        #num_patches = mask_multi.shape[1]
        #print(num_patches)

        # Apply the same positional embeddings to both
        if geotransform is not None and self.pos_embed_type == 'geo':
            # Calculate normalized centroids for the entire batch
            normalized_centroids = self.get_normalized_centroids(geotransform, device)
            # Generate positional embeddings for the entire batch
            pos_embedding = self.positional_encoding(normalized_centroids)
        
        # Number of visible patches for MS and SRTM
        num_visible_patches_multi = (mask_multi != 1).sum(dim=1).max().item()
        num_visible_patches_srtm = (mask_srtm != 1).sum(dim=1).max().item()
        assert num_visible_patches_multi + num_visible_patches_srtm == encoder_tokens.size(1), f"Separated number of tokens {num_visible_patches_multi + num_visible_patches_srtm} and total number of tokens {encoder_tokens.size(1)} do not coincide." 
        
        # Separate the encoder inputs for MS and SRTM
        ms_x = context_tokens[:, :num_visible_patches_multi]
        #print(ms_x.size())
        srtm_x = context_tokens[:, num_visible_patches_multi:num_visible_patches_multi + num_visible_patches_srtm]
        #print(srtm_x.size())
        
        # Multispectral decoding
        B, N, D = ms_x.shape
        ms_mask_tokens = self.mask_token.repeat(B, ids_restore_multi.shape[1] - N, 1)
        ms_x_ = torch.cat([ms_x, ms_mask_tokens], dim=1)
        ids_restore_multi_expanded = ids_restore_multi.unsqueeze(-1).repeat(1, 1, D)
        ms_x = torch.gather(ms_x_, dim=1, index=ids_restore_multi_expanded)
        
        # Add positional and modality embeddings
        #ms_x = ms_x + self.generate_context_embeddings(B, ms_x.device)
        if geotransform is not None and self.pos_embed_type == 'geo':
            ms_x = ms_x + pos_embedding
        else:
            ms_x = ms_x + self.pos_emb
        
        # Get queries and context
        queries_ms, context_tokens_ms = self.get_queries_and_context(ms_x, ids_restore_multi)
        
        for blk in self.decoder_blocks:
            ms_x = blk(queries_ms, context_tokens_ms)

        ms_x = self.norm(ms_x)
        ms_x = self.ms_mlp(ms_x)
        reconstructed_multispectral = self.reconstruct_image(ms_x, num_channels=self.ms_channels)

        # SRTM decoding
        B, N, D = srtm_x.shape
        srtm_mask_tokens = self.mask_token.repeat(B, ids_restore_srtm.shape[1] - N, 1)
        srtm_x_ = torch.cat([srtm_x, srtm_mask_tokens], dim=1)
        ids_restore_srtm_expanded = ids_restore_srtm.unsqueeze(-1).repeat(1, 1, D)
        srtm_x = torch.gather(srtm_x_, dim=1, index=ids_restore_srtm_expanded)
        
        # Add positional and modality embeddings
        #srtm_x = srtm_x + self.generate_context_embeddings(B, srtm_x.device)
        if geotransform is not None and self.pos_embed_type == 'geo':
            srtm_x = srtm_x + pos_embedding
        else:
            srtm_x = srtm_x + self.pos_emb
        
        # Get queries and context
        queries_srtm, context_tokens_srtm = self.get_queries_and_context(srtm_x, ids_restore_srtm)

        for blk in self.decoder_blocks:
            srtm_x = blk(queries_srtm, context_tokens_srtm)

        srtm_x = self.norm(srtm_x)
        srtm_x = self.srtm_mlp(srtm_x)
        reconstructed_srtm = self.reconstruct_image(srtm_x, num_channels=self.srtm_channels)

        return reconstructed_multispectral, reconstructed_srtm
