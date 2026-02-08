import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import math

# Local imports
from src.models.Model_Components import  SelfAttentionBlockViT, trunc_normal_

# Helper function to build 2D sine-cosine positional embeddings
def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    grid_h = torch.arange(h, dtype=torch.float32)  # height (rows, i)
    grid_w = torch.arange(w, dtype=torch.float32)  # width (cols, j)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])  # width direction → x
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])  # height direction → y
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    # In the encoder we kept the following reshape and then performed it in the forward
    # here, we simplified the process and just keep [B, (h w), embedding_dim]
    #pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w)  # Change the arrangement to NCHW
    return pos_emb

class DualModalityViTClassDecoder(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=4,
        num_classes=4,
        d_model=256,
        mlp_ratio=4,
        dec_d_model=512,
        dec_depth=8,
        dec_num_heads=16,
        pos_embed_type = "geo",
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_classes = num_classes
        self.dec_d_model = dec_d_model
        self.pos_embed_type = pos_embed_type

        # Linear projection from encoder to decoder dimension
        self.linear_proj = nn.Linear(d_model, dec_d_model)   
        self.linear_proj_combined = nn.Linear(dec_d_model*2, dec_d_model)     

        # Initialize positional embeddings
        self.h_posemb = self.image_size // patch_size
        self.w_posemb = self.image_size // patch_size
        if self.pos_embed_type in ['geo', 'absolute']:
            posemb = build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=dec_d_model)
            self.register_buffer("pos_emb", posemb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, self.dec_d_model))
            trunc_normal_(self.pos_emb, std=0.02)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))

        self.combined_decoder_blocks = nn.ModuleList([
            SelfAttentionBlockViT(dec_d_model, dec_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(dec_depth)])
        
        self.norm = norm_layer(dec_d_model)

        # MLP for classification head
        self.mlp = nn.Sequential(
        nn.LayerNorm(dec_d_model),
            nn.Linear(dec_d_model, dec_d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dec_d_model // 2, num_classes)
        )           
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # A gamma parameter to let the model learn how much to retain from the original features
        self.gamma = nn.Parameter(torch.ones(1))
   

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
        half_dim = self.dec_d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, 2, device=coords.device) * -(math.log(10000.0) / half_dim)
        )

        enc[..., 0::4] = torch.sin(coords[..., 0].unsqueeze(-1) * div_term)  # global_x
        enc[..., 1::4] = torch.cos(coords[..., 0].unsqueeze(-1) * div_term)
        enc[..., 2::4] = torch.sin(coords[..., 1].unsqueeze(-1) * div_term)  # global_y
        enc[..., 3::4] = torch.cos(coords[..., 1].unsqueeze(-1) * div_term)


        # Add static absolute 2D positional embedding (flattened)
        pos_emb_flat = self.pos_emb.reshape(1, -1, self.dec_d_model)

        return enc + pos_emb_flat


    def forward(self, encoder_tokens, mask_multi, mask_srtm, ids_restore_multi, ids_restore_srtm, geotransform):
        B = encoder_tokens.size(0)
        device = encoder_tokens.device
        #print(encoder_tokens.size())
        expected_tokens = (self.image_size // self.patch_size)**2
        #print(expected_tokens)
        
        # Linear projection decoder emb. dim.
        context_tokens = self.linear_proj(encoder_tokens)
        #print(context_tokens.size())

        # Apply the same positional embeddings to both
        if geotransform is not None and self.pos_embed_type == 'geo':
            # Calculate normalized centroids for the entire batch
            normalized_centroids = self.get_normalized_centroids(geotransform, device)
            # Generate positional embeddings for the entire batch
            pos_embedding = self.positional_encoding(normalized_centroids)
        else:
            pos_embedding = self.pos_emb
        
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
        
        # SRTM decoding
        B, N, D = srtm_x.shape
        srtm_mask_tokens = self.mask_token.repeat(B, ids_restore_srtm.shape[1] - N, 1)
        srtm_x_ = torch.cat([srtm_x, srtm_mask_tokens], dim=1)
        ids_restore_srtm_expanded = ids_restore_srtm.unsqueeze(-1).repeat(1, 1, D)
        srtm_x = torch.gather(srtm_x_, dim=1, index=ids_restore_srtm_expanded)
        

        # Combine fused features and pass to classification MLP
        x = torch.cat((ms_x, srtm_x), dim=-1)
        x = self.linear_proj_combined(x)
        x = x + pos_embedding

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x_residual = x.clone()
        for cblk in self.combined_decoder_blocks:
            x = cblk(x)
        
        x = x + (self.gamma * x_residual)
        x = self.norm(x)

        output = self.mlp(x[:, 0])   

        return output
