import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import math

from src.models.Model_Components import SelfAttentionBlockViT, PatchEmbed, build_2d_sincos_posemb, trunc_normal_

class MultiMAE_Encoder(nn.Module):
    def __init__(
        self,
        image_size=256, 
        patch_size=16, 
        multispectral_channels=4,  # Multispectral channels (R, G, B, NIR)
        srtm_channels=1,            # SRTM channels
        d_model=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        pos_embed_type="geo",
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.depth = depth
        self.pos_embed_type = pos_embed_type
        
        self.patch_embed_multi = PatchEmbed(image_size, patch_size, multispectral_channels, d_model)
        self.patch_embed_srtm = PatchEmbed(image_size, patch_size, srtm_channels, d_model)
        
        # Initialize positional embeddings
        self.h_posemb = self.w_posemb = self.image_size // patch_size
        if self.pos_embed_type == 'geo':
            #Geographic coords emb. will be added dinamically later
            self.pos_emb = nn.Parameter(build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=self.d_model), requires_grad=False)
        elif self.pos_embed_type == 'absolute':
            self.pos_emb = nn.Parameter(build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=self.d_model), requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.h_posemb, self.w_posemb, self.d_model))
            trunc_normal_(self.pos_emb, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            SelfAttentionBlockViT(d_model, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(d_model)

    # def get_normalized_centroids(self, geotransform, device):
    #     B = geotransform.shape[0]
    #     h_posemb, w_posemb = self.h_posemb, self.w_posemb

    #     # Calculate the grid of patch coordinates
    #     i_coords = torch.arange(h_posemb, device=device, dtype=torch.float32).view(1, -1, 1).repeat(B, 1, w_posemb)
    #     j_coords = torch.arange(w_posemb, device=device, dtype=torch.float32).view(1, 1, -1).repeat(B, h_posemb, 1)
        
    #     # Extract geotransform components
    #     geotransform = geotransform.view(B, 6)  # Ensure geotransform is 2D
    #     origin_x = geotransform[:, 0]
    #     pixel_width = geotransform[:, 1]
    #     origin_y = geotransform[:, 3]
    #     pixel_height = geotransform[:, 5]

    #     # Calculate the center coordinates
    #     center_x = origin_x.view(-1, 1, 1) + (i_coords * self.patch_size + self.patch_size / 2) * pixel_width.view(-1, 1, 1)
    #     center_y = origin_y.view(-1, 1, 1) + (j_coords * self.patch_size + self.patch_size / 2) * pixel_height.view(-1, 1, 1)

    #     # Normalize coordinates based on geographical boundaries
    #     min_x, max_x = -180.0, 180.0
    #     min_y, max_y = -90.0, 90.0

    #     norm_center_x = (center_x - min_x) / (max_x - min_x)
    #     norm_center_y = (center_y - min_y) / (max_y - min_y)

    #     # Flatten the coordinates
    #     coords = torch.stack([norm_center_x, norm_center_y], dim=-1).view(B, -1, 2)
    #     return coords
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
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, multispectral, srtm, geotransform, mask_ratio_ms, mask_ratio_elev, return_intermediate=False):
        B, _, H, W = multispectral.shape
        #print(B)
        #print(geotransform.shape[0])
        # Adjust geotransform tensor to match current batch size -> The last batch from the dataloader had 
        if geotransform is not None:
            geotransform = geotransform[:B]
            
        device = multispectral.device
        
        patches_multi = self.patch_embed_multi(multispectral)
        patches_srtm = self.patch_embed_srtm(srtm)

        # Ensure both patch embeddings have the correct number of patches
        assert patches_multi.shape[1] == patches_srtm.shape[1], \
            f"Mismatch in number of patches: {patches_multi.shape[1]} != {patches_srtm.shape[1]}"

        # Apply the same positional embeddings to both
        if geotransform is not None and self.pos_embed_type == 'geo':
            # Calculate normalized centroids for the entire batch
            normalized_centroids = self.get_normalized_centroids(geotransform, device)
    
            # Generate positional embeddings for the entire batch
            pos_embedding = self.positional_encoding(normalized_centroids)
            
        else:
            # Add static absolute 2D positional embedding (flattened)
            pos_embedding = self.pos_emb.reshape(1, -1, self.d_model)

        # Ensure patches and positional_embeddings have the same dimension
        assert patches_multi.shape[-1] == pos_embedding.shape[-1], \
            f"Patches feature dimension ({patches_multi.shape[-1]}) and positional embedding dimension ({pos_embedding.shape[-1]}) must match."
            
        patches_multi = patches_multi + pos_embedding
        patches_srtm = patches_srtm + pos_embedding
                
        # Masking
        patches_multi, mask_multi, ids_restore_multi = self.random_masking(patches_multi, mask_ratio_ms)
        patches_srtm, mask_srtm, ids_restore_srtm = self.random_masking(patches_srtm, mask_ratio_elev)

        # Combine patches
        x = torch.cat((patches_multi, patches_srtm), dim=1)

        intermediate_features = []  # Collect intermediate features
        # Apply Transformer blocks and collect intermediate features
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if return_intermediate:
                # Collect at 1/4, 1/2, 3/4, and final depth
                if i in [self.depth // 4, self.depth // 2, 3 * self.depth // 4, self.depth - 1]:
                    intermediate_features.append(x)
        x = self.norm(x)

        if return_intermediate:
            return {'encoder_tokens':x,
                    'intermediate_features':intermediate_features,
                    'mask_multi':mask_multi,
                    'mask_srtm':mask_srtm,
                    'ids_restore_multi':ids_restore_multi,
                    'ids_restore_srtm':ids_restore_srtm
                    }
        else:
            return x, mask_multi, mask_srtm, ids_restore_multi, ids_restore_srtm
