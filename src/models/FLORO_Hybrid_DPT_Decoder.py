import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from src.models.util.blocks import FeatureFusionBlock, _make_scratch
from src.models.Model_Components import SelfAttentionBlockViT, trunc_normal_

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

class FLORODPTDecoder(nn.Module):
    def __init__(self, 
                 image_size=256,
                 patch_size=16,
                 d_model=1024,
                 dec_d_model=768, 
                 dec_depth=4,
                 dec_num_heads=8,
                 mlp_ratio=4,
                 dpt_channels = [256, 512, 1024, 1024],
                 out_channels=1,
                 pos_embed_type='geo',
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dec_d_model = dec_d_model
        self.pos_embed_type = pos_embed_type
        self.d_model = d_model
        self.out_channels = out_channels
        self.dpt_channels = dpt_channels

        # Linear projection from encoder to decoder dimension
        self.linear_proj = nn.Linear(d_model, dec_d_model)
        self.linear_proj_int_features = nn.Linear(d_model*2, d_model)
        self.linear_proj_combined = nn.Linear(dec_d_model*2, dec_d_model) 
        
        # Initialize positional embeddings
        self.h_posemb = self.image_size // patch_size
        self.w_posemb = self.image_size // patch_size
        if self.pos_embed_type in ['geo', 'absolute']:
            posemb = build_2d_sincos_posemb(h=self.h_posemb, w=self.w_posemb, embed_dim=dec_d_model)
            self.register_buffer("pos_emb", posemb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, dec_d_model))
            trunc_normal_(self.pos_emb, std=0.02)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))
        trunc_normal_(self.mask_token, std=0.02)
        self.mask_token_feats = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token_feats, std=0.02)

        # Transformer path
        self.transformer_blocks = nn.ModuleList([
            SelfAttentionBlockViT(self.dec_d_model, num_heads=dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(dec_depth)
        ])
        self.norm = norm_layer(self.dec_d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dec_d_model),
            nn.Linear(dec_d_model, patch_size ** 2 * out_channels)
        )

        # A gamma parameter to let the model learn how much to retain from the original features
        self.gamma = nn.Parameter(torch.ones(1))

        self.feature_projs = nn.ModuleList([
            nn.Conv2d(
                in_channels=d_model,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in dpt_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=dpt_channels[0],
                out_channels=dpt_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=dpt_channels[1],
                out_channels=dpt_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=dpt_channels[3],
                out_channels=dpt_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
    
        
        # Scratch network for feature refinement
        self.scratch = _make_scratch(
            dpt_channels, 
            dec_d_model,
            groups=1,
            expand=False
        )
        
        # Refinement blocks
        self.refinenet4 = FeatureFusionBlock(dec_d_model, nn.ReLU(False))
        self.refinenet3 = FeatureFusionBlock(dec_d_model, nn.ReLU(False))
        self.refinenet2 = FeatureFusionBlock(dec_d_model, nn.ReLU(False))
        self.refinenet1 = FeatureFusionBlock(dec_d_model, nn.ReLU(False))

        # Output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(dec_d_model, dec_d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dec_d_model // 2, out_channels, kernel_size=1)
        )
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        

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
    
    # Helper function to restore tokens
    def restore_tokens(self, tokens, mask_multi, mask_srtm, ms_restore_ids, mod_restore_ids, mask_token):
        # Extract number of visible patches
        num_visible_multi = (mask_multi != 1).sum(dim=1).max().item()
        num_visible_srtm = (mask_srtm != 1).sum(dim=1).max().item()
        ms_feat = tokens[:, :num_visible_multi]
        modality_feat = tokens[:, num_visible_multi:num_visible_multi + num_visible_srtm]
        B, N, D = ms_feat.shape
        Bm, Nm, Dm = modality_feat.shape
        ms_mask_tokens = mask_token.repeat(B, ms_restore_ids.shape[1] - N, 1)
        mod_mask_tokens = mask_token.repeat(Bm, mod_restore_ids.shape[1] - Nm, 1)
        # MS intermediate features restore
        x_ms = torch.cat([ms_feat, ms_mask_tokens], dim=1)
        restore_ids_expanded = ms_restore_ids.unsqueeze(-1).repeat(1, 1, D)
        x_ms_res_feat = torch.gather(x_ms, dim=1, index=restore_ids_expanded)
        # Modality intermediate features restore
        x_mod = torch.cat([modality_feat, mod_mask_tokens], dim=1)
        mod_restore_ids_expanded = mod_restore_ids.unsqueeze(-1).repeat(1, 1, Dm)
        x_mod_res_feat = torch.gather(x_mod, dim=1, index=mod_restore_ids_expanded)
        
        restored_tokens = torch.cat([x_ms_res_feat, x_mod_res_feat], dim=-1)
        
        return restored_tokens

    def forward(self, encoded, geotransform):
        # Transformer path
        encoder_tokens = encoded['encoder_tokens']
        mask_multi = encoded['mask_multi']
        mask_srtm = encoded['mask_srtm']
        ids_restore_multi = encoded['ids_restore_multi']
        ids_restore_srtm = encoded['ids_restore_srtm']
        intermediate_features = encoded['intermediate_features']  # List of 4 [B, 512, 1024]

        B = encoder_tokens.size(0)
        device = encoder_tokens.device

        # Linear projection
        context_tokens = self.linear_proj(encoder_tokens)

        # Apply the same positional embeddings to both
        if geotransform is not None and self.pos_embed_type == 'geo':
            # Calculate normalized centroids for the entire batch
            normalized_centroids = self.get_normalized_centroids(geotransform, device)
            # Generate positional embeddings for the entire batch
            pos_embedding = self.positional_encoding(normalized_centroids)
        else:
            pos_embedding = self.pos_emb.repeat(B, 1, 1)

        # Restore full token sequence
        tokens = self.restore_tokens(
            context_tokens, 
            mask_multi, 
            mask_srtm,
            ids_restore_multi,
            ids_restore_srtm,
            self.mask_token
        )
        tokens = self.linear_proj_combined(tokens)
        tokens = tokens + pos_embedding
        residual = tokens.clone()
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = tokens + (self.gamma * residual)
        tokens = self.norm(tokens)
        tokens = self.mlp(tokens)
        #transf_out = rearrange(tokens, 'b (h w) c -> b c h w', h=self.h_posemb, w=self.w_posemb)

        # DPT-style path
        dpt_features = []
        for i, feat in enumerate(intermediate_features):
            # Restore and concatenate by last dimension
            rest_feat = self.restore_tokens(
            feat, 
            mask_multi, 
            mask_srtm,
            ids_restore_multi,
            ids_restore_srtm,
            self.mask_token_feats
            )
            # Project to d_model as the restored are 2*d_model
            rest_feat = self.linear_proj_int_features(rest_feat)
            
            # Restore spatial structure
            spatial_feat = rearrange(rest_feat, 'b (h w) c -> b c h w', 
                                    h=self.h_posemb, w=self.w_posemb)
            # Project feature to decoder dimension
            proj_feat = self.feature_projs[i](spatial_feat)
            
            # Apply conceptual resize
            resized_feat = self.resize_layers[i](proj_feat)
            dpt_features.append(resized_feat)
        
        # Process through scratch network
        layer1_rn = self.scratch.layer1_rn(dpt_features[0])
        layer2_rn = self.scratch.layer2_rn(dpt_features[1])
        layer3_rn = self.scratch.layer3_rn(dpt_features[2])
        layer4_rn = self.scratch.layer4_rn(dpt_features[3])
        
        # Refinement path (bottom-up)
        path4 = self.refinenet4(layer4_rn, size=layer3_rn.shape[2:])        
        path3 = self.refinenet3(path4, layer3_rn, size=layer2_rn.shape[2:])
        path2 = self.refinenet2(path3, layer2_rn, size=layer1_rn.shape[2:])
        path1 = self.refinenet1(path2, layer1_rn)
        
        # Final refinement
        dpt_out = self.refinenet1(path1)
        dpt_out = self.output_conv(dpt_out)  # [B, C, H, W]

        # ===== Fusion ===== #
        # Resize transformer output to match DPT output
        # transf_out = F.interpolate(transf_out, size=dpt_out.shape[2:], 
        #                           mode='bilinear', align_corners=True)# This interpolation requires too much resources
        transf_out = self.reconstruct_image(tokens, num_channels=self.out_channels)
        
        # Concatenate and fuse
        fused = torch.cat([dpt_out, transf_out], dim=1)
        fused = self.fusion_conv(fused)
        
        # Final output
        out = F.interpolate(fused, size=(self.image_size, self.image_size), 
                           mode='bilinear', align_corners=True)
        
        return out, dpt_out


    
    # @torch.no_grad()
    # def infer_geotiff(self, image_path, modality_path, input_size=256):
    #     image, transform, crs, orig_shape = self.image2tensor(image_path, input_size)
    #     ms_tensor, elevation_tensor, (orig_h, orig_w), transform, crs = self.load_dual_inputs(image_path, modality_path)

    #     # Model prediction
    #     pred = self.forward(ms_tensor, elevation_tensor, )  # [1, H', W'] ### missing implementation
    #     pred = F.interpolate(pred[:, None], size=orig_shape, mode='bilinear', align_corners=True)[0, 0]

    #     return pred.cpu().numpy(), transform, crs
    
    # def load_dual_inputs(self, image_path_ms, image_path_elevation, input_size=256):
    #     def load_image(path):
    #         with rasterio.open(path) as src:
    #             img = src.read()  # [C, H, W]
    #             transform = src.transform
    #             crs = src.crs
    #         return img, transform, crs

    #     ms_img, transform, crs = load_image(image_path_ms)
    #     elevation_img, _, _ = load_image(image_path_elevation)

    #     # Normalize
    #     ms_img = ms_img / 255.0 if ms_img.dtype == torch.uint8 else ms_img.astype(np.float32) * 0.0001 # assuming image in reflectance scaled to 10.000
    #     elevation_img = elevation_img.astype(np.float32) * 0.0001  # Assuming elevation is in meters scaled 10.000

    #     # Convert to tensors
    #     ms_tensor = torch.from_numpy(ms_img).float()  # [C, H, W]
    #     elevation_tensor = torch.from_numpy(elevation_img).float()

    #     # Resize both
    #     orig_h, orig_w = ms_tensor.shape[1:]
    #     aspect = orig_h / orig_w
    #     if aspect >= 1.0:
    #         target_h = input_size
    #         target_w = int(input_size / aspect)
    #     else:
    #         target_w = input_size
    #         target_h = int(input_size * aspect)

    #     ms_tensor = F.interpolate(ms_tensor.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=True)[0]
    #     elevation_tensor = F.interpolate(elevation_tensor.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=True)[0]

    #     ms_tensor = ms_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    #     elevation_tensor = elevation_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    #     return ms_tensor, elevation_tensor, (orig_h, orig_w), transform, crs


    # def save_geotiff(array, transform, crs, path, dtype='float32'):
    #     with rasterio.open(
    #         path, 'w',
    #         driver='GTiff',
    #         height=array.shape[0],
    #         width=array.shape[1],
    #         count=1,
    #         dtype=dtype,
    #         crs=crs,
    #         transform=transform
    #     ) as dst:
    #         dst.write(array.astype(dtype), 1)



