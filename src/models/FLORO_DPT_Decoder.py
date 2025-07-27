import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from src.models.util.blocks import FeatureFusionBlock, _make_scratch
from src.models.Model_Components import SelfAttentionBlockViT, trunc_normal_

class FLORODPTDecoder(nn.Module):
    def __init__(self, 
                 image_size=256,
                 patch_size=16,
                 d_model=1024,
                 dec_d_model=768, 
                 dpt_channels = [256, 512, 1024, 1024],
                 out_channels=1
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dec_d_model = dec_d_model
        self.d_model = d_model
        self.out_channels = out_channels
        self.dpt_channels = dpt_channels

        # Linear projection from encoder to decoder dimension
        self.linear_proj_int_features = nn.Linear(d_model*2, d_model)
        
        # Mask token
        self.mask_token_feats = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token_feats, std=0.02)

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

    def forward(self, encoded):
        # Transformer path
        mask_multi = encoded['mask_multi']
        mask_srtm = encoded['mask_srtm']
        ids_restore_multi = encoded['ids_restore_multi']
        ids_restore_srtm = encoded['ids_restore_srtm']
        intermediate_features = encoded['intermediate_features']  # List of 4 [B, 512, 1024]

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

        return dpt_out


    
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



