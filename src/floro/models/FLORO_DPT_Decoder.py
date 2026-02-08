import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from floro.models.util.blocks import FeatureFusionBlock, _make_scratch
from floro.models.Model_Components import SelfAttentionBlockViT, trunc_normal_

class FLORODPTDecoder(nn.Module):
    def __init__(self, 
                 image_size: int= 256,
                 patch_size: int=16,
                 d_model: int=1024,
                 dec_d_model: int=768, 
                 dpt_channels: list= [256, 512, 1024, 1024],
                 out_channels: int=1,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dec_d_model = dec_d_model
        self.d_model = d_model
        self.out_channels = out_channels
        self.dpt_channels = dpt_channels
        self.dropout = dropout

        # Linear projection from encoder to decoder dimension
        self.linear_proj_int_features = nn.Linear(d_model*2, d_model)
        
        self.h_posemb = self.image_size // patch_size
        self.w_posemb = self.image_size // patch_size
 
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
        self.refinenet0 = FeatureFusionBlock(dec_d_model, nn.ReLU(False))

        # Output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(dec_d_model, dec_d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p=self.dropout),
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
        path1 = self.refinenet1(path2, layer1_rn) # [B,C,image_size //2,image_size //2]
        path0 = self.refinenet0(path1, size=(self.image_size,self.image_size))
        
        # Final refinement
        
        dpt_out = self.output_conv(path0)  # [B, C, H, W]
        #upsampled_features = F.interpolate(dpt_out, scale_factor=2, mode='bilinear', align_corners=False) # [B,C,image_size,image_size]

        return dpt_out#upsampled_features




