import torch
import torch.nn as nn
import torch.nn.functional as F


class FLOROLinearClassDecoder(nn.Module):
    def __init__(
        self,
        num_classes=10,
        d_model=256,
        image_size=64,
        patch_size=8,
        use_intermediate_features=True,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_intermediate_features = use_intermediate_features

        # Fuse MS + modality features when both are present
        self.linear_proj_int_features = nn.Linear(d_model * 2, d_model)

        # If using 4 intermediate layers, concatenated channel dim becomes 4*d_model
        fused_channels = 4 * d_model if use_intermediate_features else d_model

        # Project concatenated multi-layer features
        self.feature_fusion = nn.Conv2d(fused_channels, d_model, kernel_size=1)

        # Spatial class activation maps
        self.classifier = nn.Conv2d(d_model, num_classes, kernel_size=1)

    def _tokens_to_map(self, x):
        """
        Convert patch tokens [B, N, D] -> feature map [B, D, H, W]
        """
        B, N, D = x.shape
        assert N == self.num_patches, (
            f"Expected {self.num_patches} patch tokens, got {N}"
        )
        x = x.view(B, self.grid_size, self.grid_size, D)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _fuse_ms_mod_tokens(self, feat, n_ms, n_mods, use_mods):
        """
        feat: [B, N_total, D]
        Returns fused patch tokens [B, N_ms, D]
        """
        ms_vis = feat[:, :n_ms, :]  # [B, N_ms, D]

        if use_mods and n_mods > 0:
            mods_vis = feat[:, n_ms:n_ms + n_mods, :]  # [B, N_mods, D]

            # If modality tokens are one-per-patch, shapes should match.
            # If not, adapt this logic depending on your encoder design.
            if mods_vis.shape[1] == ms_vis.shape[1]:
                fused = torch.cat([ms_vis, mods_vis], dim=-1)  # [B, N_ms, 2D]
                fused = self.linear_proj_int_features(fused)   # [B, N_ms, D]
            else:
                # fallback: use MS tokens only if token counts differ
                fused = ms_vis
        else:
            fused = ms_vis

        return fused

    def forward(self, encoder_out: dict, return_cam=False):
        """
        Expected keys in encoder_out:
          - "intermediate_features": list of 4 tensors [B, N_total, D]
            OR
          - "encoder_tokens": final tensor [B, N_total, D]
          - "n_ms_tokens"
          - "n_modalities_tokens"
          - "use_modalities"
        """
        n_ms = int(encoder_out["n_ms_tokens"])
        n_mods = int(encoder_out["n_modalities_tokens"])
        use_mods = bool(encoder_out["use_modalities"])

        if self.use_intermediate_features:
            intermediate_features = encoder_out["intermediate_features"]
            assert isinstance(intermediate_features, (list, tuple)), \
                "encoder_out['intermediate_features'] must be a list/tuple"
            assert len(intermediate_features) == 4, \
                f"Expected 4 intermediate features, got {len(intermediate_features)}"

            feat_maps = []
            for feat in intermediate_features:
                assert feat.size(1) == n_ms + n_mods, \
                    f"Feature token count mismatch: got {feat.size(1)}, expected {n_ms+n_mods}"

                fused_tokens = self._fuse_ms_mod_tokens(feat, n_ms, n_mods, use_mods)
                feat_map = self._tokens_to_map(fused_tokens)  # [B, D, H, W]
                feat_maps.append(feat_map)

            final_features = torch.cat(feat_maps, dim=1)     # [B, 4D, H, W]
            final_features = self.feature_fusion(final_features)  # [B, D, H, W]

        else:
            feat = encoder_out["encoder_tokens"]
            assert feat.size(1) == n_ms + n_mods, \
                f"Feature token count mismatch: got {feat.size(1)}, expected {n_ms+n_mods}"

            fused_tokens = self._fuse_ms_mod_tokens(feat, n_ms, n_mods, use_mods)
            final_features = self._tokens_to_map(fused_tokens)  # [B, D, H, W]

        # Class activation maps: [B, C, H, W]
        cam = self.classifier(final_features)

        # Global average pooling over spatial grid -> logits [B, C]
        logits = cam.mean(dim=(2, 3))

        if return_cam:
            return (logits, cam)

        return logits