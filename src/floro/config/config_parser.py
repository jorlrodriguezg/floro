import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Pretraining FLORO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---------------------------
    # Logging (W&B)
    # ---------------------------
    parser.add_argument("--wb_project", default="Pretraining_floro_geo", type=str, help="Wandb project")
    parser.add_argument("--run_name", default="Initial run", type=str, help="Wandb run name")

    # ---------------------------
    # Distributed training
    # ---------------------------
    parser.add_argument('--distributed', action='store_true', help="Enable distributed training")
    parser.add_argument("--rank", default=int(os.getenv("LOCAL_RANK", 0)), type=int, help="Local rank")
    parser.add_argument("--world_size", default=int(os.getenv("WORLD_SIZE", 1)), type=int, help="World size")
    parser.add_argument("--node_rank", default=int(os.getenv("NODE_RANK", 0)), type=int, help="Node rank")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")

    # ---------------------------
    # Data loading
    # ---------------------------
    parser.add_argument("--workers", default=4, type=int, help="Number of dataloader workers")
    parser.add_argument("--train_path", default="./data/train", type=str, help="Path to the training data root")
    parser.add_argument("--val_path", default="./data/val", type=str, help="Path to the validation data root")

    # Multi-folder dataset (expects root/subfolder/*.tif)
    parser.add_argument(
        "--data_layout",
        default="multifolder",
        choices=["flat", "multifolder"],
        help="Dataset layout: flat=root/*.tif, multifolder=root/<source>/*.tif"
    )
    parser.add_argument(
        "--sources",
        default="uav_rgb_dsm,uav_ms_dsm,france_dtm,s2_dem_s1",
        type=str,
        help="Comma-separated source folder names used when data_layout=multifolder"
    )

    # ---------------------------
    # Training
    # ---------------------------
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--masking_amount", default=0.75, type=float, help="Percentage of patches to be masked")
    parser.add_argument("--mask_ratio_ms", default=0.75, type=float, help="Percentage of multispectral patches to be masked")
    parser.add_argument("--mask_ratio_mod", default=0.70, type=float, help="Percentage of modality patches to be masked")
    parser.add_argument("--mask_ratio_jitter", default=0.0, type=float, help="Optional jitter for varying mask regime")
    parser.add_argument("--lambda_ms", default=1.0, type=float, help="Importance of multispectral input")
    parser.add_argument("--lambda_mod", default=0.5, type=float, help="Importance of modality input")
    parser.add_argument("--lambda_sar", default=0.5, type=float, help="Importance of SAR input")
    parser.add_argument("--accumulation_steps", default=4, type=int, help="Gradient accumulation steps")

    # Augmentations
    parser.add_argument("--rotate_type", default="fixed90", choices=["none", "fixed90", "free"], type=str, help="Rotation type")
    parser.add_argument("--blur_sigma", default=1.0, type=float, help="Max gaussian blur sigma")
    parser.add_argument("--noise_max", default=0.2, type=float, help="Max gaussian noise std (approx)")

    # Band dropping / modality dropout
    parser.add_argument("--use_band_drop", action="store_true", help="Enable BandDropping transform")
    parser.add_argument("--band_drop_p_apply", default=0.7, type=float, help="Probability to apply band dropping per sample")
    parser.add_argument("--band_drop_bgr", default=0.05, type=float, help="Drop prob for BGR")
    parser.add_argument("--band_drop_re", default=0.10, type=float, help="Drop prob for RE")
    parser.add_argument("--band_drop_nir", default=0.10, type=float, help="Drop prob for NIR (B8)")
    parser.add_argument("--band_drop_nir2", default=0.10, type=float, help="Drop prob for NIR2 (B8A)")
    parser.add_argument("--band_drop_swir", default=0.10, type=float, help="Drop prob for SWIR")
    parser.add_argument("--band_drop_elev", default=0.15, type=float, help="Drop prob for elevation")
    parser.add_argument("--band_drop_sar", default=0.10, type=float, help="Drop prob for SAR")
    parser.add_argument("--band_drop_keep_one_opt", action="store_true", help="Ensure at least one optical group remains")
    parser.set_defaults(band_drop_keep_one_opt=True)

    # Normalization stats paths (or pass via code)
    parser.add_argument("--stats_path", default="", type=str, help="Path to npz with means/stds (optional)")

    # ---------------------------
    # Model
    # ---------------------------
    parser.add_argument("--task", default="self_supervised", choices=["self_supervised", "chm", "class"], type=str, help="Task")
    parser.add_argument("--images_size", default=256, type=int, help="Input image size")
    parser.add_argument("--patch_size", default=16, type=int, help="Patch size")

    # IMPORTANT: Encoder inputs and Decoder outputs are different. Decoder won't predict validity channels!
    parser.add_argument("--image_channels", default=13, type=int, help="Packed image channels (with validity)")
    parser.add_argument("--modality_channels", default=5, type=int, help="Packed modalities channels (with validity)")
    parser.add_argument("--output_image_channels", default=8, type=int, help="Packed image channels (with validity)")
    parser.add_argument("--output_modality_channels", default=3, type=int, help="Packed modalities channels (with validity)")

    parser.add_argument("--d_model", default=1024, type=int, help="Encoder dim")
    parser.add_argument("--depth", default=24, type=int, help="Encoder depth")
    parser.add_argument("--num_heads", default=16, type=int, help="Num heads")
    parser.add_argument("--mlp_ratio", default=4, type=int, help="MLP ratio")

    parser.add_argument("--dec_d_model", default=768, type=int, help="Decoder dim")
    parser.add_argument("--dec_depth", default=2, type=int, help="Decoder depth")

    parser.add_argument("--pos_embed", default="absolute", choices=["absolute", "geo"], type=str, help="Position embedding type")

    # ---------------------------
    # Checkpointing / saving
    # ---------------------------
    parser.add_argument("--save_dir", default="./checkpoints", type=str, help="Directory to save models")

    parser.add_argument("--save_checkpoint", action="store_true", help="Save checkpoints")
    parser.add_argument("--no_save_checkpoint", action="store_false", dest="save_checkpoint", help="Do not save checkpoints")
    parser.set_defaults(save_checkpoint=True)

    parser.add_argument("--save_only_master", action="store_true", help="Only save from rank 0")
    parser.set_defaults(save_only_master=True)

    # Resume / load checkpoints
    parser.add_argument("--train_checkpoint", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--no_train_checkpoint", action="store_false", dest="train_checkpoint", help="Train from scratch")
    parser.set_defaults(train_checkpoint=False)

    parser.add_argument("--encoder_ckp", default="", type=str, help="Encoder checkpoint path")
    parser.add_argument("--decoder_ckp", default="", type=str, help="Decoder checkpoint path")

    # ---------------------------
    # Device
    # ---------------------------
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")


    # ---------------------------
    # Gradient scaling with autocast
    # ---------------------------
    autocast_group = parser.add_mutually_exclusive_group()
    autocast_group.add_argument('--use_autocast', dest='use_autocast', action='store_true', help='Enable automatic mixed precision (AMP) via autocast (default).')
    autocast_group.add_argument('--no_autocast', dest='use_autocast', action='store_false', help='Disable automatic mixed precision (AMP).')
    parser.set_defaults(use_autocast=True)


    return parser