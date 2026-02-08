import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Train ViT with CHM and Classification Decoders', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wb_project', default="Finetuning", type=str, help="Project name for Wandb")
    parser.add_argument('--run_name', default="Finetuning", type=str, help="Run name for Wandb")
    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true', help="Enable distributed training")
    parser.add_argument('--rank', default=os.getenv('LOCAL_RANK', 0), type=int, help='Rank of the process in distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Total number of processes to run')
    parser.add_argument('--node_rank', default=0, type=int, help='Rank of the node for multi-node distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU to use for training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Dataloaidng
    parser.add_argument('--workers', default=2, type=int, help='Number of workers for dataloading')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--dataset",
        type=str,
        default="UAV",
        choices=["UAV", "Potsdam"],
        help="Dataset for finetuning: "
             "'UAV' default UAV dataset for plant segmentation, "
             "'Potsdam' Potsdam dataset. "
    )
    
    # Basic Training Parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1e-6, type=float, help='Default learning rate')
    parser.add_argument('--lr_enc', default=1e-6, type=float, help='Learning rate for encoder finetune | Ideally lower than that of the decoder')
    parser.add_argument('--lr_dec', default=1e-4, type=float, help='Learning rate for decoder traing')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay (L2 penalty)')
    parser.add_argument('--masking_ms', default=0.0, type=float, help='Percentage of MS patches to be masked')
    parser.add_argument('--masking_modality', default=0.0, type=float, help='Percentage of modality patches to be masked')    
    parser.add_argument('--lambda_bg', default=0.1, type=float, help='Contribution of background loss to combined loss')

    # --- Segmentation-specific knobs ---
    parser.add_argument('--ignore_index', default=0, type=int, help='Label value to ignore in loss/metrics (void/unknown pixels). Default: 0')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing factor for Cross-Entropy. Range [0,1]. Default: 0.0')
    parser.add_argument('--class_weights', default=None, type=float, nargs='+',
                    help='Optional per-class weights for multi-class CE (space-separated). '
                         'Example: --class_weights 1.0 2.0 0.5 ; Omit for None')
    parser.add_argument('--lambda_dice', default=0.3, type=float, help='Weight for Dice loss mixed with CE. Typical 0.3–0.6; 0.0 disables Dice.')
    
    # Model Specific Parameters
    parser.add_argument('--images_size', default=256, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=16, type=int, help='Size of the patches')
    parser.add_argument('--input_channels', default=4, type=int, help='Number of input channels (e.g., RGB+NIR)')
    parser.add_argument('--elev_channels', default=1, type=int, help='Number of elevation channels (e.g., DSM+DTM)')
    parser.add_argument('--d_model', default=1024, type=int, help='Dimension of the model')
    parser.add_argument('--depth', default=24, type=int, help='Depth of the encoder')
    parser.add_argument('--num_heads', default=16, type=int, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='Ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--pos_embed_type', default='absolute', type=str, help='Position embedding type: absolute | geo')

    # Decoder Specific Parameters
    parser.add_argument('--num_channels', default=1, type=int, help='Number of output channels')
    parser.add_argument('--dec_d_model', default=768, type=int, help='Dimension of the decoder model')
    parser.add_argument('--dpt_channels', nargs='+', default=[256, 512, 1024, 1024], type=int, help='DPT channels')
    
    # Finetuning regime
    parser.add_argument('--warmup_epochs', default=50, type=int, help='Number of warm-up epochs')
    parser.add_argument('--hold_epochs', default=800, type=int, help='Number of epochs to hold the max LR after warmup (if any)')
    parser.add_argument('--alpha', default=1e-6, type=float, help='Final learning rate factor after cosine decay')
    parser.add_argument(
        "--train_mode",
        type=str,
        default="decoder_only",
        choices=["decoder_only", "resume_decoder", "finetune_both"],
        help="Training mode: "
             "'decoder_only' to train decoder from scratch while encoder is fixed, "
             "'resume_decoder' to continue training decoder alone from a checkpoint, "
             "'finetune_both' to fine-tune both encoder and decoder with distinct LRs."
    )
    parser.add_argument('--accumulation_steps', default=4, type=int, help='Number of steps to accumulate gradients')

    # Gradient scaling with autocast
    autocast_group = parser.add_mutually_exclusive_group()
    autocast_group.add_argument('--use_autocast', dest='use_autocast', action='store_true', help='Enable automatic mixed precision (AMP) via autocast (default).')
    autocast_group.add_argument('--no_autocast', dest='use_autocast', action='store_false', help='Disable automatic mixed precision (AMP).')
    parser.set_defaults(use_autocast=True)


    # Paths and Saving
    parser.add_argument('--save_checkpoint', action='store_true', dest='save_checkpoint', help='Whether to save checkpoint')
    parser.add_argument('--no_save_checkpoint', action='store_false', dest='save_checkpoint', help='Do not save checkpoint')
    parser.set_defaults(save_checkpoint=True)
    parser.add_argument('--train_path', default='', type=str, help='Path to the training data')
    parser.add_argument('--val_path', default='', type=str, help='Path to the validation data')
    parser.add_argument('--save_dir', default='./checkpoints', type=str, help='Directory to save models')
    parser.add_argument('--save_every', default=10, type=int, help='Save checkpoint every X epochs')
    parser.add_argument('--log_name', default='', type=str, help='Log name to store training metrics')

    # Train from checkpoints
    parser.add_argument('--train_checkpoint', action='store_true', help='Whether to continue training from checkpoint')
    parser.add_argument('--no_train_checkpoint', action='store_false', dest='use_abs_pos', help='Train from scratch')
    parser.set_defaults(train_checkpoint=False)
    parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint to continue training')
    parser.add_argument('--checkpoint_dec', default='', type=str, help='Checkpoint to continue decoder-only training')
    
    # Device Configuration
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')

    return parser
