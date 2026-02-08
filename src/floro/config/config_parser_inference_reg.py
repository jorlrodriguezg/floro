import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Inference CHM and Biomass DPT decoders', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wb_project', default="InferenceDPT", type=str, help="Project name for Wandb")
    parser.add_argument('--run_name', default="InferenceDPT", type=str, help="Run name for Wandb")
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

    # Batch | Single image inference
    parser.add_argument('--mode', default='single', choices=['single', 'batch'], help='Run mode: single image or batch')

    # Paths and input parameters
    parser.add_argument('--image_directory', default='', type=str, help='Path to directory with input images (for batch mode)')
    parser.add_argument('--modality_directory', default='', type=str, help='Path to directory with modality files (for batch mode)')
    parser.add_argument('--image_path', default='', type=str, help='Path to the image data')
    parser.add_argument('--modality_path', default='', type=str, help='Path to the modality data')
    parser.add_argument('--bands', nargs='+', default=[1, 2, 3, 4], type=int, help='Bands to be used as input, indexing is 1-based (i.e. --bands 1 2 5 8) | FLORO supports 4 spectral bands (Blue, Green, Red, NIR)')

    # Sliding window inference parameters
    parser.add_argument('--tile_size', default=256, type=int, help='Tile size for inference | Currently FLORO only receives 256 x 256')
    parser.add_argument('--stride', default=200, type=int, help='Stride for sliding window | Determines the overlap between windows (overlap = image_size - stride)')

    # Basic Inference Parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for batched inference')
    parser.add_argument('--masking_ms', default=0.0, type=float, help='Percentage of MS patches to be masked')
    parser.add_argument('--masking_modality', default=0.0, type=float, help='Percentage of modality patches to be masked')    
    parser.add_argument('--lambda_bg', default=0.1, type=float, help='Contribution of background loss to combined loss')
    try:
        bool_action = argparse.BooleanOptionalAction
    except AttributeError:
        bool_action = None
    if bool_action:
        parser.add_argument("--save_std", action=bool_action, default=True,
                            help="Save a companion Standard Deviation GeoTIFF (default: True).")
    else:
        parser.add_argument("--save_std", action="store_true",
                            help="Save a companion Standard Deviation GeoTIFF.")
        parser.add_argument("--no-save_std", dest="save_std",
                            action="store_false", help="Do not save standard deviation map.")
        parser.set_defaults(save_std=True)

    # Target weights
    parser.add_argument('--chm_alpha', default=1.0, type=float, help='Contribution of CHM loss to combined loss')
    parser.add_argument('--biomass_beta', default=1.0, type=float, help='Contribution of Biomass loss to combined loss')
    parser.add_argument('--nitrogen_gamma', default=1.0, type=float, help='Contribution of Nitrogen loss to combined loss')
    parser.add_argument('--carbon_delta', default=1.0, type=float, help='Contribution of Carbon content loss to combined loss')
    
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
    
    # Saving
    parser.add_argument('--output_dir', default='', type=str, help='Directory to save models')
    parser.add_argument('--log_name', default='', type=str, help='Log name to store metrics')

    # Train from checkpoints
    parser.add_argument('--encoder_checkpoint', default='', type=str, help="Encoder's checkpoint | Or single file Enc-Dec checkpoint")
    parser.add_argument('--decoder_checkpoint', default='', type=str, help="Decoder's checkpoint")
    
    # Device Configuration
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')

    return parser
