import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Train ViT with CHM and Classification Decoders', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Distributed training parameters
    parser.add_argument('--wb_project', default="Pretraining_floro_geo", type=str, help='Wandb project')
    parser.add_argument('--run_name', default="Initial run", type=str, help='Wandb run name')

    # Distributed training parameters
    parser.add_argument('--distributed', default=None, type=str, help='"distributed" to use distributed training')
    parser.add_argument('--rank', default=os.getenv('LOCAL_RANK', 0), type=int, help='Rank of the process in distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Total number of processes to run')
    parser.add_argument('--node_rank', default=0, type=int, help='Rank of the node for multi-node distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU to use for training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Dataloaidng
    parser.add_argument('--workers', default=0, type=int, help='Number of workers for dataloading')
    
    # Basic Training Parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the encoder')
    parser.add_argument('--lr_chm_dec', default=1e-4, type=float, help='Learning rate for the CHM decoder')
    parser.add_argument('--lr_class_dec', default=1e-4, type=float, help='Learning rate for the classification decoder')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay (L2 penalty)')
    parser.add_argument('--masking_amount', default=0.75, type=float, help='Percentage of patches to be masked')
    
    # Model Specific Parameters
    parser.add_argument('--task', default='self_supervised', type=str, help='Task to train (self_supervised, chm, class)')
    parser.add_argument('--images_size', default=256, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=16, type=int, help='Size of the patches')
    parser.add_argument('--input_channels', default=4, type=int, help='Number of input channels (e.g., RGB+NIR)')
    parser.add_argument('--elev_channels', default=1, type=int, help='Number of elevation channels (e.g., DSM+DTM)')
    parser.add_argument('--d_model', default=1024, type=int, help='Dimension of the model')
    parser.add_argument('--depth', default=24, type=int, help='Depth of the encoder')
    parser.add_argument('--num_heads', default=16, type=int, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='Ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of classes for classification decoder')
    parser.add_argument('--dec_d_model', default=768, type=int, help='Decoder model dimension')
    parser.add_argument('--dec_depth', default=2, type=int, help='Depth of the decoders')
    parser.add_argument('--pos_embed', default='absolute', type=str, help='Position embedding type: absolute | geo')

    # Paths and Saving
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save checkpoint')
    parser.add_argument('--no_save_checkpoint', action='store_false', dest='use_abs_pos', help='Do not save checkpoint')
    parser.set_defaults(save_checkpoint=True)
    parser.add_argument('--train_path', default='./data/train', type=str, help='Path to the training data')
    parser.add_argument('--val_path', default='./data/val', type=str, help='Path to the validation data')
    parser.add_argument('--save_dir', default='./checkpoints', type=str, help='Directory to save models')

    # Train from checkpoints
    parser.add_argument('--train_checkpoint', action='store_true', help='Whether to continue training from checkpoint')
    parser.add_argument('--no_train_checkpoint', action='store_false', dest='use_abs_pos', help='Train from scratch')
    parser.set_defaults(train_checkpoint=False)
    parser.add_argument('--encoder_ckp', default='', type=str, help='Encoder checkpoint to continue training')
    parser.add_argument('--decoder_ckp', default='', type=str, help='Decoder checkpoint to continue training')
    parser.add_argument('--accumulation_steps', default=4, type=int, help='Number of steps to accumulate gradients')
    

    # Device Configuration
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')

    return parser
