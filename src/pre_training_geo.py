import os
import time
import datetime
import torch
import torch.nn as nn
from src.models.FLORO_Encoder import MultiMAE_Encoder
from src.models.FLORO_SSDecoder import SatMultiMAEDecoder

from src.utils.training_utils import train_one_epoch_geo, validate_one_epoch_geo, save_model, save_model_from_checkpoint
from src.data.get_dataloaders import get_dataloaders_geo
from src.config.config_parser import get_args_parser
from src.utils.various import get_world_size, get_rank, init_distributed_mode

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import wandb  # <-- 1) Import W&B

def main(args):
    if args.train_checkpoint:
        if os.path.exists(args.encoder_ckp):
            print("Encoder checkpoint found in path.")
        else:
            raise Exception("Encoder checkpoint not found!")
            
        if os.path.exists(args.decoder_ckp):
            print("Decoder checkpoint found in path.")
        else:
            raise Exception("Decoder checkpoint not found!")
    
    init_distributed_mode(args)
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 2) Initialize W&B outside of the data-loading code
    wandb.init(
        project=args.wb_project,
        config=vars(args)
    )
    if args.run_name:
        wandb.run.name = args.run_name

    if args.distributed == "distributed":
        train_dataset, val_dataset = get_dataloaders_geo(
            batch_size=args.batch_size,
            path_to_data=args.train_path,
            test_size=0.2,
            workers=args.workers,
            device=args.device,
            task=args.task,
            distributed=args.distributed
        )

        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers
        )
    else:
        train_dataloader, val_dataloader = get_dataloaders_geo(
            batch_size=args.batch_size,
            path_to_data=args.train_path,
            test_size=0.2,
            workers=args.workers,
            device=args.device,
            task=args.task,
            distributed=None
        )

    model_enc = MultiMAE_Encoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        multispectral_channels=args.input_channels,
        srtm_channels=args.elev_channels,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        pos_embed_type="geo",
    ).to(device)

    model_dec = SatMultiMAEDecoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        ms_channels=args.input_channels,  # Multispectral channels
        srtm_channels=args.elev_channels,  # SRTM channels
        d_model=args.d_model,
        mlp_ratio=args.mlp_ratio,
        dec_d_model=args.dec_d_model,
        dec_depth=args.dec_depth,
        dec_num_heads=args.num_heads,
        pos_embed_type="geo",
        norm_layer=nn.LayerNorm
    ).to(device)

    epoch_ini = 0
    # 3) (Optional) watch the models
    wandb.watch(model_enc, log="all", log_freq=100)
    wandb.watch(model_dec, log="all", log_freq=100)

    if args.train_checkpoint:
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                checkpoint_enc = torch.load(args.encoder_ckp, map_location=device)
                model_enc.load_state_dict(checkpoint_enc['model_state_dict'], strict=False)
                epoch_ini = epoch_enc = int(checkpoint_enc['epoch']) + 1
                loss_enc = checkpoint_enc['loss']
                torch.cuda.empty_cache()
                checkpoint_dec = torch.load(args.decoder_ckp, map_location=device)
                model_dec.load_state_dict(checkpoint_dec['model_state_dict'], strict=False)
                epoch_dec = checkpoint_dec['epoch'] + 1
                loss_dec = checkpoint_dec['loss']
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading checkpoints: {e}")

    if args.distributed == "distributed":
        model_enc = torch.nn.parallel.DistributedDataParallel(model_enc, device_ids=[args.gpu])
        model_dec = torch.nn.parallel.DistributedDataParallel(model_dec, device_ids=[args.gpu])

    optimizer_enc = torch.optim.AdamW(model_enc.parameters(), lr=args.lr, betas=(0.9, 0.95))
    optimizer_dec = torch.optim.AdamW(model_dec.parameters(), lr=args.lr, betas=(0.9, 0.95))

    criterion = nn.MSELoss()
    scaler = GradScaler()

    total_epochs = epoch_ini + args.epochs
    for epoch in range(epoch_ini, total_epochs):
        print(f"\nTraining epoch {epoch}\n")
        try:
            current_lr_enc = optimizer_enc.param_groups[0]['lr']
            current_lr_dec = optimizer_dec.param_groups[0]['lr']
        except:
            current_lr_enc = ""
            current_lr_dec = ""
            print("Learning rates could not be saved!")

        loss = train_one_epoch_geo(model_enc, model_dec, train_dataloader, optimizer_enc, optimizer_dec, criterion, device, scaler, args)
        val_loss = validate_one_epoch_geo(model_enc, model_dec, val_dataloader, criterion, device, args)

        # 4) Log metrics to W&B
        metrics = {
            "epoch": epoch,
            "train_loss": loss,
            "val_loss": val_loss,
        }
        if current_lr_enc is not None:
            metrics["lr_encoder"] = current_lr_enc
        if current_lr_dec is not None:
            metrics["lr_decoder"] = current_lr_dec
        wandb.log(metrics)
        
        if args.save_checkpoint:
            if args.train_checkpoint:
                save_path_enc = args.encoder_ckp
                save_path_dec = args.decoder_ckp
                path_saved_enc = save_model_from_checkpoint(model_enc, optimizer_enc, epoch, loss, save_path_enc)
                path_saved_dec = save_model_from_checkpoint(model_dec, optimizer_dec, epoch, loss, save_path_dec)
                print("********************************")
                print(f"\nModels successfully saved:\n")
                print(f"Encoder location: {path_saved_enc}\n")
                print(f"Decoder location: {path_saved_dec}\n")
                print("********************************")
            else:
                save_path = args.save_dir if args.save_dir else "./checkpoints"
                print(f"\nSaving models to: {save_path}")
                path_saved_enc = save_model(model_enc, optimizer_enc, epoch, loss, save_path, model_name="SatMultiMAEViT", suffix="Encoder", dt=dt)
                path_saved_dec = save_model(model_dec, optimizer_dec, epoch, loss, save_path, model_name="SatMultiMAEViT", suffix="Decoder", dt=dt)
                print("********************************")
                print(f"\nModels successfully saved:\n")
                print(f"Encoder location: {path_saved_enc}\n")
                print(f"Decoder location: {path_saved_dec}\n")
                print("********************************")
        else:
            print("Checkpoint not saved.")
    wandb.finish()
    print("Training completed.")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)