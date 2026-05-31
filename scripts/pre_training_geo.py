import os
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
import wandb

from floro.models.encoder import FLOROGeoEncoder
from floro.models.mae_decoder import FLOROSSDecoder
from floro.utils.training_utils import train_one_epoch_geo_ss, validate_one_epoch_geo_ss
from floro.utils.checkpoint_utils import save_model, save_model_from_checkpoint
from floro.data.get_dataloaders import get_dataloaders_pretraining
from floro.config.config_parser import get_args_parser
from floro.utils.dist_utils import get_world_size, get_rank, init_distributed_mode


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


def main(args):
    # --------- DDP init ----------
    init_distributed_mode(args)
    use_ddp = (args.distributed == "distributed")

    # choose device
    if args.device == "cuda" and torch.cuda.is_available():
        # LOCAL_RANK is the per-node GPU index; this is what you want for set_device
        local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "rank", 0)))

        # If you pass --gpu explicitly, let it override, but default to local_rank
        gpu_id = local_rank if args.gpu is None else int(args.gpu)

        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device("cpu")
        gpu_id = None

    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --------- Checkpoints existence ----------
    if args.train_checkpoint:
        if not os.path.exists(args.encoder_ckp):
            raise FileNotFoundError("Encoder checkpoint not found!")
        if not os.path.exists(args.decoder_ckp):
            raise FileNotFoundError("Decoder checkpoint not found!")

    # --------- W&B init (rank 0 only) ----------
    if is_main_process():
        wandb.init(project=args.wb_project, config=vars(args))
        if args.run_name:
            wandb.run.name = args.run_name
    else:
        # disable wandb on non-master ranks
        os.environ["WANDB_MODE"] = "disabled"

    # --------- Data ----------
    # IMPORTANT: args.file_path must exist in your parser or remove it from here.
    # I'll guard it:
    file_path = getattr(args, "file_path", "")

    if args.val_path == "" or args.val_path is None:
        # Split train into train/val
        train_dataset, val_dataset = get_dataloaders_pretraining(
            path_to_data=args.train_path,
            test_size=0.3
        )
    else:
        # Use explicit train/val folders (no splitting)
        train_dataset, _ = get_dataloaders_pretraining(
            path_to_data=args.train_path,
            test_size=0.0
        )
        val_dataset, _ = get_dataloaders_pretraining(
            path_to_data=args.val_path,
            test_size=0.0
        )

    loader_kwargs = dict(batch_size=args.batch_size, pin_memory=True)
    if args.workers > 0:
        loader_kwargs.update(num_workers=args.workers, persistent_workers=True, prefetch_factor=2)
    else:
        loader_kwargs.update(num_workers=0)

    if use_ddp:
        num_tasks = get_world_size()
        global_rank = get_rank()

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False
        )

        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    if is_main_process():
        print(f"Dataset size (num. batches): Train -> [{len(train_dataloader)}] Val -> [{len(val_dataloader)}]")

    # --------- Models ----------
    model_enc = FLOROGeoEncoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        multispectral_channels=args.image_channels, 
        modalities_channels=args.modality_channels,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        pos_embed_type=args.pos_embed,
    ).to(device)

    model_dec = FLOROSSDecoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        out_ms_channels=args.output_image_channels,
        out_mod_channels=args.output_modality_channels,
        d_model=args.d_model,
        mlp_ratio=args.mlp_ratio,
        dec_d_model=args.dec_d_model,
        dec_depth=args.dec_depth,
        dec_num_heads=args.num_heads,
        pos_embed_type=args.pos_embed,
        norm_layer=nn.LayerNorm,
    ).to(device)

    # --------- Resume ----------
    epoch_ini = 0
    if args.train_checkpoint:
        checkpoint_enc = torch.load(args.encoder_ckp, map_location="cpu")
        checkpoint_dec = torch.load(args.decoder_ckp, map_location="cpu")
        model_enc.load_state_dict(checkpoint_enc["model_state_dict"], strict=False)
        model_dec.load_state_dict(checkpoint_dec["model_state_dict"], strict=False)
        epoch_ini = int(checkpoint_enc.get("epoch", 0)) + 1

    # --------- DDP wrap ----------
    if use_ddp:
        gpu_id = args.gpu if args.gpu is not None else int(getattr(args, "rank", 0))
        model_enc = torch.nn.parallel.DistributedDataParallel(model_enc, device_ids=[gpu_id], find_unused_parameters=False)
        model_dec = torch.nn.parallel.DistributedDataParallel(model_dec, device_ids=[gpu_id], find_unused_parameters=False)

    # --------- Optimizers / loss / scaler ----------
    optimizer_enc = torch.optim.AdamW(model_enc.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    optimizer_dec = torch.optim.AdamW(model_dec.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    criterion = nn.MSELoss()
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # (Optional) watch only on master and preferably before DDP wrap (or skip entirely)
    if is_main_process():
        # wandb.watch can slow training; keep it light
        wandb.watch(model_enc if not hasattr(model_enc, "module") else model_enc.module, log="gradients", log_freq=500)
        wandb.watch(model_dec if not hasattr(model_dec, "module") else model_dec.module, log="gradients", log_freq=500)

    # --------- Train loop ----------
    total_epochs = epoch_ini + args.epochs
    for epoch in range(epoch_ini, total_epochs):
        if use_ddp:
            train_dataloader.sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\nTraining epoch {epoch}\n")

        loss = train_one_epoch_geo_ss(
            model_enc, model_dec, train_dataloader,
            optimizer_enc, optimizer_dec, criterion, device, scaler, args
        )
        val_loss = validate_one_epoch_geo_ss(model_enc, model_dec, val_dataloader, criterion, device, args)

        # Logging (rank 0 only)
        if is_main_process():
            metrics = {
                "epoch": epoch,
                "train_loss": float(loss),
                "val_loss": float(val_loss),
                "lr_encoder": optimizer_enc.param_groups[0]["lr"],
                "lr_decoder": optimizer_dec.param_groups[0]["lr"],
            }
            wandb.log(metrics)

        # Saving (rank 0 only by your updated save_model)
        if args.save_checkpoint and is_main_process():
            save_path = args.save_dir if args.save_dir else "./checkpoints"
            os.makedirs(save_path, exist_ok=True)

            if args.train_checkpoint:
                # keep same naming convention but append epoch in your helper
                path_saved_enc = save_model_from_checkpoint(model_enc, optimizer_enc, epoch, loss, args.encoder_ckp)
                path_saved_dec = save_model_from_checkpoint(model_dec, optimizer_dec, epoch, loss, args.decoder_ckp)
            else:
                path_saved_enc = save_model(model_enc, optimizer_enc, epoch, loss, save_path,
                                            model_name="FLORO", suffix="Encoder", dt=dt, only_master=True)
                path_saved_dec = save_model(model_dec, optimizer_dec, epoch, loss, save_path,
                                            model_name="FLORO", suffix="Decoder", dt=dt, only_master=True)

            print("********************************")
            print("Models successfully saved:")
            print("Encoder:", path_saved_enc)
            print("Decoder:", path_saved_dec)
            print("********************************")

    if is_main_process():
        wandb.finish()
        print("Training completed.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)