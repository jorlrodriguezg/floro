import os
import numpy as np
import csv
import datetime
import torch
import torch.nn as nn
import wandb  # <-- 1) Import W&B

from floro.models.FLORO_Encoder import MultiMAE_Encoder
from floro.models.FLORO_DPT_Decoder import FLORODPTDecoder

from floro.config.config_parser_finetune_dpt_seg import get_args_parser

from floro.utils.training_utils import save_checkpoint
from floro.utils.finetuning_utils import finetune_one_epoch_geo_seg_dpt, validate_one_epoch_geo_seg_dpt

from floro.utils.dist_utils import get_world_size, get_rank, init_distributed_mode
from floro.utils.checkpoint_utils import load_weights_strip_module, _strip_module_prefix, _extract_state_dict

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
import random

def log_epoch_data(epoch_run, training_loss, validation_loss, learning_rate, learning_rate_dec, csv_file='training_log.csv'):
    """
    Appends the given epoch's data to a CSV file. If the file doesn't exist or is empty,
    writes the header first.
    
    :param epoch_run: (int) The current epoch number
    :param training_loss: (float) The training loss for this epoch
    :param validation_loss: (float) The validation loss for this epoch
    :param learning_rate: (float) The current learning rate
    :param csv_file: (str) Path to the CSV file where data will be logged
    """
    fieldnames = ['Epoch', 'Training', 'Validation', 'lr_encoder', 'lr_decoder']

    # Use append mode to add a new row for each epoch
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # If the file is empty, write the header first
        if f.tell() == 0:
            writer.writeheader()
        
        # Write the epoch data
        writer.writerow({
            'Epoch': epoch_run,
            'Training': training_loss,
            'Validation': validation_loss,
            'lr_encoder': learning_rate,
            'lr_decoder': learning_rate_dec
        })

# For reproducibility only
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    if args.dataset == "UAV":
        from floro.data.get_dataloaders_finetuning import get_dataloaders_seg
        print("Dataloader for UAV.")
    elif args.dataset =="Potsdam":
        from floro.data.get_dataloaders_finetuning import get_dataloaders_potsdam as get_dataloaders_seg
        print("Dataloader for Potsdam.")

    if args.train_checkpoint:
        if os.path.exists(args.checkpoint):
            print(args.checkpoint)
            print("Encoder checkpoint found in path.")
        else:
            raise Exception("Encoder checkpoint not found!")
            
        if os.path.exists(args.checkpoint_dec):
            print(args.checkpoint_dec)
            print("Decoder checkpoint found in path.")
    
    init_distributed_mode(args)
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 2) Initialize W&B outside of the data-loading code
    wandb.init(
        project=args.wb_project,
        #entity="halokaust",  # optional
        config=vars(args)
    )
    if args.run_name:
        wandb.run.name = args.run_name

    if args.val_path =='':
        
        train_dataset, val_dataset = get_dataloaders_seg(
            path_to_data=args.train_path,
            test_size=0.3    
        )
    else:
        train_dataset, _ = get_dataloaders_seg(
            path_to_data=args.train_path,
            test_size=0.3,
        )
        val_dataset, _ = get_dataloaders_seg(
            path_to_data=args.val_path,
            test_size=0
            
        )

    # For reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed if args.seed is not None else 42)

    # base kwargs
    loader_kwargs = dict(
        batch_size=args.batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=(torch.Generator().manual_seed(getattr(args, "seed", 42))),
    )

    # only add multiprocessing-only knobs if workers>0
    if args.workers > 0:
        loader_kwargs.update(
            num_workers=args.workers,
            persistent_workers=True,
            prefetch_factor=1,
        )
    else:
        loader_kwargs.update(num_workers=0)  # no prefetch_factor / persistent_workers


    if args.distributed:            
        print("---\nUsing distributed training.\n---")
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
            drop_last=True
            )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            **loader_kwargs
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False
            )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            **loader_kwargs
        )
        print(f'Dataset size (num. batches): Train -> [{len(train_dataloader)}] Test -> [{len(val_dataloader)}]')
    else:
        print("---\nUsing single GPU or non-distributed training.\n---")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **loader_kwargs
            )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, **loader_kwargs
            )
        print(f'Dataset size (num. batches): Train -> [{len(train_dataloader)}] Test -> [{len(val_dataloader)}]')
    
    # Define the warm-up + cosine decay function. Make sure it “sees” total_epochs.
    def warmup_schedule_with_cosine_decay(epoch):
        # Linear warm-up for the first 40 epochs, then constant learning rate
        warmup = args.warmup_epochs
        hold = args.hold_epochs
        alpha = args.alpha
        if epoch == 0:
            return 0
        if epoch < warmup:
            return (epoch + 1) / warmup  # Gradually scale from 1e-6 to 1e-4
        if epoch < warmup + hold:
            return 1.0
        else:
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup - hold) / float(args.epochs - warmup - hold)))
            return (1.0 - alpha) * cosine_decay + alpha

    model_enc = MultiMAE_Encoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        multispectral_channels=args.input_channels,
        srtm_channels=args.elev_channels,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        pos_embed_type=args.pos_embed_type,
    ).to(device)

    model_dec = FLORODPTDecoder(
        image_size=args.images_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        dec_d_model=args.dec_d_model, 
        dpt_channels=args.dpt_channels, 
        out_channels=args.num_channels
    ).to(device)

    epoch_ini = 0
    # 3) (Optional) watch the models
    # This won't work with zero-backwadr on skip
    #wandb.watch(model_enc, log="all", log_freq=100)
    #wandb.watch(model_dec, log="all", log_freq=100)

    # Modified optimizer setup with mode selection
    scheduler_enc = None
    scheduler_dec = None
    if args.train_mode == 'finetune_both':
        # optimizer = torch.optim.AdamW(
        #     [
        #         {'params': model_enc.parameters(), 'lr': args.lr_enc},
        #         {'params': model_dec.parameters(), 'lr': args.lr_dec} 
        #     ],
        #     # The global lr here is often ignored if you explicitly set it in groups,
        #     # but can serve as a default if some group doesn't define 'lr'.
        #     lr=args.lr,  
        #     betas=(0.9, 0.95)
        # )
        optimizer_enc = torch.optim.AdamW(model_enc.parameters(), lr=args.lr_enc, betas=(0.9, 0.95))
        optimizer_dec = torch.optim.AdamW(model_dec.parameters(), lr=args.lr_dec, betas=(0.9, 0.95))
        # Single scheduler for joint optimization
        scheduler_enc = torch.optim.lr_scheduler.LambdaLR(optimizer_enc, lr_lambda=warmup_schedule_with_cosine_decay)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec, lr_lambda=warmup_schedule_with_cosine_decay) 
    else:  # decoder-only mode
        # # Freeze encoder parameters
        # for param in model_enc.parameters():
        #     param.requires_grad = False
        optimizer_enc = None
        optimizer_dec = torch.optim.AdamW(model_dec.parameters(), lr=args.lr_dec, betas=(0.9, 0.95))
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec, lr_lambda=warmup_schedule_with_cosine_decay)

    # Modified checkpoint loading section
    if args.train_checkpoint:
        torch.cuda.empty_cache()
        try:
            # Load combined checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            for k in checkpoint:
                print(k)
                
            if args.checkpoint_dec != "":
                try:
                    checkpoint_dec = torch.load(args.checkpoint_dec, map_location=device, weights_only=True)
                    for k in checkpoint_dec:
                        print(k)
                except:
                    checkpoint_dec = None
                    print(F"Decoder checkpoint not loaded.")
            else:
                checkpoint_dec = None
            
            # Load model states (handle both single and multi-model formats)
            if 'encoder_state_dict' in checkpoint:
                model_enc.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
                model_dec.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
                epoch_ini = epoch_enc = int(checkpoint['epoch']) + 1
            else:  # backward compatibility
                #model_enc.load_state_dict(checkpoint['model_state_dict']['encoder'], strict=True)
                #model_dec.load_state_dict(checkpoint['model_state_dict']['decoder'], strict=True)
                model_enc.load_state_dict(checkpoint['model_state_dict'], strict=True)
                if checkpoint_dec is not None:
                    model_dec.load_state_dict(checkpoint_dec['model_state_dict'], strict=True)
                    epoch_ini = epoch_enc = int(checkpoint_dec['epoch']) + 1

            # Load optimizer state with compatibility checks
            try:
                if 'optimizer_enc_state_dict' in checkpoint:  # multi-optimizer format
                    # Handle multi-optimizer format if needed
                    current_opt_state_enc = optimizer_enc.state_dict()
                    saved_opt_state_enc = checkpoint['optimizer_enc_state_dict']
                    
                    # 1. Filter parameter mismatches
                    filtered_param_groups_enc = []
                    for saved_group, current_group in zip(saved_opt_state_enc['param_groups'], current_opt_state_enc['param_groups']):
                        filtered_params = []
                        for p in saved_group['params']:
                            if p in current_group['params']:
                                filtered_params.append(p)
                        filtered_group = {**saved_group, 'params': filtered_params}
                        filtered_param_groups_enc.append(filtered_group)
                    
                    # 2. Filter state entries
                    filtered_state_enc = {
                        k: v for k, v in saved_opt_state_enc['state'].items() 
                        if k in current_opt_state_enc['state']
                    }
                    
                    # Apply filtered state
                    optimizer_enc.load_state_dict({
                        'state': filtered_state_enc,
                        'param_groups': filtered_param_groups_enc
                    })

                    #### Decoder
                    #####
                    current_opt_state_dec = optimizer_dec.state_dict()
                    if checkpoint_dec is not None:
                        saved_opt_state_dec = checkpoint_dec['optimizer_enc_state_dict']
                    
                        # 1. Filter parameter mismatches
                        filtered_param_groups_dec = []
                        for saved_group, current_group in zip(saved_opt_state_dec['param_groups'], current_opt_state_dec['param_groups']):
                            filtered_params = []
                            for p in saved_group['params']:
                                if p in current_group['params']:
                                    filtered_params.append(p)
                            filtered_group = {**saved_group, 'params': filtered_params}
                            filtered_param_groups_dec.append(filtered_group)
                        
                        # 2. Filter state entries
                        filtered_state_dec = {
                            k: v for k, v in saved_opt_state_dec['state'].items() 
                            if k in current_opt_state_dec['state']
                        }
                        
                        # Apply filtered state
                        optimizer_dec.load_state_dict({
                            'state': filtered_state_dec,
                            'param_groups': filtered_param_groups_dec
                        })
                    
                else:
                    # Filter incompatible parameters when changing modes
                    current_opt_state = optimizer_dec.state_dict()
                    saved_opt_state = checkpoint['optimizer_state_dict']
                    
                    # 1. Filter parameter mismatches
                    filtered_param_groups = []
                    for saved_group, current_group in zip(saved_opt_state['param_groups'], current_opt_state['param_groups']):
                        filtered_params = []
                        for p in saved_group['params']:
                            if p in current_group['params']:
                                filtered_params.append(p)
                        filtered_group = {**saved_group, 'params': filtered_params}
                        filtered_param_groups.append(filtered_group)
                    
                    # 2. Filter state entries
                    filtered_state = {
                        k: v for k, v in saved_opt_state['state'].items() 
                        if k in current_opt_state['state']
                    }
                    
                    # Apply filtered state
                    optimizer_dec.load_state_dict({
                        'state': filtered_state,
                        'param_groups': filtered_param_groups
                    })
                    
                # Override loaded LRs with current args
                if args.train_mode == 'finetune_both':
                    # for i, param_group in enumerate(optimizer_enc.param_groups):
                        # if i == 0:  # encoder group
                        #     param_group['lr'] = args.lr_enc
                        # elif i == 1:  # decoder group
                        #     param_group['lr'] = args.lr_dec
                    for param_group in optimizer_enc.param_groups:
                        param_group['lr'] = args.lr_enc

                    for param_group in optimizer_dec.param_groups:
                        param_group['lr'] = args.lr_dec

                else:
                    for param_group in optimizer_dec.param_groups:
                        param_group['lr'] = args.lr_dec
                        
            except (KeyError, ValueError) as e:
                print(f"Partial optimizer loading: {e}")
                # Initialize fresh optimizer if architecture changed
                pass
            
            # Load scheduler state
            if args.train_mode == "decoder_only":
                if checkpoint_dec is not None:
                    if 'scheduler_state_dict' in checkpoint_dec:
                        scheduler_dec.load_state_dict(checkpoint_dec['scheduler_state_dict'])
                        epoch_ini = checkpoint_dec['epoch'] + 1
                    else:
                        print("No scheduler state found in checkpoint")
                else:
                    print("No scheduler state loaded")                        

            loss = checkpoint.get('loss', float('inf'))
            
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {loss:.4f}")
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    if args.distributed:
        # Encoder mode first (before DDP), so reducer sees correct requires_grad
        if args.train_mode == 'decoder_only':
            model_enc.eval()
            for p in model_enc.parameters():
                p.requires_grad_(False)
    
            # Only wrap the trainable module
            for p in model_dec.parameters():
                p.requires_grad_(True)
            model_dec = torch.nn.parallel.DistributedDataParallel(
                model_dec,
                device_ids=[args.gpu],
                output_device=args.gpu,
                find_unused_parameters=True,   # decoder path is fully used
                broadcast_buffers=False
            )
            model_dec.train()
        else:
            # Fine-tune encoder: unfreeze BEFORE DDP
            for p in model_enc.parameters():
                p.requires_grad_(True)
    
            model_enc = torch.nn.parallel.DistributedDataParallel(
                model_enc,
                device_ids=[args.gpu],
                output_device=args.gpu,
                find_unused_parameters=True,    # important for solving unused-param case
                broadcast_buffers=False
                # gradient_as_bucket_view=False  # optional: silences the stride warning
            )
            model_enc.train()
            
            for p in model_dec.parameters():
                p.requires_grad_(True)
            model_dec = torch.nn.parallel.DistributedDataParallel(
                model_dec,
                device_ids=[args.gpu],
                output_device=args.gpu,
                find_unused_parameters=True,   # all decoder params are used but there's an issue
                broadcast_buffers=False
            )
            model_dec.train()


    
    scaler = GradScaler(device=args.device) if args.use_autocast else None
    

    total_epochs = epoch_ini + args.epochs  # total number of epochs for the scheduler
    for epoch in range(epoch_ini, total_epochs):
        print(f"\nTraining epoch {epoch}\n")

        train_sampler.set_epoch(epoch)
        
        #Log losses
        if args.train_mode == 'finetune_both':
            try:
                current_lr_enc = optimizer_enc.param_groups[0]['lr']
                current_lr_dec = optimizer_dec.param_groups[0]['lr']
            except:
                current_lr_enc = ""
                current_lr_dec = ""
                print("Learning rates could not be saved!")
        else:
            current_lr_enc = ""
            current_lr_dec = optimizer_dec.param_groups[0]['lr']

        loss, train_loss_epoch = finetune_one_epoch_geo_seg_dpt(
            model_enc, model_dec, train_dataloader, optimizer_enc,
            optimizer_dec, device, args, scaler,
            scheduler_enc, scheduler_dec
            )
         
        if epoch % args.save_every == 0 or epoch == total_epochs:
            val_loss, val_loss_epoch = validate_one_epoch_geo_seg_dpt(
                model_enc, model_dec, val_dataloader,
                device, args, log_preds_wb = True
                )
        else:
            val_loss, val_loss_epoch = validate_one_epoch_geo_seg_dpt(
                model_enc, model_dec, val_dataloader, device, args
                )

        if args.log_name != "":
            log_file = args.log_name
        else:
            log_file = f"FLORO_GEO_losses_Segmentation_log_{dt}.csv"
        log_epoch_data(epoch, loss, val_loss, current_lr_enc, current_lr_dec, log_file)

        # 4) Log metrics to W&B
        metrics = {
            "epoch": epoch,
            "train_loss": loss,
            "train_ce_loss": train_loss_epoch["ce"],
            "train_dice_loss": train_loss_epoch["dice"],
            "train_miou": train_loss_epoch["miou"],
            "train_pixel_acc": train_loss_epoch["pixel_acc"],
            "val_loss": val_loss,
            "val_ce_loss": val_loss_epoch["ce"],
            "val_dice_loss": val_loss_epoch["dice"],
            "val_miou": val_loss_epoch["miou"],
            "val_pixel_acc": val_loss_epoch["pixel_acc"],
            "val_per_class_iou": val_loss_epoch["per_class_iou"]
        }
        if current_lr_enc is not None:
            metrics["lr_encoder"] = current_lr_enc
        if current_lr_dec is not None:
            metrics["lr_decoder"] = current_lr_dec
        wandb.log(metrics)
        
        

        if args.save_checkpoint:
            if epoch % args.save_every == 0 or epoch == total_epochs:
                if args.train_checkpoint:
                    if args.train_mode == "decoder_only":
                        save_path = args.checkpoint
                        path_saved = save_checkpoint(model_dec, optimizer_dec, epoch, loss, save_dir = None, prefix = None, checkpoint_path=save_path)
                        print("********************************")
                        print(f"\nDecoder Checkpopint successfully saved:\n")
                        print(f"Decoder location: {path_saved}\n")
                        print("********************************")
                    else:
                        save_path = args.checkpoint
                        path_saved = save_checkpoint({'encoder': model_enc, 'decoder': model_dec},
                                                     {'optimizer_enc':optimizer_enc, 'optimizer_dec':optimizer_dec},
                                                     epoch, loss, save_dir = None, prefix = None,
                                                     checkpoint_path=save_path)
                        print("********************************")
                        print(f"\nModels successfully saved:\n")
                        print(f"Combined checkpoint location: {path_saved}\n")
                        print("********************************")
                else:
                
                    save_path = args.save_dir if args.save_dir else "./checkpoints/FineTuneSegmentation"
                    print(f"\nSaving models to: {save_path}")

                    if args.train_mode == "decoder_only":
                        path_saved = save_checkpoint(model_dec, optimizer_dec, epoch, loss, save_dir=save_path, prefix="DecTraining", dt=dt, checkpoint_path=None)
                        print("********************************")
                        print(f"\nDecoder Checkpopint successfully saved:\n")
                        print(f"Decoder location: {path_saved}\n")
                        print("********************************")
                    else:
                        path_saved = save_checkpoint({'encoder': model_enc, 'decoder': model_dec},
                                                     {'optimizer_enc':optimizer_enc, 'optimizer_dec':optimizer_dec},
                                                     epoch, loss, save_dir=save_path,
                                                     prefix = "Finetune_", checkpoint_path=None)
                        print(f'Model checkpoint saved to: {path_saved}')
                        
                        print("********************************")
                        print(f"\nModels chackpoint successfully saved:\n")
                        print("********************************")
            else:
                print("Checkpoint not saved.")
    wandb.finish()
    print("Training completed.")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)