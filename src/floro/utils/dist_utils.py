import builtins, os, datetime, socket
import torch
import torch.distributed as dist

def setup_for_distributed(is_master: bool):
    """Silence printing on non-master ranks."""
    builtin_print = builtins.print
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)
    builtins.print = print

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def _default_master_addr():
    # best effort: hostname of rank0 node
    return os.environ.get("HOSTNAME", "127.0.0.1")

def _default_master_port():
    # pick something stable-ish if user didn't pass one
    return os.environ.get("MASTER_PORT", "29500")

def init_distributed_mode(args):
    """
    Works with both `torchrun` and SLURM (without torchrun).
    Populates args.rank, args.world_size, args.gpu and initializes the PG.
    """
    # Case 1: launched with torchrun (recommended)
    if all(k in os.environ for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]):
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        init_method = "env://"

    # Case 2: SLURM (srun) without torchrun
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
        args.world_size = n_tasks
        # Local rank may be provided by SLURM; if not, derive it
        args.gpu = int(os.environ.get("SLURM_LOCALID", args.rank % torch.cuda.device_count()))
        # Ensure MASTER_* are set
        os.environ.setdefault("MASTER_ADDR", _default_master_addr())
        os.environ.setdefault("MASTER_PORT", _default_master_port())
        # Export RANK/WORLD_SIZE/LOCAL_RANK so downstream libs behave
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["LOCAL_RANK"] = str(args.gpu)
        init_method = "env://"

    # Case 3: MPI (rare here)
    elif args.__dict__.get("dist_on_itp", False) and all(k in os.environ for k in ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"]):
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        os.environ.setdefault("MASTER_ADDR", _default_master_addr())
        os.environ.setdefault("MASTER_PORT", _default_master_port())
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["LOCAL_RANK"] = str(args.gpu)
        init_method = "env://"

    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)
        args.distributed = False
        return

    args.distributed = True

    # MUST set the device before any NCCL collective/barrier
    torch.cuda.set_device(args.gpu)

    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {init_method}, gpu {args.gpu}, world size {args.world_size}", flush=True)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

    # Optional: warm-up barrier AFTER init (now PG knows device mapping)
    dist.barrier()

    setup_for_distributed(args.rank == 0)
