# Import necessary modules
import torch
from src.data.dataloader_vegetation_uav import LoadersPreparation as LoadersPreparationSeg
from src.data.dataloader_vegetation_uav_chm import LoadersPreparation as LoadersPreparationReg
from src.data.dataloader_france_dem import LoadersPreparation as LoadersPreparationFranceDEM
from src.data.dataloading_benchmark_datasets import LoadersPreparationBench
# Add imports for your other dataloaders here

def get_dataloaders_seg(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    #mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    #std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    mean = None
    std = None
    
    train_dataloader, val_dataloader = LoadersPreparationSeg(
        path_to_data,
        batch_size=batch_size,
        test_size=test_size,
        workers=workers,
        device=device,
        mean = mean,
        std = std,
        rotate_type = 'fixed90',
        downsample_scale=None,
        blur_sigma=1.0,
        noise_max = 0.1,
        distributed = distributed,
        type="training"
        )
    
    return train_dataloader, val_dataloader

def get_dataloaders_reg(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    #mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    #std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    mean = None
    std = None
    
    train_dataloader, val_dataloader = LoadersPreparationReg(
        path_to_data,
        batch_size=batch_size,
        test_size=test_size,
        workers=workers,
        device=device,
        mean = mean,
        std = std,
        rotate_type = 'fixed90',
        downsample_scale=None,
        blur_sigma=1.0,
        noise_max = 0.1,
        distributed = distributed,
        type="training"
        )
    
    return train_dataloader, val_dataloader

def get_dataloaders_reg_dem(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    #mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    #std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    mean = None
    std = None
    
    train_dataloader, val_dataloader = LoadersPreparationFranceDEM(
        path_to_data,
        batch_size=batch_size,
        test_size=test_size,
        workers=workers,
        device=device,
        mean = mean,
        std = std,
        rotate_type = 'fixed90',
        downsample_scale=None,
        blur_sigma=1.0,
        noise_max = 0.1,
        distributed = distributed,
        type="training"
        )
    
    return train_dataloader, val_dataloader

# def get_dataloaders_potsdam(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
#     #mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
#     #std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
#     mean = None
#     std = None
    
#     train_dataloader, val_dataloader = LoadersPreparationPotsdam(
#         path_to_data,
#         batch_size=batch_size,
#         test_size=test_size,
#         workers=workers,
#         device=device,
#         mean = None,
#         std = None,
#         rotate_type = 'fixed90',
#         downsample_scale=None,
#         blur_sigma=1.5,
#         noise_max = 0.2,
#         distributed = distributed,
#         type="training"
#         )
    
#     return train_dataloader, val_dataloader

def get_dataloaders_potsdam(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    #mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    #std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    mean = None
    std = None
    
    train_dataloader, val_dataloader = LoadersPreparationBench(
        path_to_data,
        dataset_type='custom',
        test_size=test_size,
        batch_size=batch_size,
        workers=workers,
        device=device,
        mean = None,
        std = None,
        rotate_type = 'fixed90',
        downsample_scale=None,
        blur_sigma=1.5,
        noise_max = 0.2,
        distributed = distributed,
        type="training"
        )
    
    return train_dataloader, val_dataloader

def get_dataloaders_loveDA(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    #mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    #std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    mean = None
    std = None
    
    train_dataloader, val_dataloader = LoadersPreparationBench(
        path_to_data,
        dataset_type='loveda',
        test_size=test_size,
        batch_size=batch_size,
        workers=workers,
        device=device,
        mean = None,
        std = None,
        rotate_type = 'fixed90',
        downsample_scale=None,
        blur_sigma=1.1,
        noise_max = 0.2,
        distributed = distributed,
        type="training"
        )
    
    return train_dataloader, val_dataloader

def get_dataloaders_EuroSAT(batch_size, path_to_data, test_size=0.3, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    #mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    #std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    mean = None
    std = None
    
    train_dataloader, val_dataloader = LoadersPreparationBench(
        path_to_data,
        dataset_type='eurosat_ms',
        test_size=test_size,
        batch_size=batch_size,
        workers=workers,
        device=device,
        mean = None,
        std = None,
        rotate_type = None,# No rotation as using  GT
        downsample_scale=None,
        blur_sigma=1.1,
        noise_max = 0.2,
        distributed = distributed,
        type="training"
        )
    
    return train_dataloader, val_dataloader