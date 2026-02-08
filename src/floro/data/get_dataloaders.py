# Import necessary modules
import torch
from floro.data.dataloading_ss import LoadersPreparation as LoadersPreparation
from floro.data.self_supervised_dataloader import LoadersPreparation as LoadersPreparationGeo
# Add imports for your other dataloaders here

def get_dataloaders_seg(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    
    train_dataloader, val_dataloader = LoadersPreparation(
        path_to_data,
        batch_size=batch_size,
        test_size=test_size,
        workers=workers,
        device=device,
        mean = mean,
        std = std,
        rotate_type = None,
        downsample_scale=None,
        blur_sigma=1.0,
        noise_max = 0.1,
        distributed = distributed
        )
    
    return train_dataloader, val_dataloader

def get_dataloaders_geo(batch_size, path_to_data, test_size=0.2, workers=4, device='cuda',  task=None, distributed=None, **kwargs):
    
    mean = torch.tensor([0.1690, 0.2079, 0.2433, 0.3591, 0.0900])
    std = torch.tensor([0.0657, 0.0774, 0.1140, 0.1205, 0.0984])
    
    train_dataloader, val_dataloader = LoadersPreparationGeo(
        path_to_data,
        batch_size=batch_size,
        test_size=test_size,
        workers=workers,
        device=device,
        mean = mean,
        std = std,
        rotate_type = None,
        downsample_scale=None,
        blur_sigma=1.0,
        noise_max = 0.1,
        distributed = distributed
        )
    
    return train_dataloader, val_dataloader
