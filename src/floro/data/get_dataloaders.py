import torch
from floro.data.dataloading_ss import LoadersPreparation as LoadersPreparation
from floro.data.self_supervised_dataloader import LoadersPreparation as LoadersPreparationGeo
from floro.data.floro_self_supervised_dataloader import LoadersPreparation as LoadersPreparationPretraining

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

def get_dataloaders_pretraining(path_to_data, test_size=0.2,  task=None,  **kwargs):
    """
        image_mean_8: [0.2073523  0.2670019  0.28107855 0.24270512 0.2793084  0.14631674  0.20326711  0.14910233]
        image_std_8 : [0.18876277 0.21284738 0.20284666 0.15357633 0.12527913 0.17737003  0.11302987  0.09519225]
        mod_mean_3  : [2094.924  -11.424427  -18.914251]
        mod_std_3   : [1678.725  3.4653134  4.3036957]
        Optical range: 0.0 1.0
        Elev range   : -282.0 9000.0
        SAR range    : -39.997310638427734 19.99913787841797
    """
    
    train_dataset, val_dataset = LoadersPreparationPretraining(
        path_to_data,
        test_size=test_size,
        img_mean=[0.2073523, 0.2670019, 0.28107855, 0.24270512, 0.2793084, 0.14631674, 0.20326711, 0.14910233],
        img_std=[0.18876277, 0.21284738, 0.20284666, 0.15357633, 0.12527913, 0.17737003,  0.11302987,  0.09519225],
        mod_mean=[2094.924, -11.424427, -18.914251],
        mod_std=[1678.725, 3.4653134, 4.3036957],
        rotate_type = "fixed90",
        blur_sigma=1.1,
        noise_max = 0.2,
        band_drop_kwargs=kwargs.get("band_drop_kwargs"),
        )
    
    return train_dataset, val_dataset

