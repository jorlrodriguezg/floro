import os
import glob
import torch
import rasterio
import torch.nn as nn
import numpy as np
from skimage import io, transform, exposure
from scipy import ndimage as ndi
from sklearn import preprocessing
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
import math
import random as rn

#np.random.seed(0) 
#torch.manual_seed(0)

class Augmentation(object):
    def __call__(self, sample):
        r = rn.randint(0,100)

        x = sample['image']
        t = sample['target']
        
        if r % 2 == 0:
            angles = [90, 180, 270]
            angle = rn.choice(angles)
            sample['image'] = transform.rotate(x, angle, resize = False) 
            sample['target'] = transform.rotate(t, angle, resize = False)
        
        if r % 2 == 0:
            
            rx = rn.randint(0,10)
            shx = rn.randint(0,4)
            shy = rn.randint(0,4)
            
            if rx % 2 == 0:
                shx *= -1
            ry = rn.randint(0,10)
            if ry % 2 == 0:
                shy *= -1
            
            random_shiftx = (0,shx, shy)
            random_shiftt = (shx, shy)
            rng = np.random.default_rng()
            corrupted_pixels = rng.choice([False, True], size=x.shape, p=[0.01, 0.99])
            # The shift corresponds to the pixel offset relative to the reference image
            #sample['image'] = ndi.shift(x, random_shiftx)
            #sample['target'] = ndi.shift(t, random_shiftt)
            sample['image'] *= corrupted_pixels
            
        return sample     

class random_patch_removal(object):
    def __call__(self, sample):
        x = sample['image']

        # Original image dimensions and window size
        image_size = (x.shape[0], x.shape[0])
        window_size = 4
        
        # Compute the number of windows in each dimension
        windows_x, windows_y = image_size[0] // window_size, image_size[1] // window_size
        # Generate a random mask for whether to remove data in each window
        remove_mask = np.random.rand(windows_x, windows_y) < 0.5  # 50% chance
        # Upscale the mask to the original image size
        upscaled_mask = np.repeat(np.repeat(remove_mask, window_size, axis=0), window_size, axis=1)
        # Apply the mask to the image
        x[upscaled_mask] = 0  # This sets the selected windows to 0

        sample['image'] = x
        return sample
    
class MinMaxScaling(object):
    def __call__(self, sample):
        x = sample['image']
        for i in range(x.shape[2]):
            scaler = preprocessing.MinMaxScaler()
            im = x[:,:,i]
            scaler.fit(im.reshape(-1,1))
            xr = scaler.transform(im.reshape(-1,1))
            x[:,:,i] = xr.reshape(im.shape[0], im.shape[1])
            
        sample['image'] = x
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['target']
        # swap channel axis
        image = image.transpose((2, 0, 1))
        #label = label.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'target': torch.from_numpy(label).unsqueeze(0)}
        
class ImageCHMDataset(Dataset):
    def __init__(self, root_dir, transform=None, device='cpu'):
        self.transform = transform 
        self.image_files = glob.glob(root_dir + '/images/*.tif')
        self.label_files = glob.glob(root_dir + '/labels/*.tif')
        self.device = device
        
    def __len__(self):
        return len(self.image_files)
    
    def _load_label_tile(self, img_tile):
        target_path = img_tile.replace("images","labels")
        target_path = target_path.replace(".tif","_class.tif")
        target = io.imread(target_path)
        return target
    
    def __getitem__(self, idx):
        image = io.imread(self.image_files[idx])
        
        #print(self.image_files[idx])
        
        label = self._load_label_tile(self.image_files[idx])  
        
        sample = {'image': image, 'target': label.astype(np.float32)}        
        return self.transform(sample) if self.transform else sample
    
def prep_loaders(root_dir=None, batch_size=1, test_size=0.1, workers=1, device='cpu'):
    # Load dataset
    image_chm_dataset = ImageCHMDataset(root_dir=root_dir, transform=transforms.Compose([MinMaxScaling(),
                                                                                         Augmentation(),
                                                                                         random_patch_removal(),
                                                                                         ToTensor()]
                                                                                       )
                                       )

    # Split into training and validation sets
    train_size = int((1-test_size) * len(image_chm_dataset))
    test_size = len(image_chm_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(image_chm_dataset, [train_size, test_size])

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    print(f'Dataset size (num. batches): Train -> [{len(train_loader)}] Test -> [{len(valid_loader)}]')
    
    return train_loader, valid_loader