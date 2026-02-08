import os
import glob
import torch
import rasterio
import torch.nn as nn
import numpy as np
from skimage import io, transform, exposure, filters, util
from scipy import ndimage as ndi
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.measure import shannon_entropy
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
import math
import random as rn

np.random.seed(0) 
torch.manual_seed(0)


# Augmentation class
class Augmentation(object):
    def __init__(self, rotation_type='free', downsample_scale=0.5, blur_sigma=1, noise_max=0.02, type=None):
        self.rotation_type = rotation_type
        self.downsample_scale = downsample_scale
        self.blur_sigma = blur_sigma
        self.noise_max = noise_max
        self.type = type

    def __call__(self, sample):
        x = sample['image']
        e = sample['elevation']
        t = sample['target']

        # Apply rotation
        if self.rotation_type == 'free':
            angle = np.random.uniform(0, 360)
            x = transform.rotate(x, angle, resize=False)
            t = transform.rotate(t, angle, resize=False, order=0, preserve_range=True)
            e = transform.rotate(e, angle, resize=False)
            
            
        elif self.rotation_type == 'fixed90':
            angles = [90, 180, 270]
            angle = np.random.choice(angles)
            x = transform.rotate(x, angle, resize=False)
            t = transform.rotate(t, angle, resize=False, order=0, preserve_range=True)
            e = transform.rotate(e, angle, resize=False)
            

        if self.downsample_scale is not None:
            # Downsample
            height, width = x.shape[:2]
            new_height, new_width = int(height * self.downsample_scale), int(width * self.downsample_scale)
            x = transform.resize(x, (new_height, new_width))

        if self.type == "training":
            # Apply Gaussian blur
            x = filters.gaussian(x, sigma=rn.uniform(0, self.blur_sigma), preserve_range=True)
            # Add Gaussian noise
            x = util.random_noise(x, mode='gaussian', var=rn.uniform(0, self.noise_max)**2)

        sample['image'] = x
        sample['elevation'] = e
        sample['target'] = t
        
        return sample

# ToTensor class
class ToTensor(object):
    def __call__(self, sample):
        image, label, elevation, gt = sample['image'], sample['target'], sample['elevation'], sample['geo_transform']
        
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        if label.ndim == 3:
            label = label.transpose((2, 0, 1))
            
        return {'image': torch.from_numpy(image).float(),
                'target': torch.from_numpy(label).unsqueeze(0),
                'elevation': torch.from_numpy(elevation).unsqueeze(0).float(),
                'geo_transform': torch.tensor(gt, dtype=torch.float32)
               }
        
# ImageRSDataset class
class ImageRSDataset(Dataset):
    def __init__(self, root_dir,
                 transform=None,
                 mean=None,
                 std=None,
                 device='cpu'
                ):
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root_dir, '*.tif'))
        self.device = device
        if mean is not None:
            self.mean = mean.tolist()
        if std is not None:
            self.std = std.tolist()

    def __len__(self):
        return len(self.image_files)
    
    def _load_label_tile(self, img_tile):
        target_path = img_tile.replace("images","labels")
        target_path = target_path.replace("MS-DSM.tif","label.tif")
        target = io.imread(target_path)
        return target


    def __getitem__(self, idx):
        try:
            # Load and scale image
            image, gt = self.load_image(self.image_files[idx])
            
            image_ms = image[:,:,:4]
            elevation = image[:,:,4]
            
            label = self._load_label_tile(self.image_files[idx])

            sample = {'image': image_ms,
                      'elevation': elevation,
                      'target': label,
                      'geo_transform':gt}

            if self.transform:
                sample = self.transform(sample)

                sample['image'] = sample['image'].to(self.device)
                sample['elevation'] = sample['elevation'].to(self.device)
                sample['target'] = sample['target'].to(self.device)
                sample['geo_transform'] = sample['geo_transform'].to(self.device)

        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {e}")
            return None

        return sample

    def load_image(self, image_path):
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32) * 0.0001  # Scale the image
            geo_transform = src.transform
        image = np.moveaxis(image, 0, -1)  # Move the channels to the last dimension
        #image = (image - self.mean) / self.std  # Normalize the image

        return image, geo_transform.to_gdal()

# Data loading function
def LoadersPreparation(root_dir=None,
                       batch_size=1,
                       test_size=0.1,
                       workers=1,
                       device='cpu',
                       mean=None,
                       std=None,
                       rotate_type='fixed90',
                       downsample_scale=0.25,
                       blur_sigma=1.0,
                       noise_max=0.2,
                       distributed = None,
                       type = None,
                       droplast = True
                      ):
    
    image_rs_dataset = ImageRSDataset(root_dir=root_dir,
                                      transform=None,
                                      mean=mean,
                                      std=std
                                      )

    # Split into training and validation sets
    train_size = int((1 - test_size) * len(image_rs_dataset))
    test_size = len(image_rs_dataset) - train_size
    train_dataset, test_dataset = random_split(image_rs_dataset, [train_size, test_size])

    # Define transforms
    train_transform = transforms.Compose([
        Augmentation(rotation_type=rotate_type, downsample_scale=downsample_scale, blur_sigma=blur_sigma, noise_max=noise_max,type = type),
        ToTensor()
    ])

    test_transform = transforms.Compose([
        Augmentation(rotation_type=rotate_type, downsample_scale=downsample_scale, blur_sigma=blur_sigma, noise_max=noise_max),
        ToTensor()
    ])

    # Apply transforms dynamically
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    if distributed == 'distributed':
        return train_dataset, test_dataset
    else:
        # Prepare data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=droplast, num_workers=workers)
        valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=droplast, num_workers=workers)
        print(f'Dataset size (num. batches): Train -> [{len(train_loader)}] Test -> [{len(valid_loader)}]')
    
        return train_loader, valid_loader
