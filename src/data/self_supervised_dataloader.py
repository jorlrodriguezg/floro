import os
import glob
import torch
import rasterio
import torch.nn as nn
import numpy as np
from skimage import io, transform, filters, util
#from scipy import ndimage as ndi
# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from pyproj import Transformer
#from sklearn.cluster import KMeans
#from skimage.measure import shannon_entropy
from torchvision import transforms
#from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
import math
import random as rn

np.random.seed(0) 
torch.manual_seed(0)

def compute_mean_std(dataset, batch_size=32, num_workers=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize sums
    mean = torch.zeros(5)  # Assuming 4 channels (R, G, B, NIR)
    std = torch.zeros(5)
    num_batches = 0
    
    for batch in loader:
        
        data = batch['image']#scbatch['image']  # Assuming 'image' contains the image tensor
        for channel in range(data.size(3)):
            mean[channel] += np.nanmean(data[:, :, :, channel])#.mean()
            std[channel] += np.nanstd(data[:, :, :, channel])#.std()
        
        num_batches += 1
        #print(mean)
    
    # Divide by the number of samples to get the mean of means and std of stds
    mean /= num_batches
    std /= num_batches

    return mean, std

def get_mean_std_from_dataset(root_dir=None, batch_size=1, workers=1, device='cpu'):
    image_rs_dataset = DatasetMeanStdCalc(root_dir=root_dir, transform=None)

    # Compute mean and std
    mean, std = compute_mean_std(image_rs_dataset, batch_size=batch_size, num_workers=workers)
    print(f'Dataset Mean: {mean}')
    print(f'Dataset Std: {std}')
    
    return mean, std

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
        t = sample['target']
        e = sample['elevation']
        te = sample['target_elev']

        # Apply rotation
        if self.rotation_type == 'free':
            angle = np.random.uniform(0, 360)
            x = transform.rotate(x, angle, resize=False)
            t = transform.rotate(t, angle, resize=False)
            e = transform.rotate(e, angle, resize=False)
            te = transform.rotate(te, angle, resize=False)
            
        elif self.rotation_type == 'fixed90':
            angles = [90, 180, 270]
            angle = np.random.choice(angles)
            x = transform.rotate(x, angle, resize=False)
            t = transform.rotate(t, angle, resize=False)
            e = transform.rotate(e, angle, resize=False)
            te = transform.rotate(te, angle, resize=False)

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
        sample['target'] = t
        sample['elevation'] = e
        sample['target_elev'] = te
        
        return sample

# ToTensor class
class ToTensor(object):
    def __call__(self, sample):
        image, label, elevation, label_elev, gt = sample['image'], sample['target'], sample['elevation'], sample['target_elev'], sample['geo_transform']
        
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        if label.ndim == 3:
            label = label.transpose((2, 0, 1))
            
        return {'image': torch.from_numpy(image).float(),
                'target': torch.from_numpy(label).float(),
                'elevation': torch.from_numpy(elevation).unsqueeze(0).float(),
                'target_elev': torch.from_numpy(label_elev).unsqueeze(0).float(),
                'geo_transform': torch.tensor(gt, dtype=torch.float32)
               }

# ImageRSDataset class
class DatasetMeanStdCalc(Dataset):
    def __init__(self, root_dir, transform=None, device='cpu'):
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root_dir, '/images/*.tif'))
        self.device = device

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load and scale image
            image = io.imread(self.image_files[idx]).astype(np.float32) * 0.0001
            
            sample = {'image': image}

            if self.transform:
                sample = self.transform(sample)
                sample['image'] = sample['image'].to(self.device)
                
        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {e}")
            return None
        return sample

        
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
        self.mean = mean.tolist()
        self.std = std.tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load and scale image
            image, gt = self.load_image(self.image_files[idx])
            
            image_ms = image[:,:,:4]
            elevation = image[:,:,4]
            
            label = image_ms.copy()
            label_elev = elevation.copy()

            sample = {'image': image_ms,
                      'elevation': elevation,
                      'target': label,
                      'target_elev': label_elev,
                      'geo_transform':gt}

            if self.transform:
                sample = self.transform(sample)

                sample['image'] = sample['image'].to(self.device)
                sample['elevation'] = sample['elevation'].to(self.device)
                sample['target'] = sample['target'].to(self.device)
                sample['target_elev'] = sample['target_elev'].to(self.device)
                sample['geo_transform'] = sample['geo_transform'].to(self.device)

        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {e}")
            return None

        return sample

    # def load_image(self, image_path):
    #     with rasterio.open(image_path) as src:
    #         image = src.read().astype(np.float32) * 0.0001  # Scale the image -> Assuming reflectance scaled [0,10000]
    #         geo_transform = src.transform
    #     image = np.moveaxis(image, 0, -1)  # Move the channels to the last dimension
    #     #image = (image - self.mean) / self.std  # Normalize the image

    #     return image, geo_transform.to_gdal()
    def load_image(self, image_path):
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32) * 0.0001
            orig_transform = src.transform
            crs = src.crs
            
            # Reproject geotransform to EPSG:3857
            transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
            
            # Get origin and pixel size in original CRS
            origin_x = orig_transform.c
            origin_y = orig_transform.f
            pixel_width = orig_transform.a
            pixel_height = orig_transform.e  # Typically negative

            # Convert origin to Web Mercator
            origin_x_3857, origin_y_3857 = transformer.transform(origin_x, origin_y)
            
            # Calculate pixel dimensions in meters
            next_x = origin_x + pixel_width
            next_y = origin_y + pixel_height
            next_x_3857, next_y_3857 = transformer.transform(next_x, next_y)
            
            pixel_width_m = next_x_3857 - origin_x_3857
            pixel_height_m = next_y_3857 - origin_y_3857

            geo_transform = [
                origin_x_3857,   # top-left x (Easting)
                pixel_width_m,   # w-e pixel resolution
                0,               # rotation, 0 if image is "north up"
                origin_y_3857,   # top-left y (Northing)
                0,               # rotation, 0 if image is "north up"
                -abs(pixel_height_m)  # n-s pixel resolution (negative for top-down)
            ]

        image = np.moveaxis(image, 0, -1) # Move the channels to the last dimension
        return image, geo_transform

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
                       type = None
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
        valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)
        print(f'Dataset size (num. batches): Train -> [{len(train_loader)}] Test -> [{len(valid_loader)}]')
    
        return train_loader, valid_loader
