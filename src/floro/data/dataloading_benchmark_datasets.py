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
from pyproj import Transformer

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
            e = transform.rotate(e, angle, resize=False, preserve_range=True)
            
            
        elif self.rotation_type == 'fixed90':
            angles = [90, 180, 270]
            angle = np.random.choice(angles)
            x = transform.rotate(x, angle, resize=False)
            t = transform.rotate(t, angle, resize=False, order=0, preserve_range=True)
            e = transform.rotate(e, angle, resize=False, preserve_range=True)
            

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
            # Apply Gaussian blur
            e = filters.gaussian(e, sigma=rn.uniform(0, self.blur_sigma), preserve_range=True)
            # Add Gaussian noise
            e = util.random_noise(e, mode='gaussian', var=rn.uniform(0, self.noise_max)**2)

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
            label = label.transpose((2, 0, 1))  # HWC -> CHW
        # elif label.ndim == 2:
        #     label = label[np.newaxis, :, :]     # HW -> 1 x H x W
        elif label.ndim == 0:
            label = label[np.newaxis]           # Scalar -> [1]
       
        return {'image': torch.from_numpy(image).float(),
                'target': torch.from_numpy(label).unsqueeze(0),
                'elevation': torch.from_numpy(elevation).unsqueeze(0).float(),
                'geo_transform': torch.tensor(gt, dtype=torch.float32)
               }
        
class UnifiedSegmentationDataset(Dataset):
    def __init__(self, root_dir, dataset_type='custom', split='train',
                 transform=None, device='cpu', mean=None, std=None):
        """
        dataset_type: 'custom' (for ImageRSDataset-style data) or 'loveda'
        split: used only for LoveDA ('Train', 'Val', 'Test')
        """
        self.dataset_type = dataset_type.lower()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.device = device
        self.mean = mean
        self.std = std

        if self.dataset_type == 'custom':
            self.image_files = glob.glob(os.path.join(root_dir, '*.tif'))
        elif self.dataset_type == 'loveda':
            # Find all images and labels under both Urban and Rural
            self.image_files = sorted(glob.glob(os.path.join(root_dir, '*', 'images_png', '*.png')))
            self.label_files = sorted(glob.glob(os.path.join(root_dir, '*', 'masks_png', '*.png')))
        elif self.dataset_type == 'eurosat_ms':
            # Folder-per-class structure
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(os.listdir(self.root_dir)))}
            self.image_files = []
            self.labels = []

            for cls_name, idx in self.class_to_idx.items():
                class_dir = os.path.join(root_dir, cls_name)
                for img_file in glob.glob(os.path.join(class_dir, '*.tif')):
                    self.image_files.append(img_file)
                    self.labels.append(idx)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def __len__(self):
        return len(self.image_files)

    def _load_custom(self, idx):
        image_path = self.image_files[idx]
        try:
            with rasterio.open(image_path) as src:
                image = src.read().astype(np.float32) * 0.0001
                geo_transform = src.transform
            image = np.moveaxis(image, 0, -1)  # HWC

            image_ms = image[:, :, :4]
            elevation = image[:, :, 4]
            label_path = image_path.replace("images", "labels").replace("MS-DSM.tif", "label.tif")
            label = io.imread(label_path)

            sample = {
                'image': image_ms,
                'elevation': elevation,
                'target': label,
                'geo_transform': geo_transform.to_gdal()
            }

            return sample
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _load_loveda(self, idx):
        # Load RGB image and normalize
        image = io.imread(self.image_files[idx]).astype(np.float32) / 255.0  # [1024, 1024, 3]
        label = io.imread(self.label_files[idx]).astype(np.uint8)            # [1024, 1024]

        # Add dummy NIR channel (zeros)
        nir = np.zeros_like(image[:, :, 0:1])  # shape: [1024, 1024, 1]
        image_4ch = np.concatenate([image, nir], axis=-1)  # [1024, 1024, 4]

        # Select a random 256x256 tile
        patch_size = 256
        max_y = image_4ch.shape[0] - patch_size
        max_x = image_4ch.shape[1] - patch_size

        top = np.random.randint(0, max_y + 1)
        left = np.random.randint(0, max_x + 1)

        patch_image = image_4ch[top:top + patch_size, left:left + patch_size, :]  # [256, 256, 4]
        patch_label = label[top:top + patch_size, left:left + patch_size]         # [256, 256]

        sample = {
            'image': patch_image.astype(np.float32),                   # [256, 256, 4]
            'elevation': np.zeros((patch_size, patch_size), dtype=np.float32),
            'target': patch_label.astype(np.uint8),
            'geo_transform': np.zeros(6, dtype=np.float32)             # Dummy
        }

        return sample
    
    def _load_eurosat_ms(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]

        try:
            image, geo_transform = self.load_geo_image(image_path, bands=[2,3,4,8,12])# I'll try SWIR as there is no elevation

            image_ms = image[:,:,:4]
            elevation = image[:,:,4]
            image = transform.resize(image_ms, (256, 256), anti_aliasing=True)
            elevation = transform.resize(elevation, (256, 256), anti_aliasing=True)

            sample = {
                'image': image,                # shape: [H, W, 4]
                'elevation': elevation,        # shape: [H, W]
                'target': np.array(label),     # scalar int
                'geo_transform': geo_transform
            }

            return sample

        except Exception as e:
            print(f"Error loading EuroSAT_MS image {image_path}: {e}")
            return None
        
    def load_geo_image(self, image_path, bands=None):
        with rasterio.open(image_path) as src:
            if bands is not None:
                image = src.read(indexes=bands).astype(np.float32) * 0.0001
            else:
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

    def __getitem__(self, idx):
        if self.dataset_type == 'custom':
            sample = self._load_custom(idx)
        elif self.dataset_type == 'loveda':
            sample = self._load_loveda(idx)
        elif self.dataset_type == 'eurosat_ms':
            sample = self._load_eurosat_ms(idx)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        if sample is None:
            return None

        if self.transform:
            sample = self.transform(sample)

        return sample 

# Data loading function
def LoadersPreparationBench(
    root_dir,
    dataset_type='custom',
    test_size=0.1,
    device='cpu',
    mean=None,
    std=None,
    rotate_type='fixed90',
    downsample_scale=0.25,
    blur_sigma=1.0,
    noise_max=0.2,
    type=None,
    
):
    if dataset_type == 'loveda':
        # Use predefined 'Train' and 'Val' splits
        train_dataset = UnifiedSegmentationDataset(root_dir=root_dir, dataset_type='loveda', split='Train',
                                                   transform=transforms.Compose([
                                                       Augmentation(rotation_type=rotate_type,
                                                                    downsample_scale=downsample_scale,
                                                                    blur_sigma=blur_sigma,
                                                                    noise_max=noise_max,
                                                                    type=type),
                                                       ToTensor()
                                                   ]),
                                                   device=device)
        val_dataset = UnifiedSegmentationDataset(root_dir=root_dir, dataset_type='loveda', split='Val',
                                                 transform=transforms.Compose([
                                                     Augmentation(rotation_type=rotate_type,
                                                                  downsample_scale=downsample_scale,
                                                                  blur_sigma=blur_sigma,
                                                                  noise_max=noise_max),
                                                     ToTensor()
                                                 ]),
                                                 device=device)
    elif dataset_type == 'eurosat_ms':
        full_dataset = UnifiedSegmentationDataset(root_dir=root_dir, dataset_type='eurosat_ms',
                                                  transform=None, mean=mean, std=std)
        train_size = int((1 - test_size) * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size])

        train_dataset.dataset.transform = transforms.Compose([
            Augmentation(rotation_type=rotate_type,
                         downsample_scale=downsample_scale,
                         blur_sigma=blur_sigma,
                         noise_max=noise_max,
                         type=type),
            ToTensor()
        ])

        val_dataset.dataset.transform = transforms.Compose([
            Augmentation(rotation_type=rotate_type,
                         downsample_scale=downsample_scale,
                         blur_sigma=blur_sigma,
                         noise_max=noise_max),
            ToTensor()
        ])
    else:
        full_dataset = UnifiedSegmentationDataset(root_dir=root_dir, dataset_type='custom',
                                                  transform=None, mean=mean, std=std)
        train_size = int((1 - test_size) * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size])

        train_dataset.dataset.transform = transforms.Compose([
            Augmentation(rotation_type=rotate_type,
                         downsample_scale=downsample_scale,
                         blur_sigma=blur_sigma,
                         noise_max=noise_max,
                         type=type),
            ToTensor()
        ])

        val_dataset.dataset.transform = transforms.Compose([
            Augmentation(rotation_type=rotate_type,
                         downsample_scale=downsample_scale,
                         blur_sigma=blur_sigma,
                         noise_max=noise_max),
            ToTensor()
        ])

    return train_dataset, val_dataset
    

