import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

from pangaea.datasets.base import RawGeoFMDataset


class EspeletiaSceneClassification(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int | bool,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list,
        data_mean: dict[str, list[float]],
        data_std: dict[str, list[float]],
        data_min: dict[str, list[float]],
        data_max: dict[str, list[float]],
        download_url: str,
        auto_download: bool,
    ):
        super().__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = Path(root_path)
        self.split = split

        split_csv = self.root_path / split / "labels.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")

        self.df = pd.read_csv(split_csv)

        required_cols = ["patch_id", "label", "ms_path", "chm_path"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in {split_csv}")

        self.samples = self.df.to_dict("records")

    def __len__(self):
        return len(self.samples)

    def _read_tif(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
            geo_transform = src.transform
        return arr, geo_transform.to_gdal()

    def __getitem__(self, index: int):
        sample = self.samples[index]

        ms_path = self.root_path / sample["ms_path"]
        chm_path = self.root_path / sample["chm_path"]

        if not ms_path.exists():
            raise FileNotFoundError(f"MS patch not found: {ms_path}")
        if not chm_path.exists():
            raise FileNotFoundError(f"chm patch not found: {chm_path}")

        ms, gt = self._read_tif(ms_path)
        chm, _ = self._read_tif(chm_path)

        ms = ms.astype(np.float32)
        chm = chm.astype(np.float32)

        ms = torch.from_numpy(ms).float()
        chm = torch.from_numpy(chm).float()
        gt = torch.tensor(gt, dtype=torch.float32)

        # Pangaea expects C T H W
        ms = ms.unsqueeze(1)   # [6, 1, H, W]
        chm = chm.unsqueeze(0) # [1, 1, H, W]
        

        target = int(sample["label"])
        target = torch.tensor(target, dtype=torch.long)

        output = {
            "image":{
                "optical": ms,     # (C_ms, 1, H, W)
                "dem": chm,        # (1, 1, H, W)
            },
            "target": target,
            "metadata": {
                "gt": gt,
                "patch_id": sample.get("patch_id", str(index)),
                "polygon_id": sample.get("polygon_id", -1),
                "description": sample.get("description", ""),
                "ms_path": str(ms_path),
                "chm_path": str(chm_path),
            }
        }
        return output