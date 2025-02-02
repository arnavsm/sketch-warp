import os
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from functools import lru_cache


class WarpDataset(Dataset):
    def __init__(
            self, 
            csv_path, 
            data_path, 
            mode, 
            mix_within_class=False, 
            num_photo_per_class=90
    ):
        self.csv = pd.read_csv(csv_path)
        self.data_path = data_path
        self.mode = mode
        self.mix_within_class = mix_within_class
        self.unique_photo_ids = self.csv["photo"].unique()

        self.trans = transforms.ToTensor()
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.num_photo_per_class = num_photo_per_class

    def __len__(self):
        return len(self.unique_photo_ids) if self.mode != "test" else len(self.csv)

    @lru_cache(maxsize=None)
    def _load_image(self, path):
        """Load and cache an image from disk."""
        return np.array(Image.open(os.path.join(self.data_path, path)))

    def _load_train_pair(self, idx):
        photo_id = self.unique_photo_ids[idx]
        row = self.csv[self.csv["photo"] == photo_id].sample(n=1).iloc[0]

        photo = self._load_image(row["photo"])
        sketch = self._load_image(row["sketch"])
        class_idx = row["class"]
        flip = row["flip"]
        return photo, sketch, class_idx, flip

    def _load_eval_pair(self, idx):
        photo_id = self.unique_photo_ids[idx]
        row = self.csv[self.csv["photo"] == photo_id].sample(n=1).iloc[0]

        photo = self._load_image(row["photo"])
        sketch = self._load_image(row["sketch"])
        class_idx = row["class"]
        return photo, sketch, class_idx, 0

    def _load_test_pair(self, idx):
        row = self.csv.iloc[idx]

        photo = self._load_image(row["photo"])
        sketch = self._load_image(row["sketch"])

        y1 = np.array([float(v) for v in row["XA"].split(";")])
        x1 = np.array([float(v) for v in row["YA"].split(";")])
        y2 = np.array([float(v) for v in row["XB"].split(";")])
        x2 = np.array([float(v) for v in row["YB"].split(";")])

        kp1 = np.stack([x1, y1], axis=1)
        kp2 = np.stack([x2, y2], axis=1)
        return photo, sketch, kp1, kp2

    def __getitem__(self, idx):
        if self.mode == "train":
            return self._get_train_item(idx)
        elif self.mode == "eval":
            return self._get_eval_item(idx)
        elif self.mode == "test":
            return self._get_test_item(idx)
        else:
            raise NotImplementedError

    def _get_train_item(self, idx):
        if self.mix_within_class:
            photo, sketch, class_idx, flip = self._load_train_pair(idx)
            new_idx = (class_idx - 1) * self.num_photo_per_class + np.random.randint(
                self.num_photo_per_class
            )
            _, sketch, _, _ = self._load_train_pair(new_idx)
        else:
            photo, sketch, class_idx, flip = self._load_train_pair(idx)

        photo = self.trans(photo)
        sketch = self.trans(sketch)

        if flip == 1:
            photo = TF.hflip(photo)
            sketch = TF.hflip(sketch)

        return photo, sketch

    def _get_eval_item(self, idx):
        photo, sketch, class_idx, _ = self._load_eval_pair(idx)

        photo = self.norm(self.trans(photo))
        sketch = self.norm(self.trans(sketch))

        return photo, sketch

    def _get_test_item(self, idx):
        photo, sketch, kp1, kp2 = self._load_test_pair(idx)

        photo = self.norm(self.trans(photo))
        sketch = self.norm(self.trans(sketch))

        return photo, sketch, kp1, kp2
