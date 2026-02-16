import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class GlomeruliDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        # Augmentations
        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        mask = (mask > 0).astype("float32")

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask