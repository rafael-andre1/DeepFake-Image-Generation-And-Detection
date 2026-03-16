from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image
import random
import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

def resnetSizeFormat(target_size=(224, 224), grayscale_to_rgb=False):
    # Transforms to tensor & resizes for faster computation 
    pipeline = [transforms.ToTensor(), transforms.Resize(target_size)]    
    return transforms.Compose(pipeline)


class DeepFakeDataset(Dataset):
    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(self,img_dir,label,transform = None, albumentations = None, gauss = False):
        self.img_dir = img_dir
        self.label = label
        self.transform = transform
        self.albumentations = albumentations
        self.gauss = gauss

        # Assertion of Correct Read
        self.filenames = sorted(f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
        if len(self.filenames) == 0: raise RuntimeError(f"No images found in '{img_dir}' with SUPPORTED_EXTENSIONS {SUPPORTED_EXTENSIONS}")

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        image = np.array(Image.open(path).convert("RGB"))  # shape: (H, W, 3) 

        # ---- Torchvision transforms (ToTensor, Resize, etc.) --------------
        if self.transform: image = self.transform(image)

        return image, self.label
