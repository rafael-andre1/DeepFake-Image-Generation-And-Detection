from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image
import random
import torch
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

def resnetFormat(target_size=(224, 224), grayscale_to_rgb=False):
    # Transforms to tensor & resizes for faster computation 
    pipeline = [transforms.ToTensor(), transforms.Resize(target_size)]    
    return transforms.Compose(pipeline)


# To load and preserve order:

"""
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,      # preserves file order
    num_workers=0,      # simplest / deterministic
    pin_memory=torch.cuda.is_available()
)
"""

class DeepFakeDataset(Dataset):
    def __init__(self,img_dir,label,transform = None): #, albumentations = None, gauss = False):
        self.img_dir = img_dir
        self.label = label
        self.transform = transform
        self.fold = os.path.basename(os.path.normpath(img_dir))
        #self.albumentations = albumentations
        #self.gauss = gauss

        # Assertion of Correct Read (order is quite important for wiki vs inpainting)
        self.filenames = sorted(f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
        if len(self.filenames) == 0: raise RuntimeError(f"No images found in '{img_dir}' with SUPPORTED_EXTENSIONS {SUPPORTED_EXTENSIONS}")

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        image = np.array(Image.open(path).convert("RGB"))  # shape: (H, W, 3) 

        # formatting for resnet
        if self.transform: image = self.transform(image)

        return image, self.label, self.fold, self.filenames[idx]

    # WARNING: returns the np.array representation of the image
    def show(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        image = np.array(Image.open(path).convert("RGB"))  # shape: (H, W, 3) 
        plt.imshow(image)
        plt.title(f"label={self.label}, fold={self.fold}, file={self.filenames[idx]}")
        plt.show()

        print(f"label={self.label}, fold={self.fold}, file={self.filenames[idx]}")

        return image

""" Rationale

Here I basically wanted to have a dataset per subfolder, for simplicity. That would allow us to 
choose specific folds for training, which is good for:
    - cross validation
    - set size control



Another big reason is that WIKI and INPAINTING are direct counterparts of deepfakes, 
it's good to test multiple difficulty levels for the model:
    - easy: model trained wiki vs direct_counterpart and tested on another subset of wiki vs direct_counterpart
    - medium: model trained 80% on wiki vs direct_counterpart but also unrelated wiki and inpainting instances
    - hard: model trained on 50% wiki vs direct_counterpart but also unrelated wiki and inpainting instances


"""

def buildDsFolds(img_dir, range_folds=[0,5], label=0, transform=resnetFormat(), interval=True):
    if interval: fold_names = [f"{i:02d}" for i in range(range_folds[0], range_folds[1])]
    else: fold_names = [f"{i:02d}" for i in range_folds]
    n_folds = len(fold_names)

    datasets = []
    for fold in tqdm(fold_names, desc=f"Building {img_dir} dataset with {n_folds} folds"):
        datasets.append(DeepFakeDataset(os.path.join(img_dir, fold), label=label, transform=transform))

    return ConcatDataset(datasets)
    