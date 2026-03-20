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
from torchvision.models import resnet18
from tqdm.auto import tqdm
import torch.nn as nn



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
    def __init__(self, img_dir, label, transform=resnetFormat(), range_folds=[0,5], interval=True): #, albumentations = None, gauss = False):
        self.img_dir = img_dir
        self.label = label
        self.transform = transform
        self.range_folds = range_folds
        self.interval = interval
        #self.albumentations = albumentations
        #self.gauss = gauss

        # Taken from buildDsFolds!
        if interval: self.fold_names = [f"{i:02d}" for i in range(range_folds[0], range_folds[1])]
        else: self.fold_names = [f"{i:02d}" for i in range_folds]
        n_folds = len(self.fold_names)

        self.samples = []

        # Tracks progress of reading multiple folds
        for fold in tqdm(self.fold_names, desc=f"Building {img_dir} dataset with {n_folds} folds"):
            fold_dir = os.path.join(img_dir, fold)

            if not os.path.isdir(fold_dir): raise RuntimeError(f"Fold directory does not exist: '{fold_dir}'")

            # Assertion of Correct Read (order is quite important for wiki vs inpainting)
            filenames = sorted(f for f in os.listdir(fold_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
            if len(filenames) == 0: raise RuntimeError(f"No images found in '{fold_dir}' with SUPPORTED_EXTENSIONS {SUPPORTED_EXTENSIONS}")

            for fname in filenames: self.samples.append((os.path.join(fold_dir, fname), fold, fname))


        # Sanity check for correct read and build
        if len(self.samples) == 0: raise RuntimeError(f"No images found in '{img_dir}' across folds {self.fold_names}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, fold, fname = self.samples[idx]
        image = Image.open(path).convert("RGB")  # shape: (H, W, 3) 

        # formatting for resnet
        if self.transform: image = self.transform(image)

        return image, self.label, fold, fname

    # WARNING: returns the np.array representation of the image
    def show(self, idx, plot=True):
        image, label, fold, fname = self[idx]

        if plot:
            if isinstance(image, torch.Tensor):
                img_plot = image.detach().cpu().permute(1, 2, 0).numpy()
                plt.imshow(img_plot)
            else: plt.imshow(np.array(image))

            plt.title(f"label={label}, fold={fold}, file={fname}")
            plt.show()

        if isinstance(image, torch.Tensor): print(f"img_shape={image.shape}, img_type={type(image)}, label={label}, fold={fold}, file={fname}")
        else: print(f"img_shape={np.array(image).shape}, img_type={type(image)}, label={label}, fold={fold}, file={fname}")

        return image

""" Explaining buildDsFolds...

Here I basically wanted to have a dataset per subfolder, for simplicity. That would allow us to 
choose specific folds for training, which is good for:
    - cross validation
    - set size control


Another big reason is that WIKI and INPAINTING are direct counterparts of deepfakes, 
it's good to test multiple difficulty levels for the model:
    - easy: model trained wiki vs direct_counterpart and tested on another subset of wiki vs direct_counterpart
    - medium: model trained 80% on wiki vs direct_counterpart but also unrelated wiki and inpainting instances
    - hard: model trained on 50% wiki vs direct_counterpart but also unrelated wiki and inpainting instances


The problem with this is that concatenating does transform it into a single dataset for all purposes
EXCEPT retaining the inner functions, such as show and etc. Each instance retains properties, it's 
just that the class's inner functions are not preserved in the same way. For that reason, this function
is obsolete but really did help develop the most importnat bit of reading logic of the dataset.

"""

def buildDsFolds(img_dir, range_folds=[0,5], label=0, transform=resnetFormat(), interval=True):
    if interval: fold_names = [f"{i:02d}" for i in range(range_folds[0], range_folds[1])]
    else: fold_names = [f"{i:02d}" for i in range_folds]
    n_folds = len(fold_names)

    datasets = []
    for fold in tqdm(fold_names, desc=f"Building {img_dir} dataset with {n_folds} folds"):
        datasets.append(DeepFakeDataset(os.path.join(img_dir, fold), label=label, transform=transform))

    return ConcatDataset(datasets)


# Used when concatenation nukes the inner functions
def showResNetVision(loader):
    images, labels, folds, fnames = next(iter(loader))

    print("batch image tensor shape:", images.shape)
    print("batch label tensor shape:", labels.shape if hasattr(labels, "shape") else type(labels))
    print("first sample:", labels[0].item() if torch.is_tensor(labels[0]) else labels[0], folds[0], fnames[0])

    img0_tensor = images[0].detach().cpu()
    img0_plot = img0_tensor.permute(1, 2, 0).numpy()

    print("resnet sees:", img0_tensor.shape, "-> [C, H, W]")
    print("matplotlib sees:", img0_plot.shape, "-> [H, W, C]")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(img0_plot)
    axes[0].set_title("Human view\n[H, W, C]")
    axes[0].axis("off")

    axes[1].imshow(img0_tensor[0].numpy(), cmap="gray")
    axes[1].set_title(f"Channel 0\nshape={img0_tensor[0].shape}")
    axes[1].axis("off")

    axes[2].imshow(img0_tensor[1].numpy(), cmap="gray")
    axes[2].set_title(f"Channel 1\nshape={img0_tensor[1].shape}")
    axes[2].axis("off")

    axes[3].imshow(img0_tensor[2].numpy(), cmap="gray")
    axes[3].set_title(f"Channel 2\nshape={img0_tensor[2].shape}")
    axes[3].axis("off")

    axes[4].text(
        0.05, 0.5,
        f"ResNet input order:\n\n"
        f"image[0] = Red channel\n"
        f"image[1] = Green channel\n"
        f"image[2] = Blue channel\n\n"
        f"Tensor shape:\n{tuple(img0_tensor.shape)}",
        fontsize=12,
        va="center"
    )
    axes[4].axis("off")

    plt.suptitle(
        f"label={labels[0].item() if torch.is_tensor(labels[0]) else labels[0]}, "
        f"fold={folds[0]}, file={fnames[0]}"
    )
    plt.tight_layout()
    plt.show()
    