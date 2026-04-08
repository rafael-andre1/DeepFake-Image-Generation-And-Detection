import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Função de transformações para DCGAN - Devolve uma sequência de trasnformações aplicadas a cada imagem
def get_dcgan_transform(image_size=64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

# Usado para criar um Dataset Personalizado
class WikiGANDataset(Dataset):
    """
    Estrutura esperada:
        ../deepfake_data/wiki/00/*.jpg
        ../deepfake_data/wiki/01/*.jpg
        ...

    Devolve apenas a imagem transformada.
    """
    def __init__(self, img_dir, transform=None, range_folds=(0, 5), interval=True):
        self.img_dir = img_dir
        self.transform = transform

        if interval:
            self.fold_names = [f"{i:02d}" for i in range(range_folds[0], range_folds[1])]
        else:
            self.fold_names = [f"{i:02d}" for i in range_folds]

        self.samples = []

        for fold in self.fold_names:
            fold_dir = os.path.join(img_dir, fold)

            if not os.path.isdir(fold_dir):
                raise RuntimeError(f"Fold directory does not exist: '{fold_dir}'")

            filenames = sorted(
                f for f in os.listdir(fold_dir)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            )

            if len(filenames) == 0:
                raise RuntimeError(
                    f"No images found in '{fold_dir}' with supported extensions {SUPPORTED_EXTENSIONS}"
                )

            for fname in filenames:
                self.samples.append(os.path.join(fold_dir, fname))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in '{img_dir}' across folds {self.fold_names}"
            )

    # Usado para saber quantas amostras tem o dataset
    def __len__(self):
        return len(self.samples)

    # Carrega a imagem e aplica-lhe as transformações
    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    # Método apenas para a visualização da imagem
    def show(self, idx):
        import matplotlib.pyplot as plt
        image = Image.open(self.samples[idx]).convert("RGB")
        plt.imshow(image)
        plt.title(os.path.basename(self.samples[idx]))
        plt.axis("off")
        plt.show()

# Esta função é usada para criar e devolver o Dataset e o DataLoader
def build_wiki_gan_dataloader(
    img_dir,
    image_size=64,
    batch_size=128,
    num_workers=2,
    range_folds=(0, 5),
    interval=True,
    shuffle=True,
    drop_last=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    transform = get_dcgan_transform(image_size=image_size)

    dataset = WikiGANDataset(
        img_dir=img_dir,
        transform=transform,
        range_folds=range_folds,
        interval=interval
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataset, dataloader