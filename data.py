import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MEAN = [0.3403, 0.3121, 0.3214]
STD  = [0.2724, 0.2608, 0.2669]

def get_transforms(train=True, size=32):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(data_root, batch_size=64, num_workers=4, size=32):
    train_set = datasets.GTSRB(data_root, split="train", download=True,
                                transform=get_transforms(True, size))
    test_set  = datasets.GTSRB(data_root, split="test",  download=True,
                                transform=get_transforms(False, size))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def apply_corruption(imgs, corruption="gaussian_noise", severity=0.1):
    noisy = imgs.clone()
    if corruption == "gaussian_noise":
        noisy += torch.randn_like(noisy) * severity
    elif corruption == "blur":
        import torchvision.transforms.functional as F
        k = 3
        noisy = torch.stack([F.gaussian_blur(img, kernel_size=k) for img in noisy])
    elif corruption == "brightness":
        noisy = torch.clamp(noisy * (1 + severity), 0, 1)
    return noisy

