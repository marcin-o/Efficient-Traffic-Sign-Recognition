import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MEAN = [0.3403, 0.3121, 0.3214]
STD  = [0.2724, 0.2608, 0.2669]

def get_transforms(train=True, size=32):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
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


def _denormalize(imgs):
    mean = torch.tensor(MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=imgs.device).view(1, 3, 1, 1)
    return imgs * std + mean


def _normalize(imgs):
    mean = torch.tensor(MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=imgs.device).view(1, 3, 1, 1)
    return (imgs - mean) / std


def apply_corruption(imgs, corruption="gaussian_noise", severity=0.1):
    px = _denormalize(imgs).clamp(0, 1)
    if corruption == "gaussian_noise":
        px = px + torch.randn_like(px) * severity
    elif corruption == "blur":
        import torchvision.transforms.functional as F
        px = F.gaussian_blur(px, kernel_size=3)
    elif corruption == "brightness":
        px = px * (1 + severity)
    px = px.clamp(0, 1)
    return _normalize(px)

