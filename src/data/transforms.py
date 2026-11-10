from torchvision import transforms


def build_transforms(image_size=224, train=False, augment=True):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if train and augment:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(), norm,
        ]
    else:
        ops = [transforms.Resize((image_size, image_size)), transforms.ToTensor(), norm]
    return transforms.Compose(ops)
