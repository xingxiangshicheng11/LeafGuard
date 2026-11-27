from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import LeafDiseaseDataset
from .transforms import build_transforms


def build_dataloader(
    manifest_csv,
    project_root=".",
    image_size=224,
    train=False,
    augment=True,
    batch_size=32,
    num_workers=4,
    shuffle=None,
):
    if shuffle is None:
        shuffle = train
    transform = build_transforms(image_size=image_size, train=train, augment=augment)
    dataset = LeafDiseaseDataset(
        manifest_csv, project_root=project_root, transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_dataloaders(
    split_dir="splits",
    project_root=".",
    image_size=224,
    batch_size=32,
    num_workers=4,
    augment=True,
):
    split_dir = Path(split_dir)
    return {
        "train": build_dataloader(
            split_dir / "train.csv",
            project_root=project_root,
            image_size=image_size,
            train=True,
            augment=augment,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        "val": build_dataloader(
            split_dir / "val.csv",
            project_root=project_root,
            image_size=image_size,
            train=False,
            augment=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        "test": build_dataloader(
            split_dir / "test.csv",
            project_root=project_root,
            image_size=image_size,
            train=False,
            augment=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    }
