"""
Image loading shared by the training and evaluation scripts.

These helpers were previously duplicated across train.py, evaluate.py and
eval_ablation.py in three slightly different forms.
"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(root, limit=None):
    """Sorted image filenames in `root`. Sorting keeps subset selection
    deterministic across runs and machines."""
    files = sorted(f for f in os.listdir(root) if f.lower().endswith(IMAGE_EXTS))
    return files[:limit] if limit else files


def load_image(root, fname, device):
    """Load one image as a (1, 3, H, W) float tensor in [0, 1]."""
    img = Image.open(os.path.join(root, fname)).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


def load_image_path(path, device):
    """As `load_image`, but from a full path."""
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


class FlatFolderDataset(Dataset):
    """Flat directory of images, no class subfolders.

    Yields a tensor by default; set `with_name=True` to yield (tensor, filename).
    """

    def __init__(self, root_dir, transform=None, with_name=False):
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        self.with_name = with_name
        self.image_files = sorted(
            f for f in os.listdir(root_dir) if f.lower().endswith(IMAGE_EXTS)
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        image = Image.open(os.path.join(self.root_dir, name)).convert("RGB")
        image = self.transform(image)
        return (image, name) if self.with_name else image
