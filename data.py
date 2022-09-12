import random
from typing import Optional, Callable, Tuple, Any

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as F


class MultiResCIFAR10(CIFAR10):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 resolution_range: Tuple[int, int] = (32, 64)) -> None:
        """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        Args:
            root (string): Root directory of dataset where directory
                ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            resolution_range (tuple, optional): Define the range of possible resolutions of the output
            images. The default (32, 64) means the output image's size varies from 32x32 to 64x64.
        """
        super().__init__(root, train, transform, target_transform, download)
        self.resolution_range = resolution_range
        self.resolution = 32

    def change_resolution(self):
        """
        Reset the image resolution. The new value is randomly chose from the self.resolution_range.
        """
        self.resolution = random.choice(range(*self.resolution_range))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        img = F.resize(img, [self.resolution, self.resolution])
        return img, target


def create_dataloaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = MultiResCIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader
