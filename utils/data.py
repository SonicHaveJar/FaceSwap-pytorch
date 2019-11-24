import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

#https://stackoverflow.com/a/50802257
class LimitDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]

# Default load function to make things easier #TODO: ADD VAL AND TEST SETS
def autoloader(pathA, pathB, data_transforms=None, bs=32, num_workers=2, number_of_images=None):
    if data_transforms is None:
        data_transforms = transforms.Compose([
            transforms.Resize(224),#RandomResizedCrop(224)??
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    A_dataset = ImageFolder(os.path.join(pathA), data_transforms)
    B_dataset = ImageFolder(os.path.join(pathB), data_transforms)

    if not number_of_images is None:
        A_dataset = LimitDataset(A_dataset, number_of_images)
        B_dataset = LimitDataset(B_dataset, number_of_images)
    
    A_loader = DataLoader(A_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    B_loader = DataLoader(B_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    return A_loader, B_loader