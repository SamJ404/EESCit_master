from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register



def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('fixed_scale_pairsr_warp')
class FixedScalePairSRWarp(Dataset):
    def __init__(self, dataset, scale):
        self.dataset = dataset
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lr, hr = self.dataset[idx]

        return {
            'img': lr,
            'gt': hr
        }