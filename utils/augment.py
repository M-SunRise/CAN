import cv2
import torch
import random
import numpy as np
import albumentations as A
import torchvision.transforms as transforms

from PIL import Image
from utils.tools import Filter, DCT_mat
from albumentations.pytorch import ToTensorV2

class TwoTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return [v1, v2]

class OneOfTrans:
    """random select one of from the input transform list"""

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return self.base_transforms[random.randint(0,len(self.base_transforms)-1)](x)

class ALBU_AUG:
    def __init__(self, base_transform):
        self.transform = base_transform
    
    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        return self.transform(image=x)['image']

def get_augs(name="base", norm="imagenet", size=256):
    IMG_SIZE = size
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "0.5":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    if name == "None":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == 'Test':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RE":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.8,scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RandCrop":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.0), ratio=(0.9,1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RaAug":
        return OneOfTrans([
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.0), ratio=(0.9,1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ])
        ])
    elif name == "DFDC_selim":
        # dfdc 第一名数据增强方案
        return ALBU_AUG(A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.05),
            A.HorizontalFlip(),
            A.OneOf([
                A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_CUBIC),
                A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_AREA),
                A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            ], p=1.0),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Normalize(mean=tuple(mean),std=tuple(std)),
            ToTensorV2()
        ]))
    elif name == "Default":
            return OneOfTrans([
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.0), ratio=(0.9,1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ])
        ])
    elif name == 'MLFM_Base':
        maskGenerator = MLFMGenerator()

        return OneOfTrans([
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.0), ratio=(0.9,1.1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.13, 'all')),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.1, 'low')),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.05, 'mid')),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.05, 'high')),
                    transforms.Normalize(mean=mean,std=std),
                ])
            ])
    elif name == 'MLFM_All':
        maskGenerator = MLFMGenerator()

        return OneOfTrans([
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.15, 'all')),
                    transforms.Normalize(mean=mean,std=std),
                ]),
            ])
    elif name == 'MLFM_High':
        maskGenerator = MLFMGenerator()

        return OneOfTrans([
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.15, 'high')),
                    transforms.Normalize(mean=mean,std=std),
                ]),
            ])
    elif name == 'MLFM_Mid':
        maskGenerator = MLFMGenerator()

        return OneOfTrans([
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.15, 'mid')),
                    transforms.Normalize(mean=mean,std=std),
                ]),
            ])
    elif name == 'MLFM_Low':
        maskGenerator = MLFMGenerator()

        return OneOfTrans([
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                    transforms.Normalize(mean=mean,std=std),
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: maskGenerator.forward(img, 0.15, 'low')),
                    transforms.Normalize(mean=mean,std=std),
                ]),
            ])
        raise NotImplementedError
    
class MLFMGenerator():
    def __init__(self, ratio: float = 0.15, size=320, band: str = 'all') -> None:
        self.ratio = ratio
        self.band = band  # 'low', 'mid', 'high', 'all'
        self._DCT_all = torch.nn.Parameter(
            torch.tensor(DCT_mat(size)).float(), requires_grad=False
        )
        self._DCT_all_T = torch.nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1),
            requires_grad=False,
        )
        self.low_filter = Filter(size, 0, size // 2.82)
        self.middle_filter = Filter(size, size // 2.82, size // 2)
        self.high_filter = Filter(size, size // 2, size * 2)

    def forward(self, image, ratio = -1, band = None):
        if ratio != -1:
            self.ratio = ratio
        if band != None:
            self.band = band

        freq_image = self._DCT_all @ image @ self._DCT_all_T 

        c, height, width = image.shape

        if self.band == 'all':
            mask = self._create_balanced_mask(height, width, self.ratio)
            self.masked_freq_image = freq_image * mask
            masked_image_array = self._DCT_all_T @ self.masked_freq_image @ self._DCT_all

        if self.band == 'low':
            mask = self._create_balanced_mask(height, width, 1 - self.ratio)
            y = self._DCT_all_T @ self.low_filter.forward(freq_image)  @ self._DCT_all
            masked_image_array = image - y * torch.tensor(mask)

        elif self.band == 'mid':
            mask = self._create_balanced_mask(height, width, 1 - self.ratio)
            y = self._DCT_all_T @ self.middle_filter.forward(freq_image)  @ self._DCT_all
            masked_image_array = image - y * torch.tensor(mask)

        elif self.band == 'high':
            mask = self._create_balanced_mask(height, width, 1 - self.ratio)
            y = self._DCT_all_T @ self.high_filter.forward(freq_image)  @ self._DCT_all
            masked_image_array = image - y * torch.tensor(mask)

        return masked_image_array

    def _create_balanced_mask(self, height, width,  ratio):
        mask = np.ones((height, width, 3), dtype=np.float32)

        y_start, y_end = 0, height
        x_start, x_end = 0, width

        num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * ratio))
        mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
        y_indices = mask_frequencies_indices // (y_end - y_start) + y_start
        x_indices = mask_frequencies_indices % (x_end - x_start) + x_start

        mask[y_indices, x_indices, :] = 0
        return mask.transpose(2, 0, 1)
    


