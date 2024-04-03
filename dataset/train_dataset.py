import os
import cv2
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(
        self,
        df,
        mode,
        transforms=None,  # pytorch transforms
        seed = 888,
    ):

        super().__init__()
        self.df = df
        self.mode = mode
        self.transforms = transforms
        self.seed = seed

        rows = df

        print(
            "real {} fakes {} mode {}".format(
                len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode
            )
        )
        self.data = rows.values
        np.random.seed(self.seed)
        np.random.shuffle(self.data)

    def __getitem__(self, index: int):

        while(True):
            video, file, label, path, type=self.data[index]
            if(os.path.exists(path)):
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = label
            else:
                index = random.randint(0, len(self.data) - 1)
                continue

            if self.transforms is not None:
                ori_image = self.transforms(image)
                ref_image = self.transforms(image)


            return {
                "image": ori_image,
                "ref_image": ref_image,
                "label": label
            }

    def __len__(self) -> int:
        return len(self.data)



class RealFaceDataset(Dataset):
    def __init__(
        self,
        df,
        mode,
        transforms=None,
        seed = 888,
    ):

        super().__init__()
        self.df = df
        self.mode = mode
        self.transforms = transforms
        self.seed = seed
        rows = df

        print(
            "real {} fakes {} mode {}".format(
                len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode
            )
        )
        self.data = rows.values
        np.random.seed(self.seed)
        np.random.shuffle(self.data)

    def __getitem__(self, index: int):

        while(True):
            video, file, label, path, type=self.data[index]
            if(os.path.exists(path)):
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = label
            else:
                index = random.randint(0, len(self.data) - 1)
                continue

            if self.transforms is not None:
                ori_image = self.transforms(image)
                ref_image = self.transforms(image)

            return {
                "image": ori_image,
                "ref_image": ref_image,
                "label": label
            }

    def __len__(self) -> int:
        return len(self.data)

    
    
class FakeFaceDataset(Dataset):
    def __init__(
        self,
        df,
        mode,
        transforms=None,
        seed = 888,

    ):

        super().__init__()
        self.df = df
        self.mode = mode
        self.transforms = transforms
        self.seed = seed
        rows = df

        print(
            "real {} fakes {} mode {}".format(
                len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode
            )
        )
        self.data = rows.values
        np.random.seed(self.seed)
        np.random.shuffle(self.data)


    def __getitem__(self, index: int):

        while(True):
            video, file, label, path, type=self.data[index]
            if(os.path.exists(path)):
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = label
            else:
                index = random.randint(0, len(self.data) - 1)
                continue

            if self.transforms is not None:
                ori_image = self.transforms(image)
                ref_image = self.transforms(image)


            return {
                "image": ori_image,
                "ref_image": ref_image,
                "label": label,
            }


    def __len__(self) -> int:
        return len(self.data)


