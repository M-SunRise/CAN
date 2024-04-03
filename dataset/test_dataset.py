import cv2
import numpy as np

from torch.utils.data import Dataset


class Test_Dataset(Dataset):
    def __init__(self, df, transform = None):

        super().__init__()
        self.data = df.values
        np.random.seed(888)
        np.random.shuffle(self.data)
        self.transforms = transform
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        while(True):
            video, file, label, path, type = self.data[index]
            label = int(label)
            try:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error: {path} {e}")
            image = self.transforms(image)

            return {
                "image": image, 
                "label": label,
            }