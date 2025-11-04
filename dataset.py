import os
from typing import List, Tuple
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class data(Dataset):
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path: str = dataset_path
        
        self.low_res_path: str = os.path.join(self.dataset_path, "DIV2K_train_LR_x8")
        self.high_res_path: str = os.path.join(self.dataset_path, "DIV2K_train_HR")
        
        self.low_res_image: List[str] = os.listdir(self.low_res_path)
        self.high_res_image: List[str] = os.listdir(self.high_res_path)

    def read_image(self, prefix_path, path) -> torch.Tensor:
        # Read image
        img = cv2.imread(os.path.join(prefix_path, path))
        # Check if loaded successfully
        if img is None:
            raise FileNotFoundError("Image not found!")
        else:
            return torch.from_numpy(img).permute(2,0,1).to(torch.float32) / 256.0

    def __len__(self) -> int:
        return len(self.low_res_image)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        low_res: torch.Tensor = self.read_image(self.low_res_path, self.low_res_image[idx])
        _, lr_h, lr_w = low_res.shape
        high_res: torch.Tensor = self.read_image(self.high_res_path, self.high_res_image[idx])
        high_res = F.interpolate(high_res.unsqueeze(0), size=(lr_h*8, lr_w*8)).squeeze(0)

        return low_res, high_res
