import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import json

ACTIONS = ["attack", "back", "forward", "jump", "left", "right", "camera"]

class CustomImageDataset(Dataset):
    def __init__(self, episodes_dir:str):
        self.episodes_dir = episodes_dir
        self.img_paths, self.actions = self._get_data(episodes_dir)

    def _get_data(self, dir):
        img_objects = []
        for episode_folder in os.listdir(dir):
            episode_path = os.path.join(dir, episode_folder)
            img_objects.append(np.load(os.path.join(episode_path, "human_obs"), allow_pickle=True))
            with open(os.path.join(episode_path, "stats.json", "r")) as f:
                stats = json.load(f)
                for step in len(stats):
                    
                    
        frames = np.concatenate(img_objects, axis=0)
                
        


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label