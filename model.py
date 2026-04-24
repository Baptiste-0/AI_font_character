import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

CLASSES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_LABEL = {c: i for i, c in enumerate(CLASSES)} # 'A' : 0
LABEL_TO_CHAR = {i: c for i, c in enumerate(CLASSES)} # 0 : 'A'
NUM_CLASSES = len(CLASSES)

class MyDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = os.listdir(folder)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        path = os.path.join(self.folder, file)

        img = Image.open(path).convert("L")

        ch = file[0]
        if ch not in CHAR_TO_LABEL:
            raise ValueError(f"Bad label char '{ch}' for file {file}. Allowed: {CLASSES}")
        label = CHAR_TO_LABEL[ch]

        if self.transform:
            img = self.transform(img)

        return img, label

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 32x32
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 16x16
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def load_best(model, device, folder):
    files = [f for f in os.listdir(folder) if f.endswith(".pth")]

    best_loss = float("inf")
    best_file = None

    try:
        for name in files:
            loss_str = name.replace("best_model_", "").replace(".pth", "")
            loss_val = float(loss_str)
            if loss_val < best_loss:
                best_loss = loss_val
                best_file = name
    except:
        best_file = None

    if best_file:
        model.load_state_dict(torch.load(os.path.join(folder, best_file), map_location=device, weights_only=True))
        print("Loaded:", best_file, "loss:", best_loss)
    
    return best_loss
