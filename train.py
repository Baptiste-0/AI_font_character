import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

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

        label = ord(file[0]) - ord('A')
        if not (0 <= label < 26):
            raise ValueError(f"Bad label {label} for file {file}")

        if self.transform:
            img = self.transform(img)

        return img, label

train_dataset = MyDataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/train/1"), transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)
# taskkill /F /IM python.exe  <- kill all python process to end daemon in windows cmd

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 32x32
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 16x16
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 26)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def load_best(model, device, folder):
    files = [f for f in os.listdir(folder) if f.endswith(".pth")]

    best_loss = float("inf")
    best_file = None

    for name in files:
        loss_str = name.replace("best_model_", "").replace(".pth", "")
        loss_val = float(loss_str)
        if loss_val < best_loss:
            best_loss = loss_val
            best_file = name
    
    if best_file:
        model.load_state_dict(torch.load(os.path.join(folder, best_file), map_location=device, weights_only=True))
        print("Loaded:", best_file, "loss:", best_loss)
    
    return best_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    size_epoch = len(train_loader)

    best_loss = load_best(model, device, "models")

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, steps: {i} / {size_epoch}")
            
        epoch_loss /= len(train_loader)

        print(f"Epoch {epoch+1}, avg loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss

            path = f"models/best_model_{best_loss:.4f}.pth"
            torch.save(model.state_dict(), path)
            print("Saved:", path)

if __name__ == "__main__":
        data_folder = "models"
        os.makedirs(data_folder, exist_ok=True)
        train()
