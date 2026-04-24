import model as m
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

folder_train = "data/train/1"

train_dataset = m.MyDataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_train), transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)
# taskkill /F /IM python.exe  <- kill all python process to end daemon in windows cmd

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = m.CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 7
    size_epoch = len(train_loader)

    try:
        best_loss = m.load_best(model, device, "models")
    except:
        print("Wrong size model encountered")
        best_loss = float('inf')

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
            print(f"Saved: {path} ✅")
        else:
             print("Not better than the best epoch. ❌")

if __name__ == "__main__":
        data_folder = "models"
        os.makedirs(data_folder, exist_ok=True)
        train()
