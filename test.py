from train import load_best
from train import CNN

from PIL import Image
import torch
import os
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
load_best(model, device, "models")
model.eval()

test_files = [f for f in os.listdir("data/test") if f.endswith(".png")]

correct = 0

for f in test_files:
    img = Image.open(os.path.join("data/test", f))

    img = transform(img)
    img = img.unsqueeze(0)

    output = model(img.to(device))
    pred = torch.argmax(output, dim=1)

    if pred.item() == ord(os.path.basename(f)[0]) - ord('A'):
        correct+=1

print(f"Résultats bon: {correct / len(test_files) * 100:.2f}%")
