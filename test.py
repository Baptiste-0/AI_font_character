from model import *

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

correct_char = [0] * NUM_CLASSES
nb_char = [0] * NUM_CLASSES

correct = 0
ct = 0

for f in test_files:
    img = Image.open(os.path.join("data/test", f))

    img = transform(img)
    img = img.unsqueeze(0)

    output = model(img.to(device))
    pred = torch.argmax(output, dim=1)

    label = CHAR_TO_LABEL[os.path.basename(f)[0]]
    nb_char[CHAR_TO_LABEL[os.path.basename(f)[0]]] += 1

    if pred.item() == label:
        correct+=1
        correct_char[CHAR_TO_LABEL[os.path.basename(f)[0]]] += 1

    ct += 1
    if ct % 1000 == 0:
        print(f"{ct} / {len(test_files)} tested")

print(f"Résultats bon: {correct / len(test_files) * 100:.2f}%")
print("----------------------------")
for i in range(NUM_CLASSES):
    print(f"\'{CLASSES[i]}\' : {(correct_char[i] / nb_char[i]) * 100:.2f}%")