import os
from PIL import Image, ImageDraw, ImageFont
import random

data_folder = "data/train/1"
os.makedirs(data_folder, exist_ok=True)

import model

font_folder = "C:/Windows/Fonts"

fonts = [
    os.path.join(font_folder, f)
    for f in os.listdir(font_folder)
    if f.endswith(".ttf")
]

fonts.remove("C:/Windows/Fonts\\marlett.ttf")
fonts.remove("C:/Windows/Fonts\\SansSerifCollection.ttf")
fonts.remove("C:/Windows/Fonts\\segmdl2.ttf")
fonts.remove("C:/Windows/Fonts\\SegoeIcons.ttf")
fonts.remove("C:/Windows/Fonts\\symbol.ttf")
fonts.remove("C:/Windows/Fonts\\webdings.ttf")
fonts.remove("C:/Windows/Fonts\\wingding.ttf")

def increase_everything(additional_data_size):
    for i in range(additional_data_size):
        nb_font = random.randint(0, len(fonts))
        try:
            font = ImageFont.truetype(fonts[nb_font], 48)
        except:
            continue
        img = Image.new("L", (64, 64), color=255)
        draw = ImageDraw.Draw(img)

        x = random.randint(5, 25)
        y = random.randint(0, 5)
        char = random.choice(model.CLASSES)

        draw.text((x, y), char, font=font, fill=0)

        nb_img = random.randint(1, 2500)

        filename = os.path.join(data_folder, f"{char}_{os.path.basename(fonts[nb_font])}_{nb_img}.png")
        img.save(filename)

        if i % 5000 == 0:
            print(f"Created train images: {i} / {additional_data_size}")

def increase_one(char, additional_data_size):
    for i in range(additional_data_size):
        nb_font = random.randint(0, len(fonts))
        try:
            font = ImageFont.truetype(fonts[nb_font], 48)
        except:
            continue
        img = Image.new("L", (64, 64), color=255)
        draw = ImageDraw.Draw(img)

        x = random.randint(5, 25)
        y = random.randint(0, 5)

        draw.text((x, y), char, font=font, fill=0)

        nb_img = random.randint(1, 2500)

        filename = os.path.join(data_folder, f"{char}_{os.path.basename(fonts[nb_font])}_{nb_img}.png")
        img.save(filename)

        if i % 5000 == 0:
            print(f"Created train images: {i} / {additional_data_size}")


#increase_everything(30000)

# model plus faible sur ces caracteres
#increase_one('l', 1000)
#increase_one('q', 1000)
#increase_one('B', 500)
#increase_one('0', 1000)
#increase_one('O', 1000)
#increase_one('o', 1000)
#increase_one('a', 1000)
#increase_one('I', 1000)