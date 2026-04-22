import train
import os
from PIL import Image, ImageDraw, ImageFont
import random
import string

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

additional_data_size = 10000

data_folder = "data/train/1"
os.makedirs(data_folder, exist_ok=True)

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

    letter = random.choice(string.ascii_uppercase)
    draw.text((x, y), letter, font=font, fill=0)

    nb_img = random.randint(1, 2500)

    filename = os.path.join(data_folder, f"{letter}_{os.path.basename(fonts[nb_font])}_{nb_img}.png")
    img.save(filename)