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

data_folder = "data/test"
os.makedirs(data_folder, exist_ok=True)

for font_path in fonts:
    try:
        font = ImageFont.truetype(font_path, 48) # font and police size
    except:
        continue

    for letter in string.ascii_uppercase:
        img = Image.new("L", (64, 64), color=255) # Letters on 64x64 white grid
        draw = ImageDraw.Draw(img)

        x = random.randint(5, 25)
        y = random.randint(0, 5)

        draw.text((x, y), letter, font=font, fill=0)

        filename = os.path.join(data_folder, f"{letter}_{os.path.basename(font_path)}.png")
        img.save(filename)