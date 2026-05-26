"""
Generates placeholder ASL letter images (A-Z).
Run once: python scripts/generate_signs.py
Replace the PNGs with real ASL hand images for production.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent.parent / "app" / "static" / "signs"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 200
BG = (12, 12, 20)
FG = (74, 222, 128)

FONT_PATHS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/consola.ttf",
]


def load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


font = load_font(130)

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    img = Image.new("RGB", (SIZE, SIZE), color=BG)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (SIZE - w) // 2 - bbox[0]
    y = (SIZE - h) // 2 - bbox[1]
    draw.text((x, y), letter, fill=FG, font=font)
    path = OUT / f"{letter}.png"
    img.save(path)
    print(f"  {letter}.png")

print(f"\nDone. {len(list(OUT.glob('*.png')))} images in {OUT}")
print("Replace with real ASL hand photos for a production build.")
