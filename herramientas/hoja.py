"""Hoja de contacto de un PDF para revisar el diseño:  python herramientas/hoja.py archivo.pdf salida.png [dpi] [columnas]"""
import sys
import pymupdf
from PIL import Image

pdf, salida = sys.argv[1], sys.argv[2]
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 40
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 5
with pymupdf.open(pdf) as d:
    imgs = []
    for pg in d:
        pix = pg.get_pixmap(dpi=dpi)
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
w, h = imgs[0].size
filas = -(-len(imgs) // cols)
hoja = Image.new("RGB", (cols * (w + 6), filas * (h + 6)), (120, 120, 120))
for i, im in enumerate(imgs):
    hoja.paste(im, ((i % cols) * (w + 6) + 3, (i // cols) * (h + 6) + 3))
hoja.save(salida)
print(len(imgs), "páginas ->", hoja.size)
