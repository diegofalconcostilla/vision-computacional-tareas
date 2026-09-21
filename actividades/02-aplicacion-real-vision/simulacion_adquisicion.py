"""Simulación de la cadena de adquisición de imagen de una cámara de vehículo autónomo.

Toma una foto de dashcam (4K) y reproduce, paso a paso, lo que ocurre entre la escena
y los píxeles que recibe el sistema de percepción:

  1. Muestreo espacial (resolución)        -> fig1_muestreo.png
  2. Cuantización (bits por canal)         -> fig2_cuantizacion.png
  3. Mosaico Bayer y demosaico             -> fig3_bayer.png
  4. Exposición y rango dinámico (HDR)     -> fig4_exposicion.png
  5. Procesamiento: Sobel bajo ruido       -> fig5_bordes.png

Uso:
    python simulacion_adquisicion.py
    python simulacion_adquisicion.py --imagen assets/dashcam_original.jpg --salida .

Dependencias: numpy, scipy, Pillow, matplotlib, opencv-python-headless (solo para la fusión de Mertens).
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import convolve

SEMILLA = 42
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 100})


# ---------------------------------------------------------------- utilidades
def cargar(ruta):
    """Devuelve la imagen como float en [0, 1] (sRGB, HxWx3)."""
    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float64) / 255.0


def gris(img):
    """Luminancia ITU-R BT.601 sobre valores sRGB."""
    return img @ np.array([0.299, 0.587, 0.114])


def srgb_a_lineal(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lineal_a_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def reducir(img, f):
    """Submuestreo por promedio de bloques f x f (cada píxel integra la luz de su celda)."""
    h, w = img.shape[:2]
    h, w = h - h % f, w - w % f
    img = img[:h, :w]
    if img.ndim == 2:
        return img.reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    return img.reshape(h // f, f, w // f, f, img.shape[2]).mean(axis=(1, 3))


def ampliar(img, f):
    """Ampliación por vecino más cercano, para ver los píxeles individuales."""
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


def cuantizar(img, bits):
    niveles = 2 ** bits
    return np.round(img * (niveles - 1)) / (niveles - 1)


def psnr(ref, img):
    mse = np.mean((ref - img) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def entropia(g):
    """Entropía de Shannon (bits) del histograma de 256 niveles de una imagen en gris."""
    hist, _ = np.histogram(np.clip(g, 0, 1), bins=256, range=(0, 1))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", ruta)


# ---------------------------------------------------- experimento 1: muestreo
def exp1_muestreo(img, salida):
    # ROI (en píxeles de la imagen original 3840x2160) con el auto lejano y los semáforos.
    y0, y1, x0, x1 = 1568, 1856, 1264, 1856  # múltiplos de 16
    ancho_auto_4k = 138  # ancho aproximado del auto en la imagen original (medido a ojo)
    factores = [1, 2, 4, 8, 16]
    metricas = []
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(img)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=2))
    ax.set_title("Escena completa (3840×2160) y región de interés: auto lejano y semáforos")
    ax.axis("off")
    roi = img[y0:y1, x0:x1]
    for i, f in enumerate(factores):
        peq = reducir(roi, f)
        h_full, w_full = img.shape[0] // f, img.shape[1] // f
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(ampliar(peq, f), interpolation="nearest")
        ax.axis("off")
        auto = ancho_auto_4k / f
        ax.set_title(f"{w_full}×{h_full}  ({w_full * h_full / 1e6:.2f} MP)\nauto ≈ {auto:.0f} px de ancho")
        metricas.append({"resolucion": f"{w_full}x{h_full}", "megapixeles": round(w_full * h_full / 1e6, 3),
                         "ancho_auto_px": round(auto, 1)})
    fig.suptitle("Experimento 1. Muestreo espacial: menos píxeles, menos detalle para el detector", y=0.995)
    guardar(fig, salida / "fig1_muestreo.png")
    return metricas


# ------------------------------------------------- experimento 2: cuantización
def exp2_cuantizacion(img, salida):
    y0, y1, x0, x1 = 100, 700, 300, 2100  # cielo con degradado suave
    cielo = img[y0:y1, x0:x1]
    ref = cielo
    bits_lista = [8, 5, 4, 3, 2]
    metricas = []
    fig, axs = plt.subplots(2, 5, figsize=(13, 4.6), gridspec_kw={"height_ratios": [1, 0.8]})
    for j, b in enumerate(bits_lista):
        q = cuantizar(cielo, b)
        axs[0, j].imshow(q)
        axs[0, j].axis("off")
        p = psnr(ref, q)
        axs[0, j].set_title(f"{b} bits/canal → {2 ** b} niveles\nPSNR {p:.1f} dB" if b != 8 else "8 bits/canal → 256 niveles\n(referencia)")
        g = gris(q)
        axs[1, j].hist(g.ravel(), bins=256, range=(0, 1), color="#1f77b4")
        axs[1, j].set_yticks([])
        axs[1, j].set_xlabel("intensidad")
        metricas.append({"bits": b, "niveles": 2 ** b, "psnr_db": None if b == 8 else round(p, 1),
                         "megabytes_1080p_rgb": round(1920 * 1080 * 3 * b / 8 / 1e6, 2)})
    fig.suptitle("Experimento 2. Cuantización: con pocos bits aparecen bandas en el cielo y el histograma se vuelve un peine", y=1.0)
    fig.tight_layout()
    guardar(fig, salida / "fig2_cuantizacion.png")
    return metricas


# ------------------------------------------------------ experimento 3: Bayer
def mosaico_bayer(img):
    """Patrón RGGB. Devuelve el mosaico 2D (lo que mide el sensor) y las máscaras."""
    h, w = img.shape[:2]
    r = np.zeros((h, w), bool)
    g = np.zeros((h, w), bool)
    b = np.zeros((h, w), bool)
    r[0::2, 0::2] = True
    g[0::2, 1::2] = True
    g[1::2, 0::2] = True
    b[1::2, 1::2] = True
    mosaico = img[..., 0] * r + img[..., 1] * g + img[..., 2] * b
    return mosaico, (r, g, b)


def demosaico_bilineal(mosaico, mascaras):
    """Interpolación bilineal: es una convolución con dos kernels pequeños."""
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
    r, g, b = mascaras
    canales = [convolve(mosaico * m, k, mode="mirror") for m, k in ((r, k_rb), (g, k_g), (b, k_rb))]
    return np.clip(np.stack(canales, axis=-1), 0, 1)


def exp3_bayer(img, salida):
    base = reducir(img, 4)  # 960x540 para que los píxeles sean visibles en un recorte pequeño
    parche = base[270:366, 740:836]  # letrero amarillo con flecha
    mosaico, (r, g, b) = mosaico_bayer(parche)
    color_mosaico = np.zeros_like(parche)
    color_mosaico[..., 0] = mosaico * r
    color_mosaico[..., 1] = mosaico * g
    color_mosaico[..., 2] = mosaico * b
    demo = demosaico_bilineal(mosaico, (r, g, b))
    error = np.abs(parche - demo).mean(axis=2)
    p = psnr(parche, demo)
    e_max = float(np.abs(parche - demo).max())
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.4))
    paneles = [(parche, "Escena (RGB completo)"),
               (color_mosaico, "Lo que mide cada fotositio\n(1 de 3 colores por píxel)"),
               (mosaico, "Salida RAW (1 valor por píxel)"),
               (demo, f"Demosaico bilineal\nPSNR {p:.1f} dB"),
               (np.clip(error * 4, 0, 1), "Error ×4 (bordes y color)")]
    for a, (im, t) in zip(axs, paneles):
        a.imshow(ampliar(im, 3), cmap="gray" if im.ndim == 2 else None, interpolation="nearest", vmin=0, vmax=1)
        a.set_title(t)
        a.axis("off")
    fig.suptitle("Experimento 3. Mosaico Bayer RGGB: 2/3 de los valores de color son interpolados", y=1.02)
    fig.tight_layout()
    guardar(fig, salida / "fig3_bayer.png")
    return {"psnr_demosaico_db": round(p, 1), "error_maximo": round(e_max, 3),
            "fraccion_valores_interpolados": round(2 / 3, 3)}


# ---------------------------------------------------- experimento 4: exposición
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = SOBEL_X.T
GAUSS_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0


def magnitud_sobel(g):
    g = convolve(g, GAUSS_3, mode="nearest")
    gx = convolve(g, SOBEL_X, mode="nearest")
    gy = convolve(g, SOBEL_Y, mode="nearest")
    return np.hypot(gx, gy)


def exp4_exposicion(img, salida):
    """Simula un sensor de rango limitado: escala la luz (EV), satura en 1.0 y cuantiza a 8 bits.

    Limitación: la foto de origen ya es un JPEG de 8 bits, así que no contiene el rango dinámico
    real de la escena. El experimento muestra el efecto de elegir la exposición y el principio de
    combinar exposiciones (fusión de Mertens et al.), no el rendimiento de un sensor HDR real.
    """
    lin = srgb_a_lineal(reducir(img, 2))  # 1920x1080
    evs = [-2, 0, 2]
    imagenes = []
    for ev in evs:
        capt = np.clip(lin * 2.0 ** ev, 0, 1)  # el sensor se satura: no puede pasar de 1.0
        imagenes.append(cuantizar(lineal_a_srgb(capt), 8))  # ADC de 8 bits + curva gamma
    lista_u8 = [np.round(im[..., ::-1] * 255).astype(np.uint8) for im in imagenes]  # BGR para OpenCV
    fusion = cv2.createMergeMertens().process(lista_u8)[..., ::-1]
    fusion = np.clip(fusion, 0, 1)
    todas = imagenes + [fusion]
    titulos = ["Exposición −2 EV", "Exposición 0 EV", "Exposición +2 EV", "Fusión de las 3 (Mertens)"]
    # Zonas para medir detalle local: cielo con nubes (luces) y árboles a la izquierda (sombras)
    zonas = {"cielo": (slice(50, 300), slice(150, 1050)), "arboles": (slice(540, 870), slice(0, 570))}
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.6), gridspec_kw={"height_ratios": [1.5, 1]})
    metricas = []
    for j, (im, t) in enumerate(zip(todas, titulos)):
        g = gris(im)
        sat = float((g >= 0.98).mean() * 100)
        osc = float((g <= 0.04).mean() * 100)
        # contraste local = magnitud media de Sobel / luminancia media (así no premia solo el brillo)
        det = {k: float(magnitud_sobel(g[z]).mean() / max(g[z].mean(), 1e-6)) for k, z in zonas.items()}
        axs[0, j].imshow(im)
        for z, c in zip(zonas.values(), ("yellow", "cyan")):
            axs[0, j].add_patch(Rectangle((z[1].start, z[0].start), z[1].stop - z[1].start, z[0].stop - z[0].start,
                                          fill=False, ec=c, lw=1.5))
        axs[0, j].axis("off")
        axs[0, j].set_title(f"{t}\nsaturado {sat:.1f}% · oscuro {osc:.1f}%\n"
                            f"contraste local cielo {det['cielo']:.3f} · árboles {det['arboles']:.3f}")
        axs[1, j].hist(g.ravel(), bins=128, range=(0, 1), color="#444444")
        axs[1, j].set_yticks([])
        axs[1, j].set_xlabel("luminancia")
        metricas.append({"caso": t, "saturado_pct": round(sat, 2), "oscuro_pct": round(osc, 2),
                         "contraste_cielo": round(det["cielo"], 4), "contraste_arboles": round(det["arboles"], 4)})
    fig.suptitle("Experimento 4. Exposición y rango dinámico: un sensor de rango limitado no conserva a la vez las luces y las sombras", y=1.0)
    fig.tight_layout()
    guardar(fig, salida / "fig4_exposicion.png")
    return metricas


# -------------------------------------------------- experimento 5: Sobel/ruido
def exp5_bordes(img, salida):
    rng = np.random.default_rng(SEMILLA)
    # Tramo de carretera con marcas de carril, por encima del texto de fecha/GPS de la cámara
    g = gris(reducir(img, 4))[395:495, 100:700]
    casos = [("Limpia", g),
             ("Ruido σ=0.03", np.clip(g + rng.normal(0, 0.03, g.shape), 0, 1)),
             ("Ruido σ=0.10", np.clip(g + rng.normal(0, 0.10, g.shape), 0, 1)),
             ("Cuantizada a 3 bits", cuantizar(g, 3))]
    ref = magnitud_sobel(g)
    umbral = float(np.percentile(ref, 90))  # "borde" = 10 % de píxeles más fuertes de la imagen limpia
    ref_borde = ref > umbral
    metricas = []
    fig, axs = plt.subplots(4, 2, figsize=(11, 7.2))
    for i, (t, gi) in enumerate(casos):
        m = magnitud_sobel(gi)
        r = float(np.corrcoef(ref.ravel(), m.ravel())[0, 1])
        borde = m > umbral
        falsos = float((borde & ~ref_borde).sum() / borde.sum() * 100)
        perdidos = float((~borde & ref_borde).sum() / ref_borde.sum() * 100)
        axs[i, 0].imshow(gi, cmap="gray", vmin=0, vmax=1)
        axs[i, 0].set_title(t + ("" if i == 0 else f" · PSNR {psnr(g, gi):.1f} dB"), loc="left")
        axs[i, 1].imshow(np.clip(m / (2 * umbral), 0, 1), cmap="gray")
        axs[i, 1].set_title(f"Sobel · correlación con la limpia {r:.2f} · bordes falsos {falsos:.0f}% · bordes perdidos {perdidos:.0f}%",
                            loc="left")
        for a in axs[i]:
            a.axis("off")
        metricas.append({"caso": t, "psnr_db": None if i == 0 else round(psnr(g, gi), 1),
                         "correlacion_bordes": round(r, 3), "bordes_falsos_pct": round(falsos, 1),
                         "bordes_perdidos_pct": round(perdidos, 1)})
    fig.suptitle("Experimento 5. Procesamiento digital: el ruido del sensor contamina los bordes que usa la detección de carriles", y=1.0)
    fig.tight_layout()
    guardar(fig, salida / "fig5_bordes.png")
    return metricas


# ------------------------------------------------------------------- principal
def main():
    aqui = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagen", default=str(aqui / "assets" / "dashcam_original.jpg"))
    ap.add_argument("--salida", default=str(aqui))
    args = ap.parse_args()

    salida = Path(args.salida)
    figuras = salida / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    img = cargar(args.imagen)
    print(f"Imagen: {args.imagen}  {img.shape[1]}x{img.shape[0]}")

    resultados = {
        "imagen": {"ancho": img.shape[1], "alto": img.shape[0]},
        "exp1_muestreo": exp1_muestreo(img, figuras),
        "exp2_cuantizacion": exp2_cuantizacion(img, figuras),
        "exp3_bayer": exp3_bayer(img, figuras),
        "exp4_exposicion": exp4_exposicion(img, figuras),
        "exp5_bordes": exp5_bordes(img, figuras),
    }
    ruta = salida / "resultados.json"
    ruta.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Métricas ->", ruta)
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
