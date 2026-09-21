"""Experimentos de la actividad "Algoritmos de mejoramiento de imagen en Photoshop".

Como Adobe Photoshop es de pago, se aplican con Python los algoritmos equivalentes a sus herramientas:

  Filtro > Ruido > Reducir ruido        ->  cv2.fastNlMeansDenoisingColored (medias no locales)
  Imagen > Ajustes > Niveles / Curvas   ->  estirado por percentiles + gamma
  Imagen > Ajustes > Ecualizar / Sombras ->  CLAHE sobre el canal de luminancia (L de LAB)
  Filtro > Enfocar > Máscara de enfoque ->  unsharp masking (cantidad, radio, umbral)

Genera figuras/ y resultados.json.   python experimentos.py
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import estimate_sigma

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
FIG = AQUI / "figuras"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})


# ------------------------------------------------------------ operaciones (equivalentes a Photoshop)
def reducir_ruido(bgr, h=7):
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)


def niveles(bgr, p_bajo=0.5, p_alto=99.5, gamma=0.65):
    """Niveles: punto negro/blanco por percentiles de la luminancia y tono medio (gamma < 1 aclara)."""
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(lum, [p_bajo, p_alto])
    x = np.clip((bgr.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) ** gamma
    return (x * 255).round().astype(np.uint8)


def ecualizar_local(bgr, clip=2.0, grid=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def mascara_enfoque(bgr, cantidad=0.8, radio=1.6, umbral=3):
    """Unsharp mask: g = f + cantidad·(f − desenfoque(f)); solo donde la diferencia supera el umbral."""
    f = bgr.astype(np.float32)
    desenf = cv2.GaussianBlur(f, (0, 0), radio)
    dif = f - desenf
    dif[np.abs(dif) < umbral] = 0
    return np.clip(f + cantidad * dif, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ métricas
def gris(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def entropia(g):
    p = np.bincount(g.ravel(), minlength=256) / g.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def metricas(bgr):
    g = gris(bgr)
    sigma = float(estimate_sigma(g.astype(np.float32) / 255.0) * 255)
    return {"media": round(float(g.mean()), 1), "desviacion": round(float(g.std()), 1), "entropia": round(entropia(g), 2),
            "negro_pct": round(float((g <= 8).mean() * 100), 1), "saturado_pct": round(float((g >= 247).mean() * 100), 2),
            "sigma_ruido": round(sigma, 2), "snr": round(float(g.mean()) / max(sigma, 1e-6), 1),
            "nitidez_lap": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)}


def canny_stats(bgr, bajo=60, alto=140):
    """Bordes de Canny con umbrales fijos. Mide cuánto "ruido de bordes" hay: componentes conectados diminutos (< 6 px)."""
    g = cv2.GaussianBlur(gris(bgr), (3, 3), 0)
    e = cv2.Canny(g, bajo, alto)
    n, _, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    return {"bordes_pct": round(float((e > 0).mean() * 100), 2), "componentes": int(len(areas)),
            "diminutos_pct": round(float((areas < 6).mean() * 100), 1), "longitud_mediana_px": float(np.median(areas))}, e


def histo(ax, bgr, titulo):
    for c, col in zip(range(3), ("#1f5fbf", "#2a9d3a", "#c62828")):
        ax.plot(np.bincount(bgr[..., c].ravel(), minlength=256), color=col, lw=.9, alpha=.85)
    g = np.bincount(gris(bgr).ravel(), minlength=256)
    ax.fill_between(range(256), g, color="#555", alpha=.35)
    ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_title(titulo)
    ax.set_xlabel("intensidad (gris = área; canales RGB = líneas)")


def main():
    orig = cv2.imread(str(RAIZ / "datos_compartidos" / "noche_hongkong_ruido.jpg"))
    orig = cv2.resize(orig, (1400, round(orig.shape[0] * 1400 / orig.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(AQUI / "imagen_original.jpg"), orig, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # --- cadena elegida: reducir ruido -> niveles -> ecualización local suave -> enfoque suave
    e1 = reducir_ruido(orig, h=9)
    e2 = niveles(e1)
    e3 = ecualizar_local(e2, clip=1.5)
    e4 = mascara_enfoque(e3, cantidad=0.6, radio=1.2, umbral=4)
    etapas = [("0. Original", orig), ("1. Reducir ruido", e1), ("2. Niveles + gamma", e2),
              ("3. Ecualización local (CLAHE)", e3), ("4. Máscara de enfoque", e4)]
    cv2.imwrite(str(AQUI / "imagen_mejorada.jpg"), e4, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # --- variantes para comparar el orden de las operaciones (mismos parámetros en todas)
    sin_ruido = mascara_enfoque(ecualizar_local(niveles(orig), clip=1.5), cantidad=0.6, radio=1.2, umbral=4)   # sin reducir ruido
    despues = mascara_enfoque(ecualizar_local(reducir_ruido(niveles(orig), h=9), clip=1.5), cantidad=0.6, radio=1.2, umbral=4)  # ruido tras subir sombras
    variantes = [("Original", orig), ("Sin reducir ruido", sin_ruido), ("Reducir ruido después de los niveles", despues), ("Reducir ruido al inicio (elegida)", e4)]

    res = {"etapas": {n: metricas(im) for n, im in etapas}, "variantes": {n: metricas(im) for n, im in variantes}, "canny": {}}
    bordes = {}
    for n, im in variantes:
        res["canny"][n], bordes[n] = canny_stats(im)
    (AQUI / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    # Fig 1: original vs mejorada
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    ax[0].imshow(rgb(orig)); ax[0].set_title("Imagen original (ISO alto, poca luz)"); ax[1].imshow(rgb(e4)); ax[1].set_title("Imagen mejorada (cadena de 4 técnicas)")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig1_original_vs_mejorada.png", dpi=150); plt.close(fig)

    # Fig 2: paso a paso, recorte ampliado (letrero y fachada)
    h, w = orig.shape[:2]
    y0, y1, x0, x1 = int(h * .30), int(h * .55), int(w * .52), int(w * .78)
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.3))
    for a, (n, im) in zip(ax, etapas):
        a.imshow(rgb(im)[y0:y1, x0:x1], interpolation="nearest"); a.set_title(n); a.axis("off")
    fig.suptitle("Recorte ampliado de la misma región tras cada técnica", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig2_pasos_recorte.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fig 3: histogramas antes/después
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    histo(ax[0], orig, "Histograma ANTES"); histo(ax[1], e4, "Histograma DESPUÉS")
    ymax = max(a.get_ylim()[1] for a in ax)
    for a in ax: a.set_ylim(0, ymax)
    fig.tight_layout(); fig.savefig(FIG / "fig3_histogramas.png", dpi=150); plt.close(fig)

    # Fig 4: histogramas por etapa
    fig, ax = plt.subplots(1, 5, figsize=(17, 2.8))
    for a, (n, im) in zip(ax, etapas):
        histo(a, im, n); a.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIG / "fig4_histogramas_etapas.png", dpi=150); plt.close(fig)

    # Fig 5: orden de las operaciones (recorte) y mapas de bordes de Canny
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.4))
    for c, (n, im) in enumerate(variantes):
        ax[0, c].imshow(rgb(im)[y0:y1, x0:x1], interpolation="nearest"); ax[0, c].set_title(n, fontsize=8.5)
        ax[1, c].imshow(bordes[n][y0:y1, x0:x1], cmap="gray", interpolation="nearest")
        ax[1, c].set_title(f"Canny: {res['canny'][n]['componentes']} componentes, {res['canny'][n]['diminutos_pct']}% diminutos", fontsize=8.5)
        ax[0, c].axis("off"); ax[1, c].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig5_orden_operaciones.png", dpi=150); plt.close(fig)

    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
