"""Convolución y extracción de líneas en escenas de conducción.

Aplica Sobel, Prewitt, Laplaciano y Canny a dos imágenes de carretera, detecta líneas con Hough y compara
qué tan útil es cada detector para encontrar carriles. Además muestra una capa convolucional "tipo CNN".
    python experimentos.py   ->  figuras/, resultados.json
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
FIG = AQUI / "figuras"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})

# Trapecio de la calzada como fracciones (x, y) de la imagen; se excluye la franja inferior con la fecha/GPS impresa por la cámara
ESCENAS = {"Autopista I-75 (vehículo adelante)": ("autopista_i75.jpg", [(0.00, 0.925), (0.36, 0.80), (0.50, 0.80), (0.86, 0.925)]),
           "Autopista I-10 (salida)": ("autopista_i10.jpg", [(0.00, 0.925), (0.40, 0.80), (0.58, 0.80), (0.95, 0.925)])}


def mascara_roi(h, w, poly):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array([(int(x * w), int(y * h)) for x, y in poly], np.int32)], 255)
    return m

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
LAPLACIANO = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)


def cargar(nombre, ancho=1280):
    ruta = AQUI / "datos" / nombre                      # copia local (así funciona dentro del zip de entrega)
    if not ruta.exists():
        ruta = RAIZ / "datos_compartidos" / nombre
    img = cv2.imread(str(ruta))
    return cv2.resize(img, (ancho, round(img.shape[0] * ancho / img.shape[1])), interpolation=cv2.INTER_AREA)


def gradiente(g, kx):
    gx = cv2.filter2D(g, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.filter2D(g, cv2.CV_32F, kx.T, borderType=cv2.BORDER_REFLECT_101)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def detectores(bgr, roi):
    """Devuelve mapas de respuesta (float) y mapas binarios de borde, calculados solo en la región de la calzada."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.GaussianBlur(g, (5, 5), 1.2).astype(np.float32)
    resp = {"Sobel": gradiente(s, SOBEL_X)[0], "Prewitt": gradiente(s, PREWITT_X)[0],
            "Laplaciano": np.abs(cv2.filter2D(s, cv2.CV_32F, LAPLACIANO, borderType=cv2.BORDER_REFLECT_101))}
    sel = roi > 0
    binarios = {}
    for k, v in resp.items():
        t = np.percentile(v[sel], 92)                       # mismo criterio para todos: el 8 % de píxeles más fuertes
        b = (v > t).astype(np.uint8) * 255
        binarios[k] = b
    binarios["Canny"] = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.2), 50, 150)
    resp["Canny"] = binarios["Canny"].astype(np.float32)
    for k in binarios:
        binarios[k] = cv2.bitwise_and(binarios[k], roi)
    return g, resp, binarios


def hough(binario):
    lin = cv2.HoughLinesP(binario, 1, np.pi / 180, threshold=45, minLineLength=45, maxLineGap=12)
    return [] if lin is None else [tuple(map(int, l)) for l in np.asarray(lin).reshape(-1, 4)]   # (N,1,4) en OpenCV 4, (N,4) en OpenCV 5


def plausible(seg):
    """Un segmento es plausible como marca de carril si su inclinación está entre 20° y 80° respecto a la horizontal."""
    x1, y1, x2, y2 = seg
    ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    ang = ang if ang <= 90 else 180 - ang
    return 20 <= ang <= 80


def main():
    res = {}
    for esc, (arch, poly) in ESCENAS.items():
        bgr = cargar(arch)
        h, w = bgr.shape[:2]
        roi = mascara_roi(h, w, poly)
        ys = [int(y * h) for _, y in poly]
        y0, y1 = min(ys), max(ys)
        a0, a1 = max(y0 - 50, 0), min(y1 + 25, h)          # franja que se muestra
        g, resp, bin_ = detectores(bgr, roi)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        contorno = np.array([(int(x * w), int(y * h)) for x, y in poly], np.int32)
        clave = arch.split(".")[0]

        # --- Fig A: escena completa con la zona de análisis + respuesta de cada detector en la franja de la calzada
        fig = plt.figure(figsize=(12, 9.4))
        gs = fig.add_gridspec(6, 1, height_ratios=[3.4, 1, 1, 1, 1, 1], hspace=0.28)
        vista = rgb.copy(); cv2.polylines(vista, [contorno], True, (255, 230, 0), 4)
        ax = fig.add_subplot(gs[0]); ax.imshow(vista); ax.set_title(f"{esc}: escena y zona de calzada analizada (amarillo)"); ax.axis("off")
        ax = fig.add_subplot(gs[1]); ax.imshow(rgb[a0:a1]); ax.set_title("Franja de calzada (referencia)"); ax.axis("off")
        for i, k in enumerate(("Sobel", "Prewitt", "Laplaciano", "Canny")):
            v = resp[k]; v = v / np.percentile(v[roi > 0], 99.5) if k != "Canny" else v / 255
            ax = fig.add_subplot(gs[2 + i]); ax.imshow(np.clip(v, 0, 1)[a0:a1], cmap="gray"); ax.set_title(k); ax.axis("off")
        fig.savefig(FIG / f"figA_respuestas_{clave}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

        # --- Fig B: líneas de Hough sobre cada detector (solo calzada)
        stats = {}
        fig, axs = plt.subplots(4, 1, figsize=(12, 7.2))
        for a, k in zip(axs, ("Sobel", "Prewitt", "Laplaciano", "Canny")):
            segs = hough(bin_[k])
            buenos = [s_ for s_ in segs if plausible(s_)]
            out = rgb.copy()
            for s_ in segs:
                cv2.line(out, (s_[0], s_[1]), (s_[2], s_[3]), (255, 60, 60) if not plausible(s_) else (40, 255, 40), 4)
            a.imshow(out[a0:a1]); a.axis("off")
            long_ok = float(sum(np.hypot(s_[2] - s_[0], s_[3] - s_[1]) for s_ in buenos))
            a.set_title(f"{k}: {len(segs)} segmentos, {len(buenos)} plausibles como carril", fontsize=9)
            stats[k] = {"segmentos": len(segs), "plausibles": len(buenos), "longitud_plausible_px": round(long_ok),
                        "borde_pct": round(float((bin_[k][roi > 0] > 0).mean() * 100), 2)}
        fig.suptitle(f"{esc}: líneas de Hough (verde = inclinación plausible de carril, rojo = descartada)", y=1.0, fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / f"figB_hough_{clave}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        res[esc] = stats

    # --- Fig C: una capa convolucional "tipo CNN": banco de 8 orientaciones -> ReLU -> max-pooling
    bgr = cargar("autopista_i75.jpg")
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32) / 255
    base = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)         # Sobel horizontal
    angulos = [0, 45, 90, 135]
    bancos = []
    for ang in angulos:
        M = cv2.getRotationMatrix2D((1, 1), ang, 1.0)
        k = cv2.warpAffine(base, M, (3, 3), flags=cv2.INTER_LINEAR)
        bancos.append(k)
    y0 = int(g.shape[0] * 0.74)
    fig, ax = plt.subplots(3, 4, figsize=(15, 5.6))
    for j, (ang, k) in enumerate(zip(angulos, bancos)):
        conv = cv2.filter2D(g, cv2.CV_32F, k)[y0:]
        relu = np.maximum(conv, 0)
        pool = cv2.resize(relu, (relu.shape[1] // 2, relu.shape[0] // 2), interpolation=cv2.INTER_AREA)   # aproximación de pooling por promedio
        pool = relu[:relu.shape[0] // 2 * 2, :relu.shape[1] // 2 * 2].reshape(relu.shape[0] // 2, 2, relu.shape[1] // 2, 2).max(axis=(1, 3))
        ax[0, j].imshow(k, cmap="coolwarm", vmin=-2, vmax=2)
        for (r, c), v in np.ndenumerate(k): ax[0, j].text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8)
        ax[0, j].set_title(f"Kernel orientado {ang}°")
        ax[1, j].imshow(np.clip(conv / np.percentile(np.abs(conv), 99.5), -1, 1), cmap="gray"); ax[1, j].set_title("Convolución")
        ax[2, j].imshow(np.clip(pool / np.percentile(pool, 99.5), 0, 1), cmap="gray"); ax[2, j].set_title("ReLU + max-pooling 2×2")
        for a in ax[:, j]: a.axis("off")
    fig.suptitle("Una capa convolucional de una CNN: banco de kernels → no linealidad → reducción de resolución", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "figC_capa_cnn.png", dpi=140); plt.close(fig)

    (AQUI / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
