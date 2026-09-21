"""Comparación de métodos clásicos de segmentación en una resonancia cerebral con tumor.

Imagen: resonancia T1 con contraste con un meningioma cuyo contorno fue dibujado a mano en rojo por quien la publicó (Wikimedia Commons, CC BY-SA 3.0).
Ese contorno se usa como verdad aproximada; los algoritmos reciben la imagen SIN la línea roja (reconstruida con inpainting).
Los métodos son semiautomáticos: reciben un punto semilla dentro del tumor (un clic), como en una herramienta clínica de asistencia.
    python experimentos.py -> figuras/, datos/, resultados.json
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel, threshold_multiotsu, threshold_otsu
from skimage.segmentation import mark_boundaries, slic, watershed

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
FIG, DAT = AQUI / "figuras", AQUI / "datos"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
RES = {}


def fuente(nombre):
    ruta = DAT / nombre
    return ruta if ruta.exists() else RAIZ / "datos_compartidos" / nombre


def cargar():
    bgr = cv2.imread(str(fuente("mri_meningioma.jpg")))
    b, g, r = [bgr[..., i].astype(int) for i in range(3)]
    rojo = ((r > 130) & (g < 110) & (b < 110)).astype(np.uint8)
    rojo_d = cv2.dilate(rojo, np.ones((3, 3), np.uint8))
    cerrado = cv2.morphologyEx(rojo_d, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    gt = ndi.binary_fill_holes(cerrado > 0)
    limpio = cv2.inpaint(bgr, rojo_d * 255, 3, cv2.INPAINT_TELEA)
    gris = cv2.cvtColor(limpio, cv2.COLOR_BGR2GRAY)
    dist = ndi.distance_transform_edt(gt)
    semilla = tuple(int(v) for v in np.unravel_index(np.argmax(dist), dist.shape))     # (fila, columna) del punto más interior
    return gris, gt.astype(np.uint8), semilla


def componente_semilla(mascara, semilla):
    """Se queda con el componente conexo que contiene la semilla (o el más cercano a ella)."""
    m = (mascara > 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    l = lab[semilla]
    if l == 0:
        d = ndi.distance_transform_edt(lab == 0, return_indices=True)[1]
        l = lab[d[0][semilla], d[1][semilla]]
    return (lab == l).astype(np.uint8)


def limpiar(m, k=3):
    return cv2.morphologyEx(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


# ------------------------------------------------------------------ métodos
def m_otsu(g, semilla):
    t = threshold_multiotsu(g, classes=3)
    return componente_semilla(limpiar(g > t[-1]), semilla), {"umbrales": [int(x) for x in t]}


def m_otsu2(g, semilla):
    t = threshold_otsu(g)
    return componente_semilla(limpiar(g > t), semilla), {"umbral": int(t)}


def m_kmeans(g, semilla, k=4):
    z = g.reshape(-1, 1).astype(np.float32)
    cv2.setRNGSeed(0)
    _, etiq, centros = cv2.kmeans(z, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 5, cv2.KMEANS_PP_CENTERS)
    top = int(np.argmax(centros))
    m = (etiq.reshape(g.shape) == top)
    return componente_semilla(limpiar(m), semilla), {"centros": sorted(int(c) for c in centros.ravel())}


def m_meanshift(g, semilla):
    filtrada = cv2.pyrMeanShiftFiltering(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), sp=10, sr=18, maxLevel=1)
    fg = cv2.cvtColor(filtrada, cv2.COLOR_BGR2GRAY)
    t = threshold_multiotsu(fg, classes=3)
    return componente_semilla(limpiar(fg > t[-1]), semilla), {"colores_distintos": int(len(np.unique(fg)))}


def m_slic(g, semilla, n=350):
    seg = slic(g, n_segments=n, compactness=0.08, channel_axis=None, start_label=0, sigma=1.0)
    medias = ndi.mean(g, labels=seg, index=np.arange(seg.max() + 1))
    t = threshold_multiotsu(medias.astype(np.float32), classes=3)[-1]
    m = np.isin(seg, np.nonzero(medias > t)[0])
    return componente_semilla(limpiar(m), semilla), {"superpixeles": int(seg.max() + 1)}


def m_watershed(g, semilla):
    """Watershed controlado por marcadores: el tumor se marca con un disco en la semilla y el fondo con los píxeles muy oscuros."""
    suave = cv2.GaussianBlur(g, (0, 0), 1.5)
    grad = sobel(suave.astype(np.float32))
    marcas = np.zeros(g.shape, np.int32)
    marcas[suave < 0.25 * threshold_otsu(g)] = 1           # fondo: aire / negro
    cv2.circle(marcas.view(np.uint8) if False else marcas, (semilla[1], semilla[0]), 7, 2, -1)
    # marcadores de "otro tejido": píxeles de intensidad media-baja (materia cerebral) separados del tumor brillante
    t = threshold_multiotsu(g, classes=3)
    marcas[(suave > 0.5 * t[0]) & (suave < 0.9 * t[1]) & (marcas == 0)] = 1
    lab = watershed(grad, marcas)
    return componente_semilla(limpiar(lab == 2), semilla), {}


def watershed_sin_marcadores(g):
    grad = sobel(cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32))
    lab = watershed(grad)
    return lab, int(lab.max())


METODOS = {"Otsu (2 clases)": m_otsu2, "Multi-Otsu (3 clases)": m_otsu, "K-means (K=4)": m_kmeans, "Mean Shift": m_meanshift, "SLIC (350 superpíxeles)": m_slic, "Watershed con marcadores": m_watershed}


def metricas(pred, gt):
    p, g = pred > 0, gt > 0
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    return {"dice": float(2 * tp / max(2 * tp + fp + fn, 1)), "iou": float(tp / max(tp + fp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)), "exhaustividad": float(tp / max(tp + fn, 1)), "area_pct": float(p.mean() * 100)}


def borde(ax, m, color, lw=1.4):
    ax.contour(m.astype(float), levels=[0.5], colors=color, linewidths=lw)


def main():
    g, gt, semilla = cargar()
    cv2.imwrite(str(DAT / "resonancia_sin_contorno.png"), g)
    cv2.imwrite(str(DAT / "verdad_aproximada.png"), gt * 255)
    RES["imagen"] = {"tam": list(g.shape), "semilla_fila_col": list(semilla), "tumor_pct": round(float(gt.mean() * 100), 2), "otsu2_umbral": int(threshold_otsu(g))}

    # fig1: imagen, histograma y verdad
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    ax[0].imshow(g, cmap="gray"); ax[0].set_title("Resonancia T1 con contraste (contorno rojo eliminado)"); ax[0].plot(semilla[1], semilla[0], "c+", ms=14)
    ax[1].imshow(g, cmap="gray"); borde(ax[1], gt, "lime"); ax[1].set_title("Verdad aproximada: contorno del tumor (dibujado a mano)")
    ax[2].hist(g[g > 5].ravel(), 128, color="#444"); ax[2].axvline(threshold_otsu(g), color="r", label="Otsu (2 clases)")
    for i, t in enumerate(threshold_multiotsu(g, classes=3)): ax[2].axvline(t, color="orange", ls="--", label="multi-Otsu" if i == 0 else None)
    ax[2].hist(g[gt > 0].ravel(), 128, color="lime", alpha=.6, label="píxeles del tumor"); ax[2].set_title("Histograma (sin el fondo negro)"); ax[2].legend(); ax[2].set_yticks([])
    for a in ax[:2]: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig1_imagen_y_verdad.png", dpi=140); plt.close(fig)

    # fig2: resultados sin ruido
    res0 = {}
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.6))
    ax[0, 0].imshow(g, cmap="gray"); ax[0, 0].set_title("Original"); borde(ax[0, 0], gt, "lime")
    for a, (nombre, f) in zip(ax.ravel()[1:], METODOS.items()):
        m, extra = f(g, semilla)
        mt = metricas(m, gt); res0[nombre] = {**{k: round(v, 3) for k, v in mt.items()}, **extra}
        a.imshow(g, cmap="gray"); borde(a, gt, "lime"); borde(a, m, "red")
        a.set_title(f"{nombre}\nDice {mt['dice']:.2f} · P {mt['precision']:.2f} · R {mt['exhaustividad']:.2f}")
    lab, n_ws = watershed_sin_marcadores(g)
    ax[1, 3].imshow(mark_boundaries(np.dstack([g] * 3), lab, color=(1, 0.2, 0.2)) if n_ws < 3000 else np.dstack([g] * 3)); ax[1, 3].set_title(f"Watershed SIN marcadores: {n_ws} regiones")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Verde: contorno de referencia del tumor. Rojo: resultado de cada método", y=0.995)
    fig.tight_layout(); fig.savefig(FIG / "fig2_metodos.png", dpi=120); plt.close(fig)
    RES["sin_ruido"] = res0
    RES["sobresegmentacion"] = {"watershed_sin_marcadores_regiones": n_ws, "slic_superpixeles": res0["SLIC (350 superpíxeles)"]["superpixeles"]}

    # sensibilidad al ruido
    sigmas = [0, 10, 25, 40]
    metodos_ruido = {n: f for n, f in METODOS.items() if n != "Otsu (2 clases)"}    # Otsu de 2 clases se excluye: su resultado depende de un artefacto de conectividad (ver informe)
    dice = {n: [] for n in metodos_ruido}; masks_25 = {}
    rng = np.random.default_rng(0)
    for s in sigmas:
        gn = np.clip(g.astype(np.float32) + rng.normal(0, s, g.shape), 0, 255).astype(np.uint8) if s else g
        for nombre, f in metodos_ruido.items():
            try:
                m, _ = f(gn, semilla)
            except Exception:
                m = np.zeros_like(gt)
            dice[nombre].append(round(metricas(m, gt)["dice"], 3))
            if s == 25: masks_25[nombre] = m
        if s == 25: g25 = gn
    RES["ruido"] = {"sigmas": sigmas, "dice": dice}
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    for n, d in dice.items(): ax[0].plot(sigmas, d, "o-", label=n)
    ax[0].set_xlabel("desviación estándar del ruido gaussiano añadido (niveles de gris)"); ax[0].set_ylabel("coeficiente de Dice"); ax[0].set_title("Sensibilidad al ruido"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.02)
    ax[1].imshow(g25, cmap="gray"); borde(ax[1], gt, "lime")
    for (n, m), c in zip(masks_25.items(), ("red", "orange", "cyan", "magenta", "yellow")): borde(ax[1], m, c, 1.0)
    ax[1].set_title("Con ruido σ=25: contorno de referencia (verde) y de cada método"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig3_ruido.png", dpi=130); plt.close(fig)

    # segunda imagen sin verdad: glioblastoma T1 con contraste (solo cualitativo)
    g2 = cv2.imread(str(fuente("mri_glioblastoma_t1km.jpg")), 0)
    sem2 = (int(g2.shape[0] * 0.55), int(g2.shape[1] * 0.42))
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.0))
    ax[0].imshow(g2, cmap="gray"); ax[0].plot(sem2[1], sem2[0], "c+", ms=12); ax[0].set_title("Glioblastoma T1 con contraste (sin verdad)")
    for a, (nombre) in zip(ax[1:], ("Multi-Otsu (3 clases)", "K-means (K=4)", "SLIC (350 superpíxeles)")):
        try:
            m, _ = METODOS[nombre](g2, sem2)
        except Exception:
            m = np.zeros_like(g2)
        a.imshow(g2, cmap="gray"); borde(a, m, "red"); a.set_title(nombre + f"\nárea {m.mean() * 100:.1f} %")
    for a in ax: a.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "fig4_segunda_imagen.png", dpi=130); plt.close(fig)

    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(RES, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
