"""Segmentación de Otsu en tres imágenes médicas en escala de grises: radiografía de tórax, tomografía de cráneo y microscopía de fluorescencia.

Para cada una: imagen original, histograma con el umbral, imagen segmentada y explicación del umbral (medias y pesos de las clases, eficacia η).
Además se mide la estabilidad del umbral ante ruido y el efecto de un suavizado previo.
    python experimentos.py -> figuras/, resultados.json
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, threshold_otsu

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
    rx = cv2.imread(str(fuente("chest_xray.jpg")), 0)
    rx = cv2.resize(rx, (900, round(rx.shape[0] * 900 / rx.shape[1])), interpolation=cv2.INTER_AREA)
    ct = cv2.imread(str(fuente("ct_cerebro_axial.png")), 0)[:, :355]              # se recorta el corte axial (sin el recuadro de la vista sagital)
    nu = cv2.imread(str(fuente("celulas_hela_dapi.jpg")))[..., 1]                # canal verde = núcleos (DAPI)
    return {"Radiografía de tórax": rx, "TC de cráneo (corte axial)": ct, "Microscopía de fluorescencia (núcleos)": nu}


def explicar(g):
    t = threshold_otsu(g)
    a, b = g[g <= t].astype(float), g[g > t].astype(float)
    w0, w1 = a.size / g.size, b.size / g.size
    sb = w0 * w1 * (a.mean() - b.mean()) ** 2
    return {"umbral": int(t), "media_clase_oscura": round(float(a.mean()), 1), "media_clase_clara": round(float(b.mean()), 1),
            "peso_clase_oscura_pct": round(w0 * 100, 1), "peso_clase_clara_pct": round(w1 * 100, 1), "eta": round(float(sb / g.astype(float).var()), 3)}


def dice(a, b):
    a, b = a > 0, b > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def pulmones(mascara_oscura):
    """Zonas oscuras grandes cuyo centroide cae en la parte central de la imagen (el aire exterior queda en los laterales); se conservan las dos mayores.
    Se abre con un elemento grande para romper las conexiones finas entre los pulmones y el aire exterior."""
    m = cv2.morphologyEx(mascara_oscura.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    h, w = m.shape
    cand = [(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if 0.18 * w < cen[i][0] < 0.82 * w and 0.15 * h < cen[i][1] < 0.85 * h and st[i, cv2.CC_STAT_AREA] > 0.02 * h * w]
    cand.sort(reverse=True)
    out = np.zeros_like(m)
    for _, i in cand[:2]: out[lab == i] = 1
    return ndi.binary_fill_holes(out).astype(np.uint8)


def main():
    imgs = cargar()
    resumen = {}
    fig, ax = plt.subplots(3, 4, figsize=(19, 12))
    for fila, (nombre, g) in enumerate(imgs.items()):
        info = explicar(g)
        t = info["umbral"]
        ax[fila, 0].imshow(g, cmap="gray"); ax[fila, 0].set_title(f"{nombre}\n{g.shape[1]}×{g.shape[0]} px")
        ax[fila, 1].hist(g.ravel(), 128, color="#444"); ax[fila, 1].axvline(t, color="r", label=f"Otsu, t = {t}")
        if "TC" in nombre:
            tm = threshold_multiotsu(g, classes=3)
            for k, tt in enumerate(tm): ax[fila, 1].axvline(tt, color="orange", ls="--", label="multi-Otsu (3 clases)" if k == 0 else None)
            info["multiotsu_umbrales"] = [int(x) for x in tm]
        ax[fila, 1].set_yscale("log"); ax[fila, 1].legend(fontsize=8); ax[fila, 1].set_title(f"Histograma (escala log), η = {info['eta']}")
        seg = g > t
        ax[fila, 2].imshow(seg, cmap="gray"); ax[fila, 2].set_title("Segmentación de Otsu (clara = píxeles > t)")
        # tercera columna: resultado de interés
        if "Radiografía" in nombre:
            pul = pulmones(~seg)
            info["pulmones_pct_imagen"] = round(float(pul.mean() * 100), 1)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); vis[pul > 0] = (0.55 * vis[pul > 0] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
            ax[fila, 3].imshow(vis[..., ::-1]); ax[fila, 3].set_title(f"Zonas oscuras centrales (pulmones + fuga a las esquinas): {info['pulmones_pct_imagen']} %")
        elif "TC" in nombre:
            tm = threshold_multiotsu(g, classes=3)
            reg = np.digitize(g, bins=tm)
            info["multiotsu_pct_clases"] = [round(float((reg == k).mean() * 100), 1) for k in range(3)]
            ax[fila, 3].imshow(reg, cmap="viridis"); ax[fila, 3].set_title("Multi-Otsu: 0 aire/fondo, 1 tejido blando, 2 hueso")
        else:
            m = cv2.morphologyEx(seg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            areas = st[1:, cv2.CC_STAT_AREA]
            info["nucleos"] = int(n - 1); info["area_mediana_px"] = float(np.median(areas)); info["fondo_pct"] = round(float((~seg).mean() * 100), 1)
            ax[fila, 3].imshow(g, cmap="gray"); ax[fila, 3].contour(m.astype(float), levels=[0.5], colors="lime", linewidths=.6)
            ax[fila, 3].set_title(f"Contornos de {info['nucleos']} núcleos detectados")
        for a in (ax[fila, 0], ax[fila, 2], ax[fila, 3]): a.axis("off")
        resumen[nombre] = info
    fig.tight_layout(); fig.savefig(FIG / "fig1_resultados.png", dpi=110); plt.close(fig)
    RES["imagenes"] = resumen

    # ---- estabilidad ante ruido y efecto del suavizado previo (Dice entre la segmentación con ruido y la de la imagen limpia)
    rng = np.random.default_rng(0)
    estab = {}
    for nombre, g in imgs.items():
        base = g > threshold_otsu(g)
        estab[nombre] = {}
        for s in (10, 25, 40):
            gn = np.clip(g.astype(np.float32) + rng.normal(0, s, g.shape), 0, 255).astype(np.uint8)
            fila = {}
            for etiqueta, pre in (("sin suavizar", gn), ("gaussiano σ=1.5", cv2.GaussianBlur(gn, (0, 0), 1.5)), ("mediana 5×5", cv2.medianBlur(gn, 5))):
                t = threshold_otsu(pre); fila[etiqueta] = {"umbral": int(t), "dice_vs_limpia": round(dice(pre > t, base), 3)}
            estab[nombre][f"σ={s}"] = fila
    RES["estabilidad_ruido"] = estab
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    for a, (nombre, d) in zip(ax, estab.items()):
        xs = list(d.keys())
        for etiqueta, c in (("sin suavizar", "#c62828"), ("gaussiano σ=1.5", "#1f77b4"), ("mediana 5×5", "#2a9d3a")):
            a.plot(xs, [d[x][etiqueta]["dice_vs_limpia"] for x in xs], "o-", color=c, label=etiqueta)
        a.set_title(nombre, fontsize=9); a.set_ylabel("Dice frente a la segmentación sin ruido"); a.set_ylim(0.3, 1.02); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig2_ruido.png", dpi=130); plt.close(fig)
    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(RES, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
