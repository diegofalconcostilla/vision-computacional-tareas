"""Panorama con keypoints y descriptores: SIFT y ORB (SURF no está disponible en OpenCV estándar por patente).

Como no se dispone de fotos propias con solapamiento, se generan 3 vistas de una foto real 4K con giros exactos de la cámara (cada vista es una
homografía de la foto original, es decir, una cámara que solo rota, como al hacer un panorama en el teléfono), con distinta exposición.
Al conocerse la geometría verdadera, se puede medir la calidad del emparejamiento y de la unión.
    python experimentos.py -> figuras/, datos/, resultados.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
try:
    from herramientas import puntos as P
except ImportError:              # dentro del zip de entrega, puntos.py está junto al script
    import puntos as P

FIG, DAT = AQUI / "figuras", AQUI / "datos"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
RES = {}
W, H_ = 1280, 720


def rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def crear_vistas():
    src = cv2.imread(str(RAIZ / "datos_compartidos" / "calle_oak_alden.jpg"))
    hs, ws = src.shape[:2]
    fs = 950.0                               # la foto fuente mide 1920 px de ancho: focal supuesta de 950 px (campo de visión ≈ 90°)
    Ks = P.matriz_K(ws, hs, fs)
    fv = 1200.0
    Kv = P.matriz_K(W, H_, fv)
    yaws = (15.0, 0.0, -15.0)               # con este convenio, yaw > 0 mira hacia la izquierda de la escena
    exp = (0.85, 1.0, 1.12)                 # la cámara ajusta la exposición en cada toma
    vistas, Hs = [], []
    for yaw, br in zip(yaws, exp):
        R = P.rotacion(yaw=yaw)
        Hsv = Kv @ R @ np.linalg.inv(Ks)     # foto original -> vista
        v = cv2.warpPerspective(src, Hsv, (W, H_), flags=cv2.INTER_LINEAR)
        v = np.clip(v.astype(np.float32) * br, 0, 255)
        v = np.clip(v + np.random.default_rng(int(yaw) + 100).normal(0, 2.0, v.shape), 0, 255).astype(np.uint8)
        vistas.append(v); Hs.append(Hsv)
    return vistas, Hs


def homografia_entre(Hi, Hj):
    """Homografía verdadera de la vista i a la vista j."""
    return Hj @ np.linalg.inv(Hi)


def peso_borde(mask):
    d = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return d / max(d.max(), 1)


def unir(vistas, Hs_a_centro):
    """Lleva cada vista al plano de la central, compensa la exposición y mezcla con pesos que decrecen hacia los bordes."""
    h, w = vistas[0].shape[:2]
    esq = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    todas = np.vstack([P.proyectar(esq, Hc) for Hc in Hs_a_centro])
    xmin, ymin = np.floor(todas.min(axis=0)); xmax, ymax = np.ceil(todas.max(axis=0))
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
    tam = (int(xmax - xmin), int(ymax - ymin))
    warp = [cv2.warpPerspective(v, T @ Hc, tam) for v, Hc in zip(vistas, Hs_a_centro)]
    masks = [cv2.warpPerspective(np.full(v.shape[:2], 255, np.uint8), T @ Hc, tam) for v, Hc in zip(vistas, Hs_a_centro)]
    # compensación de exposición: la vista central es la referencia; la ganancia se calcula en la zona de solape
    ganancias = [1.0, 1.0, 1.0]
    for i in (0, 2):
        solape = (masks[i] > 0) & (masks[1] > 0)
        ganancias[i] = float(warp[1][solape].mean() / max(warp[i][solape].mean(), 1e-3))
    def mezcla(compensar):
        acum = np.zeros(warp[0].shape, np.float64); pesos = np.zeros(warp[0].shape[:2], np.float64)
        for k in range(3):
            g = ganancias[k] if compensar else 1.0
            wk = peso_borde(masks[k]) ** 2 + 1e-6 * (masks[k] > 0)
            acum += warp[k].astype(np.float64) * g * wk[..., None]; pesos += wk
        return np.clip(acum / np.maximum(pesos, 1e-9)[..., None], 0, 255).astype(np.uint8)
    return mezcla(True), mezcla(False), ganancias, masks


def main():
    vistas, Hs = crear_vistas()
    nombres = ["izquierda", "centro", "derecha"]
    for n, v in zip(nombres, vistas): cv2.imwrite(str(DAT / f"foto_{n}.jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ---- Parte 1: las fotos
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
    for a, v, n in zip(ax, vistas, nombres):
        a.imshow(rgb(v)); a.set_title(f"Toma {n}"); a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig1_tomas.png", dpi=140); plt.close(fig)

    # ---- Parte 2: keypoints con SIFT y ORB en cada toma, por región
    grises = [P.gris(v) for v in vistas]
    kp = {}
    for m in ("sift", "orb"):
        kp[m] = [P.detectar(g, m, 3000)[0] for g in grises]
    RES["keypoints"] = {m: [len(k) for k in kp[m]] for m in kp}
    filas, cols = 4, 6
    def dens(kps):
        d = np.zeros((filas, cols))
        for k in kps: d[min(int(k.pt[1] / H_ * filas), filas - 1), min(int(k.pt[0] / W * cols), cols - 1)] += 1
        return d
    d_sift = dens(kp["sift"][1])
    RES["densidad_sift_centro"] = d_sift.astype(int).tolist()
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.2))
    for j, (v, n) in enumerate(zip(vistas, nombres)):
        for i, m in enumerate(("sift", "orb")):
            vis = cv2.drawKeypoints(v, kp[m][j], None, color=(0, 255, 0) if m == "sift" else (0, 200, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if m == "sift" else 0)
            ax[i, j].imshow(rgb(vis)); ax[i, j].set_title(f"{m.upper()} en la toma {n}: {len(kp[m][j])} keypoints"); ax[i, j].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig2_keypoints.png", dpi=130); plt.close(fig)

    # densidad por región y efecto de la iluminación
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.9))
    ax[0].imshow(rgb(vistas[1]), alpha=.4); ax[0].imshow(d_sift, cmap="viridis", alpha=.65, extent=(0, W, H_, 0), aspect="auto")
    for i in range(filas):
        for j in range(cols): ax[0].text((j + .5) * W / cols, (i + .5) * H_ / filas, int(d_sift[i, j]), ha="center", va="center", color="white", fontsize=8)
    ax[0].set_title("Keypoints SIFT por región (toma central)"); ax[0].axis("off")
    brillos = [0.25, 0.5, 1.0, 1.6, 2.5]
    cuentas = {"sift": [], "orb": []}; correctos = {"sift": [], "orb": []}
    base = vistas[1]
    for b in brillos:
        vb = np.clip(base.astype(np.float32) * b, 0, 255).astype(np.uint8)
        for m in ("sift", "orb"):
            r, _ = P.evaluar(base, vb, np.eye(3), m, n=3000, ratio=0.8)
            cuentas[m].append(r["kp2"]); correctos[m].append(r["correctos"])
    RES["iluminacion"] = {"brillos": brillos, "keypoints": cuentas, "correctos": correctos}
    for m, c in (("sift", "#2a9d3a"), ("orb", "#e08a00")):
        ax[1].plot(brillos, cuentas[m], "o-", color=c, label=f"{m.upper()}: keypoints en la imagen alterada")
        ax[1].plot(brillos, correctos[m], "s--", color=c, alpha=.6, label=f"{m.upper()}: correspondencias correctas con la original")
    ax[1].set_xscale("log"); ax[1].set_xlabel("factor de brillo aplicado a la toma central"); ax[1].set_title("Efecto de la iluminación"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig3_regiones_iluminacion.png", dpi=140); plt.close(fig)

    # ---- Parte 3: matching entre tomas adyacentes
    pares = [(0, 1), (1, 2)]
    matching = {}
    fig, ax = plt.subplots(4, 1, figsize=(15, 17.5))
    fila = 0
    homos = {}
    for (i, j) in pares:
        Hv = homografia_entre(Hs[i], Hs[j])
        for m in ("sift", "orb"):
            ratio = 0.75 if m == "sift" else 0.8
            r, (k1, k2, ms) = P.evaluar(vistas[i], vistas[j], Hv, m, n=3000, ratio=ratio)
            matching[f"{nombres[i]}-{nombres[j]} {m}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
            ax[fila].imshow(rgb(P.dibujar_matches(vistas[i], k1, vistas[j], k2, ms, Hv, max_n=70)))
            ax[fila].set_title(f"{m.upper()} entre {nombres[i]} y {nombres[j]}: {r['correctos']} correctas de {r['matches']} ({r['precision']*100:.0f} %); 70 mejores (verde = correcta, rojo = errónea)")
            ax[fila].axis("off"); fila += 1
            if m == "sift":
                pts1 = np.float32([k1[a.queryIdx].pt for a in ms]); pts2 = np.float32([k2[a.trainIdx].pt for a in ms])
                Hest, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
                homos[(i, j)] = Hest
    fig.tight_layout(); fig.savefig(FIG / "fig4_matching.png", dpi=110); plt.close(fig)
    RES["matching"] = matching

    # ---- Parte 4: panorama (SIFT + RANSAC + exposición + mezcla) y comparación con cv2.Stitcher
    H_0_1 = homos[(0, 1)]                    # izquierda -> centro
    H_2_1 = np.linalg.inv(homos[(1, 2)])     # derecha -> centro
    Hs_c = [H_0_1, np.eye(3), H_2_1]
    t0 = time.perf_counter()
    pano, pano_sin, gan, masks = unir(vistas, Hs_c)
    t_unir = time.perf_counter() - t0
    cv2.imwrite(str(DAT / "panorama.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # error de las homografías estimadas frente a la verdad (en las esquinas de la vista)
    esq = np.array([[0, 0], [W, 0], [W, H_], [0, H_]], np.float64)
    H_v_0_1 = homografia_entre(Hs[0], Hs[1]); H_v_2_1 = homografia_entre(Hs[2], Hs[1])
    err = {"izquierda->centro": float(np.linalg.norm(P.proyectar(esq, H_0_1) - P.proyectar(esq, H_v_0_1), axis=1).mean()),
           "derecha->centro": float(np.linalg.norm(P.proyectar(esq, H_2_1) - P.proyectar(esq, H_v_2_1), axis=1).mean())}
    t0 = time.perf_counter()
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    estado, pano_cv = stitcher.stitch(vistas)
    t_cv = time.perf_counter() - t0
    if estado == cv2.Stitcher_OK: cv2.imwrite(str(DAT / "panorama_cv2_stitcher.jpg"), pano_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # visibilidad de la costura: diferencia media entre tomas en la zona de solape, antes y después de compensar la exposición
    def dif_solape(wa, wb, ma, mb, ga=1.0):
        s = (ma > 0) & (mb > 0)
        return float(np.abs(wa[s].astype(np.float32) * ga - wb[s].astype(np.float32)).mean())
    hh = pano.shape[:2]
    Tt = np.eye(3)
    xs = np.vstack([P.proyectar(esq, Hc) for Hc in Hs_c]); Tt[0, 2] = -np.floor(xs[:, 0].min()); Tt[1, 2] = -np.floor(xs[:, 1].min())
    w_i = [cv2.warpPerspective(v, Tt @ Hc, (hh[1], hh[0])) for v, Hc in zip(vistas, Hs_c)]
    d_sin = dif_solape(w_i[0], w_i[1], masks[0], masks[1]); d_con = dif_solape(w_i[0], w_i[1], masks[0], masks[1], gan[0])
    RES["panorama"] = {"tam": [int(pano.shape[1]), int(pano.shape[0])], "ganancias": [round(g, 3) for g in gan], "error_homografia_px": {k: round(v, 3) for k, v in err.items()},
                       "dif_solape_sin_compensar": round(d_sin, 2), "dif_solape_compensada": round(d_con, 2), "t_union_s": round(t_unir, 2),
                       "stitcher_ok": bool(estado == cv2.Stitcher_OK), "stitcher_tam": [int(pano_cv.shape[1]), int(pano_cv.shape[0])] if estado == cv2.Stitcher_OK else None, "t_stitcher_s": round(t_cv, 2)}
    fig, ax = plt.subplots(3 if estado == cv2.Stitcher_OK else 2, 1, figsize=(15, 11))
    ax[0].imshow(rgb(pano_sin)); ax[0].set_title("Panorama sin compensar exposición (se nota el escalón de brillo entre tomas)")
    ax[1].imshow(rgb(pano)); ax[1].set_title("Panorama propio: SIFT + RANSAC + compensación de exposición + mezcla con peso decreciente")
    if estado == cv2.Stitcher_OK: ax[2].imshow(rgb(pano_cv)); ax[2].set_title("Resultado de cv2.Stitcher (referencia)")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig5_panorama.png", dpi=120); plt.close(fig)

    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(RES, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
