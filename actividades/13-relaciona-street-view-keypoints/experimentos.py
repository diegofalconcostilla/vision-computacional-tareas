"""Keypoints y matching entre dos vistas consecutivas de una calle (idea de Google Street View).

Google Street View no se usa (sus imágenes tienen derechos y condiciones que no permiten descargarlas ni procesarlas de forma automática):
se emplea una fotografía de calle CC0 y una segunda vista con un ligero cambio de perspectiva de verdad conocida.
    python experimentos.py -> figuras/, resultados.json, datos/
"""
import json
import sys
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
    from herramientas import puntos as P  # noqa: E402
except ImportError:            # dentro del zip de entrega, puntos.py está junto al script
    import puntos as P

FIG, DAT = AQUI / "figuras", AQUI / "datos"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
METODOS = ["harris", "orb", "sift"]
NOMBRE = {"harris": "Harris + parche", "orb": "ORB", "sift": "SIFT"}
RES = {}


def main():
    esc = cv2.resize(cv2.imread(str(RAIZ / "datos_compartidos" / "calle_ashley.jpg")), (1280, 720), interpolation=cv2.INTER_AREA)
    esc = esc[:660]          # se descarta la franja inferior: fecha y ubicación impresas por la dashcam (no forman parte de la escena)
    h, w = esc.shape[:2]
    FOCO = 900
    H = P.homografia_rotacion(w, h, FOCO, yaw=7, pitch=-2, roll=1.5)
    v2 = P.vista(esc, H, brillo=0.95, ruido=2)
    cv2.imwrite(str(DAT / "vista_1.jpg"), esc, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(str(DAT / "vista_2.jpg"), v2, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ---- fig1: las dos vistas
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    ax[0].imshow(cv2.cvtColor(esc, cv2.COLOR_BGR2RGB)); ax[0].set_title("Vista 1 (foto: M. Rivera, CC0)")
    ax[1].imshow(cv2.cvtColor(v2, cv2.COLOR_BGR2RGB)); ax[1].set_title("Vista 2: giro de la cámara de 7° (yaw), −2° (pitch) y 1.5° (roll)")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig1_vistas.png", dpi=140); plt.close(fig)

    # ---- evaluación de los tres métodos
    salidas = {}
    for m in METODOS:
        r, (k1, k2, ms) = P.evaluar(esc, v2, H, m, n=2000, ratio=0.8)
        salidas[m] = (r, k1, k2, ms)
        RES[m] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}

    # ---- fig2: keypoints y mapa de dónde caen (SIFT) + calidad por región
    k1 = salidas["sift"][1]
    ms = salidas["sift"][3]
    filas, cols = 4, 6
    dens = np.zeros((filas, cols)); ok = np.zeros((filas, cols)); mal = np.zeros((filas, cols))
    for k in k1:
        dens[min(int(k.pt[1] / h * filas), filas - 1), min(int(k.pt[0] / w * cols), cols - 1)] += 1
    k2 = salidas["sift"][2]
    for mt in ms:
        p1 = np.array(k1[mt.queryIdx].pt); p2 = np.array(k2[mt.trainIdx].pt)
        bien = np.linalg.norm(P.proyectar([p1], H)[0] - p2) <= 3.0
        (ok if bien else mal)[min(int(p1[1] / h * filas), filas - 1), min(int(p1[0] / w * cols), cols - 1)] += 1
    RES["regiones_sift"] = {"filas": filas, "cols": cols, "keypoints": dens.astype(int).tolist(), "correctos": ok.astype(int).tolist(), "erroneos": mal.astype(int).tolist()}

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    vis = esc.copy()
    for k in k1[:800]: cv2.circle(vis, (int(k.pt[0]), int(k.pt[1])), 3, (0, 220, 255), -1)
    ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"{len(k1)} keypoints SIFT (se dibujan 800)")
    for a, mat, t, cm in ((ax[1], dens, "Keypoints SIFT por región", "viridis"), (ax[2], mal, "Correspondencias erróneas por región (SIFT)", "Reds")):
        a.imshow(cv2.cvtColor(esc, cv2.COLOR_BGR2RGB), alpha=0.35)
        im = a.imshow(mat, cmap=cm, alpha=0.65, extent=(0, w, h, 0), aspect="auto")
        for i in range(filas):
            for j in range(cols): a.text((j + .5) * w / cols, (i + .5) * h / filas, int(mat[i, j]), ha="center", va="center", color="white" if cm == "viridis" else "black", fontsize=8)
        a.set_title(t)
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig2_keypoints_regiones.png", dpi=140); plt.close(fig)

    # ---- fig3: correspondencias
    fig, ax = plt.subplots(3, 1, figsize=(15, 13.2))
    for a, m in zip(ax, METODOS):
        r, kk1, kk2, mm = salidas[m]
        a.imshow(cv2.cvtColor(P.dibujar_matches(esc, kk1, v2, kk2, mm, H, max_n=70), cv2.COLOR_BGR2RGB)); a.axis("off")
        a.set_title(f"{NOMBRE[m]}: {r['correctos']} correctas de {r['matches']} ({r['precision']*100:.1f} %), {r['inliers_ransac']} inliers RANSAC; se muestran las 70 mejores")
    fig.tight_layout(); fig.savefig(FIG / "fig3_correspondencias.png", dpi=130); plt.close(fig)

    # ---- barrido del cambio de perspectiva (giro yaw creciente)
    yaws = [2, 5, 10, 15, 20, 30, 40]
    barrido = {m: [] for m in METODOS}
    for y in yaws:
        Hy = P.homografia_rotacion(w, h, FOCO, yaw=y)
        vy = P.vista(esc, Hy, brillo=0.95, ruido=2)
        for m in METODOS:
            r, _ = P.evaluar(esc, vy, Hy, m, n=2000, ratio=0.8)
            barrido[m].append({"yaw": y, "correctos": r["correctos"], "matches": r["matches"], "precision": round(r["precision"], 3),
                               "repetibilidad": round(r["repetibilidad"], 3)})
    RES["barrido_yaw"] = barrido
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.9))
    for m in METODOS:
        ax[0].plot(yaws, [b["correctos"] for b in barrido[m]], "o-", label=NOMBRE[m])
        ax[1].plot(yaws, [b["precision"] * 100 for b in barrido[m]], "o-", label=NOMBRE[m])
        ax[2].plot(yaws, [b["repetibilidad"] * 100 for b in barrido[m]], "o-", label=NOMBRE[m])
    for a, t, yl in zip(ax, ("Correspondencias correctas", "Precisión de las correspondencias (%)", "Repetibilidad del detector (%)"), ("número", "%", "%")):
        a.set_title(t); a.set_xlabel("giro de la cámara (yaw, grados)"); a.set_ylabel(yl); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig4_cambio_perspectiva.png", dpi=140); plt.close(fig)

    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: RES[k] for k in METODOS}, indent=1, ensure_ascii=False))
    print(json.dumps(RES["barrido_yaw"], ensure_ascii=False))
    print("keypoints:\n", dens.astype(int), "\nerróneos:\n", mal.astype(int), "\ncorrectos:\n", ok.astype(int))


if __name__ == "__main__":
    main()
