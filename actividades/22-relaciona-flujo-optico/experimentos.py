"""Flujo óptico sparse (Lucas-Kanade) y denso (Farnebäck) sobre una cine de resonancia cardíaca de cuatro cámaras.

Parte A. Video real (50 cuadros, un ciclo cardíaco): trayectorias LK, mapas densos de Farnebäck, curva de movimiento, tiempos y acuerdo entre métodos.
Parte B. Validación con flujo verdadero conocido (campo sintético aplicado a un cuadro real): error de punto final (EPE) de cada método
        según la amplitud del movimiento y el ruido.
    python experimentos.py -> figuras/, datos/ (videos), resultados.json
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
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
FIG, DAT = AQUI / "figuras", AQUI / "datos"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
RES = {}

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def fuente():
    ruta = DAT / "resonancia_cine_4c.gif"
    return ruta if ruta.exists() else RAIZ / "datos_compartidos" / "resonancia_cine_4c.gif"


def cargar_cuadros():
    im = Image.open(fuente())
    cuadros = []
    for i in range(im.n_frames):
        im.seek(i); cuadros.append(np.array(im.convert("L")))
    return cuadros


def rastrear(frames, pts0, fb_umbral=1.0):
    """Pirámide de Lucas-Kanade cuadro a cuadro con verificación adelante-atrás; devuelve trayectorias (N, T, 2) y máscara de vida."""
    N, T = len(pts0), len(frames)
    tray = np.full((N, T, 2), np.nan, np.float32)
    vivo = np.zeros((N, T), bool)
    tray[:, 0] = pts0[:, 0, :]; vivo[:, 0] = True
    p = pts0.copy().astype(np.float32)
    activo = np.ones(N, bool)
    for t in range(1, T):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[t - 1], frames[t], p, None, **LK)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(frames[t], frames[t - 1], p1, None, **LK)
        err = np.linalg.norm(p0b[:, 0] - p[:, 0], axis=1)
        ok = activo & (st[:, 0] == 1) & (st2[:, 0] == 1) & (err < fb_umbral)
        activo = ok
        tray[ok, t] = p1[ok, 0]; vivo[ok, t] = True
        p = np.where(ok[:, None, None], p1, p)
    return tray, vivo


def hsv_flujo(flujo, max_mag=None):
    mag, ang = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
    hsv = np.zeros(flujo.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    m = max_mag or max(np.percentile(mag, 99), 1e-6)
    hsv[..., 2] = np.clip(mag / m * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mag, ang


def muestrear(flujo, pts):
    """Flujo en posiciones subpíxel (interpolación bilineal)."""
    fx = cv2.remap(flujo[..., 0], pts[:, 0:1].astype(np.float32).reshape(1, -1), pts[:, 1:2].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)
    fy = cv2.remap(flujo[..., 1], pts[:, 0:1].astype(np.float32).reshape(1, -1), pts[:, 1:2].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)
    return np.stack([fx[0], fy[0]], axis=1)


def parte_a():
    frames = cargar_cuadros()
    T = len(frames); h, w = frames[0].shape
    RES["video"] = {"cuadros": T, "alto": h, "ancho": w, "fps_original": 25}
    f0 = frames[0]
    pts0 = cv2.goodFeaturesToTrack(f0, maxCorners=200, qualityLevel=0.03, minDistance=9, blockSize=7)
    RES["sparse_puntos_iniciales"] = int(len(pts0))

    # ---------- sparse
    t0 = time.perf_counter()
    tray, vivo = rastrear(frames, pts0)
    t_sparse = (time.perf_counter() - t0) / (T - 1)
    supervivientes = vivo.sum(axis=0)
    completos = vivo[:, -1]
    deriva = np.linalg.norm(tray[completos, -1] - tray[completos, 0], axis=1)         # el video es un ciclo: debería volver al inicio
    despl = np.nanmax(np.linalg.norm(tray - tray[:, :1], axis=2), axis=1)               # máximo desplazamiento respecto del inicio
    RES["sparse"] = {"supervivientes_ultimo": int(completos.sum()), "supervivientes_por_cuadro_min": int(supervivientes.min()),
                     "deriva_media_px": round(float(deriva.mean()), 2), "deriva_mediana_px": round(float(np.median(deriva)), 2),
                     "desplazamiento_max_mediana_px": round(float(np.nanmedian(despl[completos])), 2), "desplazamiento_max_p95_px": round(float(np.nanpercentile(despl[completos], 95)), 2),
                     "tiempo_ms_por_par": round(t_sparse * 1000, 2)}

    # ---------- denso
    flujos = []
    t0 = time.perf_counter()
    for t in range(1, T):
        flujos.append(cv2.calcOpticalFlowFarneback(frames[t - 1], frames[t], None, **FB))
    t_dense = (time.perf_counter() - t0) / (T - 1)
    flujos = np.array(flujos)                                                            # (T-1, h, w, 2)
    mags = np.linalg.norm(flujos, axis=3)
    # región del corazón: donde el video tiene movimiento apreciable en todo el ciclo (percentil 60 de la magnitud media temporal)
    mov_medio = mags.mean(axis=0)
    roi = mov_medio > np.percentile(mov_medio, 60)
    curva = mags[:, roi].mean(axis=1)
    curva_global = mags.reshape(T - 1, -1).mean(axis=1)
    RES["dense"] = {"tiempo_ms_por_par": round(t_dense * 1000, 2), "mag_media_global_px": round(float(mags.mean()), 3), "mag_media_roi_px": round(float(mags[:, roi].mean()), 3),
                    "mag_max_p99_px": round(float(np.percentile(mags, 99)), 2), "curva_roi_min": round(float(curva.min()), 3), "curva_roi_max": round(float(curva.max()), 3),
                    "cuadro_max_movimiento": int(np.argmax(curva) + 1), "cuadro_min_movimiento": int(np.argmin(curva) + 1), "roi_pct": round(float(roi.mean() * 100), 1)}
    # autocorrelación de la curva (periodicidad) y cierre del ciclo con el flujo denso
    pos = pts0[completos, 0, :].copy()
    pos_d = pts0[:, 0, :].copy()
    for t in range(T - 1):
        pos_d = pos_d + muestrear(flujos[t], pos_d)
    deriva_densa = np.linalg.norm(pos_d[completos] - pts0[completos, 0, :], axis=1)
    RES["dense"]["deriva_ciclo_puntos_px_media"] = round(float(deriva_densa.mean()), 2)
    # acuerdo entre métodos: posición tras 20 cuadros integrando el flujo denso vs. la trayectoria LK
    k = 20
    pos_d = pts0[:, 0, :].copy()
    for t in range(k): pos_d = pos_d + muestrear(flujos[t], pos_d)
    vivos_k = vivo[:, k]
    dif = np.linalg.norm(pos_d[vivos_k] - tray[vivos_k, k], axis=1)
    mov_k = np.linalg.norm(tray[vivos_k, k] - tray[vivos_k, 0], axis=1)
    RES["acuerdo"] = {"cuadro": k, "puntos": int(vivos_k.sum()), "dif_media_px": round(float(dif.mean()), 2), "dif_mediana_px": round(float(np.median(dif)), 2),
                      "movimiento_medio_px": round(float(mov_k.mean()), 2), "correlacion_desplazamientos": round(float(np.corrcoef(np.linalg.norm(pos_d[vivos_k] - pts0[vivos_k, 0], axis=1), mov_k)[0, 1]), 3)}

    # ---------- figuras
    fases = [0, 12, 25, 37]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    for a, t in zip(ax, fases):
        a.imshow(frames[t], cmap="gray"); a.set_title(f"Cuadro {t}")
        if t == 0: a.plot(pts0[:, 0, 0], pts0[:, 0, 1], "r.", ms=4); a.set_title("Cuadro 0 y puntos de Shi-Tomasi (rojo)")
        a.axis("off")
    fig.suptitle("Cine de resonancia cardíaca de cuatro cámaras (50 cuadros = un ciclo). Fuente: Doregan, CC BY-SA 4.0", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig1_cuadros.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    ax[0].imshow(f0, cmap="gray")
    cmap = plt.get_cmap("turbo")
    for i in np.nonzero(completos)[0]:
        tr = tray[i]; c = cmap(min(despl[i] / max(np.percentile(despl[completos], 98), 1e-6), 1))
        ax[0].plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.1)
    ax[0].set_title(f"Trayectorias LK durante el ciclo ({int(completos.sum())} de {len(pts0)} puntos sobreviven)"); ax[0].axis("off")
    ax[1].plot(supervivientes, "o-", ms=3); ax[1].set_xlabel("cuadro"); ax[1].set_ylabel("puntos rastreados"); ax[1].set_title("Puntos que sobreviven (verificación adelante-atrás < 1 px)"); ax[1].grid(alpha=.3)
    ax[2].hist(deriva, 20, color="#1f77b4"); ax[2].set_xlabel("distancia entre posición final e inicial (px)"); ax[2].set_title(f"Cierre del ciclo: mediana {np.median(deriva):.2f} px"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_sparse.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(17, 8.4))
    vmax = float(np.percentile(mags, 99))
    for j, t in enumerate(fases):
        rgb, mag, ang = hsv_flujo(flujos[min(t, T - 2)], vmax)
        ax[0, j].imshow(rgb); ax[0, j].set_title(f"Flujo denso entre cuadros {min(t, T-2)} y {min(t, T-2) + 1} (color = dirección, brillo = magnitud)", fontsize=8)
        ax[1, j].imshow(frames[min(t, T - 2)], cmap="gray")
        ys, xs = np.mgrid[6:h:14, 6:w:14]
        u, v = flujos[min(t, T - 2)][ys, xs, 0], flujos[min(t, T - 2)][ys, xs, 1]
        ax[1, j].quiver(xs, ys, u, v, color="yellow", angles="xy", scale_units="xy", scale=0.25, width=0.003)
        ax[1, j].set_title("Vectores (flechas ×4)")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig3_dense.png", dpi=125); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 3.9))
    ax[0].plot(np.arange(1, T), curva, "o-", ms=3, label="región con más movimiento"); ax[0].plot(np.arange(1, T), curva_global, "s--", ms=3, alpha=.7, label="toda la imagen")
    ax[0].set_xlabel("cuadro"); ax[0].set_ylabel("magnitud media del flujo (px/cuadro)"); ax[0].set_title("Movimiento a lo largo del ciclo"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].imshow(frames[0], cmap="gray"); im = ax[1].imshow(np.ma.masked_where(~roi, mov_medio), cmap="magma", alpha=.7); ax[1].set_title(f"Región con más movimiento ({RES['dense']['roi_pct']} % de la imagen)"); ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.04, label="magnitud media (px)")
    ax[2].scatter(np.linalg.norm(pos_d[vivos_k] - pts0[vivos_k, 0], axis=1), mov_k, s=10); lim = max(mov_k.max(), 1)
    ax[2].plot([0, lim], [0, lim], "r--", lw=1); ax[2].set_xlabel("desplazamiento con flujo denso integrado (px)"); ax[2].set_ylabel("desplazamiento LK (px)")
    ax[2].set_title(f"Acuerdo entre métodos tras {k} cuadros (r = {RES['acuerdo']['correlacion_desplazamientos']})"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig4_comparacion.png", dpi=140); plt.close(fig)

    # ---------- videos (ampliados 2x)
    def escribir(nombre, cuadros):
        vw = cv2.VideoWriter(str(DAT / nombre), cv2.VideoWriter_fourcc(*"mp4v"), 12, (w * 2, h * 2))
        for c in cuadros: vw.write(c)
        vw.release()
    orig = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), (w * 2, h * 2), interpolation=cv2.INTER_CUBIC) for f in frames]
    sp = []
    for t in range(T):
        c = orig[t].copy()
        for i in np.nonzero(completos)[0]:
            tr = tray[i, max(0, t - 12):t + 1]
            if len(tr) > 1: cv2.polylines(c, [np.round(tr * 2).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(c, tuple(np.round(tray[i, t] * 2).astype(int)), 3, (0, 0, 255), -1)
        sp.append(c)
    dn = []
    for t in range(T):
        fl = flujos[min(t, T - 2)]
        rgb, _, _ = hsv_flujo(fl, vmax)
        dn.append(cv2.addWeighted(orig[t], 0.45, cv2.cvtColor(cv2.resize(rgb, (w * 2, h * 2)), cv2.COLOR_RGB2BGR), 0.55, 0))
    escribir("video_original.mp4", orig); escribir("video_sparse_lk.mp4", sp); escribir("video_denso_farneback.mp4", dn)
    cv2.imwrite(str(DAT / "captura_sparse.png"), sp[25]); cv2.imwrite(str(DAT / "captura_denso.png"), dn[25])
    return frames


def campo_verdad(h, w, s):
    """Campo de flujo hacia atrás B (definido en la imagen 1): mezcla suave de contracción radial, rotación y traslación."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = (xs - w / 2) / 200, (ys - h / 2) / 200
    bx = s * (0.6 * rx - 0.3 * ry + 0.4); by = s * (0.6 * ry + 0.3 * rx - 0.2)
    return np.stack([bx, by], axis=2).astype(np.float32)


def crear_par(f0, s, ruido=0.0, semilla=0):
    h, w = f0.shape
    B = campo_verdad(h, w, s)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    f1 = cv2.remap(f0, xs + B[..., 0], ys + B[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)     # I1(y) = I0(y + B(y))
    rng = np.random.default_rng(semilla)
    a, b = f1.astype(np.float32), f0.astype(np.float32)
    if ruido > 0:
        a = np.clip(a + rng.normal(0, ruido, a.shape), 0, 255); b = np.clip(b + rng.normal(0, ruido, b.shape), 0, 255)
    return np.clip(b, 0, 255).astype(np.uint8), np.clip(a, 0, 255).astype(np.uint8), B


def evaluar_par(f0, f1, B):
    """Se estima el flujo de la imagen 1 a la imagen 0, cuyo valor verdadero en cada píxel de la imagen 1 es B."""
    h, w = f1.shape
    pts = cv2.goodFeaturesToTrack(f1, maxCorners=200, qualityLevel=0.03, minDistance=9, blockSize=7)
    out = {}
    if pts is None: return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(f1, f0, pts, None, **LK)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(f0, f1, p1, None, **LK)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb[:, 0] - pts[:, 0], axis=1) < 1.0)
    gt_pts = muestrear(B, pts[:, 0, :])
    est_lk = (p1[:, 0] - pts[:, 0])
    epe_lk = np.linalg.norm(est_lk - gt_pts, axis=1)
    fb = cv2.calcOpticalFlowFarneback(f1, f0, None, **FB)
    est_fb_pts = muestrear(fb, pts[:, 0, :])
    epe_fb_pts = np.linalg.norm(est_fb_pts - gt_pts, axis=1)
    epe_fb_all = np.linalg.norm(fb - B, axis=2)
    return {"lk_epe_media": float(epe_lk[ok].mean()) if ok.any() else float("nan"), "lk_puntos_ok": int(ok.sum()), "lk_puntos": int(len(pts)),
            "fb_epe_mismos_puntos": float(epe_fb_pts[ok].mean()) if ok.any() else float("nan"), "fb_epe_imagen": float(epe_fb_all.mean()), "fb_epe_p90": float(np.percentile(epe_fb_all, 90)),
            "mov_max": float(np.linalg.norm(B, axis=2).max()), "mov_medio": float(np.linalg.norm(B, axis=2).mean())}


def parte_b(frames):
    f0 = frames[0]
    amps = [0.5, 1, 2, 4, 8, 16, 32]
    barrido = []
    for s in amps:
        a, b, B = crear_par(f0, s)
        r = evaluar_par(a, b, B); r["s"] = s; barrido.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    RES["validacion_amplitud"] = barrido
    # ¿El fallo de Farnebäck con movimientos grandes es del método o de la configuración? (OpenCV limita la pirámide: detiene los niveles
    # cuando la imagen baja de 32 px, así que con 3, 5 o 6 niveles el resultado es idéntico; lo que sí cambia es el tamaño de ventana)
    ajustes = {"base (winsize 15)": dict(FB), "winsize 31": dict(FB, winsize=31), "winsize 51": dict(FB, winsize=51)}
    param = []
    for s_ in (16, 32):
        a_, b_, B_ = crear_par(f0, s_)
        fila = {"s": s_, "mov_max": round(float(np.linalg.norm(B_, axis=2).max()), 1)}
        for nombre, kw in ajustes.items():
            fl = cv2.calcOpticalFlowFarneback(b_, a_, None, **kw)
            fila[nombre] = round(float(np.linalg.norm(fl - B_, axis=2).mean()), 3)
        param.append(fila)
    RES["farneback_ventana"] = param
    ruidos = [0, 5, 10, 20, 30]
    bruido = []
    for n in ruidos:
        a, b, B = crear_par(f0, 2, ruido=n)
        r = evaluar_par(a, b, B); r["ruido"] = n; bruido.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    RES["validacion_ruido"] = bruido

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    mm = [b["mov_max"] for b in barrido]
    ax[0].plot(mm, [b["lk_epe_media"] for b in barrido], "o-", label="Lucas-Kanade (puntos rastreados)")
    ax[0].plot(mm, [b["fb_epe_mismos_puntos"] for b in barrido], "s-", label="Farnebäck (mismos puntos)")
    ax[0].plot(mm, [b["fb_epe_imagen"] for b in barrido], "^--", label="Farnebäck (toda la imagen)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("movimiento máximo verdadero (px)"); ax[0].set_ylabel("error de punto final medio (px)"); ax[0].set_title("Error según la amplitud del movimiento"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
    ax[1].plot(ruidos, [b["lk_epe_media"] for b in bruido], "o-", label="Lucas-Kanade"); ax[1].plot(ruidos, [b["fb_epe_mismos_puntos"] for b in bruido], "s-", label="Farnebäck (mismos puntos)")
    ax[1].plot(ruidos, [b["fb_epe_imagen"] for b in bruido], "^--", label="Farnebäck (toda la imagen)")
    ax[1].set_xlabel("ruido gaussiano añadido (desv. est., niveles)"); ax[1].set_ylabel("error de punto final medio (px)"); ax[1].set_title("Error según el ruido (movimiento máx. ≈ 2.9 px)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    a, b, B = crear_par(f0, 4)
    fb = cv2.calcOpticalFlowFarneback(b, a, None, **FB)
    err = np.linalg.norm(fb - B, axis=2)
    im = ax[2].imshow(err, cmap="inferno", vmax=max(np.percentile(err, 99), 0.5)); ax[2].set_title("Error de Farnebäck por píxel (movimiento máx. ≈ 5.9 px)"); ax[2].axis("off")
    plt.colorbar(im, ax=ax[2], fraction=0.04, label="px")
    fig.tight_layout(); fig.savefig(FIG / "fig5_validacion.png", dpi=140); plt.close(fig)


def main():
    frames = parte_a()
    parte_b(frames)
    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(RES, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
