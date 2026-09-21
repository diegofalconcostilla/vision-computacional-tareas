"""Estabilización digital con procesamiento en frecuencia.

Parte A. Imagen con desenfoque por movimiento: espectro de Fourier, estimación del movimiento (cepstrum) y restauración (filtro de Wiener).
Parte B. Video con vibración: movimiento entre cuadros por correlación de fase (Fourier), filtrado pasa bajas de la trayectoria y compensación.
Los datos de entrada son reales (fotografía CC0); la vibración y el desenfoque se simulan con parámetros conocidos para poder medir el resultado.
    python experimentos.py  ->  figuras/, resultados.json, datos/
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
FIG, DAT = AQUI / "figuras", AQUI / "datos"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
RES = {}


def fuente(nombre):
    ruta = DAT / nombre
    return ruta if ruta.exists() else RAIZ / "datos_compartidos" / nombre


def psf_movimiento(L, ang_deg, tam=None):
    tam = tam or (int(L) | 1) + 4
    k = np.zeros((tam, tam), np.float32)
    c = tam // 2
    dx, dy = np.cos(np.radians(ang_deg)) * (L - 1) / 2, -np.sin(np.radians(ang_deg)) * (L - 1) / 2
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return k / k.sum()


def otf(psf, forma):
    """Función de transferencia (FFT de la PSF centrada en el origen)."""
    p = np.zeros(forma, np.float64)
    p[:psf.shape[0], :psf.shape[1]] = psf
    p = np.roll(p, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    return np.fft.fft2(p)


def espectro_log(img):
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img.astype(float)))))


def parte_a():
    g = cv2.imread(str(fuente("calle_patterson.jpg")), 0)
    g = cv2.resize(g, (960, 540), interpolation=cv2.INTER_AREA)
    nitida = g.astype(np.float64)
    L, ang = 31, 25.0
    H = otf(psf_movimiento(L, ang), g.shape)
    rng = np.random.default_rng(2)
    borrosa = np.real(np.fft.ifft2(np.fft.fft2(nitida) * H)) + rng.normal(0, 2.0, g.shape)
    borrosa_u8 = np.clip(borrosa, 0, 255)
    cv2.imwrite(str(DAT / "imagen_nitida.png"), g)
    cv2.imwrite(str(DAT / "imagen_movida.png"), borrosa_u8.astype(np.uint8))

    # --- espectro y cepstrum: el desenfoque lineal deja rayas oscuras (ceros del sinc) y un pico negativo en el cepstrum a distancia L
    S_n, S_b = espectro_log(nitida), espectro_log(borrosa_u8)
    ventana = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    logF = np.log(np.abs(np.fft.fft2(borrosa_u8 * ventana)) + 1e-3)
    cep = np.fft.fftshift(np.real(np.fft.ifft2(logF)))
    cy, cx = np.array(cep.shape) // 2
    m = np.ones_like(cep, bool); m[cy - 6:cy + 7, cx - 6:cx + 7] = False        # excluye el centro
    lim = np.abs(np.mgrid[0:cep.shape[0], 0:cep.shape[1]][0] - cy) < 80; lim &= np.abs(np.mgrid[0:cep.shape[0], 0:cep.shape[1]][1] - cx) < 80
    zona = np.where(m & lim, cep, np.inf)
    iy, ix = np.unravel_index(np.argmin(zona), zona.shape)
    dy, dx = iy - cy, ix - cx
    L_est = float(np.hypot(dx, dy))
    ang_est = float(np.degrees(np.arctan2(-dy, dx)) % 180)
    if ang_est > 90 and abs(ang_est - 180 - ang) < abs(ang_est - ang): ang_est -= 180
    ang_est = float(ang_est % 180)
    RES["A_estimacion"] = {"L_real": L, "L_estimada": round(L_est, 1), "angulo_real": ang, "angulo_estimado": round(ang_est, 1)}

    # --- restauración con filtro de Wiener: F = conj(H) / (|H|^2 + K) * G ; se prueban varias K (oráculo) y una K por regla
    Hest = otf(psf_movimiento(int(round(L_est)) | 1, ang_est), g.shape)
    G = np.fft.fft2(borrosa_u8)
    def wiener(Hm, K): return np.clip(np.real(np.fft.ifft2(np.conj(Hm) / (np.abs(Hm) ** 2 + K) * G)), 0, 255)
    Ks = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    tabla = {}
    for K in Ks:
        r = wiener(H, K)
        tabla[K] = (psnr(nitida, r, data_range=255), ssim(nitida, r, data_range=255), r)
    Kbest = max(Ks, key=lambda k: tabla[k][0])
    r_est = wiener(Hest, Kbest)
    inverso = np.clip(np.real(np.fft.ifft2(np.conj(H) / (np.abs(H) ** 2 + 1e-12) * G)), 0, 255)
    RES["A_restauracion"] = {
        "movida": {"psnr": round(psnr(nitida, borrosa_u8, data_range=255), 2), "ssim": round(ssim(nitida, borrosa_u8, data_range=255), 3)},
        "inverso_directo": {"psnr": round(psnr(nitida, inverso, data_range=255), 2), "ssim": round(ssim(nitida, inverso, data_range=255), 3)},
        "wiener_K": {f"{K:g}": {"psnr": round(v[0], 2), "ssim": round(v[1], 3)} for K, v in tabla.items()},
        "mejor_K": Kbest,
        "wiener_con_PSF_estimada": {"psnr": round(psnr(nitida, r_est, data_range=255), 2), "ssim": round(ssim(nitida, r_est, data_range=255), 3)}}
    cv2.imwrite(str(DAT / "imagen_restaurada.png"), tabla[Kbest][2].astype(np.uint8))

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].imshow(nitida, cmap="gray"); ax[0, 0].set_title("Imagen nítida (referencia)")
    ax[0, 1].imshow(borrosa_u8, cmap="gray"); ax[0, 1].set_title(f"Con desenfoque por movimiento (L={L} px, {ang:.0f}°) + ruido")
    ax[0, 2].imshow(np.log1p(np.abs(np.fft.fftshift(H))), cmap="gray"); ax[0, 2].set_title("|H(u,v)| del desenfoque (función sinc)")
    ax[1, 0].imshow(S_n, cmap="gray"); ax[1, 0].set_title("Espectro de la nítida (log)")
    ax[1, 1].imshow(S_b, cmap="gray"); ax[1, 1].set_title("Espectro de la movida: rayas oscuras")
    rec = cep[cy - 60:cy + 61, cx - 60:cx + 61]
    ax[1, 2].imshow(rec, cmap="gray", vmin=cep[iy, ix] * 1.3, vmax=-cep[iy, ix] * 1.3); ax[1, 2].plot(ix - (cx - 60), iy - (cy - 60), "r+", ms=14)
    ax[1, 2].set_title(f"Cepstrum: pico negativo → L={L_est:.1f}, ángulo={ang_est:.0f}°")
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig1_espectro_movimiento.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 3.9))
    for a, im, t in zip(ax, (borrosa_u8, inverso, tabla[Kbest][2], r_est),
                        ("Movida", f"Filtro inverso (K≈0)\nPSNR {psnr(nitida, inverso, data_range=255):.1f} dB",
                         f"Wiener (PSF real), K={Kbest:g}\nPSNR {tabla[Kbest][0]:.1f} dB", f"Wiener (PSF estimada), K={Kbest:g}\nPSNR {psnr(nitida, r_est, data_range=255):.1f} dB")):
        a.imshow(im[150:400, 250:700], cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig2_restauracion.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def parte_b():
    src = cv2.imread(str(fuente("carretera_bemiss_4k.jpg")))
    src = cv2.resize(src, (2400, 1350), interpolation=cv2.INTER_AREA)
    W, H, N, fps = 960, 540, 150, 30
    t = np.arange(N) / fps
    rng = np.random.default_rng(4)
    pan_x = 400 + 600 * t / t[-1]                                  # movimiento intencional lento
    pan_y = 300 + 60 * np.sin(2 * np.pi * t / t[-1] * 0.5)
    def vib(a):                                                     # vibración de 3 a 9 Hz + un poco de ruido
        v = sum(ai * np.sin(2 * np.pi * f * t + p) for ai, f, p in zip(a, (3.2, 5.7, 8.6), rng.uniform(0, 6.28, 3)))
        return v + rng.normal(0, 0.6, N)
    jx, jy = vib((7, 4, 2.5)), vib((6, 3.5, 2))
    px, py = pan_x + jx, pan_y + jy                                 # trayectoria verdadera de la cámara (en px de la fuente)
    frames = []
    for i in range(N):
        M = np.float32([[1, 0, -px[i]], [0, 1, -py[i]]])
        frames.append(cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)

    # --- movimiento entre cuadros por correlación de fase (cociente de espectros cruzados, Kuglin & Hines, 1975)
    d = np.array([cv2.phaseCorrelate(gray[i - 1], gray[i], win)[0] for i in range(1, N)])   # (dx, dy) del contenido entre i-1 e i
    est = np.vstack([[0, 0], np.cumsum(d, axis=0)])
    est_x, est_y = -est[:, 0], -est[:, 1]                           # el contenido se mueve al revés que la cámara
    est_x += px[0]; est_y += py[0]
    err_traj = np.hypot(est_x - px, est_y - py)

    # --- filtrado pasa bajas de la trayectoria en el dominio de la frecuencia (gaussiano en frecuencia), fc en Hz
    def suavizar(x, fc):
        # La FFT supone una señal periódica: una rampa (el paneo) crea un salto artificial entre el último y el primer valor.
        # Se resta la recta que une los extremos, se filtra en frecuencia y se vuelve a sumar.
        tend = np.linspace(x[0], x[-1], len(x))
        Fx = np.fft.rfft(x - tend); f = np.fft.rfftfreq(len(x), 1 / fps)
        return np.fft.irfft(Fx * np.exp(-(f ** 2) / (2 * fc ** 2)), n=len(x)) + tend
    fc = 0.8
    sx_est, sy_est = suavizar(est_x, fc), suavizar(est_y, fc)
    sx_true, sy_true = suavizar(px, fc), suavizar(py, fc)
    # Convención: el cuadro i muestra src(x + p_i). Para simular una cámara en la posición suave s_i hay que remuestrear
    # el cuadro en x + (s_i - est_i): es una traslación del contenido de -(s_i - est_i) = -corr_i.
    corr_x, corr_y = sx_est - est_x, sy_est - est_y
    estab = [cv2.warpAffine(frames[i], np.float32([[1, 0, -corr_x[i]], [0, 1, -corr_y[i]]]), (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT) for i in range(N)]
    # Posición efectiva de la cámara tras compensar = p + corr (si la estimación fuera perfecta, sería exactamente la suave)
    ef_x, ef_y = px + corr_x, py + corr_y
    antes = np.hypot(px - sx_true, py - sy_true)          # vibración: distancia entre la trayectoria real y su versión suave
    despues = np.hypot(ef_x - sx_true, ef_y - sy_true)
    def mad(fr):
        c = (slice(int(H * .12), int(H * .88)), slice(int(W * .12), int(W * .88)))
        return float(np.mean([np.abs(fr[i][c].astype(np.float32) - fr[i - 1][c].astype(np.float32)).mean() for i in range(1, len(fr))]))
    a0 = 60
    ideal = [cv2.warpAffine(src, np.float32([[1, 0, -pan_x[i]], [0, 1, -pan_y[i]]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for i in range(N)]
    RES["B"] = {"cuadros": N, "fps": fps, "fc_hz": fc,
                "rms_vibracion_antes_px": round(float(np.sqrt(np.mean(antes ** 2))), 2), "rms_vibracion_despues_px": round(float(np.sqrt(np.mean(despues ** 2))), 2),
                "error_estimacion_trayectoria_rms_px": round(float(np.sqrt(np.mean(err_traj ** 2))), 3),
                "dif_media_entre_cuadros_antes": round(mad(frames), 2), "dif_media_entre_cuadros_despues": round(mad(estab), 2),
                "dif_media_entre_cuadros_limite_ideal": round(mad(ideal), 2)}
    # barrido de la frecuencia de corte
    barrido = {}
    for f_c in (0.3, 0.8, 2.0, 4.0, 7.0):
        sxe, sye = suavizar(est_x, f_c), suavizar(est_y, f_c); sxt, syt = suavizar(px, f_c), suavizar(py, f_c)
        ex, ey = px + (sxe - est_x), py + (sye - est_y)
        barrido[f_c] = {"rms_residual_px": round(float(np.sqrt(np.mean(np.hypot(ex - sxt, ey - syt) ** 2))), 2),
                        "rms_vs_pan_ideal_px": round(float(np.sqrt(np.mean(np.hypot(ex - pan_x, ey - pan_y) ** 2))), 2)}
    RES["B_barrido_fc"] = barrido

    cv2.imwrite(str(DAT / "cuadro_vibrado.jpg"), frames[a0]); cv2.imwrite(str(DAT / "cuadro_estabilizado.jpg"), estab[a0])
    for nombre, fr in (("video_original", frames), ("video_estabilizado", estab)):
        vw = cv2.VideoWriter(str(DAT / f"{nombre}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        for f in fr: vw.write(f)
        vw.release()

    # --- figuras
    fig, ax = plt.subplots(2, 2, figsize=(14, 7.2))
    ax[0, 0].plot(t, px - px.mean(), label="cámara real (con vibración)", lw=1); ax[0, 0].plot(t, est_x - px.mean(), "--", label="estimada (corr. de fase)", lw=1)
    ax[0, 0].plot(t, sx_true - px.mean(), label=f"suavizada (pasa bajas {fc} Hz)", lw=2); ax[0, 0].set_title("Trayectoria horizontal de la cámara"); ax[0, 0].set_xlabel("s"); ax[0, 0].set_ylabel("px"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    f = np.fft.rfftfreq(N, 1 / fps)
    quitar = lambda x: x - np.linspace(x[0], x[-1], len(x))          # sin la rampa del paneo, para ver la vibración
    ax[0, 1].semilogy(f, np.abs(np.fft.rfft(quitar(px))) ** 2 / N, label="antes"); ax[0, 1].semilogy(f, np.abs(np.fft.rfft(quitar(ef_x))) ** 2 / N, label="después")
    for f0 in (3.2, 5.7, 8.6): ax[0, 1].axvline(f0, color="gray", ls=":", lw=.8)
    ax[0, 1].axvline(fc, color="r", ls="--", lw=.8, label=f"corte {fc} Hz"); ax[0, 1].set_title("Espectro de la trayectoria horizontal (sin rampa; punteado = vibración)"); ax[0, 1].set_xlabel("Hz"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    ax[1, 0].plot(t, antes, label="vibración residual antes"); ax[1, 0].plot(t, despues, label="después"); ax[1, 0].set_title("Error respecto a la trayectoria suave deseada"); ax[1, 0].set_xlabel("s"); ax[1, 0].set_ylabel("px"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    ax[1, 1].plot(list(barrido), [v["rms_residual_px"] for v in barrido.values()], "o-", label="residuo vs. trayectoria suavizada al mismo corte")
    ax[1, 1].plot(list(barrido), [v["rms_vs_pan_ideal_px"] for v in barrido.values()], "s-", label="distancia al movimiento intencional verdadero"); ax[1, 1].set_xscale("log")
    ax[1, 1].set_title("Elección de la frecuencia de corte"); ax[1, 1].set_xlabel("frecuencia de corte (Hz)"); ax[1, 1].set_ylabel("RMS (px)"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig3_trayectoria.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(15, 6.6))
    for j, i in enumerate((a0, a0 + 1, a0 + 2)):
        ax[0, j].imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)[150:420, 300:800]); ax[0, j].set_title(f"Original, cuadro {i}")
        ax[1, j].imshow(cv2.cvtColor(estab[i], cv2.COLOR_BGR2RGB)[150:420, 300:800]); ax[1, j].set_title(f"Estabilizado, cuadro {i}")
        for a in ax[:, j]: a.axis("off"); a.axhline(120, color="yellow", lw=.6); a.axvline(200, color="yellow", lw=.6)
    fig.suptitle("Misma región de tres cuadros consecutivos (las líneas amarillas marcan una posición fija de referencia)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig4_cuadros.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    parte_a()
    parte_b()
    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(RES, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
