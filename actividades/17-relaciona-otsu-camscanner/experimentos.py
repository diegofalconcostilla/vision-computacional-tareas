"""Segmentación clásica de un documento fotografiado (flujo tipo CamScanner).

Etapas: bordes (Canny) / umbral (Otsu) -> región conectada y contorno -> esquinas -> corrección de perspectiva -> binarización adaptativa.
Parte A. Tres fotografías reales de recibos (dominio público / CC BY-SA) con perspectiva, sombras y fondos difíciles.
Parte B. Escenas sintéticas con esquinas y texto conocidos para medir cada etapa (tipo de fondo y fuerza de la sombra).
    python experimentos.py -> figuras/, resultados.json
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_otsu, threshold_sauvola

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
FIG, DAT = AQUI / "figuras", AQUI / "datos"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
RES = {}
K_ELIP = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def fuente(nombre):
    ruta = DAT / nombre
    return ruta if ruta.exists() else RAIZ / "datos_compartidos" / nombre


# ------------------------------------------------------------------- detección del documento
def reducir(img, lado=900):
    h, w = img.shape[:2]
    e = lado / max(h, w)
    return (cv2.resize(img, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA) if e < 1 else img.copy()), e


def ordenar(p):
    p = np.asarray(p, np.float32).reshape(4, 2)
    s, d = p.sum(axis=1), np.diff(p, axis=1)[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], np.float32)     # tl, tr, br, bl


def cuadrilatero(mascara):
    """Contorno externo mayor -> casco convexo -> aproximación poligonal a 4 vértices. Devuelve (esquinas, exacto)."""
    cs, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None, False
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * mascara.shape[0] * mascara.shape[1]: return None, False
    hull = cv2.convexHull(c); per = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(hull, eps * per, True)
        if len(ap) == 4: return ordenar(ap), True
    return ordenar(cv2.boxPoints(cv2.minAreaRect(hull))), False


def mascara_otsu(gris):
    b = cv2.GaussianBlur(gris, (7, 7), 0)
    t, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K_ELIP(25))              # el texto oscuro deja huecos: se cierran
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K_ELIP(15))               # quita puentes finos con el fondo
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cs: cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    return out, int(t)


def bordes_canny(gris):
    b = cv2.GaussianBlur(gris, (5, 5), 0)
    mediana = np.median(b)
    e = cv2.Canny(b, int(max(0, 0.66 * mediana)), int(min(255, 1.33 * mediana)))          # umbrales a partir de la mediana
    e = cv2.dilate(e, K_ELIP(3), iterations=2)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, K_ELIP(21))
    cs, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(e)
    if cs: cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    return out, e


def detectar(img, metodo):
    """Devuelve (esquinas en píxeles de img, exacto, máscara reducida). metodo: 'otsu' o 'canny'."""
    peq, e = reducir(img)
    gris = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY) if peq.ndim == 3 else peq
    if metodo == "otsu": m, _ = mascara_otsu(gris)
    else: m, _ = bordes_canny(gris)
    q, ok = cuadrilatero(m)
    return (None if q is None else q / e), ok, m


def rectificar(img, esq, tam=None):
    tl, tr, br, bl = esq
    if tam is None:
        W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))); H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    else:
        W, H = tam
    if tam is None: dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    else: dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])                    # misma convención con que se generó la escena sintética
    M = cv2.getPerspectiveTransform(esq.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------- binarización
def bin_fija(g, t=128): return (g < t).astype(np.uint8) * 255
def bin_otsu(g):
    b = cv2.GaussianBlur(g, (3, 3), 0); return (b < threshold_otsu(b)).astype(np.uint8) * 255
def bin_adaptativa(g, bloque=41, c=15):
    return cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bloque, c)
def bin_sauvola(g, ventana=41, k=0.2):
    b = cv2.GaussianBlur(g, (3, 3), 0); return (b < threshold_sauvola(b, window_size=ventana, k=k)).astype(np.uint8) * 255
def limpiar(m, k=2):
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def f1(pred, gt, tol=1):
    """F1 con tolerancia espacial (trazos finos): un píxel predicho es correcto si está a <= tol px del texto verdadero, y un píxel de texto se
    recupera si hay un píxel predicho a <= tol px. Devuelve (F1, fracción de píxeles de fondo -a más de 2 px del texto- marcados como texto)."""
    from scipy.ndimage import distance_transform_edt as edt
    p, g = pred > 0, gt > 0
    dg = edt(~g); dp = edt(~p)
    prec = float((dg[p] <= tol).mean()) if p.any() else 0.0
    rec = float((dp[g] <= tol).mean()) if g.any() else 0.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lejos = dg > 2
    return f, float((p & lejos).sum() / max(lejos.sum(), 1))


def alinear(pred, gt):
    """Traslada `pred` para compensar un pequeño desplazamiento global (error de esquinas), por correlación de fase de las máscaras suavizadas."""
    a = cv2.GaussianBlur(gt.astype(np.float32) / 255, (0, 0), 3); b = cv2.GaussianBlur(pred.astype(np.float32) / 255, (0, 0), 3)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(pred, M, (pred.shape[1], pred.shape[0]), flags=cv2.INTER_NEAREST)


# ------------------------------------------------------------------- escenas sintéticas
LINEAS = ["FACTURA 2026-0481", "Cliente: Laboratorio Central S.A.", "Concepto                Cantidad      Importe", "Reactivos de tincion (kit)      2         120.50",
          "Placas de microscopio         10          45.00", "Guantes de nitrilo (caja)       4          38.40", "Filtros para pipeta            6          27.90",
          "Subtotal                                  231.80", "Impuestos (16%)                            37.09", "TOTAL A PAGAR                            268.89",
          "Pago a 30 dias. Gracias por su compra.", "Este documento es una prueba con texto propio."]


def documento_limpio(W=850, H=1100):
    img = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(img)
    f1_, f2 = ImageFont.truetype("arial.ttf", 34), ImageFont.truetype("arial.ttf", 22)
    for i, l in enumerate(LINEAS):
        d.text((50, 70 + i * 62), l, fill=25, font=f1_ if i == 0 else f2)
    d.rectangle([40, 190, 810, 600], outline=40, width=2)
    d.line([40, 250, 810, 250], fill=40, width=2)
    d.ellipse([600, 860, 780, 1040], outline=60, width=4)
    return np.array(img)


def fondo(tipo, W, H, rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if tipo == "oscuro":
        b = 45 + 15 * np.sin(xx / 400) - 30 * (((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W))
    elif tipo == "granito":            # base clara con motas oscuras y claras: parecido al papel en brillo
        b = np.full((H, W), 150, np.float32)
        for _ in range(2500):
            c = rng.uniform(0, [W, H]); r = rng.uniform(3, 14); v = rng.choice([-70, 60])
            R_ = int(3 * r) + 1; x0, x1 = max(int(c[0]) - R_, 0), min(int(c[0]) + R_ + 1, W); y0, y1 = max(int(c[1]) - R_, 0), min(int(c[1]) + R_ + 1, H)
            b[y0:y1, x0:x1] += np.exp(-((xx[y0:y1, x0:x1] - c[0]) ** 2 + (yy[y0:y1, x0:x1] - c[1]) ** 2) / (2 * (r / 2) ** 2)) * v       # solo en una ventana local
    else:                              # madera de brillo medio con vetas
        b = 120 + 25 * np.sin(yy / 18 + 6 * np.sin(xx / 260)) + 12 * np.sin(yy / 5)
    return np.clip(b + rng.normal(0, 4, b.shape), 0, 255).astype(np.float32)


def escena(clean, tipo, sombra, semilla, W=1200, H=1600):
    rng = np.random.default_rng(semilla)
    ch, cw = clean.shape
    base = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    cx, cy = W / 2, H / 2
    ancho = rng.uniform(0.55, 0.68) * W; alto = ancho * ch / cw
    quad = np.float32([[cx - ancho / 2, cy - alto / 2], [cx + ancho / 2, cy - alto / 2], [cx + ancho / 2, cy + alto / 2], [cx - ancho / 2, cy + alto / 2]])
    quad += rng.uniform(-0.06, 0.06, (4, 2)) * [W, H] * [1, 0.6]                       # perspectiva
    ang = np.radians(rng.uniform(-14, 14))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    quad = ((quad - [cx, cy]) @ R.T + [cx, cy] + rng.uniform(-40, 40, 2)).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(base, quad)
    papel = cv2.warpPerspective(clean.astype(np.float32), Hm, (W, H), flags=cv2.INTER_CUBIC)
    m = cv2.warpPerspective(np.full(clean.shape, 255, np.uint8), Hm, (W, H)) > 128
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dirx, diry = np.cos(rng.uniform(0, 2 * np.pi)), np.sin(rng.uniform(0, 2 * np.pi))
    rampa = ((xx - cx) * dirx + (yy - cy) * diry) / (0.5 * W)
    c2 = rng.uniform([cx - 250, cy - 350], [cx + 250, cy + 350])
    mancha = np.exp(-((xx - c2[0]) ** 2 + (yy - c2[1]) ** 2) / (2 * 190 ** 2))
    luz = np.clip(1 - sombra * (0.45 * (rampa + 1) / 2 + 0.75 * mancha), 0.15, 1.2)
    fon = fondo(tipo, W, H, rng)
    img = np.where(m, papel * luz, fon * (0.9 + 0.1 * luz))
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), quad


# ------------------------------------------------------------------- parte A: recibos reales
def parte_a():
    recibos = [("Recibo arrugado en mesa oscura (Vienna)", "recibo_viena.jpg"), ("Recibo sobre granito moteado (Bankia)", "recibo_bankia.jpg"), ("Recibo inclinado y recortado (Alves)", "recibo_alves.jpg")]
    filas = []
    fig, ax = plt.subplots(len(recibos), 6, figsize=(21, 9.6 * 1.0))
    A = {}
    for i, (nombre, arch) in enumerate(recibos):
        img = cv2.imread(str(fuente(arch)))
        peq, e = reducir(img, 1000)
        gris = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
        m_o, t = mascara_otsu(gris)
        m_c, edges = bordes_canny(gris)
        q_o, ok_o = cuadrilatero(m_o); q_c, ok_c = cuadrilatero(m_c)
        iou = float(((m_o > 0) & (m_c > 0)).sum() / max(((m_o > 0) | (m_c > 0)).sum(), 1))
        ancho_doc = {}
        for k, (q, ok) in (("otsu", (q_o, ok_o)), ("canny", (q_c, ok_c))):
            if q is not None:
                tocan = int(sum(1 for p in q if p[0] < 3 or p[1] < 3 or p[0] > peq.shape[1] - 4 or p[1] > peq.shape[0] - 4))
                ancho_doc[k] = {"cuadrilatero_exacto": bool(ok), "esquinas_en_el_borde_de_la_imagen": tocan,
                                "area_doc_pct": round(float(cv2.contourArea(q.astype(np.float32)) / (peq.shape[0] * peq.shape[1]) * 100), 1)}
            else:
                ancho_doc[k] = None
        A[nombre] = {"umbral_otsu": t, "iou_mascaras_otsu_canny": round(iou, 3), **{k: v for k, v in ancho_doc.items()}}
        rgb = cv2.cvtColor(peq, cv2.COLOR_BGR2RGB)
        ax[i, 0].imshow(rgb); ax[i, 0].set_title(nombre, fontsize=8.5)
        ax[i, 1].imshow(edges, cmap="gray"); ax[i, 1].set_title("Canny + dilatación + cierre")
        ax[i, 2].imshow(m_o, cmap="gray"); ax[i, 2].set_title(f"Otsu (t={t}) + morfología + mayor región")
        vis = rgb.copy()
        for q, c in ((q_o, (0, 200, 0)), (q_c, (255, 0, 0))):
            if q is not None: cv2.polylines(vis, [q.astype(np.int32)], True, c, 5)
        ax[i, 3].imshow(vis); ax[i, 3].set_title("Esquinas: Otsu (verde), Canny (rojo)")
        q = q_o if q_o is not None else q_c
        if q is not None:
            plano = rectificar(peq, q); g2 = cv2.cvtColor(plano, cv2.COLOR_BGR2GRAY)
            ax[i, 4].imshow(g2, cmap="gray"); ax[i, 4].set_title(f"Rectificado ({plano.shape[1]}×{plano.shape[0]}) y gris")
            ba = limpiar(bin_adaptativa(g2, 41, 12)); bo = bin_otsu(g2)
            ax[i, 5].imshow(255 - ba, cmap="gray"); ax[i, 5].set_title("Binarización adaptativa (Otsu global en tabla)")
            A[nombre]["texto_pct_adaptativa"] = round(float((ba > 0).mean() * 100), 1); A[nombre]["texto_pct_otsu_global"] = round(float((bo > 0).mean() * 100), 1)
            A[nombre]["plano_tam"] = [int(plano.shape[1]), int(plano.shape[0])]
            cv2.imwrite(str(DAT / f"escaneo_{i + 1}.png"), 255 - ba)
        for a in ax[i]: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig1_recibos_reales.png", dpi=100); plt.close(fig)
    RES["reales"] = A


# ------------------------------------------------------------------- parte B: evaluación controlada
def parte_b():
    clean = documento_limpio()
    gt_txt = (clean < 128).astype(np.uint8) * 255
    fondos = ["oscuro", "granito", "madera"]
    sombras = [0.0, 0.3, 0.6]
    N = 8
    det = {}; muestras = {}
    for tipo in fondos:
        for s in sombras:
            errs = {"otsu": [], "canny": []}; exacto = {"otsu": 0, "canny": 0}
            for sem in range(N):
                img, quad = escena(clean, tipo, s, sem)
                for metodo in ("otsu", "canny"):
                    q, ok, _ = detectar(img, metodo)
                    if q is None: errs[metodo].append(np.nan); continue
                    e = float(np.linalg.norm(ordenar(q) - ordenar(quad), axis=1).mean()); errs[metodo].append(e); exacto[metodo] += int(ok)
                if sem == 0: muestras[(tipo, s)] = (img, quad)
            det[f"{tipo}|{s}"] = {m: {"error_medio_px": round(float(np.nanmean(v)), 1) if np.isfinite(v).any() else None,
                                       "exito_pct": round(100 * float(np.mean([np.isfinite(x) and x < 20 for x in v])), 1), "cuadrilatero_exacto_pct": round(100 * exacto[m] / N, 1)} for m, v in errs.items()}
    RES["deteccion"] = det

    # binarización con esquinas verdaderas (aísla el efecto de la iluminación)
    metodos = {"Umbral fijo 128": bin_fija, "Otsu global": bin_otsu, "Adaptativo gaussiano": bin_adaptativa, "Sauvola": bin_sauvola}
    bres = {m: {} for m in list(metodos) + ["Adaptativo + apertura 2×2"]}
    ejemplo_bin = None
    for s in sombras:
        acum = {m: [] for m in bres}; fps = {m: [] for m in bres}
        for tipo in fondos:
            for sem in range(4):
                img, quad = escena(clean, tipo, s, 100 + sem)
                plano = rectificar(img, ordenar(quad), (850, 1100))
                for m, f in metodos.items():
                    r = f(plano); v, fp = f1(r, gt_txt); acum[m].append(v); fps[m].append(fp)
                    if m == "Adaptativo gaussiano":
                        r2 = limpiar(r); v2, fp2 = f1(r2, gt_txt); acum["Adaptativo + apertura 2×2"].append(v2); fps["Adaptativo + apertura 2×2"].append(fp2)
                if tipo == "madera" and sem == 0 and s in (0.0, 0.6): ejemplo_bin = ejemplo_bin or {}; ejemplo_bin[s] = (plano, {m: f(plano) for m, f in metodos.items()})
        for m in bres: bres[m][str(s)] = {"f1": round(float(np.mean(acum[m])), 3), "falsos_positivos_fondo_pct": round(float(np.mean(fps[m]) * 100), 2)}
    RES["binarizacion"] = bres

    # pipeline completo con detección automática
    completo = {}
    for s in sombras:
        v_o, v_c, v_m = [], [], []
        for tipo in fondos:
            for sem in range(4):
                img, quad = escena(clean, tipo, s, 200 + sem)
                res = []
                for metodo in ("otsu", "canny"):
                    q, ok, _ = detectar(img, metodo)
                    if q is None: res.append(0.0); continue
                    plano = rectificar(img, ordenar(q), (850, 1100))
                    res.append(f1(alinear(limpiar(bin_adaptativa(plano)), gt_txt), gt_txt, tol=2)[0])
                v_o.append(res[0]); v_c.append(res[1]); v_m.append(max(res))
        completo[str(s)] = {"otsu": round(float(np.mean(v_o)), 3), "canny": round(float(np.mean(v_c)), 3), "mejor_de_ambos": round(float(np.mean(v_m)), 3)}
    RES["pipeline_completo_f1"] = completo

    # ---- figuras
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i, tipo in enumerate(fondos):
        for j, s in enumerate(sombras):
            img, quad = muestras[(tipo, s)]
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for metodo, c in (("otsu", (0, 220, 0)), ("canny", (255, 60, 60))):
                q, ok, _ = detectar(img, metodo)
                if q is not None: cv2.polylines(vis, [q.astype(np.int32)], True, c, 6)
            cv2.polylines(vis, [quad.astype(np.int32)], True, (255, 255, 0), 2)
            d = det[f"{tipo}|{s}"]
            ax[i, j].imshow(vis); ax[i, j].set_title(f"fondo {tipo}, sombra {s}\nerror Otsu {d['otsu']['error_medio_px']} px · Canny {d['canny']['error_medio_px']} px", fontsize=9); ax[i, j].axis("off")
    fig.suptitle("Escenas sintéticas: esquinas verdaderas (amarillo), detectadas con Otsu (verde) y con Canny (rojo)", y=0.995)
    fig.tight_layout(); fig.savefig(FIG / "fig2_escenas.png", dpi=80); plt.close(fig)

    fig, ax = plt.subplots(2, 5, figsize=(21, 9.4))
    for i, s in enumerate((0.0, 0.6)):
        plano, rr = ejemplo_bin[s]
        ax[i, 0].imshow(plano, cmap="gray", vmin=0, vmax=255); ax[i, 0].set_title(f"Documento rectificado (sombra {s})")
        for a, (m, r) in zip(ax[i, 1:], rr.items()):
            v, fp = f1(r, gt_txt); a.imshow(255 - r, cmap="gray"); a.set_title(f"{m}: F1 {v:.2f}")
        for a in ax[i]: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig3_binarizacion.png", dpi=80); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(3); w = 0.2
    for k, m in enumerate(metodos):
        ax[0].bar(x + (k - 1.5) * w, [bres[m][str(s)]["f1"] for s in sombras], w, label=m)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"sombra {s}" for s in sombras]); ax[0].set_ylabel("F1 del texto"); ax[0].set_title("Binarización tras rectificar (esquinas verdaderas)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    for k, (metodo, c) in enumerate((("otsu", "#2a9d3a"), ("canny", "#c62828"))):
        ax[1].bar(x + (k - .5) * 0.3, [np.mean([det[f'{t}|{s}'][metodo]['exito_pct'] for t in fondos]) for s in sombras], 0.3, color=c, label=f"detección con {metodo}")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"sombra {s}" for s in sombras]); ax[1].set_ylabel("% de escenas con esquinas correctas (error < 20 px)"); ax[1].set_title("Detección del documento (promedio de los 3 fondos)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y"); ax[1].set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(FIG / "fig4_resumen.png", dpi=130); plt.close(fig)


def main():
    parte_a()
    parte_b()
    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(RES, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
