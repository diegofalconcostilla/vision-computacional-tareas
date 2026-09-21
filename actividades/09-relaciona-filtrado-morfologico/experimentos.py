"""Operaciones morfológicas para detectar y separar células en un frotis de sangre.

Imagen real: frotis de sangre normal (Chambers, CC BY-SA 3.0): glóbulos rojos que se tocan, dos-tres leucocitos y plaquetas.
Como no hay un conteo de referencia, el pipeline se valida con una imagen sintética de conteo conocido; en la real se comparan
estimadores independientes (erosión, transformada de distancia y área).
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
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
FIG, DAT = AQUI / "figuras", AQUI / "datos"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
RES = {}
EL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def fuente(nombre):
    ruta = DAT / nombre
    return ruta if ruta.exists() else RAIZ / "datos_compartidos" / nombre


# ---------------------------------------------------------------- segmentación y morfología
def binarizar(g):
    """Células = píxeles más oscuros que el umbral de Otsu (canal verde, donde el contraste es mayor)."""
    s = cv2.GaussianBlur(g, (0, 0), 1.0)
    return (s < threshold_otsu(s)).astype(np.uint8), int(threshold_otsu(s))


def n_comp(m, area_min=1):
    lab = label(m, connectivity=2)
    return int(sum(1 for r in regionprops(lab) if r.area >= area_min))


def rellenar_huecos_pequenos(m, area_max):
    """Rellena solo los huecos de área < area_max (la zona pálida del centro de la célula), no los espacios entre células."""
    huecos = ndi.binary_fill_holes(m) & (m == 0)
    lab = label(huecos, connectivity=1)
    out = m.copy()
    for r in regionprops(lab):
        if r.area < area_max: out[lab == r.label] = 1
    return out.astype(np.uint8)


def area_celula_tipica(m):
    """Mediana del área de los componentes aislados y redondos (los que probablemente son una sola célula)."""
    lab = label(m, connectivity=2)
    a = [r.area for r in regionprops(lab) if r.area > 150 and r.solidity > 0.93 and r.eccentricity < 0.6]
    return float(np.median(a)) if a else float("nan")


def contar_por_erosion(m, k, area_min):
    e = cv2.erode(m, EL, iterations=k) if k else m
    return n_comp(e, area_min)


def contar_por_distancia(m, r_tip):
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dist_s = cv2.GaussianBlur(dist, (0, 0), 1.5)
    picos = peak_local_max(dist_s, min_distance=max(int(0.55 * r_tip), 3), threshold_abs=0.5 * r_tip, labels=m.astype(int), exclude_border=False)
    marc = np.zeros(m.shape, np.int32)
    for i, (y, x) in enumerate(picos, 1): marc[y, x] = i
    lab = watershed(-dist_s, marc, mask=m.astype(bool))
    return len(picos), lab


def pipeline(rgb):
    g = rgb[..., 1]
    m0, t = binarizar(g)
    a0 = area_celula_tipica(cv2.morphologyEx(m0, cv2.MORPH_OPEN, EL))
    r_tip = float(np.sqrt(a0 / np.pi))
    m1 = cv2.morphologyEx(m0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))       # quita plaquetas y basura
    m2 = rellenar_huecos_pequenos(m1, 0.35 * a0)                                                            # rellena el centro pálido
    m3 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, EL)                                                          # une bordes con pequeñas grietas
    return {"g": g, "umbral": t, "m0": m0, "m1": m1, "m2": m2, "m3": m3, "area_tip": a0, "r_tip": r_tip}


def contar(rgb):
    P = pipeline(rgb)
    m, a0, r = P["m3"], P["area_tip"], P["r_tip"]
    area_min = 0.3 * a0
    res = {"area_tipica_px": round(a0, 1), "radio_tipico_px": round(r, 1), "umbral_otsu": P["umbral"],
           "componentes_binarizado_crudo": n_comp(P["m0"]), "componentes_pequenos_crudo_menor_30pct": int(sum(1 for q in regionprops(label(P['m0'], connectivity=2)) if q.area < area_min)),
           "componentes_tras_apertura": n_comp(P["m1"], 1), "componentes_pequenos_tras_apertura": int(sum(1 for q in regionprops(label(P['m1'], connectivity=2)) if q.area < area_min)), "componentes_tras_rellenar_y_cierre": n_comp(P["m3"], 1),
           "area_celulas_pct": round(float(m.mean() * 100), 1)}
    curva = [contar_por_erosion(m, k, max(0.05 * a0, 20)) for k in range(0, 13)]
    k_opt = int(np.argmax(curva))
    n_dist, lab_ws = contar_por_distancia(m, r)
    n_area = float(m.sum() / a0)
    lab = label(m, connectivity=2)
    tam = [r_.area / a0 for r_ in regionprops(lab) if r_.area >= area_min]
    res.update({"curva_erosion": curva, "k_optimo": k_opt, "n_erosion_optima": int(curva[k_opt]), "n_distancia": int(n_dist), "n_area": round(n_area, 1),
                "componentes_tras_limpiar_(area>=30%)": len(tam),
                "componentes_de_1_celula_pct": round(100 * float(np.mean([0.6 <= t <= 1.5 for t in tam])), 1), "componentes_grupos_(>1.5_celdas)": int(sum(t > 1.5 for t in tam)),
                "mayor_grupo_celulas": round(float(max(tam)), 1)})
    return P, res, lab_ws


# ---------------------------------------------------------------- imagen sintética de conteo conocido
def sintetica(n=90, solape=0.0, ancho=700, alto=520, semilla=0):
    """Discos con centro pálido sobre fondo claro. `solape` = fracción del diámetro que se superponen (0 = solo se tocan; >0 = se traslapan)."""
    rng = np.random.default_rng(semilla)
    centros, radios = [], []
    intentos = 0
    while len(centros) < n and intentos < 60000:
        intentos += 1
        r = rng.normal(13, 1.6); c = rng.uniform(r + 3, [ancho - r - 3, alto - r - 3])
        ok = True
        for c2, r2 in zip(centros, radios):
            d = np.hypot(*(c - c2))
            if d < (r + r2) * (1 - solape) - 0.01: ok = False; break
        if ok:
            # aceptar preferentemente posiciones cercanas a otras células (para que se toquen)
            vec = [np.hypot(*(c - c2)) - (r + r2) for c2, r2 in zip(centros, radios)]
            if not centros or min(vec) < 6 or rng.random() < 0.15:
                centros.append(c); radios.append(r)
    img = np.full((alto, ancho), 235, np.float32)
    yy, xx = np.mgrid[0:alto, 0:ancho]
    for c, r in zip(centros, radios):
        d = np.hypot(xx - c[0], yy - c[1])
        cel = np.clip(1 - (d / r) ** 6, 0, 1)
        img = np.minimum(img, 235 - cel * (110 + 25 * rng.random()) * (0.75 + 0.25 * np.clip(d / r, 0, 1)))       # más pálido en el centro
        pal = np.clip(1 - (d / (0.38 * r)) ** 2, 0, 1); img = img + pal * 35 * cel
    for _ in range(25):                                                                                          # plaquetas (basura)
        c = rng.uniform(5, [ancho - 5, alto - 5]); r = rng.uniform(2, 3.5)
        img = np.minimum(img, 235 - np.clip(1 - (np.hypot(xx - c[0], yy - c[1]) / r) ** 4, 0, 1) * 120)
    img = cv2.GaussianBlur(img, (0, 0), 1.1) + rng.normal(0, 4, img.shape)
    g = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g]), len(centros)


def dice_(a, b):
    a, b = a > 0, b > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def main():
    rgb = cv2.cvtColor(cv2.imread(str(fuente("frotis_sangre.jpg"))), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    P, res, lab_ws = contar(rgb)
    RES["frotis"] = {"tam": [w, h], **res}

    # ---- fig1: imagen y binarización
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].imshow(rgb); ax[0].set_title(f"Frotis de sangre ({w}×{h} px). Foto: K. Chambers, CC BY-SA 3.0"); ax[0].axis("off")
    ax[1].hist(P["g"].ravel(), 128, color="#444"); ax[1].axvline(P["umbral"], color="r", label=f"Otsu, t = {P['umbral']}"); ax[1].set_title("Histograma del canal verde"); ax[1].legend(); ax[1].set_yticks([])
    ax[2].imshow(P["m0"], cmap="gray"); ax[2].set_title(f"Binarización de Otsu: {res['componentes_binarizado_crudo']} componentes"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig1_imagen_binarizacion.png", dpi=140); plt.close(fig)

    # ---- fig2: etapas morfológicas sobre un recorte
    y0, y1, x0, x1 = 130, 400, 250, 620
    sl = (slice(y0, y1), slice(x0, x1))
    m3 = P["m3"]; k_opt = res["k_optimo"]
    er = cv2.erode(m3, EL, iterations=k_opt)
    sk = skeletonize(m3 > 0)
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.2))
    paneles = [("Imagen original (recorte)", rgb[sl], None),
               ("(a) Binarización de Otsu", P["m0"][sl], "gray"),
               ("(b) Apertura 5×5: quita plaquetas y basura", P["m1"][sl], "gray"),
               ("(c) Relleno de huecos pequeños: centro pálido", P["m2"][sl], "gray"),
               ("(d) Cierre 3×3", m3[sl], "gray"),
               (f"(e) Erosión ×{k_opt}: separa células que se tocan", er[sl], "gray")]
    for a, (t, im, cm) in zip(ax.ravel(), paneles):
        a.imshow(im, cmap=cm); a.set_title(t); a.axis("off")
    esq = np.dstack([m3 * 90] * 3).astype(np.uint8); esq[sk] = (255, 60, 60)
    ax[1, 2].imshow(cv2.dilate(esq, np.ones((2, 2), np.uint8))[sl]); ax[1, 2].set_title("(f) Esqueleto del resultado (rojo)"); ax[1, 2].axis("off")
    ov = rgb.copy(); bd = cv2.morphologyEx(np.where(lab_ws > 0, lab_ws, 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ov[bd] = (0, 200, 0)
    ax[1, 3].imshow(ov[sl]); ax[1, 3].set_title("(g) Separación por transformada de distancia + watershed (verde)"); ax[1, 3].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig2_etapas.png", dpi=125); plt.close(fig)

    # ---- fig3: conteo según erosión, con la referencia por distancia y por área
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(range(len(res["curva_erosion"])), res["curva_erosion"], "o-", label="componentes tras erosionar")
    ax[0].axhline(res["n_distancia"], color="g", ls="--", label=f"transformada de distancia ({res['n_distancia']})"); ax[0].axhline(res["n_area"], color="r", ls=":", label=f"área total / área típica ({res['n_area']:.0f})")
    ax[0].axvline(k_opt, color="gray", lw=.8); ax[0].set_xlabel("iteraciones de erosión (elemento elíptico 3×3)"); ax[0].set_ylabel("objetos contados"); ax[0].set_title("Conteo en el frotis real"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    lab = label(m3, connectivity=2); tam = [r_.area / res["area_tipica_px"] for r_ in regionprops(lab) if r_.area >= 0.3 * res["area_tipica_px"]]
    ax[1].hist(tam, bins=np.arange(0, 8.5, .25), color="#1f77b4"); ax[1].axvline(1, color="r", ls="--", lw=.8); ax[1].set_xlabel("área del componente / área típica de una célula"); ax[1].set_ylabel("componentes"); ax[1].set_title("Tamaño de los componentes conectados (1 = una célula)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig3_conteo.png", dpi=140); plt.close(fig)

    # ---- fig4: resultado final sobre toda la imagen
    ov = rgb.copy(); bd = cv2.morphologyEx(np.where(lab_ws > 0, lab_ws, 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ov[bd] = (0, 180, 0)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.2))
    ax[0].imshow(rgb); ax[0].set_title("Original"); ax[1].imshow(ov); ax[1].set_title(f"Células separadas (watershed): {res['n_distancia']} objetos"); [a.axis("off") for a in ax]
    fig.tight_layout(); fig.savefig(FIG / "fig4_resultado.png", dpi=125); plt.close(fig)

    # ---- validación con imágenes sintéticas de conteo conocido
    val = []
    ejemplos = {}
    for solape in (0.0, 0.1, 0.25):
        for semilla in range(4):
            img, n_real = sintetica(n=90, solape=solape, semilla=semilla)
            Pp, rr, lab_s = contar(img)
            mm = Pp["m3"]
            val.append({"solape": solape, "semilla": semilla, "n_real": n_real, "crudo": rr["componentes_binarizado_crudo"], "tras_limpieza": rr["componentes_tras_limpiar_(area>=30%)"],
                        "erosion": rr["n_erosion_optima"], "k": rr["k_optimo"], "distancia": rr["n_distancia"], "area": rr["n_area"], "componentes_pequenos_crudo": rr["componentes_pequenos_crudo_menor_30pct"],
                        "curva": rr["curva_erosion"]})
            if semilla == 0: ejemplos[solape] = (img, lab_s, Pp)
    RES["validacion_sintetica"] = val
    def err(clave):
        out = {}
        for s in (0.0, 0.1, 0.25):
            e = [abs(v[clave] - v["n_real"]) / v["n_real"] * 100 for v in val if v["solape"] == s]
            b = [(v[clave] - v["n_real"]) / v["n_real"] * 100 for v in val if v["solape"] == s]
            out[str(s)] = {"error_abs_medio_pct": round(float(np.mean(e)), 1), "sesgo_pct": round(float(np.mean(b)), 1)}
        return out
    RES["errores_sinteticos"] = {k: err(k) for k in ("crudo", "tras_limpieza", "erosion", "distancia", "area")}

    fig, ax = plt.subplots(2, 3, figsize=(17, 8.4))
    for j, s in enumerate((0.0, 0.1, 0.25)):
        img, lab_s, Pp = ejemplos[s]
        ax[0, j].imshow(img); ax[0, j].set_title(f"Sintética, solape {int(s*100)} % (90 células reales)"); ax[0, j].axis("off")
        ov = img.copy(); bd = cv2.morphologyEx(np.where(lab_s > 0, lab_s, 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0; ov[bd] = (0, 200, 0)
        v0 = [v for v in val if v["solape"] == s and v["semilla"] == 0][0]
        ax[1, j].imshow(ov); ax[1, j].set_title(f"Watershed: {v0['distancia']} · erosión: {v0['erosion']} · crudo: {v0['crudo']}"); ax[1, j].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig5_sintetica.png", dpi=115); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    etiquetas = {"crudo": "binarización sin morfología", "tras_limpieza": "tras apertura, relleno y cierre", "erosion": "erosión con el mejor k", "distancia": "transformada de distancia + watershed", "area": "área total / área típica"}
    x = np.arange(3); ancho = 0.16
    for i, (k, t) in enumerate(etiquetas.items()):
        ax[0].bar(x + (i - 2) * ancho, [RES["errores_sinteticos"][k][str(s)]["error_abs_medio_pct"] for s in (0.0, 0.1, 0.25)], ancho, label=t)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["solo se tocan", "solape 10 %", "solape 25 %"]); ax[0].set_ylabel("error absoluto medio del conteo (%)"); ax[0].set_title("Error de conteo en imágenes sintéticas (12 imágenes)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, axis="y")
    for s, c in ((0.0, "#1f77b4"), (0.1, "#2a9d3a"), (0.25, "#c62828")):
        v = [v for v in val if v["solape"] == s and v["semilla"] == 0][0]
        ax[1].plot(v["curva"], "o-", color=c, ms=3, label=f"solape {int(s*100)} % (real: {v['n_real']})")
    ax[1].set_xlabel("iteraciones de erosión"); ax[1].set_ylabel("componentes"); ax[1].set_title("Conteo tras erosionar (una imagen por nivel de solape)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig6_validacion.png", dpi=140); plt.close(fig)

    # ---- segunda imagen (frotis con iluminación desigual y células pálidas), solo cualitativa
    rgb2 = cv2.cvtColor(cv2.imread(str(fuente("frotis_wright.jpg"))), cv2.COLOR_BGR2RGB)
    rgb2 = cv2.resize(rgb2, (1000, round(rgb2.shape[0] * 1000 / rgb2.shape[1])), interpolation=cv2.INTER_AREA)
    g2 = rgb2[..., 1]
    m_g, t_g = binarizar(g2)                                                                       # umbral global
    fondo = cv2.GaussianBlur(g2.astype(np.float32), (0, 0), 60)
    corr = np.clip(g2.astype(np.float32) - fondo + 128, 0, 255).astype(np.uint8)                   # se resta el fondo (iluminación desigual)
    m_c, t_c = binarizar(corr)
    a2 = 3000.0
    m_cl = cv2.morphologyEx(m_c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m_cl = rellenar_huecos_pequenos(m_cl, 4000)
    RES["segunda_imagen"] = {"tam": list(rgb2.shape[:2][::-1]), "umbral_global": t_g, "umbral_tras_restar_fondo": t_c, "pct_fg_global": round(float(m_g.mean() * 100), 1),
                             "pct_fg_corregido": round(float(m_c.mean() * 100), 1), "pct_fg_final": round(float(m_cl.mean() * 100), 1), "componentes_global": n_comp(m_g), "componentes_final": n_comp(m_cl, 500)}
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    for a, im, t in ((ax[0], rgb2, "Frotis con tinción Wright (foto: A. K. Chaurasiya, CC BY-SA 4.0)"), (ax[1], m_g, f"Otsu global: {RES['segunda_imagen']['pct_fg_global']} % primer plano"),
                     (ax[2], m_c, f"Otsu tras restar el fondo: {RES['segunda_imagen']['pct_fg_corregido']} %"), (ax[3], m_cl, "Apertura 9×9 + relleno de huecos")):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray"); a.set_title(t, fontsize=8.5); a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig7_segunda_imagen.png", dpi=130); plt.close(fig)

    (AQUI / "resultados.json").write_text(json.dumps(RES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in RES.items() if k != "validacion_sintetica"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
