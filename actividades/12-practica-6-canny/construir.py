"""Construye la entrega de la Práctica 6 (Canny). La actividad pide experimentar con parámetros y reportar hallazgos."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas import cuaderno as cu  # noqa: E402
from herramientas.reporte import paginas_pdf, zip_entrega  # noqa: E402

COMP = RAIZ / "datos_compartidos"
DATA = AQUI / "codigo" / "data"
NOMBRE = "practica_6_canny"


def preparar_datos():
    DATA.mkdir(parents=True, exist_ok=True)
    def guarda(origen, destino, ancho):
        im = cv2.imread(str(COMP / origen))
        im = cv2.resize(im, (ancho, round(im.shape[0] * ancho / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(DATA / destino), im, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return im
    im = guarda("autopista_i75.jpg" if (COMP / "autopista_i75.jpg").exists() else "carretera_bemiss_4k.jpg", "image.jpg", 480)
    np.save(DATA / "img.npy", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))       # el notebook base lo espera (lo crea la práctica 2)
    guarda("calle_oak_alden.jpg", "escena_urbana.jpg", 640)           # muchas líneas
    guarda("celulas_hela_dapi.jpg", "textura_celulas.jpg", 640)       # textura fina
    guarda("microscopia_levaduras.jpg", "microscopia.jpg", 640)
    guarda("autopista_i10.jpg", "autopista.jpg", 640)                 # pocas líneas


EJ = [
    cu.md("""
# Experimentos de la actividad

La actividad no pide funciones nuevas sino **modificar parámetros para comprender el código** y reportar los hallazgos. Se hace en tres pasos:

1. [Sigma y tamaño del kernel](#sigma): qué cambia al modificar σ, y qué pasa con el tamaño del kernel fijo.
2. [Canny con y sin remoción de ruido](#ruido): con una verdad de bordes conocida (imagen sintética) se miden precisión, exhaustividad y F1.
3. [Imágenes con distinta cantidad de líneas y textura](#imagenes).
4. [Hallazgos](#hallazgos).

Para barrer muchos parámetros se usa `cv2.Canny` (rápido); el notebook base implementa cada etapa a mano con bucles y es demasiado lento para hacerlo decenas de veces.
Antes de los experimentos se corrigieron tres problemas del notebook base (ver «Hallazgos»).
"""),
    cu.code("""
import cv2, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
plt.rcParams.update({"font.size": 9})

def gauss_k(g, sigma, tam=None):
    # Gaussiano con tamaño de kernel proporcional a sigma (6σ+1, impar) salvo que se indique otro
    if sigma <= 0: return g
    tam = tam or (int(2 * np.ceil(3 * sigma) + 1))
    return cv2.GaussianBlur(g, (tam, tam), sigma)

def canny(g, sigma=1.4, bajo=50, alto=120, tam=None):
    return cv2.Canny(gauss_k(g, sigma, tam), bajo, alto)

def prf(deteccion, verdad, tol=2):
    # precisión = detectados a ≤ tol px de un borde verdadero; exhaustividad = bordes verdaderos a ≤ tol px de un detectado
    d, v = deteccion > 0, verdad > 0
    dist_v = distance_transform_edt(~v); dist_d = distance_transform_edt(~d)
    p = float((dist_v[d] <= tol).mean()) if d.any() else 0.0
    r = float((dist_d[v] <= tol).mean()) if v.any() else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
"""),
    cu.md("""
<a id="sigma"></a>
## 1. Modificar σ y el tamaño del kernel

En el notebook base, `sigma = 10` se usa con `kernel_size = 3`. Un kernel de 3×3 solo puede representar bien un gaussiano de σ ≲ 1: si σ es mayor, los tres coeficientes por fila se vuelven casi iguales y el resultado es un simple promedio de 3×3, **por muy grande que sea σ**. Se comprueba y se compara con un kernel cuyo tamaño crece con σ (6σ+1).
"""),
    cu.code("""
gris = cv2.imread("data/image.jpg", 0)
sigmas = [0.5, 1, 2, 3, 5, 10]
fig, ejes = plt.subplots(3, len(sigmas), figsize=(19, 7.6))
filas = []
for j, s in enumerate(sigmas):
    fijo = cv2.GaussianBlur(gris, (3, 3), s)                       # como en el notebook base: kernel 3x3
    adaptado = gauss_k(gris, s)                                    # kernel de tamaño 6σ+1
    e_fijo, e_ad = cv2.Canny(fijo, 40, 100), cv2.Canny(adaptado, 40, 100)
    filas.append((s, (e_fijo > 0).mean() * 100, (e_ad > 0).mean() * 100, np.abs(fijo.astype(int) - gauss_k(gris, 0.5).astype(int)).mean(), np.abs(adaptado.astype(int) - gris.astype(int)).mean()))
    ejes[0, j].imshow(adaptado, cmap="gray"); ejes[0, j].set_title(f"σ={s}, kernel {int(2*np.ceil(3*s)+1)}×{int(2*np.ceil(3*s)+1)}")
    ejes[1, j].imshow(e_fijo, cmap="gray"); ejes[1, j].set_title(f"Canny, kernel fijo 3×3")
    ejes[2, j].imshow(e_ad, cmap="gray"); ejes[2, j].set_title(f"Canny, kernel adaptado")
for a in ejes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
print(f"{'σ':>5s} {'bordes % (kernel 3×3)':>22s} {'bordes % (kernel 6σ+1)':>24s}")
for s, ef, ea, _, _ in filas: print(f"{s:5.1f} {ef:22.2f} {ea:24.2f}")
"""),
    cu.md("""
<a id="ruido"></a>
## 2. Canny con y sin remoción de ruido

Se construye una imagen sintética con bordes conocidos (rectángulo, círculo y triángulo con niveles de gris distintos, y una textura suave de fondo) y se le añade ruido gaussiano de distinta intensidad. La verdad de bordes es el contorno de cada figura. Se mide con una tolerancia de 2 píxeles.
"""),
    cu.code("""
def imagen_sintetica(n=320, ruido=0, semilla=0):
    rng = np.random.default_rng(semilla)
    g = np.full((n, n), 110, np.float32)
    g += cv2.GaussianBlur(rng.normal(0, 14, (n, n)).astype(np.float32), (0, 0), 6) * 2         # textura suave de fondo
    etiquetas = np.zeros((n, n), np.uint8)
    cv2.rectangle(etiquetas, (30, 40), (140, 150), 1, -1); cv2.circle(etiquetas, (225, 100), 55, 2, -1)
    cv2.fillPoly(etiquetas, [np.array([[80, 290], [160, 180], [240, 290]])], 3)
    for k, v in ((1, 200), (2, 50), (3, 170)): g[etiquetas == k] = v
    verdad = cv2.morphologyEx(etiquetas, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    g = g + rng.normal(0, ruido, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8), (verdad > 0).astype(np.uint8) * 255

niveles = [0, 10, 25, 40]
sig_lista = [0, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0]
tabla = {}
for nr in niveles:
    g, v = imagen_sintetica(ruido=nr)
    for s in sig_lista: tabla[(nr, s)] = prf(canny(g, s, 50, 120), v)
print("F1 de Canny (umbrales 50/120) según el ruido de la imagen (filas) y σ del suavizado previo (columnas; 0 = sin suavizar)\\n")
print(f"{'ruido σn':>9s} " + " ".join(f"σ={s:<5}" for s in sig_lista))
for nr in niveles: print(f"{nr:9d} " + " ".join(f"{tabla[(nr, s)][2]:7.3f}" for s in sig_lista))
mejor = {nr: max(sig_lista, key=lambda s: tabla[(nr, s)][2]) for nr in niveles}
print("\\nMejor σ por nivel de ruido:", mejor)
print("\\nPrecisión / exhaustividad en la imagen con ruido σn=25:")
for s in sig_lista: print(f"  σ={s:<4} precisión {tabla[(25, s)][0]:.3f}  exhaustividad {tabla[(25, s)][1]:.3f}")
"""),
    cu.code("""
g, v = imagen_sintetica(ruido=25)
fig, ejes = plt.subplots(2, 5, figsize=(18, 7.4))
ejes[0, 0].imshow(g, cmap="gray"); ejes[0, 0].set_title("Imagen con ruido σn=25"); ejes[1, 0].imshow(v, cmap="gray"); ejes[1, 0].set_title("Verdad de bordes")
for j, s in enumerate((0, 1.4, 3.0, 5.0), start=1):
    e = canny(g, s, 50, 120); p, r, f = prf(e, v)
    ejes[0, j].imshow(gauss_k(g, s), cmap="gray"); ejes[0, j].set_title("sin suavizar" if s == 0 else f"suavizada σ={s}")
    ejes[1, j].imshow(e, cmap="gray"); ejes[1, j].set_title(f"Canny: P={p:.2f} R={r:.2f} F1={f:.2f}")
for a in ejes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
"""),
    cu.md("""
<a id="imagenes"></a>
## 3. Imágenes con distinta cantidad de líneas y textura

Con σ=1.4 y umbrales fijos, se aplica Canny a cuatro imágenes de naturaleza distinta y se mide la **densidad de bordes** (porcentaje de píxeles marcados) y el número de componentes conexos.
"""),
    cu.code("""
imagenes = {"Autopista (pocas líneas)": "data/autopista.jpg", "Escena urbana (muchas líneas)": "data/escena_urbana.jpg",
            "Células fluorescentes (textura fina)": "data/textura_celulas.jpg", "Microscopía (fondo liso)": "data/microscopia.jpg"}
fig, ejes = plt.subplots(2, 4, figsize=(19, 6.6))
print(f"{'imagen':38s} {'σ=1.4: bordes %':>16s} {'componentes':>12s} {'σ=3: bordes %':>14s}")
for j, (nombre, ruta) in enumerate(imagenes.items()):
    g = cv2.imread(ruta, 0); e = canny(g, 1.4, 50, 120); e3 = canny(g, 3.0, 50, 120)
    n_comp = cv2.connectedComponents(e, connectivity=8)[0] - 1
    print(f"{nombre:38s} {(e > 0).mean() * 100:16.2f} {n_comp:12d} {(e3 > 0).mean() * 100:14.2f}")
    ejes[0, j].imshow(g, cmap="gray"); ejes[0, j].set_title(nombre); ejes[1, j].imshow(e, cmap="gray"); ejes[1, j].set_title(f"Canny σ=1.4: {(e > 0).mean() * 100:.1f} % de bordes")
for a in ejes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
"""),
    cu.md("""
<a id="hallazgos"></a>
## 4. Hallazgos

__HALLAZGOS__

## Referencias

- Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8*(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851
- Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4.ª ed.). Pearson. (Cap. 10.)
- Sobel, I., & Feldman, G. (1968). *A 3x3 isotropic gradient operator for image processing* [Presentación]. Stanford Artificial Intelligence Project (SAIL).
- Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools, 25*(11), 120–123.
- Imágenes: ver `LICENCIAS.md` (Wikimedia Commons: Rivera, CC BY 4.0 y CC0; Arnatbio, CC BY-SA 4.0; Chaurasiya, CC0). La imagen sintética se genera por código.
"""),
]

HALLAZGOS = """
**1. Efecto de σ y del tamaño del kernel.**
- Al aumentar σ el suavizado previo elimina detalle fino y, con él, líneas. Con un kernel adaptado (6σ+1) el porcentaje de píxeles de borde baja de 17.9 % (σ=0.5) a 12.0 % (σ=1), 4.2 % (σ=2), 1.4 % (σ=3), 0.07 % (σ=5) y 0 % (σ=10): con σ=10 la imagen queda tan borrosa que ningún gradiente supera los umbrales.
- **Con el kernel fijo de 3×3 del notebook base, σ deja de tener efecto a partir de σ≈2**: el porcentaje de bordes se estanca en 13.1 % (13.25 %, 13.07 %, 13.05 %, 13.05 % para σ = 2, 3, 5 y 10). Un kernel de 3×3 solo representa un gaussiano con σ ≲ 1; para σ mayor se convierte en un promedio uniforme. Por eso los valores «1, 3, 5, 10, 20» que sugiere el notebook no producen resultados distintos si no se aumenta también `kernel_size`.
- Regla práctica: el tamaño del kernel debe ser al menos 6σ+1.

**2. Con y sin remoción de ruido.** En una imagen sintética con bordes conocidos (tolerancia de 2 px) se midió el F1 de Canny con umbrales fijos:
- **Sin suavizar, el ruido arruina el resultado**: con ruido de σn=10 el F1 es 0.17 y con σn=25 es 0.13, porque la exhaustividad es 1.0 (se detectan todos los bordes reales) pero la precisión es de solo 0.07 con σn=25 (más del 90 % de lo detectado es ruido).
- Con el suavizado adecuado el F1 llega a 1.000 con σn=10 (σ=1.0), 1.000 con σn=25 (σ=1.4) y 0.993 con σn=40 (σ=2.0). **El σ óptimo crece con el ruido** (0.7–1.0 → 1.4 → 2.0).
- Un suavizado excesivo también falla: con σn=25 y σ=3 la exhaustividad cae a 0.16 y con σ=5 se pierde todo. Ojo: con umbrales fijos, un σ grande reduce la magnitud del gradiente, así que en la imagen limpia con σ=3 o σ=5 el F1 es 0.000 aunque el borde sigue ahí; al suavizar más hay que bajar también los umbrales.
- Conclusión: **la remoción de ruido no es opcional**; el primer paso de Canny es necesario, y su intensidad debe ajustarse al ruido y a los umbrales.

**3. Imágenes con distinta cantidad de líneas y textura** (σ=1.4, umbrales 50/120):
- La escena urbana, con muchas líneas (ventanas, cables, señales), es la que tiene más bordes (7.0 % de píxeles, 327 componentes); la autopista, con pocas líneas, 4.9 % (211 componentes).
- La imagen de células, de textura fina, tiene una densidad de bordes menor (3.4 %) pero **es la que más bordes conserva al subir σ a 3 (1.2 %)**, porque el contorno de las células es un borde amplio; en cambio, la microscopía de fondo liso cae a 0.06 %, pues sus contornos son de bajo contraste.
- Un solo par de umbrales no sirve para todas las imágenes: las de bajo contraste requieren umbrales menores, y las texturadas, mayores para no llenarse de bordes espurios.

**4. Correcciones al notebook base** (afectan también al Colab actual):
- `scipy.ndimage.filters` ya no existe (se usa `scipy.ndimage`).
- La función `threshold` marca los píxeles débiles con el valor 25, pero `hysteresis` los buscaba con 75 por defecto, de modo que **la histéresis no promovía ningún píxel débil**; se cambió el valor por defecto a 25.
- `data/img.npy` lo crea el notebook de la práctica 2; aquí se genera junto con `data/image.jpg`.

**Limitaciones.** La imagen sintética es sencilla (tres figuras de bordes limpios), por lo que el F1 alcanza 1.0; con imágenes reales no hay una verdad de bordes objetiva y se recurre a la inspección visual y a la densidad de bordes. Los barridos usan `cv2.Canny`, cuya implementación coincide en concepto con la del notebook base pero no es idéntica pixel a pixel.
"""


def main():
    preparar_datos()
    nb = cu.cargar(AQUI / "enunciado" / "3_sobel_canny_edge_detection.ipynb")
    print("parches:", cu.parchear(nb))
    # scipy.ndimage.filters fue retirado en SciPy reciente
    cu.reemplazar(nb, "from scipy.ndimage.filters import convolve\nfrom scipy.ndimage.filters import gaussian_filter as gauss\nfrom scipy.ndimage.filters import median_filter as med",
                  "from scipy.ndimage import convolve\nfrom scipy.ndimage import gaussian_filter as gauss\nfrom scipy.ndimage import median_filter as med")
    cu.reemplazar(nb, "ndimage.filters.convolve(img, Kx)", "ndimage.convolve(img, Kx)")
    cu.reemplazar(nb, "ndimage.filters.convolve(img, Ky)", "ndimage.convolve(img, Ky)")
    # threshold() marca los píxeles débiles con 25 pero hysteresis() los buscaba con 75: la histéresis no hacía nada
    cu.reemplazar(nb, "def hysteresis(img, weak = 75, strong=255):", "def hysteresis(img, weak = 25, strong=255):")
    cu.agregar(nb, [c if c.cell_type != "markdown" else c for c in EJ])
    for c in nb.cells:
        if c.cell_type == "markdown" and "__HALLAZGOS__" in c.source:
            c.source = c.source.replace("__HALLAZGOS__", HALLAZGOS or "(pendiente)")
    cu.ejecutar(nb, AQUI / "codigo", timeout=1500)
    cu.guardar(nb, AQUI / "codigo" / f"{NOMBRE}.ipynb")
    shutil.copy(COMP / "LICENCIAS.md", AQUI / "codigo" / "LICENCIAS.md")
    pdf = cu.a_pdf(nb, AQUI / "Practica6_Canny.pdf", "Práctica 6. Extracción de líneas con el algoritmo de Canny",
                   "Efecto de σ, del suavizado previo y del tipo de imagen")
    n = zip_entrega(AQUI / "Practica6_Canny.zip", AQUI, ["codigo", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
