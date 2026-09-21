"""Construye la entrega de la Práctica 7 (Harris y matching de puntos de interés): dos notebooks + PDF combinado."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas import cuaderno as cu  # noqa: E402
from herramientas import puntos as P  # noqa: E402
from herramientas.reporte import paginas_pdf, zip_entrega  # noqa: E402

COMP = RAIZ / "datos_compartidos"
CODIGO = AQUI / "codigo"
DATA = CODIGO / "data"


def preparar_datos():
    DATA.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAIZ / "herramientas" / "puntos.py", CODIGO / "puntos.py")
    # tablero de ajedrez con perspectiva (data/chessboard.jpg del notebook base no se incluyó con la actividad)
    n, cas = 9, 60
    tab = np.zeros((n * cas, n * cas), np.uint8)
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0: tab[i * cas:(i + 1) * cas, j * cas:(j + 1) * cas] = 255
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [n * cas, 0], [n * cas, n * cas], [0, n * cas]]), np.float32([[90, 60], [560, 40], [600, 430], [50, 400]]))
    cv2.imwrite(str(DATA / "chessboard.jpg"), cv2.cvtColor(cv2.warpPerspective(tab, M, (640, 480), borderValue=128), cv2.COLOR_GRAY2BGR))
    # escena para el matching (reemplaza a elon_1 / elon_2, que no se incluyeron) y una segunda vista con verdad conocida
    esc = cv2.resize(cv2.imread(str(COMP / "calle_oak_alden.jpg")), (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DATA / "escena_1.jpg"), esc, [cv2.IMWRITE_JPEG_QUALITY, 93])
    H = P.homografia_rotacion(1280, 720, 900, yaw=12, pitch=-4, roll=8)
    cv2.imwrite(str(DATA / "escena_2.png"), P.vista(esc, H, brillo=0.85, ruido=3))
    # objeto para Harris: letrero (recorte de una foto CC0)
    big = cv2.imread(str(COMP / "carretera_bemiss_4k.jpg"))
    cv2.imwrite(str(DATA / "letrero.jpg"), big[1000:1500, 2950:3420], [cv2.IMWRITE_JPEG_QUALITY, 95])


# =========================================================================== notebook de Harris
EJ_H = [
    cu.md("""
# Experimentos: el detector de Harris bajo distintas condiciones

La actividad pide probar el detector con un objeto fotografiado bajo **distinta iluminación** y **distintos ángulos**. Como aquí no se dispone de fotos propias de cada condición, se parte de una fotografía real de un letrero (CC0, ver `LICENCIAS.md`)
y se generan las demás vistas de forma **controlada**: los cambios de iluminación se simulan con brillo, gamma y ruido, y los de ángulo con la homografía exacta de un giro del plano del objeto, de la cámara o de la escala (`puntos.py`).
Así se conoce la verdad y se puede medir la **repetibilidad**: de las esquinas detectadas en la imagen original, qué fracción reaparece (a ≤ 3 px) en la imagen transformada. Si se dispone de fotos propias, basta con reemplazar `data/letrero.jpg` y cargar cada toma como una condición adicional.

Contenido: [1. Condiciones](#cond) · [2. Parámetros de Harris](#param) · [3. Hallazgos](#hall)
"""),
    cu.code("""
import numpy as np, cv2, matplotlib.pyplot as plt
import puntos as P
plt.rcParams.update({"font.size": 9})

objeto = cv2.imread("data/letrero.jpg")
h, w = objeto.shape[:2]
gris0 = P.gris(objeto)
FOCO = 650
I = np.eye(3)
def rot(**kw): return P.homografia_rotacion(w, h, FOCO, **kw)

condiciones = {
    "original":                         (I, {}),
    "poca luz (×0.3)":                  (I, dict(brillo=0.3)),
    "sobreexpuesta (γ=0.5, ×1.5)":      (I, dict(gamma=0.5, brillo=1.5)),
    "bajo contraste + ruido":           (I, dict(brillo=0.45, offset=70, ruido=8)),
    "plano girado 20°":                 (P.homografia_plano(w, h, FOCO, 20), {}),
    "plano girado 40°":                 (P.homografia_plano(w, h, FOCO, 40), {}),
    "plano girado 60°":                 (P.homografia_plano(w, h, FOCO, 60), {}),
    "giro 30° (roll)":                  (rot(roll=30), {}),
    "escala 0.6×":                      (P.homografia_rotacion(w, h, FOCO, escala=0.6), {}),
    "plano 40° + poca luz":             (P.homografia_plano(w, h, FOCO, 40), dict(brillo=0.3)),
}
vistas = {n: P.vista(objeto, H, **kw) for n, (H, kw) in condiciones.items()}
print(f"Imagen del objeto: {w}×{h} px; esquinas de referencia: {len(P.detectar_harris(gris0, n=300))}")
"""),
    cu.md("""
<a id="cond"></a>
## 1. Repetibilidad de Harris según la condición

Se toman las 300 esquinas más fuertes de la imagen original y de cada vista (Harris con ventana de 3 px, apertura de Sobel 3 y `k = 0.04`), y se mide qué fracción de las esquinas de la original que siguen dentro de la vista reaparecen a ≤ 3 px.
"""),
    cu.code("""
k0 = P.detectar_harris(gris0, n=300)
filas = []
fig, ejes = plt.subplots(3, 4, figsize=(17, 12.5))
for a, (nombre, (H, kw)) in zip(ejes.ravel(), condiciones.items()):
    v = vistas[nombre]; k1 = P.detectar_harris(P.gris(v), n=300)
    rep, visibles = P.repetibilidad(k0, k1, H, v.shape, 3.0)
    filas.append((nombre, len(k1), visibles, rep * 100))
    vis = v.copy()
    for kp in k1[:150]: cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 0, 255), -1)
    a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); a.set_title(f"{nombre}\\nrepetibilidad {rep*100:.0f} %"); a.axis("off")
plt.tight_layout(); plt.show()
print(f"{'condición':32s} {'esquinas':>9s} {'visibles':>9s} {'repetibilidad %':>16s}")
for n, k, vs, r in filas: print(f"{n:32s} {k:9d} {vs:9d} {r:16.1f}")
"""),
    cu.md("""
<a id="param"></a>
## 2. Efecto de los parámetros del detector

La respuesta de Harris depende del tamaño de la ventana del tensor de estructura (`blockSize`) y del parámetro `k`. Se mide la repetibilidad en tres condiciones difíciles y el número de esquinas con respuesta mayor al 1 % del máximo.
"""),
    cu.code("""
dif = {"poca luz (×0.3)": (I, dict(brillo=0.3)), "plano girado 40°": (P.homografia_plano(w, h, FOCO, 40), {}), "giro 30° (roll)": (rot(roll=30), {}), "escala 0.6×": (P.homografia_rotacion(w, h, FOCO, escala=0.6), {})}
print(f"{'blockSize':>9s} {'k':>5s} {'esquinas orig.':>15s} " + " ".join(f"{n[:16]:>17s}" for n in dif))
for bloque in (2, 3, 5, 7):
    for kk in (0.04, 0.06):
        k0b = P.detectar_harris(gris0, n=300, bloque=bloque, k=kk)
        total = len(P.detectar_harris(gris0, n=100000, bloque=bloque, k=kk))
        fila = []
        for n, (H, kw) in dif.items():
            v = P.vista(objeto, H, **kw); k1 = P.detectar_harris(P.gris(v), n=300, bloque=bloque, k=kk)
            fila.append(P.repetibilidad(k0b, k1, H, v.shape, 3.0)[0] * 100)
        print(f"{bloque:9d} {kk:5.2f} {total:15d} " + " ".join(f"{x:16.1f}%" for x in fila))
"""),
    cu.md("""
<a id="hall"></a>
## 3. Hallazgos

__H1__

## Referencias

- Harris, C., & Stephens, M. (1988). A combined corner and edge detector. En *Proceedings of the 4th Alvey Vision Conference* (pp. 147–151). Alvey Vision Club.
- Schmid, C., Mohr, R., & Bauckhage, C. (2000). Evaluation of interest point detectors. *International Journal of Computer Vision, 37*(2), 151–172. https://doi.org/10.1023/A:1008199403446
- Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4.ª ed.). Pearson.
- Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools, 25*(11), 120–123.
- Imágenes: ver `LICENCIAS.md` (Wikimedia Commons, M. Rivera, CC0). El tablero `chessboard.jpg` se genera por código.
"""),
]

# =========================================================================== notebook de matching
EJ_M = [
    cu.md("""
# Experimentos: qué tan buenos son los descriptores para el matching

La actividad pide probar diferentes descriptores con variaciones de **iluminación** y **orientación**, y ver qué tan bien funciona el matching **con Harris por sí solo** o con otros como **SIFT** y **ORB**.
Como las dos fotos de personas del notebook base (`elon_1`, `elon_2`) no se incluyeron, se usa una escena urbana real (CC0) y se generan vistas con transformación conocida (`puntos.py`), de modo que cada correspondencia se puede calificar como correcta o errónea.

Se comparan tres alternativas, todas con la misma prueba de razón de Lowe (0.8) y 1500 puntos como máximo:

| Método | Detector | Descriptor | Distancia |
|---|---|---|---|
| **Harris + parche** | esquinas de Harris | parche 17×17 normalizado (media 0, norma 1) | L2 |
| **ORB** | FAST orientado | BRIEF binario girado | Hamming |
| **SIFT** | diferencia de gaussianas (invariante a escala) | histogramas de gradientes (128 valores) | L2 |

«Harris por sí solo» no tiene descriptor propio: describe únicamente *dónde* hay esquinas; aquí se le da el descriptor más sencillo posible (el parche de píxeles) para emparejarlo.
"""),
    cu.code("""
import numpy as np, cv2, matplotlib.pyplot as plt
import puntos as P
plt.rcParams.update({"font.size": 9})

escena = cv2.imread("data/escena_1.jpg")
h, w = escena.shape[:2]
FOCO = 900
def rot(**kw): return P.homografia_rotacion(w, h, FOCO, **kw)
I = np.eye(3)
condiciones = {
    "sin cambio":                          (I, {}),
    "poca luz (×0.3)":                     (I, dict(brillo=0.3)),
    "sobreexpuesta (γ=0.5, ×1.5)":         (I, dict(gamma=0.5, brillo=1.5)),
    "bajo contraste + ruido":              (I, dict(brillo=0.45, offset=70, ruido=8)),
    "giro 30° (roll)":                     (rot(roll=30), {}),
    "giro 90° (roll)":                     (rot(roll=90), {}),
    "perspectiva yaw 25°":                 (rot(yaw=25), {}),
    "escala 0.6×":                         (P.homografia_rotacion(w, h, FOCO, escala=0.6), {}),
    "giro 30° + poca luz":                 (rot(roll=30), dict(brillo=0.3)),
}
metodos = ["harris", "orb", "sift"]
res = {}
for n, (H, kw) in condiciones.items():
    v = P.vista(escena, H, **kw)
    for m in metodos:
        res[(n, m)] = P.evaluar(escena, v, H, m, n=1500, ratio=0.8)[0]
print("CORRESPONDENCIAS CORRECTAS (de las totales), precisión y matches consistentes con la homografía estimada (RANSAC)\\n")
print(f"{'condición':30s} " + " ".join(f"{m:>25s}" for m in metodos))
for n in condiciones:
    print(f"{n:30s} " + " ".join(f"{res[(n, m)]['correctos']:5d}/{res[(n, m)]['matches']:<5d} {res[(n, m)]['precision']*100:5.1f}% RS{res[(n, m)]['inliers_ransac']:<5d}" for m in metodos))
"""),
    cu.code("""
print("REPETIBILIDAD DEL DETECTOR (%) y TIEMPO (ms, detección en las dos imágenes)\\n")
print(f"{'condición':30s} " + " ".join(f"{m:>18s}" for m in metodos))
for n in condiciones:
    print(f"{n:30s} " + " ".join(f"{res[(n, m)]['repetibilidad']*100:8.1f}% {res[(n, m)]['t_deteccion_ms']:6.0f}ms" for m in metodos))
print("\\nError medio de la homografía estimada (px, en las 4 esquinas; NaN si RANSAC no encontró modelo)\\n")
print(f"{'condición':30s} " + " ".join(f"{m:>10s}" for m in metodos))
for n in condiciones: print(f"{n:30s} " + " ".join(f"{res[(n, m)]['error_H_px']:10.2f}" for m in metodos))
"""),
    cu.code("""
# Visualización: giro de 30° con poca luz (el caso más difícil de la tabla). Verde = correspondencia correcta, rojo = errónea.
H, kw = condiciones["giro 30° + poca luz"]
v = P.vista(escena, H, **kw)
fig, ejes = plt.subplots(3, 1, figsize=(15, 14))
for a, m in zip(ejes, metodos):
    r, (k1, k2, ms) = P.evaluar(escena, v, H, m, n=1500, ratio=0.8)
    a.imshow(cv2.cvtColor(P.dibujar_matches(escena, k1, v, k2, ms, H, max_n=60), cv2.COLOR_BGR2RGB))
    a.set_title(f"{m.upper()}: {r['correctos']} correctas de {r['matches']} ({r['precision']*100:.0f} %); se muestran las 60 mejores"); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    cu.md("""
## Hallazgos

__H2__

## Referencias

- Harris, C., & Stephens, M. (1988). A combined corner and edge detector. En *Proceedings of the 4th Alvey Vision Conference* (pp. 147–151). Alvey Vision Club.
- Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision, 60*(2), 91–110. https://doi.org/10.1023/B:VISI.0000029664.99615.94
- Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. En *2011 International Conference on Computer Vision* (pp. 2564–2571). IEEE. https://doi.org/10.1109/ICCV.2011.6126544
- Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM, 24*(6), 381–395. https://doi.org/10.1145/358669.358692
- Mikolajczyk, K., & Schmid, C. (2005). A performance evaluation of local descriptors. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 27*(10), 1615–1630. https://doi.org/10.1109/TPAMI.2005.188
- OpenCV. (s. f.). *Feature detection and description*. https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html
- Imágenes: ver `LICENCIAS.md` (Wikimedia Commons, M. Rivera, CC0).
"""),
]

TEXTO = {"H1": """
**Iluminación.**
- Harris resiste bien los cambios de brillo lineales: con la luz reducida a ×0.3 la repetibilidad es de **96.7 %** (241 esquinas visibles y 242 detectadas). Un cambio de brillo multiplica los gradientes, pero no mueve el lugar donde la respuesta es máxima, y al quedarse con las 300 esquinas más fuertes el umbral se ajusta solo.
- Lo que sí lo daña es lo **no lineal**: con la imagen sobreexpuesta (γ=0.5, ×1.5) la repetibilidad cae a **44.8 %**, porque las luces se saturan (el letrero es de un amarillo brillante) y las esquinas de las zonas saturadas desaparecen; con bajo contraste y ruido llega a 71.0 % y se detectan 300 esquinas (el límite), muchas de ellas espurias.

**Ángulo.**
- Al girar el plano del objeto (escorzo) la repetibilidad baja de forma gradual: **67.4 % (20°), 62.8 % (40°) y 56.6 % (60°)**. Una esquina deja de tener el mismo aspecto cuando la vecindad se deforma. Con poca luz añadida (40° + ×0.3) es de 64.9 %, es decir, la luz influye mucho menos que el ángulo.
- Con un giro de 30° en el plano de la imagen la repetibilidad es de 69.9 %. En teoría Harris es invariante a rotación (el determinante y la traza del tensor de estructura no cambian al girar), así que la pérdida se debe al remuestreo con interpolación, a que 22 de las 241 esquinas salen del encuadre y a que la posición se mide a ±3 px.
- Con una escala de 0.6× cae a **34.9 %**: Harris **no es invariante a la escala**. Al reducir la imagen, una esquina se convierte en una curva suave dentro de la misma ventana. Esa es la razón de que SIFT trabaje en un espacio de escalas.

**Parámetros.** Aumentar `blockSize` mejora la repetibilidad frente al giro (63.2 % con 2, 69.9 % con 3, 74.6 % con 5 y 77.5 % con 7, con k = 0.04) y algo frente a la poca luz (94.3 % → 98.0 %), pero reduce el número de esquinas totales (212 → 201) y suaviza su localización. `k` (0.04 frente a 0.06) tiene un efecto pequeño y sin patrón consistente (diferencias de unos pocos puntos). Frente a la escala ningún ajuste ayuda (32–39 %).

**Limitación.** Las vistas son sintéticas a partir de una sola fotografía (sin paralaje, sombras ni reflejos reales). Con fotos propias del mismo objeto bajo el sol y en penumbra, y desde varios ángulos, se espera un comportamiento similar, con las diferencias que traigan las sombras y los reflejos.
""", "H2": """
**Cambios de iluminación (sin cambiar la geometría).**
- Con poca luz (×0.3) los tres funcionan casi igual: Harris 1293/1296 correctas (99.8 %), ORB 1285/1309 (98.2 %) y SIFT 1215/1226 (99.1 %). Con bajo contraste y ruido, todos superan el 98 %.
- La **sobreexposición con saturación** es el caso más difícil de los cambios fotométricos: Harris conserva la precisión (98.8 %) pero con muy pocas correspondencias (323), SIFT baja a 92.3 % (262/284) y **ORB a 84.0 %** (184/219). Los descriptores binarios de ORB comparan pares de píxeles y se confunden cuando las zonas saturadas pierden textura.

**Cambios de orientación y escala.**
- **Harris por sí solo no sirve con giros**: con 30° solo produce 5 correspondencias y ninguna es correcta (0/5), y con 90° solo 3 (0/3). El descriptor de parche compara los píxeles tal como están y no es invariante a la rotación; el detector encuentra las mismas esquinas (repetibilidad de 73.5 % con 30° y 99.6 % con 90°, donde el giro exacto no interpola), pero no hay cómo reconocerlas. Con la escala de 0.6× solo logra 11/17.
- **ORB y SIFT funcionan con giros y escala**: con 30° ORB 786/827 (95.0 %) y SIFT 837/856 (97.8 %); con 90° ORB 567/624 (90.9 %) y SIFT 651/662 (98.3 %); con escala 0.6× ORB 519/534 (97.2 %) y SIFT 243/256 (94.9 %). Combinar giro y poca luz (30° + ×0.3) no cambia esa conclusión (ORB 94.5 %, SIFT 97.2 %).
- Con una **perspectiva moderada (yaw 25°)** hasta el parche de Harris sirve (219/227, 96.5 %), pero con menos correspondencias que ORB (522) o SIFT (546).

**Precisión geométrica y tiempo.**
- SIFT es el más preciso: el error de la homografía estimada (RANSAC) no supera 0.51 px en ninguna condición. ORB llega a errores de entre 0.05 y 7.6 px (el peor caso, con la perspectiva de 25°). Con solo 4 o 5 correspondencias, la homografía de Harris no tiene sentido (errores de cientos de píxeles).
- En tiempo de detección (dos imágenes de 1280×720, un solo equipo y varias ejecuciones), ORB fue el más rápido (unos 20–40 ms), Harris quedó en un punto intermedio (45–110 ms) y SIFT fue el más lento (105–220 ms), varias veces más que ORB (entre 3 y 8 veces según la condición). Los valores exactos cambian de una ejecución a otra, pero el orden se mantuvo.

**Conclusiones.** (1) Harris detecta *dónde* hay puntos estables; para emparejar hace falta un descriptor, y con uno ingenuo (píxeles del parche) no soporta rotación ni escala. (2) SIFT ofrece la mejor precisión y la mayor robustez, a cambio de tiempo de cómputo; ORB es una alternativa mucho más rápida con precisión ligeramente menor, salvo con saturación. (3) Ninguno resolvió sin errores la saturación de luces. (4) Los resultados dependen de la escena y de la prueba de razón (0.8); son de una sola escena con vistas sintéticas y deberían complementarse con fotos reales propias.
"""}


def armar(nb_path, nombre, ej, marcadores, parches, titulo, sub, pdf_nombre):
    nb = cu.cargar(nb_path)
    print(nombre, "parches:", cu.parchear(nb))
    parches(nb)
    for c in ej:
        if c.cell_type == "markdown":
            for k, v in marcadores.items():
                c.source = c.source.replace(k, v)
    cu.agregar(nb, ej)
    cu.ejecutar(nb, CODIGO, timeout=1500)
    cu.guardar(nb, CODIGO / f"{nombre}.ipynb")
    return cu.a_pdf(nb, AQUI / pdf_nombre, titulo, sub)


def parches_harris(nb):
    cu.reemplazar(nb, "from scipy.ndimage.filters import convolve", "from scipy.ndimage import convolve")


def parches_matching(nb):
    cu.reemplazar(nb, "data/elon_1.jpg", "data/escena_1.jpg", veces=3)
    cu.reemplazar(nb, "data/elon_2.png", "data/escena_2.png", veces=1)


def main():
    preparar_datos()
    shutil.copy(COMP / "LICENCIAS.md", CODIGO / "LICENCIAS.md")
    p1 = armar(AQUI / "enunciado" / "7_harris_edge_corner_detection.ipynb", "practica_7a_harris", EJ_H, {"__H1__": TEXTO["H1"]}, parches_harris,
               "Práctica 7 (parte A). Detector de esquinas de Harris", "Comportamiento bajo cambios de iluminación, ángulo y escala", "_p7a.pdf")
    p2 = armar(AQUI / "enunciado" / "9_image_matching.ipynb", "practica_7b_matching", EJ_M, {"__H2__": TEXTO["H2"]}, parches_matching,
               "Práctica 7 (parte B). Matching de puntos de interés", "Harris, ORB y SIFT con variaciones de iluminación y orientación", "_p7b.pdf")
    salida = AQUI / "Practica7_Harris_y_matching.pdf"
    doc = pymupdf.open(p1)
    doc.insert_pdf(pymupdf.open(p2))
    doc.save(salida)
    n = zip_entrega(AQUI / "Practica7_Harris_y_matching.zip", AQUI, ["codigo", salida.name])
    print("PDF", paginas_pdf(salida), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
