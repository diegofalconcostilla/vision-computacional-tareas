"""Construye la entrega de la Práctica 1 (procesamiento básico de imágenes).

    python construir.py

Genera: codigo/practica_1_procesamiento_basico.ipynb (ejecutado), Practica1_Procesamiento_basico.pdf y .zip
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas import cuaderno as cu  # noqa: E402
from herramientas.reporte import EQUIPO, paginas_pdf, zip_entrega  # noqa: E402

COMPARTIDOS = RAIZ / "datos_compartidos"
DATA = AQUI / "codigo" / "data"
NOMBRE = "practica_1_procesamiento_basico"


def preparar_datos():
    DATA.mkdir(parents=True, exist_ok=True)

    def copia(origen, destino, ancho):
        im = Image.open(COMPARTIDOS / origen).convert("RGB")
        im = im.resize((ancho, round(im.height * ancho / im.width)), Image.LANCZOS)
        im.save(DATA / destino, quality=92)

    copia("calle_oak_alden.jpg", "image.jpg", 960)            # imagen que usa el notebook original
    copia("calle_patterson.jpg", "calle_patterson.jpg", 960)  # imágenes "propias" para el ejercicio 1
    copia("microscopia_levaduras.jpg", "microscopia.jpg", 960)
    copia("celulas_hela_dapi.jpg", "celulas.jpg", 960)
    # radiografía y resonancia para los ejercicios de negativo y gamma
    for a, b in (("chest_xray.jpg", "radiografia_torax.jpg"), ("mri_glioblastoma_t1km.jpg", "resonancia_cerebro.jpg")):
        shutil.copy(COMPARTIDOS / a, DATA / b)
    # degradado lineal 0-255 que pide la celda de binarización
    grad = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (128, 1))
    Image.fromarray(np.stack([grad] * 3, axis=-1)).save(DATA / "linear_gradient.png")


EJERCICIOS = [
    cu.md("""
# Ejercicios de la actividad

Los cuatro ejercicios pedidos se agregan a continuación del notebook base. Las imágenes se ejecutan localmente desde la carpeta `data/`
(fotografías de dominio público o con licencia libre; ver `LICENCIAS.md`). Para usar imágenes propias basta con reemplazar los archivos de `data/`
conservando su nombre.

1. [Transformaciones fotométricas para aumento de datos](#ej1)
2. [Negativo de imagen: aplicación en radiología](#ej2)
3. [Corrección gamma: aplicación en imagenología médica y en pantallas](#ej3)
4. [Sustracción de imágenes: detección de cambios y angiografía](#ej4)
5. [Conclusión](#conclusion)
"""),
    cu.md("""
<a id="ej1"></a>
## Ejercicio 1. Transformaciones fotométricas para aumento de datos

**Investigación.** Las transformaciones *píxel a píxel* modifican el valor de cada píxel sin cambiar su posición (a diferencia de las geométricas: giros, recortes, escalado).
Cuando se emplean para *aumentar* un conjunto de entrenamiento (*data augmentation*), simulan cambios de iluminación, de cámara o de color que el modelo
encontrará en el mundo real, sin necesidad de fotografiar de nuevo (Shorten & Khoshgoftaar, 2019). Como la etiqueta de la imagen no cambia (un letrero sigue siendo un letrero),
son una forma barata de mejorar la robustez. Se aplican tres:

| # | Transformación | Modelo | Qué simula |
|---|---|---|---|
| 1 | Brillo y contraste lineal | $g(x,y) = \\alpha\\, f(x,y) + \\beta$ | Cambios de exposición o de iluminación |
| 2 | Perturbación de color en HSV | $H' = H + \\Delta h,\\; S' = k_s S,\\; V' = k_v V$ | Balance de blancos, saturación distinta entre cámaras |
| 3 | Ruido gaussiano aditivo | $g = f + \\eta,\\; \\eta \\sim \\mathcal{N}(0,\\sigma^2)$ | Ruido del sensor con ISO alto o poca luz |
"""),
    cu.code("""
import cv2, numpy as np, matplotlib.pyplot as plt

def leer_rgb(ruta):
    return cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB)

def brillo_contraste(img, alfa, beta):
    # g = alfa*f + beta, con saturación a [0, 255]
    return np.clip(alfa * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

def jitter_hsv(img, dh, ks, kv):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)   # H en [0,180) en OpenCV
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * ks, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * kv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def ruido_gaussiano(img, sigma, semilla=0):
    rng = np.random.default_rng(semilla)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

propias = {"Calle 1": "data/image.jpg", "Calle 2": "data/calle_patterson.jpg", "Microscopía": "data/microscopia.jpg"}
columnas = [("Original", lambda i: i),
            ("Brillo/contraste\\nα=1.4, β=−30", lambda i: brillo_contraste(i, 1.4, -30)),
            ("HSV\\nΔh=+12, ks=1.6, kv=0.85", lambda i: jitter_hsv(i, 12, 1.6, 0.85)),
            ("Ruido gaussiano\\nσ=25", lambda i: ruido_gaussiano(i, 25))]

fig, ejes = plt.subplots(len(propias), len(columnas), figsize=(15, 8.2))
for r, (nombre, ruta) in enumerate(propias.items()):
    img = leer_rgb(ruta)
    for c, (titulo, f) in enumerate(columnas):
        ejes[r, c].imshow(f(img)); ejes[r, c].axis("off")
        if r == 0: ejes[r, c].set_title(titulo)
    ejes[r, 0].text(-0.02, 0.5, nombre, transform=ejes[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=11)
plt.tight_layout(); plt.show()
"""),
    cu.code("""
# Estadísticas: cómo cambia la distribución de intensidades con cada transformación (imagen "Calle 1")
img = leer_rgb("data/image.jpg")
filas = []
for titulo, f in columnas:
    g = cv2.cvtColor(f(img), cv2.COLOR_RGB2GRAY).astype(float)
    filas.append((titulo.replace("\\n", " "), g.mean(), g.std(), (g >= 250).mean() * 100, (g <= 5).mean() * 100))
print(f"{'Transformación':46s} {'media':>7s} {'desv.':>7s} {'%saturado':>10s} {'%negro':>7s}")
for t, m, s, sat, neg in filas:
    print(f"{t:46s} {m:7.1f} {s:7.1f} {sat:10.2f} {neg:7.2f}")
"""),
    cu.md("""
**Discusión.** El brillo/contraste aumenta la desviación estándar (más contraste) pero desplaza la media y crea píxeles saturados o negros que ya no se pueden recuperar;
el ruido gaussiano casi no cambia la media, pero altera la textura fina (importante para redes que dependen de bordes); y la perturbación HSV cambia el color sin tocar las formas.
Estas variantes obligan al modelo a apoyarse en la *forma* del objeto y no en un color o una exposición concretos. Hay que elegir parámetros verosímiles:
un ruido de σ=25 es plausible para una foto nocturna, pero en una imagen médica podría ocultar una lesión real y cambiar el diagnóstico, así que en ese dominio el aumento fotométrico se usa con más cautela.
"""),
    cu.md("""
<a id="ej2"></a>
## Ejercicio 2. Negativo de imagen: aplicación en radiología

**Investigación.** El negativo de una imagen con $L$ niveles es $s = L - 1 - r$. Gonzalez y Woods (2018) señalan que es especialmente adecuado para *realzar detalles blancos o grises
inmersos en regiones oscuras* de una imagen, en particular cuando las zonas oscuras dominan; su ejemplo es una mamografía digital, donde las lesiones pequeñas son más fáciles de ver en la versión negativa.
Otra aplicación con valor específico es la **digitalización de películas fotográficas y radiográficas**: el escáner captura el negativo y hay que invertirlo para obtener la imagen positiva.

**Demo.** Se invierte una radiografía de tórax (CC0) y se compara una región del campo pulmonar, donde las estructuras finas (vasos y bronquios) son de bajo contraste.
"""),
    cu.code("""
rx = cv2.imread("data/radiografia_torax.jpg", cv2.IMREAD_GRAYSCALE)
neg = 255 - rx
h, w = rx.shape
roi = (slice(int(h*0.18), int(h*0.50)), slice(int(w*0.10), int(w*0.42)))   # campo pulmonar derecho del paciente

fig, ejes = plt.subplots(2, 3, figsize=(14, 9))
for c, (t, im) in enumerate((("Radiografía original", rx), ("Negativo", neg))):
    ejes[0, c].imshow(im, cmap="gray", vmin=0, vmax=255); ejes[0, c].set_title(t); ejes[0, c].axis("off")
    ejes[1, c].imshow(im[roi], cmap="gray", vmin=0, vmax=255); ejes[1, c].set_title(f"{t}: región ampliada"); ejes[1, c].axis("off")
ejes[0, 2].hist(rx.ravel(), 128, color="#444", alpha=.7, label="original"); ejes[0, 2].hist(neg.ravel(), 128, color="#c33", alpha=.6, label="negativo")
ejes[0, 2].set_title("Histogramas (el negativo es el reflejo)"); ejes[0, 2].legend()
ejes[1, 2].axis("off")
ejes[1, 2].text(0, .6, f"media original:  {rx.mean():.1f}\\nmedia negativo:  {neg.mean():.1f}\\n"
                       f"desv. original:  {rx.std():.1f}\\ndesv. negativo:  {neg.std():.1f}", family="monospace", fontsize=12)
plt.tight_layout(); plt.show()
"""),
    cu.md("""
**Discusión.** La desviación estándar no cambia (el negativo solo refleja el histograma), de modo que no hay más información: lo que mejora es la *percepción*.
El ojo humano discrimina mejor pequeñas diferencias de luminancia en fondos claros que en fondos muy oscuros, por lo que algunos especialistas prefieren revisar ciertas regiones en negativo.
Por eso el negativo es útil como paso de visualización, no como algoritmo de mejora de la información.
"""),
    cu.md("""
<a id="ej3"></a>
## Ejercicio 3. Corrección gamma

**Investigación.** La transformación de potencia $s = c\\, r^{\\gamma}$ (con $r \\in [0,1]$) tiene dos usos prácticos. (1) **Compensación de dispositivos de visualización:** los monitores
reproducen la intensidad con una respuesta aproximadamente $V^{2.2}$, por lo que las imágenes se *codifican* con el exponente inverso; sin esa corrección se verían demasiado oscuras.
(2) **Mejora de contraste en imágenes oscuras o lavadas:** con $\\gamma < 1$ se expanden las intensidades bajas (se ven detalles en sombras); con $\\gamma > 1$ se comprimen las bajas y se expanden las altas.
Gonzalez y Woods (2018) lo ilustran con una resonancia magnética de columna y una imagen aérea lavada. Aquí se aplica a una resonancia cerebral T1 con contraste, cuya materia gris y estructuras profundas quedan en la zona oscura del histograma.
"""),
    cu.code("""
mri = cv2.imread("data/resonancia_cerebro.jpg", cv2.IMREAD_GRAYSCALE)

def gamma(img, g, c=1.0):
    return np.clip(c * (img / 255.0) ** g, 0, 1)

gammas = [1.0, 0.6, 0.4, 0.3, 2.0]
fig, ejes = plt.subplots(2, len(gammas), figsize=(16, 6.6))
for c, g in enumerate(gammas):
    out = gamma(mri, g)
    ejes[0, c].imshow(out, cmap="gray", vmin=0, vmax=1); ejes[0, c].axis("off")
    ejes[0, c].set_title("original (γ=1)" if g == 1 else f"γ = {g}")
    ejes[1, c].hist(out.ravel(), 64, color="#345"); ejes[1, c].set_yticks([]); ejes[1, c].set_xlim(0, 1)
plt.tight_layout(); plt.show()

r = np.linspace(0, 1, 256)
plt.figure(figsize=(5.2, 4.6))
for g in gammas: plt.plot(r, r**g, label=f"γ={g}")
plt.plot(r, r, "k:", lw=.8); plt.xlabel("entrada r"); plt.ylabel("salida s"); plt.title("Curvas de la transformación gamma"); plt.legend(); plt.grid(alpha=.3); plt.show()
"""),
    cu.code("""
# Demo de la compensación de pantalla: una rampa lineal de luz se ve "correcta" solo si se codifica con γ≈1/2.2
rampa = np.tile(np.linspace(0, 1, 512), (80, 1))
codificada = rampa ** (1 / 2.2)
fig, ejes = plt.subplots(1, 2, figsize=(11, 2.2))
ejes[0].imshow(rampa, cmap="gray", vmin=0, vmax=1); ejes[0].set_title("Rampa lineal sin codificar"); ejes[0].axis("off")
ejes[1].imshow(codificada, cmap="gray", vmin=0, vmax=1); ejes[1].set_title("Codificada con γ = 1/2.2 (como sRGB)"); ejes[1].axis("off")
plt.show()
"""),
    cu.md("""
**Discusión.** Con γ=0.4 se revela la textura de los tejidos que en la imagen original casi no se distingue, a costa de perder contraste en las zonas ya claras (el tumor comienza a saturarse);
con γ=2 ocurre lo contrario. La elección del exponente depende de la tarea: no existe un valor universal. Es la razón de que los sistemas de visión suelan normalizar la iluminación antes de aplicar un detector.
"""),
    cu.md("""
<a id="ej4"></a>
## Ejercicio 4. Sustracción de imágenes

**Investigación.** La diferencia $g(x,y) = f_1(x,y) - f_2(x,y)$ elimina lo que es común a ambas imágenes y deja únicamente lo que cambió. Dos aplicaciones clásicas:
- **Detección de cambios / movimiento** en videovigilancia y tráfico: se resta un fondo de referencia del fotograma actual y se umbraliza (Gonzalez & Woods, 2018).
- **Angiografía por sustracción digital (DSA):** se resta una imagen "máscara" (antes de inyectar contraste) de la imagen "en vivo" (con contraste), de modo que el hueso y los tejidos desaparecen y solo quedan los vasos (Gonzalez & Woods, 2018, modo máscara).

**Demo.** Detección de cambios con dos fotogramas. Como aquí no hay un video, el segundo fotograma se construye con la misma escena (calle, CC0) más una pieza recortada que se desplaza,
y con ruido gaussiano distinto en cada toma para imitar el sensor. Así se conoce la verdad: dónde está realmente el objeto.
"""),
    cu.code("""
base = cv2.cvtColor(cv2.imread("data/image.jpg"), cv2.COLOR_BGR2GRAY).astype(np.float32)
H, W = base.shape
rng = np.random.default_rng(3)

# "objeto" = un parche real de la escena (una zona con detalle) pegado en dos posiciones
parche = base[int(H*.62):int(H*.62)+70, int(W*.30):int(W*.30)+110].copy()
def fotograma(x, y):
    f = base.copy(); f[y:y+parche.shape[0], x:x+parche.shape[1]] = parche
    return np.clip(f + rng.normal(0, 4, f.shape), 0, 255)

f1 = fotograma(80, 60)      # objeto arriba a la izquierda
f2 = fotograma(190, 90)     # el objeto se movió
diff = np.abs(f2 - f1)
mascara = (cv2.GaussianBlur(diff, (5, 5), 0) > 25).astype(np.uint8)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # quita puntos de ruido aislados

fig, ejes = plt.subplots(1, 4, figsize=(17, 3.6))
for a, im, t in zip(ejes, (f1, f2, diff, mascara), ("Fotograma 1", "Fotograma 2", "|F2 − F1|", "Cambio detectado (umbral 25)")):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()

verdad = np.zeros_like(mascara)
for (x, y) in ((80, 60), (190, 90)): verdad[y:y+parche.shape[0], x:x+parche.shape[1]] = 1
inter = (mascara & verdad).sum(); union = (mascara | verdad).sum()
print(f"IoU entre la máscara detectada y las dos posiciones reales del objeto: {inter/union:.2f}")
"""),
    cu.md("""
**Discusión.** La sustracción separa lo que cambió, pero es sensible al ruido: sin umbral ni suavizado aparecen puntos por todo el fondo, y por eso se combina con filtrado y apertura morfológica.
Además el resultado marca *ambas* posiciones (donde estaba y donde está el objeto), no solo la nueva. En una angiografía real la máscara y la imagen en vivo deben estar perfectamente alineadas: un pequeño movimiento del paciente
genera artefactos que obligan a registrar las imágenes antes de restarlas.
"""),
    cu.md("""
<a id="conclusion"></a>
## Conclusión

- Las operaciones píxel a píxel son las más simples del procesamiento digital de imágenes y, aun así, sirven para tareas tan distintas como **generar variantes de entrenamiento** (ejercicio 1),
  **mejorar la lectura de una radiografía** (2), **compensar la respuesta de una pantalla o realzar tejidos oscuros** (3) y **detectar cambios** (4).
- Ninguna de ellas agrega información: reasignan valores. El negativo conserva la desviación estándar exactamente; gamma y el brillo/contraste solo desplazan la información a otra parte del histograma, y pueden perderla si saturan.
- La calidad del resultado depende siempre del **histograma de partida**: una transformación útil en una imagen oscura (γ<1) daña una imagen ya brillante.
- Para conjuntos de datos, el aumento fotométrico debe conservar la etiqueta y ser verosímil; en imagen médica se usa con más cautela que en fotografía general.
- Hallazgos sobre el notebook base (ambos afectan también al Colab actual): (a) `Image.ANTIALIAS` ya no existe en Pillow reciente y se cambió por `Image.LANCZOS`;
  (b) con NumPy 2 la suma `np.max(orig_img) + 1` sobre `uint8` desborda (255 + 1 = 0), el logaritmo de 0 es infinito y la imagen de la transformación logarítmica salía completamente negra;
  se corrigió convirtiendo a `float64` antes de aplicar el logaritmo.

## Referencias

- Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4.ª ed.). Pearson.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*, Artículo 60. https://doi.org/10.1186/s40537-019-0197-0
- Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools*, 25(11), 120–123.
- OpenCV. (s. f.). *Histogram equalization* [Tutorial]. https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
- Imágenes: ver `LICENCIAS.md` (Wikimedia Commons: Rivera, Stillwaterising, Hellerhoff, Arnatbio, Chaurasiya; licencias CC0, CC BY-SA).
"""),
]


def main():
    preparar_datos()
    (AQUI / "codigo").mkdir(exist_ok=True)
    nb = cu.cargar(AQUI / "enunciado" / "1_basic_operations.ipynb")
    parches = cu.parchear(nb)
    print("parches:", parches)
    # NumPy 2: uint8 + 1 desborda (255+1=0) y la transformación logarítmica salía completamente negra
    cu.reemplazar(nb, "c = 255 / np.log(1 + np.max(orig_img))", "c = 255 / np.log(1 + np.max(orig_img.astype(np.float64)))")
    cu.reemplazar(nb, "log_img = c * (np.log(orig_img + 1))", "log_img = c * (np.log(orig_img.astype(np.float64) + 1))")
    cu.agregar(nb, EJERCICIOS)
    cu.ejecutar(nb, AQUI / "codigo")
    cu.formato_entrega(nb)  # sin negritas en las reflexiones + declaración de uso de IA
    ruta_nb = cu.guardar(nb, AQUI / "codigo" / f"{NOMBRE}.ipynb")
    shutil.copy(RAIZ / "datos_compartidos" / "LICENCIAS.md", AQUI / "codigo" / "LICENCIAS.md")
    pdf = cu.a_pdf(nb, AQUI / "Practica1_Procesamiento_basico.pdf",
                   "Práctica 1. Procesamiento básico de imágenes",
                   "Transformaciones píxel a píxel: fotométricas, negativo, gamma y sustracción",
                   integrantes=EQUIPO, fecha="27/9/2026")
    n = zip_entrega(AQUI / "Practica1_Procesamiento_basico.zip", AQUI, ["codigo", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
