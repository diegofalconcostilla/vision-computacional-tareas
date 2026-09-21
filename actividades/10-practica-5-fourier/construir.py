"""Construye la entrega de la Práctica 5 (mejoramiento en el dominio de Fourier)."""
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
CODIGO = AQUI / "codigo"
DATA = CODIGO / "data"
NOMBRE = "practica_5_fourier"


def preparar_datos():
    DATA.mkdir(parents=True, exist_ok=True)
    g = cv2.imread(str(COMP / "calle_patterson.jpg"), 0)
    cv2.imwrite(str(DATA / "image.jpg"), cv2.resize(g, (480, 270), interpolation=cv2.INTER_AREA))
    rx = cv2.imread(str(COMP / "chest_xray.jpg"), 0)
    cv2.imwrite(str(DATA / "radiografia.png"), cv2.resize(rx, (512, round(rx.shape[0] * 512 / rx.shape[1])), interpolation=cv2.INTER_AREA))
    # left01.jpg (tablero de calibración del ejemplo de OpenCV) no se incluyó con la actividad: se genera uno propio con perspectiva
    n, cas = 9, 60
    tab = np.zeros((n * cas, n * cas), np.uint8)
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0: tab[i * cas:(i + 1) * cas, j * cas:(j + 1) * cas] = 255
    src = np.float32([[0, 0], [n * cas, 0], [n * cas, n * cas], [0, n * cas]])
    dst = np.float32([[90, 60], [560, 40], [600, 430], [50, 400]])
    M = cv2.getPerspectiveTransform(src, dst)
    izq = cv2.warpPerspective(tab, M, (640, 480), borderValue=128)
    cv2.imwrite(str(CODIGO / "left01.jpg"), izq)


EJ = [
    cu.md("""
# Ejercicios de la actividad

En el notebook base se implementan los tres filtros básicos con bucles sobre cada píxel. Aquí se reescriben de forma **vectorizada** (mucho más rápida) y se usan en dos aplicaciones, cada una medida contra una imagen de referencia limpia:

1. [Ejercicio a) Filtro pasa bajas: reducción de ruido en una radiografía](#a)
2. [Ejercicio b) Filtro pasa altas: recuperar nitidez (énfasis de alta frecuencia)](#b)
3. [Conclusión](#conclusion)

Los tres filtros, con $D(u,v)$ la distancia al centro del espectro y $D_0$ la frecuencia de corte:

| Filtro | Pasa bajas $H(u,v)$ | Pasa altas |
|---|---|---|
| Ideal | $1$ si $D \\le D_0$, $0$ si no | $1 - H_{LP}$ |
| Butterworth (orden $n$) | $1 / \\left(1 + (D/D_0)^{2n}\\right)$ | $1 - H_{LP}$ |
| Gaussiano | $e^{-D^2/(2D_0^2)}$ | $1 - H_{LP}$ |

El filtro ideal corta de golpe y produce **oscilaciones (efecto de anillo o *ringing*)** en la imagen filtrada; el Butterworth transiciona más suave (menos anillo cuanto menor $n$) y el gaussiano no produce anillo.
"""),
    cu.code("""
import numpy as np, cv2, matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
plt.rcParams.update({"font.size": 9})

def malla_distancia(forma):
    filas, cols = forma
    v, u = np.meshgrid(np.arange(cols) - cols / 2, np.arange(filas) - filas / 2)
    return np.hypot(u, v)

def pasa_bajas(forma, D0, tipo, n=2):
    D = malla_distancia(forma)
    if tipo == "ideal": return (D < D0).astype(float)      # igual que idealFilterLP del notebook base (distancia estricta)
    if tipo == "butterworth": return 1 / (1 + (D / D0) ** (2 * n))
    if tipo == "gaussiano": return np.exp(-(D ** 2) / (2 * D0 ** 2))
    raise ValueError(tipo)

def pasa_altas(forma, D0, tipo, n=2): return 1 - pasa_bajas(forma, D0, tipo, n)

def filtrar(img, H):
    F = np.fft.fftshift(np.fft.fft2(img.astype(float)))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))

def espectro(img): return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img.astype(float)))))

# Comprobación de que la versión vectorizada coincide con la del notebook base (filtro ideal, D0=50)
base_lp = idealFilterLP(50, img.shape)
print("Máxima diferencia con idealFilterLP del notebook base:", np.abs(base_lp - pasa_bajas(img.shape, 50, "ideal")).max())
"""),
    cu.md("""
<a id="a"></a>
## Ejercicio a) Filtro pasa bajas: reducción de ruido en una radiografía

**Investigación.** El ruido (del detector, de la cuantización o del bajo número de fotones) es un fenómeno de **alta frecuencia**: cambia de un píxel al siguiente. Los bordes y detalles también son de alta frecuencia, pero la energía de una imagen natural se concentra en las bajas frecuencias (espectro con un pico central que decae).
Un filtro pasa bajas elimina las componentes por encima de $D_0$ y con ello atenúa el ruido, a costa de suavizar los bordes (Gonzalez & Woods, 2018, cap. 4). En radiología se usa para reducir el ruido cuántico de imágenes de baja dosis antes de segmentar o medir.

**Demo.** A una radiografía de tórax (CC0) se le agrega ruido gaussiano (σ = 25 niveles), y se prueba cada filtro con distintas frecuencias de corte, midiendo **PSNR** y **SSIM** frente a la radiografía original sin ruido.
"""),
    cu.code("""
rx = cv2.imread("data/radiografia.png", 0).astype(float)
rng = np.random.default_rng(3)
ruidosa = np.clip(rx + rng.normal(0, 25, rx.shape), 0, 255)

cortes = [15, 30, 50, 80, 120]
tipos = ["ideal", "butterworth", "gaussiano"]
tabla = {}
for t in tipos:
    for D0 in cortes:
        s = np.clip(filtrar(ruidosa, pasa_bajas(rx.shape, D0, t)), 0, 255)
        tabla[(t, D0)] = (psnr(rx, s, data_range=255), ssim(rx, s, data_range=255), s)

print(f"Imagen con ruido: PSNR = {psnr(rx, ruidosa, data_range=255):.2f} dB, SSIM = {ssim(rx, ruidosa, data_range=255):.3f}\\n")
print(f"{'filtro':13s} " + " ".join(f"D0={D:<4d}" for D in cortes) + "   (PSNR en dB / SSIM)")
for t in tipos:
    print(f"{t:13s} " + " ".join(f"{tabla[(t, D)][0]:5.2f}/{tabla[(t, D)][1]:.2f}" for D in cortes))
mejor = {t: max(cortes, key=lambda D: tabla[(t, D)][0]) for t in tipos}
print("\\nMejor D0 por filtro:", mejor)
"""),
    cu.code("""
fig, ejes = plt.subplots(2, 4, figsize=(17, 8.4))
ejes[0, 0].imshow(rx, cmap="gray", vmin=0, vmax=255); ejes[0, 0].set_title("Original (referencia)")
ejes[0, 1].imshow(ruidosa, cmap="gray", vmin=0, vmax=255); ejes[0, 1].set_title("Con ruido gaussiano σ=25")
ejes[0, 2].imshow(espectro(rx), cmap="gray"); ejes[0, 2].set_title("Espectro de la original (log)")
ejes[0, 3].imshow(espectro(ruidosa), cmap="gray"); ejes[0, 3].set_title("Espectro con ruido: se \\"llena\\" de energía en alta frecuencia")
for a, t in zip(ejes[1, :3], tipos):
    D = mejor[t]; p, s_, im = tabla[(t, D)]
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(f"{t}, D0={D}: {p:.2f} dB")
# perfil radial del espectro: cuánta energía queda por debajo de cada frecuencia
Dm = malla_distancia(rx.shape).astype(int)
def perfil(img):
    P = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    return np.bincount(Dm.ravel(), P.ravel()) / np.bincount(Dm.ravel())
for nombre, im in (("original", rx), ("con ruido", ruidosa)): ejes[1, 3].semilogy(perfil(im)[:200], label=nombre)
ejes[1, 3].set_xlabel("frecuencia radial D"); ejes[1, 3].set_title("Potencia espectral por frecuencia"); ejes[1, 3].legend(); ejes[1, 3].grid(alpha=.3)
for a in ejes.ravel()[:7]: a.axis("off")
plt.tight_layout(); plt.show()

# Efecto de anillo (ringing): un borde en escalón limpio filtrado con D0=15; se mide el sobreimpulso y el subimpulso
paso = np.zeros((256, 256)); paso[:, 128:] = 255
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
print(f"{'filtro':13s} {'sobreimpulso':>13s} {'subimpulso':>11s}   (respecto al salto de 255 niveles)")
for t in tipos:
    fila = filtrar(paso, pasa_bajas(paso.shape, 15, t))[128]
    print(f"{t:13s} {(fila.max() - 255) / 255 * 100:12.1f}% {(0 - fila.min()) / 255 * 100:10.1f}%")
    ax[0].plot(np.arange(90, 170), fila[90:170], label=t)
ax[0].axvline(128, color="k", lw=.6, ls=":"); ax[0].set_title("Perfil horizontal de un borde en escalón tras pasa bajas (D0=15)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].imshow(np.clip(filtrar(paso, pasa_bajas(paso.shape, 15, "ideal")), -60, 315), cmap="gray"); ax[1].set_title("Filtro ideal: las bandas paralelas al borde son el anillo"); ax[1].axis("off")
plt.tight_layout(); plt.show()
"""),
    cu.md("__DISC_A__"),
    cu.md("""
<a id="b"></a>
## Ejercicio b) Filtro pasa altas: recuperar nitidez con énfasis de alta frecuencia

**Investigación.** Un filtro pasa altas conserva los detalles finos y bordes y elimina el fondo suave (el término de frecuencia cero, que es el brillo medio). Solo, produce una imagen de bordes casi negra; por eso, para **mejorar la nitidez** se usa el *filtrado de énfasis de alta frecuencia*:
$H_{hfe}(u,v) = a + b\\,H_{HP}(u,v)$, con $a \\ge 0$ y $b > a$ (Gonzalez & Woods, 2018, cap. 4). Con $a=1$ y $b>0$ equivale a la máscara de enfoque: $g = f + b\\,\\text{HP}(f)$. Se aplica en radiografías, imágenes satelitales y microscopía para realzar bordes y en preprocesamiento antes de detectar contornos.

**Demo.** La radiografía se desenfoca con un gaussiano (σ = 2 px) para simular una toma borrosa, y se intenta recuperar con cada filtro pasa altas (D0 = 30, a = 1, b = 1.5). Como se conoce la imagen nítida original, se mide PSNR, SSIM y un índice de nitidez (varianza del Laplaciano).
"""),
    cu.code("""
def nitidez(g): return float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var())

borrosa = cv2.GaussianBlur(rx, (0, 0), 2.0)
print(f"{'imagen':30s} {'PSNR':>7s} {'SSIM':>7s} {'nitidez':>9s}")
print(f"{'nítida (referencia)':30s} {'':>7s} {'':>7s} {nitidez(rx):9.1f}")
print(f"{'borrosa (σ=2)':30s} {psnr(rx, borrosa, data_range=255):7.2f} {ssim(rx, borrosa, data_range=255):7.3f} {nitidez(borrosa):9.1f}")
res = {}
for t in tipos:
    for D0 in (20, 30, 50):
        salida = np.clip(borrosa + 1.5 * filtrar(borrosa, pasa_altas(rx.shape, D0, t)), 0, 255)
        res[(t, D0)] = salida
        print(f"{t + ', D0=' + str(D0):30s} {psnr(rx, salida, data_range=255):7.2f} {ssim(rx, salida, data_range=255):7.3f} {nitidez(salida):9.1f}")
"""),
    cu.code("""
fig, ejes = plt.subplots(2, 4, figsize=(17, 8.4))
ejes[0, 0].imshow(rx, cmap="gray", vmin=0, vmax=255); ejes[0, 0].set_title("Nítida (referencia)")
ejes[0, 1].imshow(borrosa, cmap="gray", vmin=0, vmax=255); ejes[0, 1].set_title("Borrosa (σ=2)")
for a, t in zip(ejes.ravel()[2:5], tipos):
    solo = filtrar(borrosa, pasa_altas(rx.shape, 30, t)); a.imshow(np.clip(solo * 4 + 128, 0, 255), cmap="gray", vmin=0, vmax=255); a.set_title(f"Solo pasa altas {t} (×4, +128)")
for a, t in zip(ejes.ravel()[5:8], tipos):
    a.imshow(res[(t, 30)], cmap="gray", vmin=0, vmax=255); a.set_title(f"Borrosa + 1.5·HP {t}, D0=30")
for a in ejes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()

h, w = rx.shape; y0, y1, x0, x1 = int(h * .05), int(h * .30), int(w * .42), int(w * .75)
fig, ejes = plt.subplots(1, 5, figsize=(19, 3.4))
for a, (t, im) in zip(ejes, (("nítida", rx), ("borrosa", borrosa), ("ideal", res[("ideal", 30)]), ("butterworth", res[("butterworth", 30)]), ("gaussiano", res[("gaussiano", 30)]))):
    a.imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255); a.set_title(t + " (recorte)"); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    cu.md("__DISC_B__"),
    cu.md("""
<a id="conclusion"></a>
## Conclusión

__CONCLUSION__

## Referencias

- Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4.ª ed.). Pearson. (Cap. 4: Filtering in the frequency domain.)
- Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation, 19*(90), 297–301. https://doi.org/10.1090/S0025-5718-1965-0178586-1
- Butterworth, S. (1930). On the theory of filter amplifiers. *Experimental Wireless and the Wireless Engineer, 7*, 536–541.
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing, 13*(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Imágenes: ver `LICENCIAS.md` (Wikimedia Commons: Rivera y Stillwaterising, CC0). El tablero `left01.jpg` es generado por código.
"""),
]

DISC = {
    "A": """
**Resultados y discusión.**
- **Reducción de ruido.** La radiografía con ruido tiene 20.26 dB de PSNR y 0.160 de SSIM. El mejor resultado por PSNR se obtiene con el Butterworth y D0=50 (31.61 dB, +11.4 dB), seguido del gaussiano (31.52 dB) y del ideal (30.68 dB). El espectro de potencia lo explica: el ruido blanco añade una meseta de potencia constante en toda la banda de alta frecuencia, mientras que la potencia de la radiografía decae con la frecuencia; un corte en la zona donde ambas curvas se separan elimina el ruido conservando casi toda la energía de la imagen.
- **La frecuencia de corte es un compromiso.** Con D0=15 el filtrado borra detalle (27.5–28.7 dB) y con D0=120 deja pasar casi todo el ruido (26.7–27.8 dB); el óptimo por PSNR está cerca de D0=50. Además, PSNR y SSIM no coinciden: el SSIM, que mide la estructura, alcanza su máximo con D0=30 (0.87 con Butterworth y gaussiano), un suavizado más fuerte que el que prefiere el PSNR.
- **Efecto de anillo (*ringing*).** En un borde en escalón filtrado con D0=15 el filtro ideal produce un sobreimpulso y un subimpulso de 9.0 % del salto; el Butterworth (n=2), de 3.3 %; y el gaussiano, de 0 %. En la radiografía esto se traduce en que, con el mismo D0, el ideal siempre queda por debajo de los otros dos (por ejemplo, 30.04 dB frente a 30.85 y 31.15 dB con D0=30). El filtro ideal es útil para entender el concepto, pero no para aplicaciones reales.
- El precio de cualquier pasa bajas es suavizar los bordes y las estructuras finas (vasos, costillas), por lo que en imagen médica se combina con un criterio de calidad diagnóstica y no solo con una métrica.
""",
    "B": """
**Resultados y discusión.**
- La imagen desenfocada tiene 37.21 dB de PSNR, 0.938 de SSIM y un índice de nitidez de 0.6, frente a 48.7 de la imagen nítida. Todas las variantes con énfasis de alta frecuencia **aumentan la nitidez** (entre 4.9 y 18.8), pero **ninguna alcanza el valor de la imagen original**.
- El **PSNR baja en todos los casos** (29.3 a 34.7 dB frente a 37.2 dB): sumar componentes de alta frecuencia no reconstruye exactamente las que el desenfoque eliminó, y amplifica el ruido de cuantización y produce sobreimpulsos. El **SSIM mejora** con Butterworth y gaussiano cuando D0 ≥ 30 (0.948–0.953 frente a 0.938), pero **empeora con el filtro ideal** (0.881–0.916) por el anillo. Una imagen más «nítida» no es necesariamente más parecida a la original.
- Con D0 mayores (50) los resultados son mejores que con D0 pequeños (20): un corte bajo aplica el realce a una banda demasiado ancha, incluyendo frecuencias medias donde hay más ruido que detalle. El **gaussiano** es el mejor (34.65 dB y 0.952 con D0=50) y el ideal el peor, en coherencia con el ejercicio anterior.
- El filtro pasa altas por sí solo (figura de bordes) devuelve una imagen casi negra con solo los bordes: el brillo medio desaparece. Para que sirva como realce hay que sumarlo a la imagen original, que es lo que hace el énfasis de alta frecuencia.
- Si la causa del desenfoque se conoce (aquí, un gaussiano de σ=2), la técnica adecuada es la **deconvolución** (filtro inverso o de Wiener), que divide por la respuesta en frecuencia del desenfoque en lugar de sumar un filtro pasa altas genérico.
""",
}
CONCLUSION = """
- En el dominio de Fourier, filtrar es **multiplicar** el espectro por una función de transferencia; la misma operación en el dominio espacial exige una convolución. El costo se reduce con la FFT y permite diseñar el filtro pensando en frecuencias.
- **Pasa bajas = quitar ruido y detalle.** Con ruido gaussiano de σ=25, el mejor filtro (Butterworth, D0=50) subió el PSNR de 20.26 a 31.61 dB. La frecuencia de corte es un compromiso entre ruido residual (corte alto) y desenfoque (corte bajo).
- **Pasa altas = realzar detalle.** El énfasis de alta frecuencia aumentó la nitidez de una imagen borrosa (de 0.6 a 4.9–18.8), pero redujo el PSNR y solo mejoró el SSIM con los filtros suaves; para desenfoques conocidos conviene la deconvolución.
- **Ideal, Butterworth y gaussiano.** El filtro ideal produce anillo (9.0 % de sobreimpulso en un escalón), el Butterworth de orden 2 lo reduce a 3.3 % y el gaussiano lo elimina, y en todas las pruebas el ideal fue el peor. La elección práctica suele ser el gaussiano o el Butterworth.
- **Métricas.** PSNR y SSIM pueden preferir parámetros distintos; conviene usar ambas y mirar la imagen.
- Limitaciones: una sola radiografía y ruido sintético gaussiano; el ruido real de un equipo es más complejo (dependiente de la señal, correlacionado). Además, se comprobó que la versión vectorizada del filtro ideal coincide exactamente con la del notebook base (diferencia máxima 0), y se generó un tablero `left01.jpg` propio porque el original no venía con la actividad.
"""


def main():
    preparar_datos()
    nb = cu.cargar(AQUI / "enunciado" / "5_frequency_domain.ipynb")
    print("parches:", cu.parchear(nb))
    celdas = []
    for c in EJ:
        if c.cell_type == "markdown":
            for k, v in (("__DISC_A__", DISC.get("A", "(pendiente)")), ("__DISC_B__", DISC.get("B", "(pendiente)")),
                         ("__CONCLUSION__", CONCLUSION or "(pendiente)")):
                c.source = c.source.replace(k, v)
        celdas.append(c)
    cu.agregar(nb, celdas)
    cu.ejecutar(nb, CODIGO)
    cu.guardar(nb, CODIGO / f"{NOMBRE}.ipynb")
    shutil.copy(COMP / "LICENCIAS.md", CODIGO / "LICENCIAS.md")
    pdf = cu.a_pdf(nb, AQUI / "Practica5_Fourier.pdf", "Práctica 5. Mejoramiento en el dominio de Fourier",
                   "Filtros pasa bajas y pasa altas: ideal, Butterworth y gaussiano")
    n = zip_entrega(AQUI / "Practica5_Fourier.zip", AQUI, ["codigo", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
