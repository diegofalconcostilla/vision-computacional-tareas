"""Informe PDF: Método de segmentación de Otsu aplicado a imágenes médicas."""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
I = R["imagenes"]
E = R["estabilidad_ruido"]
RX, TC, NU = I["Radiografía de tórax"], I["TC de cráneo (corte axial)"], I["Microscopía de fluorescencia (núcleos)"]
NRX, NTC, NNU = "Radiografía de tórax", "TC de cráneo (corte axial)", "Microscopía de fluorescencia (núcleos)"


def main():
    r = Reporte("Segmentación de Otsu en imágenes médicas",
                "Radiografía de tórax, tomografía de cráneo y microscopía de fluorescencia", carpeta=AQUI,
                entrega="Reporte breve en PDF + archivo comprimido con el código")
    r.portada()

    r.h2("1. Introducción")
    r.p("La <strong>segmentación de imágenes</strong> médicas consiste en separar las regiones de interés (un órgano, un tejido, una lesión o un grupo de células) del resto de la imagen. Es la base de la medición cuantitativa: volúmenes, áreas, conteos y formas. "
        "Sus aplicaciones principales en salud incluyen la planificación de cirugía y radioterapia, el seguimiento de la evolución de una enfermedad, el conteo de células en patología y microscopía, y como paso previo de los sistemas de diagnóstico asistido por computadora (Pham et al., 2000). "
        "El método de Otsu (1979) es una de las técnicas más simples y usadas: calcula automáticamente el umbral de intensidad que mejor separa los píxeles en dos clases. Este reporte lo aplica a tres imágenes médicas en escala de grises y analiza qué resultados da y qué limitaciones tiene.")

    r.h2("2. Metodología")
    r.p("<strong>Imágenes.</strong> Tres imágenes con licencia libre (Wikimedia Commons): una radiografía de tórax posteroanterior (Stillwaterising, CC0), un corte axial de tomografía computarizada de cráneo normal (Häggström, CC0) y una fotografía de microscopía de fluorescencia de núcleos de células HeLa teñidos con DAPI (Arnatbio, CC BY-SA 4.0). "
        "Las tres se trabajaron en escala de grises (de la microscopía se usó el canal verde, donde están los núcleos).")
    r.p("<strong>Método.</strong> Se usó <code>threshold_otsu</code> de scikit-image (van der Walt et al., 2014). Otsu elige el umbral <em>t</em> que maximiza la varianza entre clases σ<sub>B</sub>²(t) = ω₀(t)·ω₁(t)·[μ₀(t) − μ₁(t)]², "
        "donde ω y μ son la proporción y la media de las dos clases separadas por <em>t</em>. Como medida de calidad se calcula la <strong>separabilidad η = σ<sub>B</sub>² / σ<sub>T</sub>²</strong>, entre 0 y 1, propuesta en el mismo artículo (Otsu, 1979). "
        "Además se midió la estabilidad del umbral cuando se añade ruido gaussiano (σ = 10, 25 y 40 niveles) y el efecto de un suavizado previo, usando el coeficiente de Dice contra la segmentación de la imagen sin ruido.")

    r.h2("3. Resultados")
    r.fig("figuras/fig1_resultados.png", "Para cada imagen (filas): original, histograma en escala logarítmica con el umbral de Otsu (rojo), imagen segmentada y un resultado de interés.", alto=190)
    r.tabla(["Imagen", "Umbral t", "Media clase oscura", "Media clase clara", "Peso oscura / clara (%)", "Eficacia η"],
            [[n, str(d["umbral"]), f"{d['media_clase_oscura']}", f"{d['media_clase_clara']}", f"{d['peso_clase_oscura_pct']} / {d['peso_clase_clara_pct']}", f"{d['eta']}"] for n, d in I.items()],
            "Umbral obtenido y descripción de las dos clases que separa.", num=(1, 2, 3, 5))
    r.h3("Explicación del umbral obtenido")
    r.ul([
        f"<strong>Radiografía de tórax (t = {RX['umbral']}, η = {RX['eta']}).</strong> El histograma no tiene dos picos separados sino una meseta ancha con dos «jorobas» (una hacia 70 y otra hacia 150): la de intensidades bajas corresponde al pulmón (aire, que atenúa poco los rayos X) y la de altas al mediastino, huesos y abdomen. Otsu coloca el umbral en el valle entre ambas ({RX['umbral']}), y reparte la imagen casi por la mitad ({RX['peso_clase_oscura_pct']} % frente a {RX['peso_clase_clara_pct']} %), con medias de {RX['media_clase_oscura']} y {RX['media_clase_clara']}.",
        f"<strong>TC de cráneo (t = {TC['umbral']}, η = {TC['eta']}).</strong> El histograma tiene un gran pico en 0 (el fondo negro fuera de la cabeza; nótese la escala logarítmica), una campana ancha centrada en 120 (el tejido blando del cerebro) y una cola de valores altos (el hueso del cráneo). Con dos clases, Otsu coloca el umbral entre el fondo y el tejido: la clase oscura tiene media {TC['media_clase_oscura']} (fondo, casi negro) y la clara media {TC['media_clase_clara']} (cabeza). Separa «cabeza» de «fondo», pero no distingue cerebro de hueso. Con <strong>multi-Otsu de tres clases</strong> los umbrales son {TC['multiotsu_umbrales'][0]} y {TC['multiotsu_umbrales'][1]}, y aparecen tres regiones: fondo y aire ({TC['multiotsu_pct_clases'][0]} % de la imagen), tejido blando ({TC['multiotsu_pct_clases'][1]} %) y hueso ({TC['multiotsu_pct_clases'][2]} %).",
        f"<strong>Microscopía de fluorescencia (t = {NU['umbral']}, η = {NU['eta']}).</strong> El histograma es muy asimétrico: el {NU['peso_clase_oscura_pct']} % de los píxeles es fondo casi negro (media {NU['media_clase_oscura']}) y solo el {NU['peso_clase_clara_pct']} % pertenece a los núcleos (media {NU['media_clase_clara']}). Aunque las clases son muy desiguales, están muy separadas en intensidad y Otsu funciona bien, con la mayor eficacia de las tres ({NU['eta']}).",
    ])
    r.h3("Estructuras segmentadas")
    r.ul([
        f"<strong>Núcleos celulares:</strong> quedaron aislados con contornos nítidos; tras una apertura morfológica de 3×3 se contaron {NU['nucleos']} objetos, con un área mediana de {NU['area_mediana_px']:.0f} px. Los núcleos que se tocan se cuentan como uno, por lo que el conteo es una aproximación.",
        "<strong>Cabeza en la TC:</strong> el cráneo y el cerebro se separan del fondo, y con multi-Otsu también se separa el hueso del tejido blando. Los ventrículos (más oscuros que el resto del cerebro) aparecen como huecos en la máscara de dos clases, porque su intensidad cae por debajo del umbral.",
        f"<strong>Pulmones en la radiografía:</strong> las zonas de aire pulmonar aparecen como dos grandes regiones oscuras, y el mediastino, la columna, las costillas y el abdomen quedan como clase clara. En el panel de la derecha, la máscara de zonas oscuras centrales ({RX['pulmones_pct_imagen']} % de la imagen) contiene los dos pulmones, pero <strong>también se «fuga» hacia las esquinas superiores</strong>, fuera del cuerpo, donde el aire tiene una intensidad parecida. Por eso ese porcentaje no equivale al área de los pulmones.",
    ])
    r.h3("Estabilidad frente al ruido")
    r.fig("figuras/fig2_ruido.png", "Coincidencia (Dice) entre la segmentación de una imagen con ruido y la de la imagen sin ruido, con y sin suavizado previo.")
    filas = []
    for n, d in E.items():
        for s in ("σ=10", "σ=25", "σ=40"):
            filas.append([n, s, f"{d[s]['sin suavizar']['umbral']}", f"{d[s]['sin suavizar']['dice_vs_limpia']:.3f}", f"{d[s]['gaussiano σ=1.5']['dice_vs_limpia']:.3f}", f"{d[s]['mediana 5×5']['dice_vs_limpia']:.3f}"])
    r.tabla(["Imagen", "Ruido", "Umbral con ruido", "Dice sin suavizar", "Dice con gaussiano σ=1.5", "Dice con mediana 5×5"], filas,
            "Estabilidad del resultado de Otsu ante ruido gaussiano (Dice frente a la segmentación sin ruido).", num=(2, 3, 4, 5))

    r.h2("4. Discusión")
    r.h3("¿Qué estructuras fueron segmentadas correctamente?")
    r.p(f"Los <strong>núcleos celulares</strong> (fondo oscuro, objetos brillantes y muy contrastados) y la <strong>cabeza frente al fondo</strong> en la TC, además de la separación aire/hueso/tejido blando con tres clases. Los pulmones son visibles como zonas oscuras en la radiografía, aunque sin delimitación limpia (ver la fuga a las esquinas). "
        "Es decir, funciona bien cuando las estructuras tienen intensidades claramente diferentes entre sí.")
    r.h3("¿Qué limitaciones se encontraron?")
    r.ul([
        "<strong>Solo dos clases (o las que se indiquen):</strong> con dos clases la TC no separa cerebro de hueso; con tres, sí, pero no distingue sustancia gris de blanca ni una lesión de intensidad parecida.",
        "<strong>No usa la información espacial:</strong> dos regiones distintas con intensidad parecida (los pulmones y el aire fuera del cuerpo en la radiografía) quedan en la misma clase.",
        "<strong>Los histogramas no siempre son bimodales</strong> (la radiografía tiene una meseta ancha); un valor de η alto no garantiza que los grupos correspondan a lo que se quiere segmentar.",
        "<strong>Sensibilidad al ruido:</strong> sin suavizar, el umbral se desplaza y la segmentación cambia, sobre todo con ruido fuerte (ver más abajo).",
        "<strong>Rótulos y artefactos:</strong> las anotaciones dentro de la imagen (la «L» de la radiografía, el marcador de lado) se segmentan como estructura; hay que enmascararlos.",
    ])
    d40 = {n: E[n]["σ=40"] for n in E}
    r.p(f"Sobre el ruido: con σ = 40 la coincidencia sin suavizar baja a {d40[NRX]['sin suavizar']['dice_vs_limpia']:.2f} (radiografía), {d40[NTC]['sin suavizar']['dice_vs_limpia']:.2f} (TC) y {d40[NNU]['sin suavizar']['dice_vs_limpia']:.2f} (núcleos); la peor es la de fluorescencia, con clases muy desiguales en las que unos pocos píxeles de fondo con ruido superan el umbral. "
        f"Un suavizado previo la recupera a {d40[NRX]['gaussiano σ=1.5']['dice_vs_limpia']:.2f}, {d40[NTC]['gaussiano σ=1.5']['dice_vs_limpia']:.2f} y {d40[NNU]['gaussiano σ=1.5']['dice_vs_limpia']:.2f} con un filtro gaussiano (σ = 1.5), y a valores parecidos con un filtro de mediana. Filtrar antes de umbralizar es un paso casi obligatorio con imágenes ruidosas.")
    r.h3("¿Cómo podría utilizarse esta técnica en un sistema médico real?")
    r.p("Como componente de un flujo de procesamiento: (1) <strong>preprocesar</strong> (suavizado o filtro de mediana); (2) <strong>umbralizar</strong> con Otsu o multi-Otsu; (3) <strong>limpiar</strong> con operaciones morfológicas y análisis de componentes conexos; y (4) <strong>medir</strong> (área, conteo, densidad). "
        "Es útil, por ejemplo, para contar núcleos o células en patología digital, para separar el hueso del tejido blando en una TC como paso previo de una reconstrucción, o para extraer una máscara inicial de los pulmones que un especialista corrige. "
        "En sistemas actuales, esta máscara suele usarse como <em>preprocesamiento</em> o para generar etiquetas iniciales de modelos de aprendizaje profundo, no como diagnóstico final.")
    r.h3("¿Qué ventajas ofrece la automatización del análisis de imágenes?")
    r.p("<strong>Velocidad</strong> (la segmentación de Otsu tarda milisegundos), <strong>reproducibilidad</strong> (el mismo algoritmo da el mismo resultado, sin variabilidad entre observadores), <strong>capacidad para procesar grandes volúmenes</strong> (cientos de campos de microscopía o miles de cortes de TC) y <strong>objetividad en la medición</strong> (áreas y conteos con criterios fijos). "
        "Su riesgo es la falsa confianza: un método automático puede fallar sin advertirlo (como la fuga a las esquinas en la radiografía), por lo que debe validarse y, en clínica, revisarse por una persona especialista.")

    r.h2("5. Conclusiones")
    r.p("Otsu es un método rápido, sin parámetros y muy eficaz cuando la imagen tiene dos poblaciones de intensidad claramente separadas, como los núcleos fluorescentes (η = 0.848). Con imágenes que tienen más de dos tejidos (TC) o intensidades solapadas en regiones distintas (radiografía), requiere multi-Otsu, información espacial u otros métodos. "
        "Un suavizado previo mejora mucho la estabilidad ante el ruido. Como limitaciones del trabajo, no se contó con una segmentación de referencia hecha por especialistas, por lo que los resultados de las tres imágenes se evaluaron de forma cualitativa y con conteos; la evaluación de la estabilidad sí se hizo de forma cuantitativa.")

    r.refs([
        "Häggström, M. (s. f.). <em>CT of a normal brain, axial 18</em> [Imagen; CC0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:CT_of_a_normal_brain,_axial_18.png",
        "Arnatbio. (s. f.). <em>HELA cells stained with DAPI and Phalloidin</em> [Imagen; CC BY-SA 4.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:HELA_cells_stained_with_DAPI_and_Phalloidin.tiff",
        "Liao, P.-S., Chen, T.-S., &amp; Chung, P.-C. (2001). A fast algorithm for multilevel thresholding. <em>Journal of Information Science and Engineering, 17</em>(5), 713–727.",
        "OpenAI. (2026). <em>Segmentación con método de Otsu</em> [Imagen generada con IA]. Material de la actividad, Visión computacional para imágenes y video. ChatGPT.",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. <em>IEEE Transactions on Systems, Man, and Cybernetics, 9</em>(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076",
        "Pham, D. L., Xu, C., &amp; Prince, J. L. (2000). Current methods in medical image segmentation. <em>Annual Review of Biomedical Engineering, 2</em>, 315–337. https://doi.org/10.1146/annurev.bioeng.2.1.315",
        "Stillwaterising. (s. f.). <em>Chest Xray PA 3-8-2010</em> [Imagen; CC0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png",
        "van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., &amp; Yu, T. (2014). scikit-image: Image processing in Python. <em>PeerJ, 2</em>, e453. https://doi.org/10.7717/peerj.453",
    ])
    pdf = r.guardar("Otsu_imagenologia_medica")
    n = zip_entrega(AQUI / "Otsu_imagenologia_medica.zip", AQUI, ["experimentos.py", "informe.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
