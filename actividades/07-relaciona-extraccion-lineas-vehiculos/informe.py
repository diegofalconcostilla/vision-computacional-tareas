"""Informe PDF: Convolución y extracción de líneas en visión artificial (vehículos autónomos)."""
import json
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
I75, I10 = R["Autopista I-75 (vehículo adelante)"], R["Autopista I-10 (salida)"]


def pct(d):
    return f"{100 * d['plausibles'] / d['segmentos']:.0f} %"


def fila(nombre, k):
    return [nombre, str(I75[k]["segmentos"]), f"{I75[k]['plausibles']} ({pct(I75[k])})", str(I10[k]["segmentos"]),
            f"{I10[k]['plausibles']} ({pct(I10[k])})"]


def main():
    (AQUI / "datos").mkdir(exist_ok=True)
    for n in ("autopista_i75.jpg", "autopista_i10.jpg"):
        shutil.copy(RAIZ / "datos_compartidos" / n, AQUI / "datos" / n)

    r = Reporte("Convolución y extracción de líneas en visión artificial",
                "De píxeles a decisiones: detección de carriles con Sobel, Prewitt, Laplaciano y Canny", carpeta=AQUI,
                entrega="PDF con imágenes y análisis + archivo comprimido con el código")
    r.portada()

    r.h2("1. Introducción")
    r.p("Un vehículo que se mantiene en su carril necesita saber, cuadro tras cuadro, dónde están las marcas viales. Esa información no viene dada: se obtiene de una imagen, y el primer paso para extraerla es "
        "encontrar <strong>bordes y líneas</strong>, es decir, lugares donde la intensidad cambia bruscamente. La herramienta matemática para hacerlo es la <strong>convolución</strong> de la imagen con un pequeño <em>kernel</em>. "
        "Este trabajo aplica cuatro operadores de detección de bordes a dos escenas reales de autopista, extrae líneas con la transformada de Hough, compara qué encuentra cada operador y "
        "relaciona el resultado con las redes neuronales convolucionales (CNN) que hoy hacen esta tarea en los vehículos autónomos.")

    r.h2("2. Parte 1. Observa y analiza")
    r.p("El gráfico resumen de la actividad describe el flujo <strong>imagen → kernel convolucional → extracción de características → interpretación visual</strong>. "
        "La convolución desliza el kernel por la imagen y, en cada posición, suma los productos entre los coeficientes y los píxeles vecinos. Según los coeficientes, el resultado es un suavizado o una medida de cambio de intensidad:")
    r.tabla(["Operador", "Kernel(es) 3×3 (filas separadas por “;”)", "Qué mide"],
            [["Sobel", "Gx = [−1 0 1; −2 0 2; −1 0 1], Gy = Gxᵀ", "Gradiente (primera derivada) con suavizado incorporado; magnitud √(Gx² + Gy²) y dirección"],
             ["Prewitt", "Gx = [−1 0 1; −1 0 1; −1 0 1], Gy = Gxᵀ", "Gradiente con promedio uniforme de las tres filas"],
             ["Laplaciano", "[0 1 0; 1 −4 1; 0 1 0]", "Segunda derivada; responde a cambios de gradiente; sin dirección (isótropo)"],
             ["Canny", "Gauss + Sobel + supresión de no-máximos + umbrales con histéresis", "Bordes delgados (1 píxel) y conectados"]],
            "Operadores empleados (Gonzalez & Woods, 2018; Canny, 1986).")
    r.h3("¿Por qué los bordes y las líneas son importantes en una imagen?")
    r.p("Porque concentran la información estructural: los contornos de los objetos, los límites entre superficies y las marcas pintadas casi siempre aparecen como cambios de intensidad. "
        "Una imagen de 1280×720 tiene casi un millón de píxeles, pero los bordes suelen ser una pequeña fracción: en las escenas analizadas, el mapa de Canny marca solo entre 2.5 % y 2.9 % de los píxeles de la calzada. "
        "Reducir la imagen a sus bordes conserva lo esencial y descarta lo demás, lo que facilita el reconocimiento y disminuye el cómputo.")
    r.h3("¿Cómo ayuda la convolución a detectar estructuras relevantes?")
    r.p("Un kernel de gradiente calcula, en cada píxel, una estimación de la derivada de la intensidad: es grande donde hay una transición y casi cero en las zonas uniformes. "
        "Una marca blanca sobre asfalto gris produce dos transiciones (gris→blanco y blanco→gris), y por eso el mapa de bordes muestra la marca como una franja o dos líneas paralelas. "
        "Como todo se reduce a sumas de productos, es una operación muy rápida y paralelizable, que se aplica al cuadro completo en milisegundos.")
    r.h3("¿Qué problemas podrían existir si un vehículo autónomo no detecta correctamente los carriles?")
    r.p("El vehículo perdería su referencia lateral: podría desviarse del carril sin advertirlo, no sabría si un cambio de carril es seguro o cuánto espacio tiene a cada lado, y sistemas de asistencia como la alerta de salida o el mantenimiento de carril "
        "dejarían de funcionar o darían falsas alarmas. Los errores no son iguales: una línea <em>no detectada</em> deja al sistema sin información, mientras que una línea <em>falsa</em> (una sombra, el borde de un guardarraíl o la parte trasera de otro vehículo) "
        "puede llevarlo a seguir una trayectoria equivocada. Por eso la extracción de líneas se acompaña de filtros de plausibilidad, como se hace en la parte práctica.")

    r.h2("3. Parte 2. Aplicación práctica")
    r.p("<strong>Imágenes.</strong> Dos fotografías de dashcam en autopistas de Florida (Rivera, 2025a, 2025b), con licencia CC BY 4.0: la <em>I-75</em>, con un vehículo adelante, línea blanca continua a la derecha, línea amarilla a la izquierda y marcas discontinuas; "
        "y la <em>I-10</em>, en una salida con marcas discontinuas y una línea blanca continua que se separa hacia la derecha. Las fotos se redujeron a 1280 px de ancho.")
    r.p("<strong>Procedimiento (igual para los cuatro operadores).</strong> (1) Conversión a gris y suavizado gaussiano 5×5 (σ = 1.2) para reducir el ruido. (2) Operador de bordes. "
        "(3) Binarización: para Sobel, Prewitt y Laplaciano se conserva el 8 % de píxeles con mayor respuesta dentro de la zona de calzada, para compararlos con el mismo criterio; Canny usa sus propios umbrales (50 y 150). "
        "(4) Todo se restringe a un <strong>trapecio sobre la calzada</strong> (región de interés), lo que excluye el cielo, la vegetación y la fecha y ubicación que imprime la propia cámara, que de otro modo producen cientos de líneas falsas. "
        "(5) Transformada de Hough probabilística para extraer segmentos de recta (Duda & Hart, 1972).")
    r.fig("figuras/figA_respuestas_autopista_i75.png", "Autopista I-75 (foto: M. Rivera, CC BY 4.0). Escena completa con la zona de análisis y respuesta de cada operador en la franja de la calzada.", alto=175)
    r.h3("Efecto producido por cada filtro")
    r.ul([
        "<strong>Sobel.</strong> Produce bordes gruesos y continuos. La línea blanca continua y las marcas discontinuas aparecen como trazos claros; también resaltan el contorno del vehículo, el guardarraíl y los postes. El suavizado que incorpora evita gran parte del ruido de la textura del asfalto.",
        "<strong>Prewitt.</strong> Da un resultado casi idéntico al de Sobel (en Hough, 11 segmentos plausibles en ambos en la I-75); al pesar igual las tres filas del kernel suaviza un poco menos, y la diferencia visual es mínima.",
        "<strong>Laplaciano.</strong> Marca los bordes con línea más fina (responde a la curvatura, no al gradiente) pero es el más sensible a la textura del pavimento y del pasto: su mapa es el más granuloso. En la I-10 fue el único que recuperó las marcas discontinuas del centro (figura 4).",
        "<strong>Canny.</strong> Genera bordes de un píxel de grosor, conectados y con mucho menos ruido (2.5 %–2.9 % de píxeles de borde frente a 8 % de los otros tres por construcción), y el mínimo de segmentos (10) al aplicar Hough. Es el mapa más limpio, pero en la I-10 conservó menos marcas de carril.",
    ])
    r.fig("figuras/figB_hough_autopista_i75.png", "I-75: segmentos de Hough sobre el mapa de bordes de cada operador. Verde: inclinación plausible de carril (entre 20° y 80° respecto a la horizontal); rojo: descartado. El vehículo de adelante genera segmentos horizontales descartados.")
    r.fig("figuras/figA_respuestas_autopista_i10.png", "Autopista I-10 (foto: M. Rivera, CC BY 4.0). Escena, zona de análisis y respuesta de cada operador.", alto=175)
    r.fig("figuras/figB_hough_autopista_i10.png", "I-10: segmentos de Hough para cada operador. Solo el Laplaciano recupera las marcas discontinuas del centro; los otros se quedan con la línea blanca continua de la derecha.")
    r.tabla(["Operador", "I-75: segmentos", "I-75: plausibles", "I-10: segmentos", "I-10: plausibles"],
            [fila("Sobel", "Sobel"), fila("Prewitt", "Prewitt"), fila("Laplaciano", "Laplaciano"), fila("Canny", "Canny")],
            "Segmentos de Hough en la zona de calzada; “plausible” = inclinación entre 20° y 80° respecto a la horizontal.", num=(1, 2, 3, 4))
    r.caja("<strong>Cómo leer estos resultados.</strong> Son dos imágenes y un criterio de plausibilidad muy simple, por lo que sirven para <em>ilustrar</em> el comportamiento de cada operador, no para declarar un ganador general. "
           "El criterio de 20°–80° descarta las líneas casi horizontales; por eso quedan fuera el borde amarillo izquierdo y el guardarraíl, que en esta cámara de gran angular se ven muy tendidos, y también algunas marcas discontinuas. "
           "Un sistema real usaría además la geometría de perspectiva (punto de fuga, vista cenital) y seguimiento entre cuadros. Lo que sí se observa con claridad es que los cuatro operadores encuentran la línea blanca continua, "
           "que Canny es el más limpio y que el Laplaciano es el más sensible al detalle fino, tanto el útil como el ruido.")

    r.h2("4. Parte 3. Relación con los vehículos autónomos")
    r.h3("¿Cómo ayudan los filtros convolucionales a detectar carriles o límites de carretera?")
    r.p("Los operadores de gradiente transforman la imagen en un mapa donde las marcas viales destacan como bordes; a partir de ahí, la <em>transformada de Hough</em> agrupa los píxeles de borde alineados en rectas y una región de interés elimina lo que no es calzada. "
        "En este trabajo esa cadena (suavizado → convolución → umbral → región de interés → Hough) recupera la línea blanca continua con los cuatro operadores, y las marcas discontinuas con alguno de ellos. "
        "Es una solución simple, rápida y explicable, pero depende de umbrales fijos, de que las marcas estén bien pintadas y de que la iluminación sea razonable.")
    r.h3("¿Por qué la extracción de líneas es importante en conducción autónoma?")
    r.p("Las marcas de carril son la referencia geométrica que permite <strong>localizarse lateralmente</strong> dentro de la vía, planificar la trayectoria y decidir si un cambio de carril es posible. "
        "A diferencia de un mapa, las líneas se ven en tiempo real y reflejan la realidad actual de la vía (obras, desvíos, desgaste). "
        "Además son una restricción de plausibilidad para otras tareas: un objeto que invade la línea es más relevante que uno fuera de la calzada.")
    r.h3("¿Cómo se relaciona la convolución con las Redes Neuronales Convolucionales (CNN)?")
    r.p("Una capa convolucional de una CNN hace exactamente la operación de este trabajo: convoluciona la imagen con varios kernels, suma un sesgo y aplica una no linealidad (habitualmente ReLU); después, un <em>pooling</em> reduce la resolución. "
        "La diferencia esencial es que en Sobel o Prewitt los coeficientes los fija una persona, mientras que en una CNN <strong>se aprenden</strong> a partir de datos mediante retropropagación (LeCun et al., 1998). "
        "Los kernels aprendidos en la primera capa de redes entrenadas en imágenes naturales resultan selectivos a frecuencia y orientación, como los detectores de bordes clásicos (Krizhevsky et al., 2012). "
        "La figura 5 reproduce el esquema con cuatro kernels orientados (0°, 45°, 90° y 135°): cada uno responde a bordes de una orientación distinta, y tras ReLU y max-pooling quedan mapas de características más pequeños que la capa siguiente combinará.")
    r.fig("figuras/figC_capa_cnn.png", "Una capa convolucional con un banco de cuatro kernels orientados (derivados del Sobel), ReLU y max-pooling 2×2 sobre la franja de la calzada de la I-75. Los kernels se giraron con interpolación, por eso algunos coeficientes no son enteros.")
    r.p("En detección de carriles, los sistemas modernos sustituyen la cadena manual por CNN que segmentan las marcas píxel a píxel, con mayor robustez ante sombras, desgaste y marcas discontinuas. "
        "Un ejemplo es LaneNet, que plantea la detección de carriles como segmentación por instancias con una red convolucional y evita las heurísticas de características hechas a mano (Neven et al., 2018). "
        "Aun así, los conceptos de esta práctica (gradiente, kernel, orientación, región de interés) siguen siendo la base para entender qué aprende esa red y cómo depurarla.")

    r.refs([
        "Bradski, G. (2000). The OpenCV library. <em>Dr. Dobb’s Journal of Software Tools, 25</em>(11), 120–123.",
        "Canny, J. (1986). A computational approach to edge detection. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8</em>(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851",
        "Duda, R. O., &amp; Hart, P. E. (1972). Use of the Hough transformation to detect lines and curves in pictures. <em>Communications of the ACM, 15</em>(1), 11–15. https://doi.org/10.1145/361237.361242",
        "Gonzalez, R. C., &amp; Woods, R. E. (2018). <em>Digital image processing</em> (4.ª ed.). Pearson.",
        "Krizhevsky, A., Sutskever, I., &amp; Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. <em>Advances in Neural Information Processing Systems, 25</em>, 1097–1105.",
        "LeCun, Y., Bottou, L., Bengio, Y., &amp; Haffner, P. (1998). Gradient-based learning applied to document recognition. <em>Proceedings of the IEEE, 86</em>(11), 2278–2324. https://doi.org/10.1109/5.726791",
        "Neven, D., De Brabandere, B., Georgoulis, S., Proesmans, M., &amp; Van Gool, L. (2018). Towards end-to-end lane detection: An instance segmentation approach. En <em>2018 IEEE Intelligent Vehicles Symposium (IV)</em> (pp. 286–291). IEEE. https://doi.org/10.1109/IVS.2018.8500547",
        "OpenAI. (2026). <em>Convolución: De píxeles a decisiones inteligentes</em> [Imagen generada con IA]. Material de la actividad, Visión computacional para imágenes y video. ChatGPT.",
        "Prewitt, J. M. S. (1970). Object enhancement and extraction. En B. S. Lipkin &amp; A. Rosenfeld (Eds.), <em>Picture processing and psychopictorics</em> (pp. 75–149). Academic Press.",
        "Rivera, M. (2025a). <em>Florida I10wb Exit 335A dashcam</em> [Fotografía; CC BY 4.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Florida_I10wb_Exit_335A_dashcam.jpg",
        "Rivera, M. (2025b). <em>Florida I75nb Exit 467 1 mile dashcam</em> [Fotografía; CC BY 4.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Florida_I75nb_Exit_467_1_mile_dashcam.jpg",
        "Sobel, I., &amp; Feldman, G. (1968). <em>A 3x3 isotropic gradient operator for image processing</em> [Presentación]. Stanford Artificial Intelligence Project (SAIL).",
    ])
    pdf = r.guardar("Extraccion_de_lineas_vehiculos_autonomos")
    n = zip_entrega(AQUI / "Extraccion_de_lineas_vehiculos_autonomos.zip", AQUI,
                    ["experimentos.py", "informe.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
