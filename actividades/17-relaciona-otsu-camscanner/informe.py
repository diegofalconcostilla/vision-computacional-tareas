"""Informe PDF: ¿Cómo funciona y se aplica la segmentación en CamScanner?"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
RE, DET, BIN, PIP = R["reales"], R["deteccion"], R["binarizacion"], R["pipeline_completo_f1"]
VI, BA, AL = (RE["Recibo arrugado en mesa oscura (Vienna)"], RE["Recibo sobre granito moteado (Bankia)"], RE["Recibo inclinado y recortado (Alves)"])
FON = {"oscuro": "fondo oscuro liso", "granito": "fondo claro moteado (granito)", "madera": "fondo de brillo medio con vetas"}


def main():
    r = Reporte("Segmentación clásica en aplicaciones de escaneo móvil",
                "Cómo un sistema tipo CamScanner detecta, endereza y limpia un documento fotografiado", carpeta=AQUI,
                entrega="PDF con capturas de cada etapa y reflexión + archivo comprimido con el código")
    r.portada()

    r.h2("1. Introducción y aclaración sobre las fotografías")
    r.p("Una aplicación de escaneo móvil convierte la foto de una hoja, tomada con perspectiva, con sombras y sobre una mesa cualquiera, en una imagen frontal, limpia y legible. Esto se logra con una cadena de técnicas clásicas de procesamiento de imágenes: "
        "<strong>(1)</strong> detección de bordes o umbralización para encontrar el documento, <strong>(2)</strong> segmentación de la región principal con análisis de contornos, <strong>(3)</strong> corrección de perspectiva a partir de las cuatro esquinas y "
        "<strong>(4)</strong> mejora del contenido con operaciones morfológicas y binarización adaptativa. Este trabajo implementa la cadena con OpenCV y evalúa cada etapa.")
    r.caja("<strong>Sobre las fotografías.</strong> La actividad pide fotografiar un documento con el teléfono. Como no se contaba con esas fotos, se usaron <strong>tres fotografías reales de recibos</strong> con licencia libre, que reproducen las dificultades del caso: "
           "un recibo arrugado sobre una mesa oscura (Grandmaster Huon, s. f.; dominio público), uno sobre un piso de granito moteado (Donald Trung, 2019a; CC BY-SA 4.0) y uno inclinado y recortado por el encuadre (Donald Trung, 2019b; CC BY-SA 4.0). "
           "Como en fotos reales no se conoce la solución correcta, además se generaron <strong>escenas sintéticas</strong> con esquinas y texto conocidos (mismo documento sobre tres fondos y con tres niveles de sombra) para medir con exactitud cada etapa. Los resultados de las fotos reales se interpretan observando las figuras; los de las sintéticas, con métricas.", aviso=True)

    r.h2("2. Análisis visual de la imagen original")
    r.p("En las tres fotografías se identifican los elementos que el enunciado pide observar:")
    r.tabla(["Elemento", "Recibo arrugado (Viena)", "Recibo sobre granito (Bankia)", "Recibo inclinado (Alves)"],
            [["Bordes del documento", "Contorno irregular por las arrugas; el borde superior es difuso.", "Bien definido; papel claro sobre fondo oscuro.", "Los bordes izquierdo y derecho son claros; el papel continúa fuera del encuadre por abajo (no hay borde inferior)."],
             ["Fondo", "Mesa negra casi uniforme.", "Granito con motas claras y oscuras, que generan muchos bordes falsos.", "Mesa negra, uniforme."],
             ["Sombras", "Zona superior más oscura y sombra de los pliegues.", "Casi uniforme.", "Degradado fuerte: la parte inferior queda más iluminada que la superior."],
             ["Ruido y reflejos", "Pliegues que actúan como reflejos y sombras sobre el texto; grano por poca luz.", "Bajo.", "Sobreexposición local que reduce el contraste del texto en la parte inferior."]],
            "Análisis visual de las fotografías originales.")

    r.h2("3. Aplicación de las técnicas (fotos reales)")
    r.p("<strong>Procedimiento.</strong> Se redujo la imagen a 1000 px. <strong>Rama Otsu:</strong> suavizado gaussiano 7×7, umbral de Otsu (el papel es más claro que el fondo), cierre 25×25 (para rellenar los huecos que deja el texto oscuro), apertura 15×15 (para cortar puentes con el fondo) y conservación de la región conectada más grande. "
        "<strong>Rama Canny:</strong> suavizado, detector de Canny con umbrales derivados de la mediana de la imagen (0.66× y 1.33×), dilatación y cierre 21×21, y contorno mayor. "
        "En ambas: casco convexo del contorno y aproximación poligonal a cuatro vértices (Douglas & Peucker, 1973), ordenamiento de las esquinas, homografía a un rectángulo (Hartley & Zisserman, 2004) y binarización.")
    r.fig("figuras/fig1_recibos_reales.png", "Etapas en cada recibo real (de izquierda a derecha): original, bordes de Canny con morfología, máscara de Otsu con morfología, esquinas detectadas por Otsu (verde) y Canny (rojo), documento rectificado en gris y binarización adaptativa.", alto=200)
    r.tabla(["Recibo", "Umbral Otsu", "IoU entre máscaras Otsu y Canny", "Área del documento (Otsu / Canny)", "Esquinas de Otsu en el borde de la imagen", "% de píxeles «texto»: adaptativa / Otsu global"],
            [[n[:n.index("(")].strip(), str(d["umbral_otsu"]), f"{d['iou_mascaras_otsu_canny']:.2f}", f"{d['otsu']['area_doc_pct']} % / {d['canny']['area_doc_pct']} %", str(d["otsu"]["esquinas_en_el_borde_de_la_imagen"]), f"{d['texto_pct_adaptativa']} % / {d['texto_pct_otsu_global']} %"] for n, d in RE.items()],
            "Resultados numéricos en las fotos reales. El área se expresa como porcentaje de la imagen.", num=(1, 2, 3, 4, 5))
    r.h3("Qué se observa en cada caso")
    r.ul([
        f"<strong>Recibo arrugado (Viena).</strong> El método de <strong>Otsu encuentra bien el documento</strong> (esquinas verdes), mientras que <strong>Canny se equivoca</strong>: las arrugas y los cambios de brillo generan muchos bordes internos que se unen con los del borde real, y el cuadrilátero rojo se extiende más allá del papel hacia una zona de sombra (IoU de máscaras de {VI['iou_mascaras_otsu_canny']:.2f}). "
        f"En la binarización, el <strong>umbral global de Otsu falla</strong>: clasifica como texto el {VI['texto_pct_otsu_global']} % del recibo (probablemente toda la zona más oscura del papel, por la sombra) frente al {VI['texto_pct_adaptativa']} % de la adaptativa, que sí separa las líneas de texto, aunque con pérdidas donde las arrugas producen sombras.",
        f"<strong>Recibo sobre granito (Bankia).</strong> Los dos métodos coinciden (IoU {BA['iou_mascaras_otsu_canny']:.2f}) y el enderezado es correcto. Aunque el fondo es moteado, el papel es claramente más claro, y el umbral de Otsu y la morfología eliminan las motas; en Canny, en cambio, las motas generan cientos de bordes pequeños, que la dilatación y el cierre absorben (segunda columna). "
        f"El texto ocupa un porcentaje casi igual con ambos binarizadores ({BA['texto_pct_adaptativa']} % y {BA['texto_pct_otsu_global']} %) porque la iluminación es uniforme.",
        f"<strong>Recibo inclinado y recortado (Alves).</strong> Es el caso donde la cadena falla, y por una razón que no es de umbral: <strong>el papel sale del encuadre por abajo</strong>. La máscara de Otsu incluye correctamente los tres lados visibles, pero {AL['otsu']['esquinas_en_el_borde_de_la_imagen']} de sus 4 esquinas están en el borde de la imagen, es decir, no son esquinas del papel; "
        f"el rectificado resultante tiene la perspectiva bien corregida en lo que se ve, pero la proporción del documento es solo una estimación a partir de la parte visible. Canny detecta el <em>marco del logotipo</em> (un cuadrilátero interno de {AL['canny']['area_doc_pct']} % del área), porque es el borde más fuerte y completo (IoU de máscaras {AL['iou_mascaras_otsu_canny']:.2f}). "
        f"La iluminación fuerte de la parte inferior beneficia a la adaptativa ({AL['texto_pct_adaptativa']} % de texto frente a {AL['texto_pct_otsu_global']} % con Otsu global).",
    ])

    r.h2("4. Evaluación controlada de cada etapa (escenas sintéticas)")
    r.p("Para responder con medidas se generó un documento de texto propio y se fotografió virtualmente 72 veces: <strong>3 fondos</strong> (oscuro liso, granito claro moteado y fondo de brillo medio con vetas), <strong>3 niveles de sombra</strong> (0, 0.3 y 0.6: un degradado más una mancha oscura sobre la hoja), perspectiva y giro aleatorios, ruido y desenfoque leve. "
        "Como las esquinas y el texto son conocidos, se mide el error de las esquinas detectadas y la calidad del texto binarizado (F1 con tolerancia de 1 px, apropiado para trazos finos).")
    r.fig("figuras/fig2_escenas.png", "Escenas sintéticas: esquinas verdaderas (amarillo), detectadas con Otsu (verde) y con Canny (rojo), para cada fondo y nivel de sombra.", alto=190)
    filas = []
    for t in ("oscuro", "granito", "madera"):
        for s in (0.0, 0.3, 0.6):
            d = DET[f"{t}|{s}"]
            filas.append([FON[t], str(s), f"{d['otsu']['error_medio_px']} px ({d['otsu']['exito_pct']:.0f} %)", f"{d['canny']['error_medio_px']} px ({d['canny']['exito_pct']:.0f} %)"])
    r.tabla(["Fondo", "Sombra", "Otsu: error de esquinas (éxito)", "Canny: error de esquinas (éxito)"], filas,
            "Detección de las esquinas del documento (8 escenas por fila; «éxito» = error medio menor de 20 px en una imagen de 1200×1600).", num=(1,))
    r.h3("¿Qué técnica permitió separar mejor el documento?")
    r.p(f"<strong>Depende del fondo y de la iluminación; ninguna es mejor en todos los casos.</strong> Con fondos lisos u ordenados (oscuro y madera), <strong>Canny es más preciso</strong> ({DET['oscuro|0.0']['canny']['error_medio_px']} px de error frente a {DET['oscuro|0.0']['otsu']['error_medio_px']} px de Otsu; el error de Otsu se debe probablemente a que el cierre y la apertura de 25 y 15 px redondean un poco las esquinas), y ambos tienen éxito en todas las escenas hasta una sombra de 0.3. "
        f"Con el fondo claro moteado, <strong>Otsu es más robusto</strong> sin sombra ({DET['granito|0.0']['otsu']['exito_pct']:.0f} % de éxito frente a {DET['granito|0.0']['canny']['exito_pct']:.0f} % de Canny, que se confunde con las motas), aunque ambos degradan con la sombra hasta {DET['granito|0.6']['otsu']['exito_pct']:.0f} % y {DET['granito|0.6']['canny']['exito_pct']:.0f} % de éxito con sombra de 0.6: "
        "el papel a la sombra queda tan oscuro como el granito, y la hipótesis de Otsu («el papel es más claro que el fondo») deja de cumplirse. Un sistema real combina ambos métodos y otros criterios (por ejemplo, verificar que el cuadrilátero tenga proporciones razonables y ángulos cercanos a 90°).")

    r.h3("¿Cómo afecta la iluminación al resultado?")
    r.fig("figuras/fig3_binarizacion.png", "Binarización del documento rectificado (fila superior: sin sombra; inferior: sombra de 0.6) con umbral fijo, Otsu global, umbral adaptativo gaussiano y Sauvola.", alto=110)
    filas = [[m] + [f"{BIN[m][str(s)]['f1']:.3f}" for s in (0.0, 0.3, 0.6)] + [f"{BIN[m]['0.6']['falsos_positivos_fondo_pct']} %"] for m in ("Umbral fijo 128", "Otsu global", "Adaptativo gaussiano", "Sauvola", "Adaptativo + apertura 2×2")]
    r.tabla(["Método de binarización", "F1 sin sombra", "F1 sombra 0.3", "F1 sombra 0.6", "Fondo marcado como texto (sombra 0.6)"], filas,
            "Calidad del texto binarizado con las esquinas verdaderas (promedio de 12 escenas por columna).", num=(1, 2, 3, 4))
    r.p(f"<strong>Sin sombra, cualquier método funciona</strong> (F1 de {BIN['Umbral fijo 128']['0.0']['f1']:.2f} a {BIN['Sauvola']['0.0']['f1']:.2f}). <strong>Con sombra, los métodos globales se derrumban</strong>: el umbral de Otsu pasa de {BIN['Otsu global']['0.0']['f1']:.2f} a {BIN['Otsu global']['0.6']['f1']:.2f} y marca como texto el {BIN['Otsu global']['0.6']['falsos_positivos_fondo_pct']} % del fondo; el umbral fijo aguanta con sombra moderada pero cae a {BIN['Umbral fijo 128']['0.6']['f1']:.2f} con sombra fuerte. "
        f"Los métodos <strong>locales</strong> (adaptativo y Sauvola) se mantienen en {BIN['Adaptativo gaussiano']['0.6']['f1']:.2f} porque calculan el umbral en cada vecindad y, por tanto, se adaptan al brillo local. Esta es la razón por la que las aplicaciones de escaneo usan binarización adaptativa y no un umbral global (Sauvola & Pietikäinen, 2000; Bradley & Roth, 2007).")
    r.h3("¿Qué operaciones ayudaron a eliminar ruido?")
    r.ul([
        "<strong>Cierre morfológico grande (25×25)</strong> sobre la máscara de Otsu: rellena los huecos del texto y de los pliegues para que el documento sea una sola región.",
        "<strong>Apertura grande (15×15)</strong> sobre esa máscara: elimina puentes finos con el fondo y motas.",
        "<strong>Selección de la región conectada de mayor área</strong>: descarta cualquier objeto adicional (una mancha, otro papel).",
        "<strong>Dilatación y cierre sobre los bordes de Canny</strong>: unen los bordes discontinuos en un contorno cerrado y absorben los bordes pequeños del fondo.",
        f"<strong>Suavizado gaussiano</strong> antes de umbralizar. <strong>Atención:</strong> una <em>apertura</em> de 2×2 <em>sobre el texto ya binarizado</em> empeoró el resultado (F1 de {BIN['Adaptativo gaussiano']['0.0']['f1']:.2f} a {BIN['Adaptativo + apertura 2×2']['0.0']['f1']:.2f}), porque los trazos de las letras miden apenas 2 o 3 px y la erosión los destruye. La morfología sirve para limpiar la máscara del <em>documento</em>, no necesariamente el <em>texto</em>; en él conviene dejar el ruido y confiar en la binarización adaptativa.",
    ])

    r.h3("Cadena completa con detección automática")
    r.tabla(["Sombra", "Detección con Otsu", "Detección con Canny", "Mejor de los dos (según la verdad)"],
            [[s, f"{PIP[s]['otsu']:.3f}", f"{PIP[s]['canny']:.3f}", f"{PIP[s]['mejor_de_ambos']:.3f}"] for s in ("0.0", "0.3", "0.6")],
            "F1 del texto tras detectar las esquinas automáticamente, enderezar y binarizar con umbral adaptativo (promedio de 12 escenas, incluyendo el fondo granito, que es el más difícil; se alinearon los resultados con una traslación global y se usó una tolerancia de 2 px). "
            "La última columna es un límite superior: elige el mejor método por escena usando la verdad, algo que un sistema real no puede hacer.", num=(0, 1, 2, 3))
    r.p(f"Con detección automática la calidad baja respecto de la obtenida con esquinas verdaderas (≈ 0.98): en total, entre {min(PIP['0.0']['otsu'], PIP['0.0']['canny']):.2f} y {max(PIP['0.0']['otsu'], PIP['0.0']['canny']):.2f} sin sombra, y hasta {min(PIP['0.6']['otsu'], PIP['0.6']['canny']):.2f}–{max(PIP['0.6']['otsu'], PIP['0.6']['canny']):.2f} con sombra fuerte. "
        "La pérdida se origina en errores de las esquinas (que desalinean el texto fino y, en el fondo granito, fallan por completo). Es decir, <strong>la detección del documento es la etapa más frágil</strong> de la cadena.")

    r.h2("5. Reflexión")
    r.h3("¿Cómo funciona la segmentación en aplicaciones de escaneo móvil?")
    r.p("Primero <em>separa el documento del fondo</em> (con bordes, con umbral o con ambos), después <em>localiza sus cuatro esquinas</em> a partir del contorno, <em>endereza</em> la imagen con una transformación proyectiva que lleva ese cuadrilátero a un rectángulo, y por último <em>mejora el contenido</em>: corrige la iluminación, aumenta el contraste y binariza de forma adaptativa. "
        "Cada etapa usa la salida de la anterior, por lo que un error al principio (esquinas mal ubicadas) arruina el resultado final, como se vio en el recibo recortado y en el fondo granito con sombra.")
    r.h3("¿Qué técnicas clásicas se identificaron?")
    r.ul(["<strong>Detección de bordes:</strong> Canny (Canny, 1986), con umbrales derivados de la mediana.",
          "<strong>Umbralización:</strong> Otsu (Otsu, 1979) para separar documento y fondo; umbral adaptativo gaussiano y Sauvola para el texto.",
          "<strong>Operaciones morfológicas:</strong> cierre, apertura y dilatación para consolidar la máscara.",
          "<strong>Regiones conectadas y contornos:</strong> seguimiento de bordes (Suzuki & Abe, 1985), región de mayor área, casco convexo y aproximación poligonal (Douglas & Peucker, 1973).",
          "<strong>Geometría proyectiva:</strong> homografía a partir de cuatro correspondencias (Hartley & Zisserman, 2004) y remuestreo bicúbico."])
    r.h3("¿Cuáles fueron las principales dificultades?")
    r.ul(["<strong>Fondos con textura o de brillo parecido al papel:</strong> el granito rompió Canny (bordes falsos) y, con sombra, también a Otsu.",
          "<strong>Sombras e iluminación desigual:</strong> hacen inútil un umbral global; la solución es local.",
          "<strong>Documentos arrugados:</strong> los pliegues generan bordes y sombras internas que confunden a Canny y ensucian la binarización.",
          "<strong>Documento recortado por el encuadre:</strong> sin las cuatro esquinas visibles, la geometría queda indeterminada; ningún método de umbral lo resuelve, y la aplicación debe pedir al usuario que vuelva a encuadrar.",
          "<strong>Elegir parámetros:</strong> tamaños de elementos morfológicos, ventana del umbral adaptativo y umbrales de Canny dependen de la resolución y del contenido; una apertura mal elegida borra el texto."])
    r.p("Los sistemas actuales combinan estas técnicas con redes neuronales que detectan el documento y sus esquinas, y con OCR para extraer el texto (Smith, 2007); aun así, entender la cadena clásica permite depurar por qué un escaneo sale mal. "
        "<strong>Limitaciones del trabajo:</strong> tres fotos reales evaluadas de forma cualitativa y una evaluación cuantitativa sobre un solo documento sintético con un tipo de texto.")

    r.refs([
        "Bradley, D., &amp; Roth, G. (2007). Adaptive thresholding using the integral image. <em>Journal of Graphics Tools, 12</em>(2), 13–21. https://doi.org/10.1080/2151237X.2007.10129236",
        "Canny, J. (1986). A computational approach to edge detection. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8</em>(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851",
        "Donald Trung. (2019a). <em>Bankia receipt, Groningen (2019) 01</em> [Fotografía; CC BY-SA 4.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Bankia_receipt,_Groningen_(2019)_01.jpg",
        "Donald Trung. (2019b). <em>Alves receipt (2019) 04</em> [Fotografía; CC BY-SA 4.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Alves_receipt_(2019)_04.jpg",
        "Douglas, D. H., &amp; Peucker, T. K. (1973). Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. <em>Cartographica, 10</em>(2), 112–122. https://doi.org/10.3138/FM57-6770-U75U-7727",
        "Gonzalez, R. C., &amp; Woods, R. E. (2018). <em>Digital image processing</em> (4.ª ed.). Pearson.",
        "Grandmaster Huon. (s. f.). <em>Grocery Store Receipt in Vienna</em> [Fotografía; dominio público]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Grocery_Store_Receipt_in_Vienna.jpg",
        "Hartley, R., &amp; Zisserman, A. (2004). <em>Multiple view geometry in computer vision</em> (2.ª ed.). Cambridge University Press.",
        "OpenAI. (2026). <em>Actividad: Segmentación clásica en camscanner</em> [Imagen generada con IA]. Material de la actividad, Visión computacional para imágenes y video. ChatGPT.",
        "OpenCV. (s. f.). <em>Image segmentation with OpenCV</em>. OpenCV Documentation. https://docs.opencv.org/",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. <em>IEEE Transactions on Systems, Man, and Cybernetics, 9</em>(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076",
        "Sauvola, J., &amp; Pietikäinen, M. (2000). Adaptive document image binarization. <em>Pattern Recognition, 33</em>(2), 225–236. https://doi.org/10.1016/S0031-3203(99)00055-2",
        "Smith, R. (2007). An overview of the Tesseract OCR engine. En <em>Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)</em> (pp. 629–633). IEEE. https://doi.org/10.1109/ICDAR.2007.4376991",
        "Suzuki, S., &amp; Abe, K. (1985). Topological structural analysis of digitized binary images by border following. <em>Computer Vision, Graphics, and Image Processing, 30</em>(1), 32–46. https://doi.org/10.1016/0734-189X(85)90016-7",
    ])
    pdf = r.guardar("CamScanner_segmentacion_documentos")
    n = zip_entrega(AQUI / "CamScanner_segmentacion_documentos.zip", AQUI, ["experimentos.py", "informe.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
