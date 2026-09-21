"""Genera el informe PDF de "Algoritmos de mejoramiento de imagen en Photoshop" a partir de resultados.json."""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
E = R["etapas"]
V = R["variantes"]
C = R["canny"]
o, f = E["0. Original"], E["4. Máscara de enfoque"]


def n(x, d=1):
    return f"{x:.{d}f}"


def mil(x):
    """Separador de miles con espacio fino: 8264 -> '8 264' (solo formatea números, no toca el texto)."""
    return f"{x:,.0f}".replace(",", " ")


def main():
    r = Reporte("Mejoramiento de imágenes con herramientas digitales",
                "Del ruido y la poca luz a una imagen apta para el análisis: técnicas equivalentes a Adobe Photoshop",
                carpeta=AQUI, entrega="PDF con imágenes, histogramas y reflexión + archivo comprimido con el código")
    r.portada()

    # ------------------------------------------------------------------ 1
    r.h2("1. Introducción y herramienta utilizada")
    r.p("El pre-procesamiento no solo hace que una fotografía se vea mejor para las personas: también determina qué tan bien puede analizarla un algoritmo. "
        "En este trabajo se toma una imagen real con <strong>iluminación deficiente y ruido</strong> (una calle de Hong Kong de noche, fotografiada con sensibilidad ISO alta) "
        "y se le aplican cuatro técnicas de mejoramiento. Después se comparan la imagen y los histogramas antes y después, y se comprueba con un detector de bordes si el resultado "
        "resulta más útil para un sistema de visión artificial.")
    r.caja("<strong>Sobre la herramienta.</strong> Adobe Photoshop es un programa de pago y el equipo no lo utilizó. La actividad permite &laquo;Photoshop o una herramienta equivalente&raquo;, "
           "así que cada operación se reprodujo con Python y OpenCV usando el mismo algoritmo que la herramienta de Photoshop correspondiente (tabla 1). "
           "El código se entrega en el archivo comprimido, de modo que todos los pasos pueden repetirse y verificarse.")
    r.tabla(["Herramienta de Photoshop", "Algoritmo empleado aquí", "Parámetros"],
            [["Filtro › Ruido › Reducir ruido", "Medias no locales (<em>Non-Local Means</em>, OpenCV)", "h = 9 (fuerza), ventana de búsqueda 21 px"],
             ["Imagen › Ajustes › Niveles (punto negro, punto blanco y tono medio)", "Estirado lineal por percentiles de la luminancia + corrección gamma", "0.5 % y 99.5 %; γ = 0.65"],
             ["Imagen › Ajustes › Ecualizar / Sombras y luces", "CLAHE (ecualización adaptativa con límite de contraste) sobre el canal L de LAB", "clipLimit 1.5, cuadrícula 8×8"],
             ["Filtro › Enfocar › Máscara de enfoque", "<em>Unsharp masking</em>: g = f + cantidad·(f − desenfoque gaussiano de f)", "cantidad 0.6, radio 1.2 px, umbral 4"]],
            "Equivalencias entre las herramientas de Photoshop (Adobe Inc., 2024) y los algoritmos usados.")

    # ------------------------------------------------------------------ 2
    r.h2("2. Parte 1. Observa y analiza")
    r.p("El gráfico resumen de la actividad plantea un flujo <strong>captura de imagen → pre-procesamiento → mejoramiento → análisis visual</strong>. "
        "Con los resultados de la parte práctica se responde a las tres preguntas de reflexión.")
    r.h3("¿Cómo afecta la iluminación a la calidad de una imagen?")
    r.p(f"La iluminación fija cuántos fotones llegan al sensor y, con ello, la relación señal-ruido. En la fotografía nocturna elegida el histograma se apila contra el negro: "
        f"el <strong>{n(o['negro_pct'])} % de los píxeles</strong> vale 8 o menos (de 255) y el brillo medio es de solo {n(o['media'])}. "
        f"La cámara compensó la poca luz subiendo la sensibilidad, y esa ganancia también amplifica el ruido del sensor: se estima un ruido de σ ≈ {n(o['sigma_ruido'], 2)} niveles. "
        f"La poca luz reduce el contraste útil y encarece el ruido, dos problemas distintos que hay que atender por separado.")
    r.h3("¿Por qué es importante mejorar una imagen antes de analizarla?")
    r.p("Los algoritmos de análisis suponen que la información que buscan (bordes, texturas, formas) es visible en los datos. Si las sombras están comprimidas cerca del negro, "
        "los bordes de los edificios no superan los umbrales de un detector; si hay ruido, aparecen bordes que no existen. En la sección 4 se cuantifica: "
        f"con los umbrales de Canny fijos, la imagen original solo registra el {n(C['Original']['bordes_pct'], 1)} % de píxeles de borde, porque gran parte de la escena está demasiado oscura, "
        f"mientras que una versión aclarada pero sin reducir el ruido llega al {n(C['Sin reducir ruido']['bordes_pct'], 1)} %, en gran parte bordes falsos.")
    r.h3("¿Qué diferencias existen entre una imagen original y una optimizada?")
    r.p(f"La imagen optimizada usa mejor el rango de intensidades (entropía de {o['entropia']} a {f['entropia']} bits, brillo medio de {n(o['media'])} a {n(f['media'])}), "
        f"tiene muchos menos píxeles hundidos en negro ({n(o['negro_pct'])} % a {n(f['negro_pct'])} %) y su relación señal-ruido mejora (de {n(o['snr'])} a {n(f['snr'])}). "
        "Lo que no cambia es la información capturada: el procesamiento la reorganiza y la hace legible, pero no inventa lo que el sensor no registró.")

    # ------------------------------------------------------------------ 3
    r.h2("3. Parte 2. Edición y análisis de la imagen", )
    r.p("<strong>Imagen seleccionada.</strong> &laquo;Stewart Road at night (high-ISO version)&raquo;, Wan Chai, Hong Kong (Wilfredor, 2013), fotografía con licencia CC0 tomada con una cámara Nikon D300. "
        "Es una escena con tres problemas reales a la vez: <em>poca luz</em> (zonas casi negras), <em>ruido de sensor</em> y <em>bajo contraste local</em> en las fachadas. "
        "Se redujo a 1400 px de ancho para trabajar. La imagen original y la procesada se guardan como <code>imagen_original.jpg</code> e <code>imagen_mejorada.jpg</code>.")
    r.fig("figuras/fig1_original_vs_mejorada.png", "Imagen original (izquierda) e imagen mejorada con las cuatro técnicas (derecha).")
    r.h3("Técnicas aplicadas, en orden")
    r.ol([
        "<strong>Reducir ruido</strong> (medias no locales). Promedia parches de la imagen que se parecen entre sí, incluso lejanos, de modo que el ruido, que es aleatorio, se cancela y las estructuras repetidas se conservan mejor que con un desenfoque simple.",
        "<strong>Niveles y tono medio.</strong> Se define el negro en el percentil 0.5 y el blanco en el 99.5 de la luminancia para aprovechar todo el rango, y una gamma de 0.65 aclara los tonos medios sin quemar las luces.",
        "<strong>Ecualización local (CLAHE).</strong> Recupera contraste dentro de cada región (fachadas, asfalto) con un límite de contraste bajo para no amplificar el ruido restante.",
        "<strong>Máscara de enfoque.</strong> Resalta los bordes sumando la diferencia entre la imagen y su versión desenfocada; el umbral evita realzar variaciones tan pequeñas que solo son ruido.",
    ])
    r.fig("figuras/fig2_pasos_recorte.png", "Recorte ampliado de una misma región después de cada técnica. El rótulo de neón y la fachada oscura muestran el efecto acumulado.")
    r.h3("Histogramas antes y después")
    r.fig("figuras/fig3_histogramas.png", "Histograma de la imagen (gris, luminancia; líneas de color, canales R, G y B) antes y después del procesamiento, con la misma escala vertical.")
    r.p(f"<strong>Antes:</strong> los tres canales forman un pico estrecho pegado al negro (entre 0 y 30 aproximadamente) con una cola muy larga y casi vacía hacia las luces; "
        f"el {n(o['negro_pct'])} % de los píxeles es negro casi puro. El pico del canal azul está algo desplazado hacia valores mayores (cerca de 8 frente a cerca de 3 en rojo y verde), lo que da una ligera dominante fría en las sombras. "
        f"<strong>Después:</strong> la masa se desplaza y se extiende hasta cerca de 120–150, el porcentaje de negros baja a {n(f['negro_pct'])} % y aparece una pequeña barra en 255 ({n(f['saturado_pct'], 2)} % de píxeles saturados, frente a {n(o['saturado_pct'], 2)} % antes), "
        "señal de que los neones y la farola quedaron algo más quemados. El aspecto dentado del histograma final es normal tras estirar un histograma: los niveles se reparten con huecos.")
    r.fig("figuras/fig4_histogramas_etapas.png", "Histograma de la luminancia y de los canales tras cada etapa.")
    filas = [[k, n(v["media"]), n(v["desviacion"]), n(v["entropia"], 2), n(v["negro_pct"]), n(v["sigma_ruido"], 2), n(v["snr"]), f"{v['nitidez_lap']:,.0f}".replace(",", " ")]
             for k, v in E.items()]
    r.tabla(["Etapa", "Media", "Desv. est.", "Entropía (bits)", "% negro (≤ 8)", "Ruido σ", "SNR (media/σ)", "Nitidez (var. Laplaciano)"], filas,
            "Métricas de la luminancia tras cada etapa. El ruido σ se estima con el método de los coeficientes wavelet (MAD); la nitidez incluye también la contribución del ruido.",
            num=range(1, 8))
    r.h3("Qué muestran las métricas etapa por etapa")
    r.ul([
        f"<strong>Reducir ruido:</strong> el ruido estimado baja de {n(E['0. Original']['sigma_ruido'], 2)} a {n(E['1. Reducir ruido']['sigma_ruido'], 2)} y la SNR se triplica ({n(E['0. Original']['snr'])} → {n(E['1. Reducir ruido']['snr'])}); "
        f"el brillo y el contraste casi no cambian. La nitidez medida baja ({E['0. Original']['nitidez_lap']:,.0f} → {E['1. Reducir ruido']['nitidez_lap']:,.0f}) porque parte de lo que se contaba como &laquo;detalle&raquo; era ruido.".replace(",", " "),
        f"<strong>Niveles + gamma:</strong> es la etapa que más mueve el histograma: el brillo medio sube de {n(E['1. Reducir ruido']['media'])} a {n(E['2. Niveles + gamma']['media'])} y los negros caen de {n(E['1. Reducir ruido']['negro_pct'])} % a {n(E['2. Niveles + gamma']['negro_pct'])} %. "
        f"Como efecto secundario el ruido casi se duplica ({n(E['1. Reducir ruido']['sigma_ruido'], 2)} → {n(E['2. Niveles + gamma']['sigma_ruido'], 2)}): aclarar amplifica todo, también el ruido.",
        f"<strong>CLAHE:</strong> añade contraste local (entropía {n(E['2. Niveles + gamma']['entropia'], 2)} → {n(E['3. Ecualización local (CLAHE)']['entropia'], 2)} bits) a costa de algo más de ruido ({n(E['3. Ecualización local (CLAHE)']['sigma_ruido'], 2)}).",
        f"<strong>Máscara de enfoque:</strong> duplica la nitidez medida ({E['3. Ecualización local (CLAHE)']['nitidez_lap']:,.0f} → {E['4. Máscara de enfoque']['nitidez_lap']:,.0f}) y sube otra vez el ruido a {n(E['4. Máscara de enfoque']['sigma_ruido'], 2)}; es la etapa más delicada.".replace(",", " "),
    ])

    # ------------------------------------------------------------------ 4
    r.h2("4. ¿Importa el orden? Efecto sobre un sistema de visión artificial")
    r.p("Se compararon tres versiones con exactamente los mismos parámetros y distinto orden del paso de reducción de ruido, y se pasó a cada una un detector de bordes de Canny "
        "con umbrales fijos (60 y 140), un paso típico de muchos sistemas de visión. Un mapa de bordes limpio conserva pocas líneas largas; uno contaminado por ruido se llena de fragmentos.")
    r.fig("figuras/fig5_orden_operaciones.png", "Arriba: la misma región con cada variante. Abajo: bordes de Canny con los mismos umbrales.")
    orden = ["Original", "Sin reducir ruido", "Reducir ruido después de los niveles", "Reducir ruido al inicio (elegida)"]
    r.tabla(["Variante", "Ruido σ", "SNR", "% píxeles de borde", "Componentes de borde", "Longitud mediana (px)"],
            [[k, n(V[k]["sigma_ruido"], 2), n(V[k]["snr"]), n(C[k]["bordes_pct"]), f"{C[k]['componentes']:,}".replace(",", " "), n(C[k]["longitud_mediana_px"], 0)] for k in orden],
            "Ruido y bordes de Canny según el orden de las operaciones.", num=range(1, 6))
    r.p(f"<strong>Resultado.</strong> Aclarar sin reducir el ruido es la peor opción: el ruido llega a σ = {n(V['Sin reducir ruido']['sigma_ruido'], 2)} (más de cuatro veces el original) y Canny devuelve "
        f"{mil(C['Sin reducir ruido']['componentes'])} fragmentos de borde, casi todos falsos." +
        f" Reducir el ruido <em>antes</em> de tocar niveles y contraste da σ = {n(V['Reducir ruido al inicio (elegida)']['sigma_ruido'], 2)}, una SNR de {n(V['Reducir ruido al inicio (elegida)']['snr'])} y {mil(C['Reducir ruido al inicio (elegida)']['componentes'])} componentes, " +
        f"frente a σ = {n(V['Reducir ruido después de los niveles']['sigma_ruido'], 2)} y {mil(C['Reducir ruido después de los niveles']['componentes'])} cuando se reduce después." +
        " La razón es que el ruido es más fácil de separar de la señal mientras la imagen aún es oscura y de bajo contraste, y que cada etapa posterior lo amplifica.")
    r.caja("<strong>Lectura prudente.</strong> La imagen original tiene <em>menos</em> bordes detectados (" + n(C['Original']['bordes_pct']) + " %) que las mejoradas (" + n(C['Reducir ruido al inicio (elegida)']['bordes_pct']) +
           " % con la cadena elegida) no porque sea más limpia, sino porque sus fachadas están demasiado oscuras para superar el umbral: la mejora <em>descubre</em> estructura real. "
           "Además, el porcentaje de componentes muy pequeños (menos de 6 px) no mejora con la cadena elegida (" + n(C['Reducir ruido al inicio (elegida)']['diminutos_pct']) + " % frente a " + n(C['Original']['diminutos_pct']) +
           " % de la original): el suavizado del ruido también fragmenta algunos bordes reales. No hay una cadena perfecta; hay compromisos.")

    # ------------------------------------------------------------------ 5
    r.h2("5. Parte 3. Reflexión y documentación")
    r.h3("¿Qué técnicas se aplicaron?")
    r.p("Reducción de ruido por medias no locales, ajuste de niveles con corrección de tono medio (gamma), ecualización adaptativa con límite de contraste (CLAHE) y máscara de enfoque, en ese orden.")
    r.h3("¿Cómo cambiaron los histogramas?")
    r.p(f"Pasaron de un pico estrecho junto al negro a una distribución mucho más ancha: el brillo medio casi se duplicó ({n(o['media'])} → {n(f['media'])}), la desviación estándar subió de {n(o['desviacion'])} a {n(f['desviacion'])} "
        f"y la entropía de {o['entropia']} a {f['entropia']} bits. A cambio aparece una pequeña acumulación de píxeles saturados en 255.")
    r.h3("¿Qué mejoras visuales se observan?")
    r.p("Se distinguen las fachadas y los edificios del fondo, el asfalto con sus marcas y los vehículos, que en la original quedaban muy oscuros; los rótulos de neón conservan el color y el grano de las zonas oscuras se reduce mucho. "
        "Como puntos débiles, algunas zonas de neón y la farola pierden algo de detalle por saturación, y quedan restos de grano fino en los cielos y las paredes más oscuras.")
    r.h3("¿Cómo podrían ayudar estas técnicas a un sistema de visión artificial?")
    r.p("Un detector de bordes, de esquinas o de objetos trabaja con diferencias de intensidad. Aclarar y ecualizar hace que esas diferencias superen los umbrales en las zonas oscuras, y reducir el ruido evita que el sistema "
        f"crea ver estructuras donde no las hay (aquí, de {mil(C['Sin reducir ruido']['componentes'])} a {mil(C['Reducir ruido al inicio (elegida)']['componentes'])} componentes de borde al reducir el ruido primero).")
    r.h3("¿Por qué el pre-procesamiento es importante antes de aplicar algoritmos de inteligencia artificial o visión computacional?")
    r.p("Porque el algoritmo solo ve números. Un modelo entrenado con imágenes bien expuestas rinde peor con imágenes oscuras o ruidosas, y no siempre existe la posibilidad de reentrenarlo. "
        "El pre-procesamiento acerca los datos nuevos a las condiciones para las que el modelo fue diseñado, hace que las medidas sean comparables entre imágenes y reduce las fuentes de error que no tienen relación con la tarea. "
        "Pero tiene un costo y un límite: cada paso puede amplificar el ruido o destruir información (como los neones saturados), por lo que conviene decidir el orden y los parámetros con métricas, como se hizo aquí, "
        "y no solo &laquo;a ojo&raquo;. En aplicaciones sensibles, como la imagen médica, además hay que verificar que el realce no altere el diagnóstico.")

    # ------------------------------------------------------------------ refs
    r.refs([
        "Adobe Inc. (2024). <em>Photoshop user guide</em>. https://helpx.adobe.com/photoshop/user-guide.html",
        "Bradski, G. (2000). The OpenCV library. <em>Dr. Dobb’s Journal of Software Tools, 25</em>(11), 120–123.",
        "Buades, A., Coll, B., &amp; Morel, J.-M. (2005). A non-local algorithm for image denoising. En <em>2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05)</em> (Vol. 2, pp. 60–65). IEEE. https://doi.org/10.1109/CVPR.2005.38",
        "Canny, J. (1986). A computational approach to edge detection. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8</em>(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851",
        "Donoho, D. L., &amp; Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. <em>Biometrika, 81</em>(3), 425–455. https://doi.org/10.1093/biomet/81.3.425",
        "Gonzalez, R. C., &amp; Woods, R. E. (2018). <em>Digital image processing</em> (4.ª ed.). Pearson.",
        "OpenCV. (2024). <em>Histogram equalization</em>. https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html",
        "van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., &amp; Yu, T. (2014). scikit-image: Image processing in Python. <em>PeerJ, 2</em>, e453. https://doi.org/10.7717/peerj.453",
        "Wilfredor. (2013). <em>Hong Kong in night (noise version)</em> [Fotografía; CC0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Hong_Kong_in_night_(noise_version).jpg",
        "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. En P. S. Heckbert (Ed.), <em>Graphics gems IV</em> (pp. 474–485). Academic Press.",
    ])
    pdf = r.guardar("Photoshop_Mejoramiento_de_imagenes")
    n_zip = zip_entrega(AQUI / "Photoshop_Mejoramiento_de_imagenes.zip", AQUI,
                        ["experimentos.py", "informe.py", "figuras", "resultados.json", "imagen_original.jpg", "imagen_mejorada.jpg", pdf.name],
                        excluir=("__pycache__",))
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n_zip, "archivos")


if __name__ == "__main__":
    main()
