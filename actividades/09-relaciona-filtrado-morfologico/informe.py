"""Informe PDF: Operaciones morfológicas para detección de células en imágenes biomédicas."""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
F, E, SI = R["frotis"], R["errores_sinteticos"], R["segunda_imagen"]
VS = R["validacion_sintetica"]
cur = F["curva_erosion"]


def ej(clave, s):
    return E[clave][str(s)]["error_abs_medio_pct"]


def main():
    r = Reporte("Operaciones morfológicas para detectar células en imágenes biomédicas",
                "Separación y conteo de glóbulos rojos en un frotis de sangre", carpeta=AQUI,
                entrega="PDF con imágenes procesadas y análisis + archivo comprimido con el código")
    r.portada()

    r.h2("1. Introducción")
    r.p("Contar y medir células es una de las tareas más frecuentes del análisis biomédico: el conteo de glóbulos en un hemograma, la carga de bacterias en un cultivo, la proporción de células en una biopsia. "
        "Automatizarla requiere <strong>segmentar</strong> la imagen, y en la práctica la binarización sola casi nunca basta: las células aparecen unidas, con huecos, junto a basura y con iluminación desigual. "
        "Las <strong>operaciones morfológicas</strong> (erosión, dilatación, apertura, cierre, esqueletización) corrigen esos defectos actuando sobre la <em>forma</em> de las regiones binarias (Serra, 1982; Gonzalez & Woods, 2018). "
        "Este trabajo las aplica a un frotis de sangre real, mide qué hace cada una y valida el procedimiento de conteo con imágenes sintéticas en las que se conoce el número exacto de células.")

    r.h2("2. Parte 1. Investigación inicial")
    r.h3("¿Qué es una imagen de microscopía digital?")
    r.p("Es la imagen que se obtiene al acoplar una cámara digital (sensor CCD o CMOS) a un microscopio: la luz que atraviesa o emite la muestra se proyecta sobre el sensor, que la muestrea en píxeles y la cuantiza a un número de niveles (Murphy & Davidson, 2012). "
        "Según la técnica puede ser de <em>campo claro</em> (muestras teñidas sobre fondo claro, como este frotis), de <em>fluorescencia</em> (estructuras brillantes sobre fondo negro) o de contraste de fase, entre otras. Cada tipo produce imágenes con propiedades distintas, y por eso el preprocesamiento debe adaptarse.")
    r.h3("¿Por qué es importante segmentar correctamente las células o estructuras biológicas?")
    r.p("Porque toda medida posterior depende de la segmentación: el conteo, el área, la forma o la intensidad de cada célula. Un error se propaga: dos células unidas que se cuentan como una subestiman el conteo (y, por ejemplo, la concentración de glóbulos), una célula partida en dos lo sobrestima, y una plaqueta contada como célula lo sobrestima. "
        "En diagnóstico, cambios en el número o en la forma de las células son indicadores clínicos, por lo que una segmentación deficiente puede llevar a conclusiones erróneas (Meijering, 2012).")
    r.h3("¿Qué problemas suelen aparecer en este tipo de imágenes?")
    r.ul(["<strong>Células conectadas o superpuestas</strong>, sobre todo en zonas gruesas del frotis.",
          "<strong>Huecos internos</strong>: el glóbulo rojo tiene una zona central más pálida que la binarización deja como un agujero.",
          "<strong>Basura y células de otro tipo</strong>: plaquetas, restos de tinción, leucocitos, de tamaño y color distintos.",
          "<strong>Ruido</strong> y <strong>desenfoque</strong> del sensor y de la óptica.",
          "<strong>Iluminación desigual</strong> y variaciones de la tinción entre zonas y entre laminillas."])
    r.h3("Imagen elegida")
    r.p(f"«Normal Adult Blood Smear» (Chambers, s. f.; CC BY-SA 3.0), un frotis de sangre de un adulto ({F['tam'][0]}×{F['tam'][1]} px) con cientos de glóbulos rojos, algunos en contacto, tres leucocitos y plaquetas dispersas. Como segunda muestra, para ver el efecto de la iluminación desigual, se usó un frotis con tinción Wright (Chaurasiya, s. f.; CC BY-SA 4.0).")

    r.h2("3. Parte 2. Aplicación de operaciones morfológicas")
    r.p("<strong>Herramienta:</strong> Python con OpenCV y scikit-image. <strong>Binarización:</strong> canal verde (donde el contraste entre las células rosadas y el fondo es mayor), suavizado gaussiano leve y umbral de Otsu (t = "
        f"{F['umbral_otsu']}); las células son los píxeles más oscuros. <strong>Área típica</strong> de una célula: {F['area_tipica_px']:.0f} px (radio de unos {F['radio_tipico_px']:.0f} px), medida como la mediana de los componentes aislados y redondos.")
    r.fig("figuras/fig1_imagen_binarizacion.png", "Frotis original, histograma del canal verde con el umbral de Otsu y binarización resultante.", alto=75)
    r.fig("figuras/fig2_etapas.png", "Etapas del procesamiento sobre un recorte: (a) binarización, (b) apertura, (c) relleno de huecos, (d) cierre, (e) erosión, (f) esqueleto y (g) separación por transformada de distancia y watershed.", alto=150)
    r.tabla(["Operación", "Qué se aplicó", "Qué cambió"],
            [["<strong>Apertura</strong> 5×5 elíptica (erosión + dilatación)", "Elimina objetos menores que el elemento y suaviza contornos, sin encoger lo demás.",
              f"De {F['componentes_binarizado_crudo']} a {F['componentes_tras_apertura']} componentes. Solo quitó los más pequeños: de los {F['componentes_pequenos_crudo_menor_30pct']} componentes menores al 30 % del área típica (plaquetas y restos) siguen {F['componentes_pequenos_tras_apertura']}, porque miden más que el elemento."],
             ["<strong>Relleno de huecos pequeños</strong> (reconstrucción)", "Rellena solo los huecos de menos de 0.35 veces el área típica (centro pálido). No se rellenaron todos los huecos para no llenar los espacios entre células.",
              "Cada glóbulo pasa de un anillo a un disco completo (figura 2c). Sin este paso, el área de cada célula se subestima y la transformada de distancia se rompe."],
             ["<strong>Cierre</strong> 3×3", "Dilatación seguida de erosión: cierra grietas estrechas en los bordes.",
              f"De {F['componentes_tras_apertura']} a {F['componentes_tras_rellenar_y_cierre']} componentes (junto con el relleno)."],
             ["<strong>Erosión</strong> iterada (elemento 3×3 elíptico)", "Encoge las regiones; los objetos unidos por contactos estrechos se separan.",
              f"El conteo casi no cambia: {cur[0]} componentes sin erosionar y entre {min(cur[:10])} y {max(cur[:10])} con 1 a 9 iteraciones (figura 3). Los contactos entre células son anchos, y la erosión los separa solo cuando ya ha reducido mucho las células."],
             ["<strong>Esqueletización</strong>", "Reduce cada región a una línea de un píxel de grosor que conserva su topología.",
              "Una célula aislada da un esqueleto casi puntual; las cadenas y los grupos dan líneas largas o ramificadas (figura 2f), lo que las delata como grupos."],
             ["<strong>Transformada de distancia + watershed</strong>", "Distancia de cada píxel al fondo; cada célula es un máximo local y el watershed traza las fronteras entre máximos.",
              f"Separa las células que se tocan: se obtienen {F['n_distancia']} objetos (figura 4)."]],
            "Operaciones aplicadas y su efecto medido en el frotis real.")
    r.fig("figuras/fig3_conteo.png", "Izquierda: conteo en el frotis real según las iteraciones de erosión, comparado con la transformada de distancia y con el área total dividida entre el área típica. Derecha: tamaño de cada componente conectado respecto del área típica.", alto=70)
    r.fig("figuras/fig4_resultado.png", "Resultado final: fronteras entre células (verde) obtenidas con transformada de distancia y watershed.", alto=115)

    r.h3("Análisis de resultados en el frotis real")
    r.p(f"<strong>Qué cambios produjo cada operación.</strong> La binarización deja {F['componentes_binarizado_crudo']} componentes, entre ellos anillos (células con agujero), restos pequeños y grupos. Tras las operaciones de limpieza, los componentes de tamaño comparable a una célula ({F['componentes_tras_limpiar_(area>=30%)']} con área mayor al 30 % de la típica) son "
        f"en un <strong>{F['componentes_de_1_celula_pct']} %</strong> células individuales (área entre 0.6 y 1.5 veces la típica) y el resto son grupos: hay <strong>{F['componentes_grupos_(>1.5_celdas)']} componentes con más de 1.5 células</strong>, el mayor de unas {F['mayor_grupo_celulas']:.0f} células. Es decir, incluso después de limpiar, el conteo directo de componentes subestima con creces el número de células.")
    r.p(f"<strong>Cómo mejoró la segmentación.</strong> Se compararon tres estimadores independientes del número de células: la <strong>erosión</strong> ({F['n_erosion_optima']}), el <strong>área total dividida entre el área típica</strong> ({F['n_area']:.0f}) y la <strong>transformada de distancia con watershed</strong> ({F['n_distancia']}). "
        f"La transformada de distancia y el área coinciden dentro de un {100 * abs(F['n_area'] - F['n_distancia']) / F['n_distancia']:.0f} % (el estimador por área depende por completo del área típica elegida; aquí da un valor mayor que el de la transformada de distancia, y en las imágenes sintéticas también tendió a sobrestimar, con un sesgo de +2 a +5 %), mientras que el conteo por erosión queda un {100 * (1 - F['n_erosion_optima'] / F['n_distancia']):.0f} % por debajo. "
        "No hay un conteo manual de referencia, por lo que estas cifras se contrastaron con la validación sintética de la sección siguiente.")
    r.p("<strong>Qué estructuras fueron eliminadas o resaltadas.</strong> Se eliminaron los agujeros del centro de los glóbulos (se rellenaron), las plaquetas más pequeñas y los contornos irregulares; se resaltó la <em>forma</em> de cada célula y las fronteras entre las que se tocan. Los leucocitos, más grandes y con núcleo, quedan dentro de la máscara como objetos de mayor tamaño: separarlos de los glóbulos rojos requiere un criterio adicional (color o tamaño).")

    r.h3("Validación con imágenes sintéticas de conteo conocido")
    r.p("Se generaron 12 imágenes con 90 células cada una (discos con centro pálido, ruido, desenfoque y plaquetas), en tres niveles de contacto: células que solo se tocan, con un traslape de 10 % del diámetro y de 25 %. El conteo verdadero se conoce, así que se puede medir el error de cada estimador.")
    r.fig("figuras/fig5_sintetica.png", "Imágenes sintéticas con 0, 10 y 25 % de traslape (arriba) y separación obtenida con watershed (abajo).", alto=115)
    r.fig("figuras/fig6_validacion.png", "Izquierda: error medio del conteo por método y nivel de traslape. Derecha: conteo tras erosionar (una imagen por nivel de traslape).", alto=70)
    r.tabla(["Método de conteo", "Solo se tocan", "Traslape 10 %", "Traslape 25 %"],
            [["Binarización sin morfología", f"{ej('crudo', 0.0)} %", f"{ej('crudo', 0.1)} %", f"{ej('crudo', 0.25)} %"],
             ["Tras apertura, relleno y cierre (componentes conexos)", f"{ej('tras_limpieza', 0.0)} %", f"{ej('tras_limpieza', 0.1)} %", f"{ej('tras_limpieza', 0.25)} %"],
             ["Erosión (mejor número de iteraciones)", f"{ej('erosion', 0.0)} %", f"{ej('erosion', 0.1)} %", f"{ej('erosion', 0.25)} %"],
             ["Área total / área típica", f"{ej('area', 0.0)} %", f"{ej('area', 0.1)} %", f"{ej('area', 0.25)} %"],
             ["Transformada de distancia + watershed", f"{ej('distancia', 0.0)} %", f"{ej('distancia', 0.1)} %", f"{ej('distancia', 0.25)} %"]],
            "Error absoluto medio del conteo (%) en 12 imágenes sintéticas de 90 células.", num=(1, 2, 3))
    r.p(f"<strong>Resultado.</strong> (1) Sin morfología, el conteo se <em>sobrestima</em> un {ej('crudo', 0.0)} % cuando las células solo se tocan, porque las plaquetas y los restos se cuentan como células; con más traslape el error se compensa por el efecto contrario (células unidas), lo que muestra que un error puede ocultar otro. "
        f"(2) La limpieza morfológica elimina la basura y corrige el conteo cuando las células solo se tocan ({ej('tras_limpieza', 0.0)} %), pero <strong>subestima</strong> cada vez más al aumentar el traslape ({ej('tras_limpieza', 0.25)} % con 25 %), porque las células unidas cuentan como una. "
        f"(3) La <strong>erosión</strong> ayuda con traslapes moderados ({ej('erosion', 0.1)} %), pero no alcanza con 25 % ({ej('erosion', 0.25)} %) y su conteo se derrumba si se erosiona demasiado (derecha de la figura 6): el número de iteraciones hay que elegirlo con cuidado. "
        f"(4) La <strong>transformada de distancia con watershed</strong> tuvo el menor error en todos los casos ({ej('distancia', 0.0)} %, {ej('distancia', 0.1)} % y {ej('distancia', 0.25)} %). "
        "<strong>Limitación importante:</strong> las células sintéticas son discos casi perfectos; con formas irregulares, células deformadas o grupos muy densos, como en la imagen real, el error será mayor, y el desempeño de la transformada de distancia se debe tomar como un límite optimista.")

    r.h3("Efecto de la iluminación desigual en otra muestra")
    r.fig("figuras/fig7_segunda_imagen.png", "Frotis con tinción Wright: binarización con umbral global, con el fondo restado, y tras apertura 9×9 y relleno de huecos.", alto=60)
    r.p(f"En la muestra con tinción Wright, el fondo no es uniforme y las células son pálidas. El umbral global (t = {SI['umbral_global']}) da {SI['pct_fg_global']} % de primer plano y {SI['componentes_global']} componentes; restar el fondo estimado con un filtro gaussiano muy amplio casi no cambia el resultado ({SI['pct_fg_corregido']} %), pues la variación de fondo es moderada. "
        f"Lo que sí ayuda es la morfología: la apertura con un elemento de 9×9 elimina el ruido de tinción y los cúmulos de bacterias pequeñas, y el relleno de huecos completa las células, quedando {SI['componentes_final']} componentes de tamaño relevante. Aquí las células son muy pálidas y la separación no es perfecta: algunas se fragmentan y otras se fusionan con restos. Muestra que cada tipo de imagen requiere ajustar los parámetros.")

    r.h2("4. Parte 3. Reflexión y análisis")
    r.h3("¿Cómo ayudan las operaciones morfológicas al análisis biomédico?")
    r.p("Convierten una binarización imperfecta en una segmentación utilizable: eliminan basura (apertura), completan estructuras (cierre y relleno), separan objetos (erosión, transformada de distancia) y resumen la forma (esqueleto). "
        "Son operaciones baratas, explicables y aplicables a muchos problemas: conteo celular, medida de vasos sanguíneos (esqueleto), limpieza de máscaras de tumores y análisis de histopatología. "
        "El caso estudiado muestra su valor y su límite: la limpieza morfológica fue esencial para pasar de 455 componentes con anillos y basura a formas de célula completas, pero para contar células que se tocan hizo falta una herramienta específica (transformada de distancia).")
    r.h3("¿Qué ventajas tiene limpiar una imagen antes de aplicar inteligencia artificial?")
    r.p("Una máscara limpia produce mejores <strong>etiquetas de entrenamiento</strong> (los modelos aprenden de las etiquetas que se les da), reduce los errores en cadena (una segmentación con basura contamina las características que se extraen de cada célula: área, forma, textura) y permite pasar a la red solo las regiones relevantes, con menos cómputo. "
        "Los modelos modernos de segmentación celular, como Cellpose (Stringer et al., 2021) o U-Net (Ronneberger et al., 2015), superan a los métodos clásicos en imágenes difíciles, pero se entrenan con datos anotados que a menudo se generan o corrigen con pasos morfológicos, y siguen usando la morfología para posprocesar sus resultados.")
    r.h3("¿Qué dificultades podrían existir si la segmentación contiene ruido o regiones mal conectadas?")
    r.ul([f"<strong>Errores de conteo:</strong> en las pruebas sintéticas, el conteo por componentes conexos sin separar las células erró hasta un {ej('tras_limpieza', 0.25)} % con un traslape del 25 %, y el conteo sin limpiar, hasta un {ej('crudo', 0.0)} % por la basura.",
          "<strong>Medidas falsas:</strong> dos células unidas se miden como una célula grande (el área y la forma serían anómalas), lo que puede confundirse con una alteración real.",
          "<strong>Errores que se compensan:</strong> células unidas y basura contada pueden dar un conteo total aparentemente correcto (visto con 25 % de traslape sin limpiar), con un resultado que es «correcto por casualidad»; por eso hay que validar con casos de conteo conocido y no solo con el total.",
          "<strong>Sensibilidad a los parámetros:</strong> el tamaño del elemento estructurante y el número de erosiones dependen de la resolución y del tamaño de las células; un parámetro inadecuado destruye las células pequeñas o deja unidas las grandes."])
    r.caja("<strong>Limitaciones del trabajo.</strong> No se dispone de un conteo manual del frotis real, por lo que su conteo se contrastó solo con estimadores independientes y con imágenes sintéticas (idealizadas). Las imágenes proceden de bancos públicos, no de un laboratorio propio, y el estudio se limitó a glóbulos rojos.")

    r.refs([
        "Chambers, K. (s. f.). <em>Normal Adult Blood Smear</em> [Fotografía; CC BY-SA 3.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Normal_Adult_Blood_Smear.JPG",
        "Chaurasiya, A. K. (s. f.). <em>Nucleated red blood cell (RBC) in a Wright’s stained peripheral blood smear (PBS) microscopy of an aplastic anemia patient</em> [Fotografía; CC BY-SA 4.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Nucleated_red_blood_cell_(RBC)_in_a_Wright%27s_stained_peripheral_blood_smear_(PBS)_microscopy_of_an_aplastic_anemia_patient.jpg",
        "Gonzalez, R. C., &amp; Woods, R. E. (2018). <em>Digital image processing</em> (4.ª ed.). Pearson.",
        "Haralick, R. M., Sternberg, S. R., &amp; Zhuang, X. (1987). Image analysis using mathematical morphology. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-9</em>(4), 532–550. https://doi.org/10.1109/TPAMI.1987.4767941",
        "ImageJ. (2024). <em>Morphological filters</em>. https://imagej.nih.gov/ij/docs/guide/",
        "Meijering, E. (2012). Cell segmentation: 50 years down the road. <em>IEEE Signal Processing Magazine, 29</em>(5), 140–145. https://doi.org/10.1109/MSP.2012.2204190",
        "Murphy, D. B., &amp; Davidson, M. W. (2012). <em>Fundamentals of light microscopy and electronic imaging</em> (2.ª ed.). Wiley-Blackwell.",
        "OpenAI. (2026). <em>Operaciones morfológicas</em> [Imagen generada con IA]. Material de la actividad, Visión computacional para imágenes y video. ChatGPT.",
        "OpenCV. (2024). <em>Morphological transformations</em>. https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. <em>IEEE Transactions on Systems, Man, and Cybernetics, 9</em>(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076",
        "Ronneberger, O., Fischer, P., &amp; Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. En N. Navab, J. Hornegger, W. M. Wells, &amp; A. F. Frangi (Eds.), <em>Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015</em> (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28",
        "Serra, J. (1982). <em>Image analysis and mathematical morphology</em>. Academic Press.",
        "Stringer, C., Wang, T., Michaelos, M., &amp; Pachitariu, M. (2021). Cellpose: A generalist algorithm for cellular segmentation. <em>Nature Methods, 18</em>(1), 100–106. https://doi.org/10.1038/s41592-020-01018-x",
        "van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., &amp; Yu, T. (2014). scikit-image: Image processing in Python. <em>PeerJ, 2</em>, e453. https://doi.org/10.7717/peerj.453",
        "Vincent, L., &amp; Soille, P. (1991). Watersheds in digital spaces: An efficient algorithm based on immersion simulations. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, 13</em>(6), 583–598. https://doi.org/10.1109/34.87344",
    ])
    pdf = r.guardar("Morfologia_celulas_biomedicas")
    n = zip_entrega(AQUI / "Morfologia_celulas_biomedicas.zip", AQUI, ["experimentos.py", "informe.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
