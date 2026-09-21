"""Informe PDF: ¿Cómo se aplica la segmentación en imágenes? Comparación de métodos clásicos en resonancia magnética."""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
S = R["sin_ruido"]
N = R["ruido"]
D = N["dice"]
SO = R["sobresegmentacion"]
IMG = R["imagen"]
ORDEN = ["Otsu (2 clases)", "Multi-Otsu (3 clases)", "K-means (K=4)", "Mean Shift", "SLIC (350 superpíxeles)", "Watershed con marcadores"]


def main():
    r = Reporte("Segmentación de tumores cerebrales en resonancia magnética",
                "Comparación de Otsu, Watershed, K-means, Mean Shift y superpíxeles (SLIC)", carpeta=AQUI,
                entrega="Reporte en PDF + archivo comprimido con el código")
    r.portada()

    r.h2("1. Introducción")
    r.p("En imagenología médica, <strong>segmentar</strong> es delimitar en la imagen las regiones de interés: un órgano, un tejido o una lesión. La delimitación permite medir su tamaño y su forma, seguir su evolución, planear una cirugía o una radioterapia, y es el punto de partida de muchos sistemas de apoyo al diagnóstico (Pham et al., 2000). "
        "En un tumor cerebral la tarea es delicada: los bordes pueden ser difusos, el tejido sano vecino puede tener una intensidad parecida y las imágenes tienen ruido. Este trabajo compara cinco métodos clásicos para separar un tumor del tejido sano en una resonancia magnética (MRI) y analiza en qué situaciones funciona cada uno.")

    r.h2("2. Métodos estudiados")
    r.tabla(["Método", "Idea", "Ventajas", "Limitaciones"],
            [["<strong>Otsu</strong> (Otsu, 1979; multinivel: Liao et al., 2001)", "Elige el umbral (o varios) que maximiza la varianza entre clases del histograma.", "Automático, rapidísimo, sin parámetros.", "Ignora la posición de los píxeles; requiere que las clases se separen por intensidad; sensible a iluminación desigual y a más de dos clases."],
             ["<strong>Watershed</strong> (Vincent & Soille, 1991)", "Interpreta el gradiente como un relieve y «inunda» las cuencas desde marcadores; las regiones se encuentran en las crestas (bordes).", "Contornos cerrados y precisos; permite guiar con marcadores.", "Sin marcadores produce una sobresegmentación enorme; sensible al ruido en el gradiente."],
             ["<strong>K-means</strong> (Lloyd, 1982; Arthur & Vassilvitskii, 2007)", "Agrupa los píxeles en K grupos de intensidad similar minimizando la distancia al centro de cada grupo.", "Sencillo, rápido, extiende a varios canales o a la posición.", "Hay que fijar K; no usa la vecindad; sensible al ruido y a la inicialización."],
             ["<strong>Mean Shift</strong> (Comaniciu & Meer, 2002)", "Desplaza cada píxel hacia el máximo local de la densidad de intensidad (y posición) hasta converger; los que llegan al mismo máximo forman una región.", "No fija el número de regiones; preserva bordes; suaviza el ruido.", "Más lento; depende de los anchos de banda espacial y de intensidad."],
             ["<strong>Superpíxeles SLIC</strong> (Achanta et al., 2012)", "Agrupa píxeles en regiones compactas de tamaño parecido (K-means local en intensidad + posición).", "Reduce la imagen a unos cientos de regiones que respetan los bordes; buen paso previo.", "Por sí mismo no distingue el tumor: hay que fusionar o clasificar los superpíxeles."]],
            "Resumen de los métodos comparados.")

    r.h2("3. Datos y metodología")
    r.p(f"<strong>Imagen.</strong> Resonancia T1 con contraste en un corte axial con un meningioma grande y brillante de la fosa posterior (Tdvorak, s. f.; CC BY-SA 3.0), de {IMG['tam'][1]}×{IMG['tam'][0]} px. Quien la publicó dibujó a mano el contorno del tumor en rojo; ese contorno se rellenó y se usa como <strong>verdad aproximada</strong> "
        f"(el tumor ocupa el {IMG['tumor_pct']} % de la imagen), y a los algoritmos se les entregó la imagen <strong>sin la línea roja</strong> (reconstruida con <em>inpainting</em>). No es una anotación clínica validada, por lo que las cifras sirven para <em>comparar métodos entre sí</em>, no como medida diagnóstica.")
    r.p("<strong>Procedimiento.</strong> Los métodos son <em>semiautomáticos</em>: reciben un punto semilla dentro del tumor (equivalente a un clic de la persona especialista), y se conserva el componente conexo que lo contiene tras una apertura y un cierre morfológicos de 3×3. "
        "Otsu multinivel, K-means (K = 4), Mean Shift y SLIC toman como «tumor» la clase de mayor intensidad; Watershed usa la semilla y el fondo como marcadores. "
        "Se mide el coeficiente de <strong>Dice</strong> (2·|A∩B| / (|A|+|B|)), la precisión y la exhaustividad respecto de la verdad aproximada, y se repite con ruido gaussiano añadido de σ = 10, 25 y 40 niveles de gris.")
    r.fig("figuras/fig1_imagen_y_verdad.png", "Imagen usada (sin el contorno rojo), con la semilla (+ cian); contorno de referencia del tumor (verde); e histograma con los umbrales de Otsu. El tumor (verde) ocupa el extremo brillante del histograma.")

    r.h2("4. Resultados")
    r.fig("figuras/fig2_metodos.png", "Resultado de cada método (rojo) frente al contorno de referencia (verde). Abajo a la derecha: Watershed sin marcadores.", alto=170)
    r.tabla(["Método", "Dice", "IoU", "Precisión", "Exhaustividad", "Área segmentada (% de la imagen)"],
            [[n, f"{S[n]['dice']:.3f}", f"{S[n]['iou']:.3f}", f"{S[n]['precision']:.3f}", f"{S[n]['exhaustividad']:.3f}", f"{S[n]['area_pct']:.1f}"] for n in ORDEN] + [["<em>Referencia (verdad aproximada)</em>", "—", "—", "—", "—", f"{IMG['tumor_pct']}"]],
            "Segmentación de la resonancia sin ruido.", num=(1, 2, 3, 4, 5))
    r.p(f"Con Otsu de dos clases, el umbral ({IMG['otsu2_umbral']}) separa la <strong>cabeza entera</strong> del fondo negro (precisión de {S['Otsu (2 clases)']['precision']:.2f}, es decir, el 87 % de lo segmentado no es tumor): el histograma tiene un pico enorme de fondo y una masa de tejido, y el tumor queda del lado del tejido. "
        "Los otros cinco métodos con semilla encuentran el tumor con un Dice de entre "
        f"{min(S[n]['dice'] for n in ORDEN[1:]):.2f} y {max(S[n]['dice'] for n in ORDEN[1:]):.2f}. Las diferencias entre ellos son pequeñas.")

    r.h3("Sensibilidad al ruido")
    r.fig("figuras/fig3_ruido.png", "Izquierda: Dice en función del ruido añadido. Derecha: contornos de cada método con ruido σ = 25.")
    r.tabla(["Método"] + [f"σ = {s}" for s in N["sigmas"]] + ["Variación (σ 0 → 40)"],
            [[n] + [f"{d:.3f}" for d in D[n]] + [f"{D[n][-1] - D[n][0]:+.3f}"] for n in D],
            "Coeficiente de Dice al añadir ruido gaussiano. Otsu de dos clases se excluye: su resultado depende de un artefacto de conectividad (ver texto).", num=range(1, 6))
    r.caja("<strong>Nota sobre Otsu de dos clases.</strong> Su Dice <em>sube</em> al añadir ruido (0.23 sin ruido); no es una virtud: sin ruido, su umbral deja la cabeza entera conectada y el componente de la semilla es toda la cabeza. El ruido rompe esa conexión y, por casualidad, el componente de la semilla queda reducido al tumor. Por eso se excluyó de la comparación de ruido.")

    r.h2("5. Análisis de resultados")
    r.h3("¿Qué método segmentó mejor la región de interés?")
    r.p(f"<strong>Watershed con marcadores</strong> obtuvo el mejor Dice ({S['Watershed con marcadores']['dice']:.3f}) y, sobre todo, la mayor precisión ({S['Watershed con marcadores']['precision']:.3f}): casi todo lo que marca es tumor, porque sus contornos se apoyan en el gradiente. Le siguen Mean Shift ({S['Mean Shift']['dice']:.3f}), Multi-Otsu ({S['Multi-Otsu (3 clases)']['dice']:.3f}), K-means ({S['K-means (K=4)']['dice']:.3f}) y SLIC ({S['SLIC (350 superpíxeles)']['dice']:.3f}). "
        "Las diferencias entre los cinco son de unas centésimas y, dado que la verdad es aproximada y hay una sola imagen, <strong>no permiten afirmar un orden definitivo</strong>: lo que sí queda claro es que Otsu de dos clases no sirve aquí y que los métodos con más de dos clases o con marcadores sí. "
        "En este caso el éxito se debe a que el meningioma con contraste es muy brillante y bien separado del tejido sano.")
    r.h3("¿Cuál fue más sensible al ruido?")
    r.p(f"<strong>K-means</strong> fue el más afectado: su Dice cayó de {D['K-means (K=4)'][0]:.3f} a {D['K-means (K=4)'][-1]:.3f} con σ = 40 ({D['K-means (K=4)'][-1] - D['K-means (K=4)'][0]:+.3f}), pues clasifica cada píxel por separado y el ruido lo mueve entre grupos. "
        f"Watershed también bajó ({D['Watershed con marcadores'][0]:.3f} → {D['Watershed con marcadores'][-1]:.3f}) porque el ruido introduce falsos bordes en el gradiente. Mean Shift (−{abs(D['Mean Shift'][-1] - D['Mean Shift'][0]):.3f}) y Multi-Otsu (−{abs(D['Multi-Otsu (3 clases)'][-1] - D['Multi-Otsu (3 clases)'][0]):.3f}) cambiaron poco, y SLIC no empeoró ({D['SLIC (350 superpíxeles)'][0]:.3f} → {D['SLIC (350 superpíxeles)'][-1]:.3f}) porque promedia la intensidad de cada superpíxel. "
        "Hay que recordar que todos los métodos usan una limpieza morfológica final y que este tumor es muy contrastado; con lesiones de bajo contraste el efecto del ruido sería mayor.")
    r.h3("¿Qué método generó sobresegmentación?")
    r.p(f"<strong>Watershed sin marcadores</strong>, el caso clásico: cada mínimo local del gradiente genera una cuenca y la imagen queda dividida en <strong>{SO['watershed_sin_marcadores_regiones']:,} regiones</strong> (esquina inferior derecha de la figura 2), imposibles de interpretar; la solución es usar marcadores, como se hizo en la versión con semilla.".replace(",", " ") +
        f" SLIC produce {SO['slic_superpixeles']} superpíxeles <em>a propósito</em> (sobresegmentación controlada) para luego fusionarlos. Además, con los métodos de umbral y con K-means aparecen pequeños agujeros dentro del tumor, donde hay zonas de intensidad algo menor.")
    r.h3("¿Cuál consideras más útil en aplicaciones médicas reales y por qué?")
    r.p("Para la asistencia a una persona especialista, un método <strong>con marcadores o semillas</strong> como Watershed (o sus variantes con contornos activos, como las que usa ITK-SNAP; Yushkevich et al., 2006) es el más útil: es interactivo, la persona controla el resultado y la precisión fue la mayor (0.993). "
        "Otsu multinivel y K-means son rápidos y sirven como preprocesamiento o primera estimación, pero dependen de que el tumor sea la clase más brillante. "
        "El siguiente resultado lo muestra.")
    r.fig("figuras/fig4_segunda_imagen.png", "Segunda resonancia (glioblastoma con realce en anillo, Hellerhoff, CC BY-SA 3.0), sin verdad de referencia. El punto cian es la semilla. Los métodos basados en intensidad no encuentran el tumor.")
    r.p("En una resonancia con un glioblastoma, cuyo tumor es heterogéneo y no es la estructura más brillante (el cráneo y la grasa lo superan), los métodos de intensidad segmentan casi nada (áreas de 0.0 % a 0.2 % de la imagen). "
        "Esto ilustra el límite general de los métodos clásicos: <strong>funcionan cuando la lesión se separa por intensidad</strong>. En la práctica actual, para tumores de apariencia variable se usan redes neuronales profundas entrenadas con imágenes anotadas, como U-Net (Ronneberger et al., 2015), evaluadas en desafíos como BraTS (Menze et al., 2015).")

    r.h2("6. Conclusiones")
    r.h3("La importancia de la segmentación en imagenología médica")
    r.p("Convierte una imagen en información cuantificable: volumen del tumor, forma, localización, respuesta a un tratamiento. Sin segmentación, el diagnóstico depende de una lectura visual que varía entre personas; con ella se puede medir de forma reproducible y comparar estudios a lo largo del tiempo. "
        "Como delimitar a mano cada corte es lento, la segmentación asistida o automática reduce el tiempo y la variabilidad (Pham et al., 2000).")
    r.h3("Integración con sistemas de inteligencia artificial para apoyo diagnóstico")
    r.p("Los métodos clásicos se integran de tres maneras: (1) como <strong>preprocesamiento</strong> (quitar el fondo, extraer el cráneo, normalizar la intensidad) antes de una red neuronal; (2) para generar <strong>etiquetas iniciales</strong> que una persona corrige y que luego sirven para entrenar modelos; y (3) como <strong>posprocesamiento</strong> (limpieza morfológica, componentes conexos) de la salida de una red. "
        "Un sistema de apoyo diagnóstico combinaría una red para proponer la región, una revisión por la persona especialista y métricas como el coeficiente de Dice para vigilar la calidad.")
    r.p("<strong>Limitaciones de este trabajo:</strong> una sola imagen con verdad aproximada dibujada a mano, un tumor muy contrastado y métodos semiautomáticos con la semilla en el centro del tumor; los resultados no pueden generalizarse a otros tumores ni a otras modalidades sin más imágenes y una anotación validada.")

    r.refs([
        "Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., &amp; Süsstrunk, S. (2012). SLIC superpixels compared to state-of-the-art superpixel methods. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, 34</em>(11), 2274–2282. https://doi.org/10.1109/TPAMI.2012.120",
        "Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. En <em>Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms</em> (pp. 1027–1035). SIAM.",
        "Comaniciu, D., &amp; Meer, P. (2002). Mean shift: A robust approach toward feature space analysis. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, 24</em>(5), 603–619. https://doi.org/10.1109/34.1000236",
        "Hellerhoff. (s. f.). <em>Glioblastoma multiforme – MRT T1KM ax</em> [Imagen; CC BY-SA 3.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Glioblastoma_multiforme_-_MRT_T1KM_ax.jpg",
        "Liao, P.-S., Chen, T.-S., &amp; Chung, P.-C. (2001). A fast algorithm for multilevel thresholding. <em>Journal of Information Science and Engineering, 17</em>(5), 713–727.",
        "Lloyd, S. (1982). Least squares quantization in PCM. <em>IEEE Transactions on Information Theory, 28</em>(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489",
        "Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., Burren, Y., Porz, N., Slotboom, J., Wiest, R., Lanczi, L., Gerstner, E., Weber, M.-A., Arbel, T., Avants, B. B., Ayache, N., Buendia, P., Collins, D. L., Cordier, N., … Van Leemput, K. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). <em>IEEE Transactions on Medical Imaging, 34</em>(10), 1993–2024. https://doi.org/10.1109/TMI.2014.2377694",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. <em>IEEE Transactions on Systems, Man, and Cybernetics, 9</em>(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076",
        "Pham, D. L., Xu, C., &amp; Prince, J. L. (2000). Current methods in medical image segmentation. <em>Annual Review of Biomedical Engineering, 2</em>, 315–337. https://doi.org/10.1146/annurev.bioeng.2.1.315",
        "Ronneberger, O., Fischer, P., &amp; Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. En N. Navab, J. Hornegger, W. M. Wells, &amp; A. F. Frangi (Eds.), <em>Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015</em> (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28",
        "Tdvorak. (s. f.). <em>Tumor Meningioma1</em> [Imagen; CC BY-SA 3.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Tumor_Meningioma1.JPG",
        "van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., &amp; Yu, T. (2014). scikit-image: Image processing in Python. <em>PeerJ, 2</em>, e453. https://doi.org/10.7717/peerj.453",
        "Vincent, L., &amp; Soille, P. (1991). Watersheds in digital spaces: An efficient algorithm based on immersion simulations. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, 13</em>(6), 583–598. https://doi.org/10.1109/34.87344",
        "Yushkevich, P. A., Piven, J., Hazlett, H. C., Smith, R. G., Ho, S., Gee, J. C., &amp; Gerig, G. (2006). User-guided 3D active contour segmentation of anatomical structures: Significantly improved efficiency and reliability. <em>NeuroImage, 31</em>(3), 1116–1128. https://doi.org/10.1016/j.neuroimage.2006.01.015",
    ])
    pdf = r.guardar("Segmentacion_MRI_tumor")
    n = zip_entrega(AQUI / "Segmentacion_MRI_tumor.zip", AQUI, ["experimentos.py", "informe.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
