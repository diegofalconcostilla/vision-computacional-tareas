"""Informe PDF: Cómo crear una imagen panorámica usando descriptores."""
import json
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
M, PAN, IL, D = R["matching"], R["panorama"], R["iluminacion"], R["densidad_sift_centro"]


def main():
    shutil.copy(RAIZ / "herramientas" / "puntos.py", AQUI / "puntos.py")
    r = Reporte("Panoramas digitales con keypoints y descriptores",
                "SIFT y ORB para detectar, emparejar y unir tres tomas con solapamiento", carpeta=AQUI,
                entrega="PDF con imágenes, keypoints, matching y reflexión + archivo comprimido con el código")
    r.portada()

    r.h2("1. Introducción y aclaración sobre las imágenes")
    r.p("La función panorámica de un teléfono parece automática, pero detrás hay una cadena de visión computacional: detectar puntos clave en cada foto, describirlos con un vector numérico, emparejarlos entre fotos consecutivas, estimar la transformación geométrica que las alinea (una homografía, con RANSAC para descartar errores) y fusionarlas en una sola imagen (Brown & Lowe, 2007). "
        "Este trabajo recorre esos pasos con dos descriptores, SIFT y ORB, y construye un panorama de tres tomas.")
    r.caja("<strong>Sobre las fotografías.</strong> La actividad pide tomar 2 o 3 fotos con el teléfono. Como aquí no se contaba con ellas, las tres tomas se <strong>generaron a partir de una fotografía de calle real</strong> (Rivera, 2025; CC0): cada toma es la vista de una cámara que solo <em>gira</em> (−15° y +15° respecto de la central, como al hacer un panorama de pie), "
           "con un ángulo de visión de unos 56° por toma y un solape aproximado de 70 % entre tomas vecinas. Además, cada toma tiene una exposición distinta (×1.12, ×1.0 y ×0.85, imitando la exposición automática) y algo de ruido. "
           "Como la geometría es exacta, se puede calificar cada correspondencia como correcta o errónea y medir el error de la unión. Es una simplificación: un giro puro no produce paralaje ni objetos que se muevan entre tomas, como sí ocurre en fotos reales. "
           "Si se cuenta con fotos propias, basta con reemplazar los archivos de la carpeta <code>datos/</code>.", aviso=True)

    r.h2("2. Parte 1. Captura de imágenes")
    r.fig("figuras/fig1_tomas.png", "Las tres tomas con solapamiento (izquierda, centro y derecha). Cada una tiene una exposición distinta. Foto original: M. Rivera, CC0.")
    r.p("Las tres tomas comparten cerca del 70 % de la escena con la vecina. Se evitaron, como pide la actividad, los movimientos bruscos, los cambios extremos de iluminación y los objetos en movimiento (el único cambio de iluminación es el de exposición entre tomas).")

    r.h2("3. Parte 2. Observación de keypoints")
    r.p("Se aplicó SIFT y ORB (SURF no está disponible en las distribuciones estándar de OpenCV por estar patentado). Se limitó cada detector a 3000 puntos por toma, de modo que <strong>el número total de puntos no dice cuál detector es más «productivo»</strong> (ambos llegan al límite en todas las tomas); "
        "lo informativo es <em>dónde</em> se ubican y cuántos resultan útiles después.")
    r.fig("figuras/fig2_keypoints.png", "Keypoints en cada toma: SIFT (arriba, círculos con tamaño y orientación) y ORB (abajo).", alto=110)
    r.fig("figuras/fig3_regiones_iluminacion.png", "Izquierda: keypoints SIFT por región en la toma central (cuadrícula de 4×6). Derecha: efecto de la iluminación sobre el número de keypoints y de correspondencias correctas al comparar la toma central con una versión más oscura o más clara de sí misma.")
    r.h3("¿Qué regiones generan más keypoints?")
    r.p(f"Los que tienen <strong>más detalle y contraste local</strong>: la fila inferior de la cuadrícula (casas, setos, vehículos y marcas del suelo) reúne entre {min(D[3])} y {max(D[3])} puntos por celda, con el máximo ({max(D[3])}) en la esquina inferior izquierda, donde hay un seto oscuro y el techo de una casa. "
        f"También destacan las ramas de los árboles sin hojas en la parte superior derecha ({D[0][5]}, {D[1][5]} y {D[1][4]} puntos por celda), que tienen mucho contraste fino.")
    r.h3("¿Qué sucede en zonas lisas o sin textura?")
    r.p(f"Casi no hay puntos: las celdas de cielo despejado de la parte superior izquierda y central tienen entre {min(D[0][:3] + D[1][:3])} y {max(D[0][:3] + D[1][:3])} keypoints. Sin variación de intensidad no hay esquinas ni bordes que anclar, y esa zona no aporta nada al alineamiento "
        "(en un panorama de un cielo puro, el algoritmo no podría alinear las fotos). Por eso las tomas panorámicas deben incluir estructuras reconocibles en la zona de solape.")
    r.h3("¿Cómo afecta la iluminación?")
    b = IL["brillos"]
    r.tabla(["Factor de brillo", "SIFT: keypoints", "SIFT: correspondencias correctas", "ORB: keypoints", "ORB: correspondencias correctas"],
            [[f"×{b[i]}", str(IL["keypoints"]["sift"][i]), str(IL["correctos"]["sift"][i]), str(IL["keypoints"]["orb"][i]), str(IL["correctos"]["orb"][i])] for i in range(len(b))],
            "Toma central comparada con una versión de sí misma con distinto brillo (límite de 3000 puntos).", num=(1, 2, 3, 4))
    r.p(f"Con la imagen <strong>oscurecida</strong> (×0.25) SIFT detecta menos puntos ({IL['keypoints']['sift'][0]}), pero casi todos se conservan como correspondencias correctas ({IL['correctos']['sift'][0]}); con ×0.5 recupera prácticamente todo ({IL['correctos']['sift'][1]} de 3000). "
        f"Aclarar la imagen tiene un efecto mucho peor: con ×1.6, las correspondencias correctas caen a {IL['correctos']['sift'][3]} en SIFT y {IL['correctos']['orb'][3]} en ORB, y con ×2.5 a {IL['correctos']['sift'][4]} y {IL['correctos']['orb'][4]}. "
        "La causa es la saturación: al multiplicar por un factor grande los píxeles claros se recortan en 255 y los detalles del cielo y de las fachadas desaparecen, mientras que oscurecer conserva la estructura (solo la comprime). "
        "En una toma real, un cambio de exposición entre fotos vecinas debilita el emparejamiento, sobre todo si hay zonas saturadas, y obliga a compensarlo al unir.")

    r.h2("4. Parte 3. Matching de descriptores")
    r.p("Se emparejaron las tomas adyacentes (izquierda–centro y centro–derecha) con fuerza bruta y la prueba de razón de Lowe (0.75 para SIFT, con distancia L2; 0.8 para ORB, con distancia de Hamming). Cada correspondencia se calificó con la homografía verdadera (correcta si cae a ≤ 3 px).")
    r.fig("figuras/fig4_matching.png", "Correspondencias entre tomas vecinas con SIFT y ORB (las 70 mejores). Verde: correcta; rojo: errónea (no aparece ninguna en las mejores).", alto=200)
    filas = []
    for clave, v in M.items():
        filas.append([clave.replace(" sift", " · SIFT").replace(" orb", " · ORB"), str(v["matches"]), str(v["correctos"]), f"{v['precision'] * 100:.1f} %", f"{v['error_H_px']:.2f}", f"{v['t_deteccion_ms']:.0f}"])
    r.tabla(["Par y descriptor", "Correspondencias", "Correctas", "Precisión", "Error de la homografía (px)", "Detección (ms, 2 imágenes)"], filas,
            "Calidad del matching entre tomas vecinas. La homografía se estima con RANSAC a partir de las correspondencias.", num=(1, 2, 3, 4, 5))
    s1, s2, o1, o2 = M["izquierda-centro sift"], M["centro-derecha sift"], M["izquierda-centro orb"], M["centro-derecha orb"]
    r.h3("Correspondencias correctas, erróneas y cantidad")
    r.p(f"SIFT encontró {s1['matches']} y {s2['matches']} correspondencias, con una precisión de {s1['precision'] * 100:.1f} % y {s2['precision'] * 100:.1f} %; ORB encontró {o1['matches']} y {o2['matches']}, con {o1['precision'] * 100:.1f} % y {o2['precision'] * 100:.1f} %. "
        f"Las erróneas de ORB ({o1['matches'] - o1['correctos']} y {o2['matches'] - o2['correctos']}) son pocas en términos absolutos y RANSAC las descarta.")
    r.h3("¿Qué descriptor produjo mejores resultados? ¿Cuál fue más rápido? ¿Qué problemas aparecieron?")
    r.ul([
        f"<strong>Mejor resultado: SIFT.</strong> Más precisión (≈ 99 % frente a ≈ 94 %) y una homografía mucho más exacta (error de {s1['error_H_px']:.2f} y {s2['error_H_px']:.2f} px, frente a {o1['error_H_px']:.2f} y {o2['error_H_px']:.2f} px de ORB). En este caso la diferencia geométrica es pequeña, y la unión visual con ambos sería aceptable.",
        f"<strong>Más rápido: ORB.</strong> La detección en las dos imágenes tardó entre {min(o1['t_deteccion_ms'], o2['t_deteccion_ms']):.0f} y {max(o1['t_deteccion_ms'], o2['t_deteccion_ms']):.0f} ms, frente a {min(s1['t_deteccion_ms'], s2['t_deteccion_ms']):.0f}–{max(s1['t_deteccion_ms'], s2['t_deteccion_ms']):.0f} ms de SIFT (unas 5 veces más). Los tiempos varían de una ejecución a otra, pero el orden se mantuvo. Para procesar en un teléfono, en tiempo real, esa diferencia es importante.",
        "<strong>Problemas:</strong> (1) las zonas sin textura no aportan puntos, y las de textura repetitiva (ramas, setos) pueden dar emparejamientos ambiguos; (2) los cambios de exposición entre tomas y la saturación reducen el número de correspondencias fiables; "
        "(3) sin RANSAC, unas pocas correspondencias erróneas bastan para estropear la homografía; (4) esta prueba no incluye objetos en movimiento ni paralaje, que en fotos reales causan «fantasmas» y desalineamientos en la unión.",
    ])

    r.h2("5. Construcción del panorama")
    r.p(f"Con las correspondencias de SIFT se estimó la homografía de cada toma lateral al plano de la central (RANSAC, umbral de 3 px). El error de esas homografías respecto de las verdaderas fue de <strong>{PAN['error_homografia_px']['izquierda->centro']:.2f} px</strong> (izquierda→centro) y <strong>{PAN['error_homografia_px']['derecha->centro']:.2f} px</strong> (derecha→centro), medido en las esquinas. "
        f"Después se aplicaron dos pasos: <strong>compensación de exposición</strong> (una ganancia por toma calculada en la zona de solape, de {PAN['ganancias'][0]:.2f} para la izquierda y {PAN['ganancias'][2]:.2f} para la derecha) y <strong>mezcla con peso decreciente hacia los bordes</strong> (<em>feathering</em>), que evita la costura visible. "
        f"La diferencia media de intensidad entre tomas en la zona de solape bajó de {PAN['dif_solape_sin_compensar']:.1f} a {PAN['dif_solape_compensada']:.1f} niveles con la compensación. El panorama mide {PAN['tam'][0]}×{PAN['tam'][1]} px y se construyó en {PAN['t_union_s']} s.")
    r.fig("figuras/fig5_panorama.png", "Arriba: panorama sin compensar la exposición. Centro: panorama propio con compensación y mezcla. Abajo: resultado del <em>Stitcher</em> de OpenCV como referencia (con proyección esférica y otro recorte).", alto=190)
    r.p("El resultado propio y el del <em>Stitcher</em> de OpenCV son comparables a simple vista, aunque este último aplica una proyección distinta y un recorte diferente. Una mejora posible sería la mezcla multibanda (Burt & Adelson, 1983), que reduce aún más las diferencias de exposición y de detalle en la zona de solape.")

    r.h2("6. Parte 4. Reflexión")
    r.h3("¿Cómo funcionan los panoramas digitales?")
    r.p("Se toman varias fotos que comparten una parte de la escena. En cada una se detectan puntos clave y se describen; al emparejarlos entre fotos vecinas aparecen las mismas partes de la escena en ambas. Con esas parejas se calcula una homografía, robusta a errores mediante RANSAC (Fischler & Bolles, 1981), que lleva una foto al plano de la otra. "
        f"Por último se fusionan las imágenes compensando la exposición y suavizando las transiciones. En este trabajo, los pares de keypoints permitieron calcular las homografías con un error de menos de {max(PAN['error_homografia_px'].values()) + 0.05:.1f} px y unir las tres tomas.")
    r.h3("¿Qué papel juegan los keypoints y los descriptores?")
    r.p("Los <strong>keypoints</strong> responden a <em>dónde</em> mirar: son lugares que se pueden localizar con precisión y encontrar de nuevo en otra foto (esquinas, manchas). Los <strong>descriptores</strong> responden a <em>qué se ve ahí</em>: un vector numérico que resume la vecindad del punto (gradientes en SIFT; comparaciones binarias en ORB) "
        "y permite comparar puntos de fotos distintas midiendo una distancia. Sin descriptores, habría que comparar imágenes píxel a píxel, algo inviable si hay cambios de escala, giro o iluminación.")
    r.h3("¿Por qué la robustez a escala y rotación es importante?")
    r.p("Porque las fotos consecutivas de un panorama no son idénticas: al girar el teléfono el encuadre rota un poco, el usuario se acerca o se aleja sin darse cuenta y la perspectiva cambia. Si el descriptor cambiara con la escala o con el giro, el mismo punto se describiría de forma distinta en cada foto y no se emparejaría. "
        "SIFT consigue la invarianza buscando puntos en un espacio de escalas y asignando a cada uno una orientación dominante (Lowe, 2004); ORB hace algo similar, con una pirámide de imágenes y una orientación calculada con el centroide de intensidad (Rublee et al., 2011). "
        "En la Práctica 7 se observó lo contrario: un descriptor de parche sencillo, sin orientación, falló por completo con un giro de 30°.")

    r.refs([
        "Bay, H., Ess, A., Tuytelaars, T., &amp; Van Gool, L. (2008). Speeded-up robust features (SURF). <em>Computer Vision and Image Understanding, 110</em>(3), 346–359. https://doi.org/10.1016/j.cviu.2007.09.014",
        "Brown, M., &amp; Lowe, D. G. (2007). Automatic panoramic image stitching using invariant features. <em>International Journal of Computer Vision, 74</em>(1), 59–73. https://doi.org/10.1007/s11263-006-0002-3",
        "Burt, P. J., &amp; Adelson, E. H. (1983). A multiresolution spline with application to image mosaics. <em>ACM Transactions on Graphics, 2</em>(4), 217–236. https://doi.org/10.1145/245.247",
        "Fischler, M. A., &amp; Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. <em>Communications of the ACM, 24</em>(6), 381–395. https://doi.org/10.1145/358669.358692",
        "Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. <em>International Journal of Computer Vision, 60</em>(2), 91–110. https://doi.org/10.1023/B:VISI.0000029664.99615.94",
        "OpenAI. (2026). <em>Actividad: Creación básica de un panorama</em> [Imagen generada con IA]. Material de la actividad, Visión computacional para imágenes y video. ChatGPT.",
        "OpenCV. (s. f.). <em>Feature detection and description</em>. OpenCV Documentation. https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html",
        "Rivera, M. (2025). <em>NB North Oak St at West Alden Ave intersection dashcam</em> [Fotografía; CC0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:NB_North_Oak_St_at_West_Alden_Ave_intersection_dashcam.jpg",
        "Rublee, E., Rabaud, V., Konolige, K., &amp; Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. En <em>2011 International Conference on Computer Vision</em> (pp. 2564–2571). IEEE. https://doi.org/10.1109/ICCV.2011.6126544",
        "Szeliski, R. (2006). Image alignment and stitching: A tutorial. <em>Foundations and Trends in Computer Graphics and Vision, 2</em>(1), 1–104. https://doi.org/10.1561/0600000009",
    ])
    pdf = r.guardar("Panorama_descriptores")
    n = zip_entrega(AQUI / "Panorama_descriptores.zip", AQUI, ["experimentos.py", "informe.py", "puntos.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
