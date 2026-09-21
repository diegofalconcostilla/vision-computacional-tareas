"""Informe PDF: Reconocimiento visual aplicado a Street View: keypoints y Harris."""
import json
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
NOM = {"harris": "Harris + parche", "orb": "ORB", "sift": "SIFT"}
B = R["barrido_yaw"]
RG = R["regiones_sift"]


def main():
    shutil.copy(RAIZ / "herramientas" / "puntos.py", AQUI / "puntos.py")
    r = Reporte("Reconocimiento visual con keypoints: la idea detrás de Street View",
                "Harris, ORB y SIFT para relacionar dos imágenes consecutivas de una calle", carpeta=AQUI,
                entrega="PDF con imágenes, keypoints y correspondencias + archivo comprimido con el código")
    r.portada()

    r.h2("1. Introducción y aclaración sobre los datos")
    r.p("Los sistemas de navegación visual, como el recorrido por calles de Google Street View, necesitan saber qué parte de una imagen corresponde a qué parte de la imagen vecina para alinearlas y moverse entre ellas con suavidad. "
        "La herramienta básica son los <strong>puntos clave</strong> (<em>keypoints</em>): lugares de la imagen que se pueden reconocer de nuevo en otra toma, como las esquinas. "
        "Este trabajo detecta esquinas con el detector de Harris, las describe y empareja con ORB y SIFT, y estudia qué regiones de una calle son buenas para reconocer y cómo afecta el cambio de perspectiva.")
    r.caja("<strong>Sobre las imágenes.</strong> No se utilizó Google Street View: las imágenes de Google tienen derechos de autor y condiciones de uso que no permiten descargarlas ni procesarlas de forma automatizada. "
           "Se empleó en su lugar una fotografía de calle tomada desde un vehículo (Rivera, 2025; licencia CC0), que tiene el mismo tipo de contenido (edificios, señales, cables, vehículos), y una <strong>segunda vista con un ligero cambio de perspectiva</strong> generada con la geometría exacta de un giro de la cámara "
           "(7° de yaw, −2° de pitch y 1.5° de roll, más un 5 % de pérdida de brillo y algo de ruido). Al ser exacta la transformación, cada correspondencia se puede calificar como correcta o errónea. "
           "Es una simplificación: un giro puro no produce paralaje, cosa que sí ocurre cuando el vehículo avanza entre dos tomas reales. Se recortó la franja inferior de la foto, donde la cámara imprime fecha y ubicación.", aviso=True)

    r.h2("2. Parte 1. Exploración visual")
    r.p("La escena elegida es un cruce urbano con edificios, semáforos, letreros comerciales, postes con cableado, vehículos y una zona de vegetación. Contiene estructuras con esquinas marcadas (ventanas, letreros, semáforos), líneas largas (cables, postes, marcas viales), "
        "superficies casi lisas (cielo) y texturas irregulares (césped seco, ramas). Las dos vistas consecutivas se muestran en la figura 1.")
    r.fig("figuras/fig1_vistas.png", "Las dos vistas consecutivas. La segunda tiene un ligero cambio de perspectiva (bordes negros: zona que la cámara ya no ve). Foto original: M. Rivera, CC0.")

    r.h2("3. Parte 2. Identificación de keypoints")
    r.h3("Observación visual")
    r.ul([
        "<strong>Esquinas:</strong> los semáforos y sus soportes, los letreros rectangulares, los bordes de las ventanas del edificio de la izquierda, las esquinas de los vehículos y el cruce de los cables con el poste.",
        "<strong>Bordes:</strong> el poste, los cables (líneas largas y finas), las marcas de la calzada y el bordillo. Los bordes largos son útiles para localizar, pero a lo largo de una línea recta hay ambigüedad (un punto del cable se parece a otro).",
        "<strong>Patrones repetitivos:</strong> las franjas del paso peatonal, las tejas del techo y los bloques de la fachada.",
        "<strong>Regiones con textura:</strong> la vegetación y el césped seco, las ramas sin hojas y la zona de comercios al fondo.",
    ])
    r.fig("figuras/fig2_keypoints_regiones.png", "Izquierda: puntos clave SIFT sobre la vista 1 (se dibujan 800). Centro: número de puntos clave SIFT por región (cuadrícula de 4×6). Derecha: correspondencias erróneas por región.", alto=105)
    r.h3("¿Qué regiones parecen más fáciles de reconocer?")
    r.p(f"Las de <strong>estructuras con esquinas y contraste</strong> a la altura de la calle: la mitad inferior de la imagen concentra la gran mayoría de los puntos clave. En la fila inferior de la cuadrícula hay entre {min(RG['keypoints'][3])} y {max(RG['keypoints'][3])} puntos por celda y "
        f"entre {min(RG['correctos'][3][:5])} y {max(RG['correctos'][3][:5])} correspondencias correctas por celda (en las cinco primeras columnas). También destaca una franja vertical que coincide con el <strong>poste y los cables</strong> (celdas de las dos filas superiores: {RG['keypoints'][0][3]} y {RG['keypoints'][1][3]} puntos), "
        f"con {RG['correctos'][0][3]} y {RG['correctos'][1][3]} correspondencias correctas.")
    r.h3("¿Qué zonas generarían más keypoints?")
    r.p(f"Las de <strong>mucho detalle a corta distancia</strong>: la fila inferior acumula {sum(RG['keypoints'][3])} de los {sum(map(sum, RG['keypoints']))} puntos clave de SIFT (el {100 * sum(RG['keypoints'][3]) / sum(map(sum, RG['keypoints'])):.0f} %), y la celda con más puntos ({max(RG['keypoints'][3])}) está en esa fila. En cambio el <strong>cielo</strong>, casi uniforme, prácticamente no produce puntos: "
        f"las diez celdas de cielo de las dos filas superiores (sin el poste) suman {sum(RG['keypoints'][0][:3]) + sum(RG['keypoints'][0][4:]) + sum(RG['keypoints'][1][:3]) + sum(RG['keypoints'][1][4:])} puntos.")
    r.h3("¿Qué regiones podrían causar problemas al algoritmo?")
    r.ul([
        "<strong>Zonas lisas</strong> (cielo, paredes sin textura, asfalto): no hay puntos que reconocer, por lo que no hay correspondencias.",
        f"<strong>Texturas irregulares y repetitivas</strong> (vegetación, césped seco): generan muchos puntos, pero con poco poder de identificación. En la celda inferior derecha, la de césped y vegetación, SIFT detectó {RG['keypoints'][3][5]} puntos, pero solo {RG['correctos'][3][5]} correspondencias correctas y {RG['erroneos'][3][5]} erróneas; en la celda de arriba ({RG['keypoints'][2][5]} puntos) solo {RG['correctos'][2][5]} correspondencias correctas.",
        f"<strong>Patrones repetidos</strong> (franjas del paso peatonal, tejas): dos partes distintas se parecen entre sí y el emparejamiento se equivoca. Las celdas con más correspondencias erróneas están en la fila inferior (hasta {max(RG['erroneos'][3])} por celda) y en la columna del poste ({RG['erroneos'][0][3]}–{RG['erroneos'][2][3]}), donde probablemente los cables paralelos se confunden entre sí.",
        "<strong>Objetos móviles</strong> (vehículos): si se mueven entre las dos tomas, sus puntos no cumplen la misma transformación que el resto de la escena y hay que descartarlos, cosa que hace RANSAC.",
    ])

    r.h2("4. Parte 3. Aplicación práctica: Harris, ORB y SIFT")
    r.p("Se detectaron hasta 2000 puntos por imagen con cada método, se emparejaron con la prueba de razón de Lowe (0.8) y cada correspondencia se calificó con la homografía verdadera (correcta si el punto proyectado cae a ≤ 3 px del emparejado). "
        "«Harris + parche» usa las esquinas de Harris y como descriptor el parche de 17×17 píxeles normalizado.")
    r.fig("figuras/fig3_correspondencias.png", "Correspondencias entre la vista 1 (izquierda) y la vista 2 (derecha) con Harris + parche, ORB y SIFT. Verde: correcta según la verdad; rojo: errónea (no aparece ninguna entre las 70 mejores).", alto=185)
    r.tabla(["Método", "Puntos en la vista 1", "Repetibilidad del detector", "Correspondencias", "Correctas", "Precisión", "Error de la homografía (px)"],
            [[NOM[m], str(R[m]["kp1"]), f"{R[m]['repetibilidad'] * 100:.1f} %", str(R[m]["matches"]), str(R[m]["correctos"]), f"{R[m]['precision'] * 100:.1f} %", f"{R[m]['error_H_px']:.2f}"] for m in ("harris", "orb", "sift")],
            "Resultados con el cambio de perspectiva ligero. La homografía se estima con RANSAC a partir de las correspondencias.", num=(1, 2, 3, 4, 5, 6))
    r.p(f"<strong>Resultados.</strong> Los tres métodos funcionan muy bien con un cambio de perspectiva ligero: entre {R['orb']['precision'] * 100:.1f} % y {R['harris']['precision'] * 100:.1f} % de las correspondencias son correctas. "
        f"Harris + parche produce menos correspondencias ({R['harris']['matches']}) pero casi todas correctas ({R['harris']['precision'] * 100:.1f} %), porque con una diferencia de perspectiva pequeña el parche de píxeles casi no cambia; ORB y SIFT dan entre {R['sift']['matches']} y {R['orb']['matches']} correspondencias. "
        f"SIFT es el más preciso geométricamente (error de la homografía de {R['sift']['error_H_px']:.2f} px), Harris queda cerca ({R['harris']['error_H_px']:.2f} px) y ORB algo por detrás ({R['orb']['error_H_px']:.2f} px). "
        "La prueba de RANSAC confirmó prácticamente todas las correspondencias correctas y descartó las erróneas.")

    r.h2("5. Parte 4. Reflexión")
    r.h3("¿Cómo puede Google Street View relacionar imágenes consecutivas?")
    r.p("Las imágenes de un recorrido se capturan a lo largo de una ruta y comparten buena parte de la escena. Si se detectan puntos clave en cada imagen, se describen con un vector numérico y se emparejan, los pares correctos indican qué puntos de la escena aparecen en ambas. "
        "Con suficientes pares se puede estimar la transformación geométrica entre las dos imágenes (aquí, una homografía con RANSAC) y, con ella, alinearlas o calcular el movimiento de la cámara. "
        "Este mismo principio es la base de la reconstrucción a partir de colecciones de fotografías (Snavely et al., 2006).")
    r.h3("¿Cuál es el papel de los keypoints?")
    r.p("Son los «anclajes» que permiten comparar imágenes sin comparar todos sus píxeles. Un buen punto clave se puede localizar con precisión y reencontrar en otra imagen (repetibilidad) y tiene un descriptor que lo distingue de los demás. "
        f"En los experimentos, la repetibilidad fue de {R['harris']['repetibilidad'] * 100:.0f} %, {R['orb']['repetibilidad'] * 100:.0f} % y {R['sift']['repetibilidad'] * 100:.0f} % (Harris, ORB y SIFT), y de ahí salieron cientos de correspondencias correctas.")
    r.h3("¿Por qué las esquinas y las texturas son importantes?")
    r.p("En un borde recto solo se sabe la posición en la dirección perpendicular a él (el «problema de apertura»): el punto puede deslizarse a lo largo de la línea sin que la imagen local cambie. En una <strong>esquina</strong> el gradiente cambia en dos direcciones y la posición queda fijada (Harris & Stephens, 1988). "
        "La <strong>textura</strong> con variaciones no repetitivas hace que el descriptor sea distinguible; sin textura (cielo) no hay nada que reconocer, y con textura demasiado repetitiva (césped, franjas) hay muchas coincidencias falsas. La figura 2 lo ilustra: mucha densidad de puntos no garantiza correspondencias fiables.")
    r.h3("¿Cómo afecta el cambio de perspectiva?")
    r.fig("figuras/fig4_cambio_perspectiva.png", "Efecto del giro de la cámara (yaw) sobre el número de correspondencias correctas, su precisión y la repetibilidad del detector.")
    filas = []
    for i, y in enumerate([b["yaw"] for b in B["sift"]]):
        filas.append([f"{y}°"] + [f"{B[m][i]['correctos']} ({B[m][i]['precision'] * 100:.0f} %)" for m in ("harris", "orb", "sift")])
    r.tabla(["Giro (yaw)", "Harris + parche", "ORB", "SIFT"], filas, "Correspondencias correctas (y precisión) según el giro de la cámara.", num=(1, 2, 3))
    r.p(f"Al aumentar el giro el número de correspondencias correctas cae con los tres métodos: de {B['harris'][0]['correctos']} a {B['harris'][-1]['correctos']} con Harris, de {B['orb'][0]['correctos']} a {B['orb'][-1]['correctos']} con ORB y de {B['sift'][0]['correctos']} a {B['sift'][-1]['correctos']} con SIFT (entre 2° y 40°). "
        f"La precisión, en cambio, baja mucho menos: de {B['sift'][0]['precision'] * 100:.0f} % a {B['sift'][-1]['precision'] * 100:.0f} % en SIFT y de {B['orb'][0]['precision'] * 100:.0f} % a {B['orb'][-1]['precision'] * 100:.0f} % en ORB. "
        "Esa caída tiene <strong>dos causas que aquí no se separan</strong>: la deformación de los descriptores por la perspectiva y la reducción del área común entre las dos tomas (con un ángulo de visión aproximado de 70° y un giro de 40°, solo se comparte poco menos de la mitad de la escena). "
        "En un recorrido de este tipo, las imágenes consecutivas se toman a poca distancia unas de otras, con poca diferencia de perspectiva, lo que favorece el emparejamiento.")

    r.h2("6. Conclusión")
    r.p("Los puntos clave permiten relacionar imágenes consecutivas de forma robusta si hay estructuras con esquinas y textura distinguible: los tres métodos probados lograron entre 94 % y 99 % de correspondencias correctas con un cambio de perspectiva ligero, y el número de correspondencias fiables baja de forma gradual al aumentar el giro. "
        "Las zonas lisas no ofrecen información, y las texturas repetitivas ofrecen muchos puntos pero pocas correspondencias fiables. Como limitación, la comparación se hizo con vistas sintéticas de una sola escena, sin paralaje ni objetos en movimiento; con imágenes reales de un recorrido habría más errores por vehículos, sombras y cambios de iluminación.")

    r.refs([
        "Bradski, G. (2000). The OpenCV library. <em>Dr. Dobb’s Journal of Software Tools, 25</em>(11), 120–123.",
        "Fischler, M. A., &amp; Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. <em>Communications of the ACM, 24</em>(6), 381–395. https://doi.org/10.1145/358669.358692",
        "Harris, C., &amp; Stephens, M. (1988). A combined corner and edge detector. En <em>Proceedings of the 4th Alvey Vision Conference</em> (pp. 147–151). Alvey Vision Club.",
        "Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. <em>International Journal of Computer Vision, 60</em>(2), 91–110. https://doi.org/10.1023/B:VISI.0000029664.99615.94",
        "Rivera, M. (2025). <em>NB N Ashley St at Connell Rd intersection dashcam</em> [Fotografía; CC0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:NB_N_Ashley_St_at_Connell_Rd_intersection_dashcam.jpg",
        "Rublee, E., Rabaud, V., Konolige, K., &amp; Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. En <em>2011 International Conference on Computer Vision</em> (pp. 2564–2571). IEEE. https://doi.org/10.1109/ICCV.2011.6126544",
        "Snavely, N., Seitz, S. M., &amp; Szeliski, R. (2006). Photo tourism: Exploring photo collections in 3D. <em>ACM Transactions on Graphics, 25</em>(3), 835–846. https://doi.org/10.1145/1141911.1141964",
        "Szeliski, R. (2022). <em>Computer vision: Algorithms and applications</em> (2.ª ed.). Springer. https://szeliski.org/Book/",
    ])
    pdf = r.guardar("Street_View_keypoints")
    n = zip_entrega(AQUI / "Street_View_keypoints.zip", AQUI, ["experimentos.py", "informe.py", "puntos.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
