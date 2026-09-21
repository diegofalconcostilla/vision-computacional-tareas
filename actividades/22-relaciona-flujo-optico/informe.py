"""Informe PDF: Flujo óptico sparse y denso para analizar movimiento cardíaco."""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
V, SP, DN, AC = R["video"], R["sparse"], R["dense"], R["acuerdo"]
VA, VR, FV = R["validacion_amplitud"], R["validacion_ruido"], R["farneback_ventana"]
cociente = DN["tiempo_ms_por_par"] / SP["tiempo_ms_por_par"]


def main():
    r = Reporte("Flujo óptico sparse y denso para analizar el movimiento cardíaco",
                "Lucas-Kanade y Farnebäck sobre una cine de resonancia magnética de cuatro cámaras", carpeta=AQUI,
                entrega="PDF con capturas y análisis + archivo comprimido con el código y los videos")
    r.portada()

    r.h2("1. Introducción")
    r.p("El <strong>flujo óptico</strong> es el campo de movimiento aparente entre dos imágenes consecutivas de una secuencia: para cada punto, un vector que indica hacia dónde se desplazó. "
        "En imagenología médica permite estudiar el movimiento del corazón (contracción y relajación de las paredes), la respiración o el desplazamiento de tejidos, con aplicaciones en la detección de zonas del miocardio que se mueven menos de lo normal (Ledesma-Carbayo et al., 2005). "
        "Este trabajo aplica los dos enfoques clásicos a una cine de resonancia magnética cardíaca: <strong>flujo sparse</strong> con Lucas-Kanade, que rastrea unos pocos puntos, y <strong>flujo denso</strong> con Farnebäck, que estima el movimiento en todos los píxeles. "
        "Además de compararlos sobre el video real, donde no se conoce el movimiento verdadero, se les midió el error sobre un campo de movimiento <em>conocido</em> aplicado a un cuadro real.")

    r.h2("2. Parte 1. Investigación previa")
    r.h3("¿Qué es el flujo óptico?")
    r.p("Se parte de la hipótesis de <em>constancia de brillo</em>: un punto conserva su intensidad al moverse, I(x + u, y + v, t + 1) = I(x, y, t). Expandiendo en serie de Taylor se obtiene la <em>ecuación de restricción del flujo óptico</em>, I<sub>x</sub>u + I<sub>y</sub>v + I<sub>t</sub> = 0. "
        "Es una sola ecuación con dos incógnitas (u, v): solo se conoce la componente del movimiento en la dirección del gradiente (problema de apertura), por lo que todo método añade una suposición adicional (Horn & Schunck, 1981).")
    r.h3("Diferencias entre flujo óptico sparse y denso")
    r.tabla(["", "Sparse (disperso)", "Denso"],
            [["Qué estima", "El movimiento de un conjunto pequeño de puntos con textura (esquinas).", "Un vector de movimiento para casi todos los píxeles."],
             ["Salida", "Trayectorias de puntos.", "Un campo (u, v) del tamaño de la imagen."],
             ["Costo", "Bajo: solo se procesan los puntos.", "Alto: proporcional al número de píxeles."],
             ["Zonas sin textura", "Se evitan al elegir los puntos.", "Hay que «rellenarlas» con regularización o suavidad."],
             ["Uso típico", "Seguimiento de objetos, odometría visual, estabilización.", "Análisis de movimiento global, deformación de tejidos, segmentación por movimiento."]],
            "Comparación conceptual.")
    r.h3("Método de Lucas-Kanade")
    r.p("Supone que el movimiento es <strong>constante dentro de una ventana pequeña</strong> alrededor de cada punto (aquí 21×21 px), lo que da un sistema sobredeterminado de ecuaciones que se resuelve por mínimos cuadrados (Lucas & Kanade, 1981). "
        "Solo funciona bien donde la matriz de gradientes de la ventana está bien condicionada, es decir, en esquinas; por eso los puntos se eligen con el criterio de Shi y Tomasi (1994). "
        "Para movimientos grandes se usa una <strong>pirámide de imágenes</strong> (Bouguet, 2001): se estima en una versión reducida y se refina en las resoluciones mayores. En este trabajo se usaron 4 niveles y una <em>verificación adelante-atrás</em> (se rastrea de un cuadro al siguiente y de vuelta; si el punto no regresa a menos de 1 px de su origen, se descarta).")
    r.h3("Método de Farnebäck")
    r.p("Aproxima la vecindad de cada píxel, en cada cuadro, con un <strong>polinomio cuadrático</strong> (expansión polinómica). Al comparar los coeficientes de los dos cuadros se despeja el desplazamiento, y se resuelve de forma piramidal, de menor a mayor resolución (Farnebäck, 2003). "
        "Sus parámetros principales son el tamaño de la ventana de promediado (<code>winsize</code>), el número de niveles de la pirámide y el suavizado de la expansión. Se usó la implementación de OpenCV con winsize = 15, 3 niveles y 3 iteraciones.")
    r.h3("Aplicaciones biomédicas del análisis de movimiento en video")
    r.ul(["<strong>Cardiología:</strong> movimiento y deformación (<em>strain</em>) de las paredes del ventrículo en ecocardiografía y resonancia cine; detección de zonas con movimiento reducido después de un infarto (Ledesma-Carbayo et al., 2005).",
          "<strong>Respiración:</strong> seguimiento del desplazamiento de órganos para planear la radioterapia o corregir imágenes.",
          "<strong>Flujo sanguíneo y ultrasonido Doppler:</strong> velocidad del flujo a partir de secuencias de imágenes.",
          "<strong>Microscopía:</strong> movilidad de células y organelos; y <strong>endoscopía:</strong> seguimiento de tejidos y de instrumentos."])

    r.h2("3. Parte 2. Selección del video")
    r.p(f"Se eligió una <strong>cine de resonancia magnética cardíaca en vista de cuatro cámaras</strong>, de un paciente con un infarto inferior crónico (Doregan, s. f.; CC BY-SA 4.0): {V['cuadros']} cuadros de {V['ancho']}×{V['alto']} px a 25 cuadros/s, en los que se ven las aurículas, los ventrículos y sus paredes en movimiento. "
        "Aparenta cubrir <strong>un ciclo cardíaco completo</strong>: las posiciones de los puntos rastreados regresan a su origen al final (ver la sección 4). Es un movimiento continuo y periódico, con la ventaja de que sirve para verificar la coherencia de los métodos (si el ciclo se cierra, la trayectoria debe volver al inicio).")
    r.fig("figuras/fig1_cuadros.png", "Cuadros 0, 12, 25 y 37 de la cine. En el primero se marcan los puntos de Shi-Tomasi que rastrea Lucas-Kanade.", alto=75)

    r.h2("4. Parte 3. Implementación")
    r.h3("Flujo óptico sparse (Lucas-Kanade)")
    r.p(f"Se detectaron {R['sparse_puntos_iniciales']} esquinas en el primer cuadro y se rastrearon cuadro a cuadro durante todo el ciclo. Con la verificación adelante-atrás sobrevivieron <strong>{SP['supervivientes_ultimo']} de {R['sparse_puntos_iniciales']} puntos</strong> ({100 * SP['supervivientes_ultimo'] / R['sparse_puntos_iniciales']:.0f} %); los demás se perdieron por salir de la zona con textura o por ambigüedad. "
        f"Como el video es un ciclo, cada punto debería terminar donde empezó: la <strong>mediana de la distancia entre la posición final y la inicial fue de {SP['deriva_mediana_px']} px</strong> (media {SP['deriva_media_px']} px, con unos pocos puntos con hasta 16 px de error, que corresponden a estructuras con poca textura). "
        f"El desplazamiento máximo mediano de un punto respecto de su inicio fue de {SP['desplazamiento_max_mediana_px']} px, y el percentil 95 de {SP['desplazamiento_max_p95_px']} px: la mayoría de los puntos, sobre estructuras estáticas o casi estáticas (pared torácica, vasos), casi no se mueve, y los que más se mueven están sobre las paredes de las cavidades cardíacas.")
    r.fig("figuras/fig2_sparse.png", "Izquierda: trayectorias de Lucas-Kanade durante el ciclo (color = desplazamiento máximo). Centro: puntos que sobreviven a la verificación adelante-atrás. Derecha: distancia entre posición final e inicial, que debería ser cero si el ciclo se cierra.", alto=70)
    r.h3("Flujo óptico denso (Farnebäck)")
    r.p(f"Se calculó el flujo entre cada par de cuadros consecutivos ({V['cuadros'] - 1} campos de {V['ancho']}×{V['alto']} vectores). La magnitud media del movimiento en toda la imagen es de solo {DN['mag_media_global_px']} px/cuadro, pero sube a {DN['mag_media_roi_px']} px/cuadro en la región de mayor movimiento (el {DN['roi_pct']} % de la imagen, definida como el percentil 60 superior de la magnitud media), "
        f"y el percentil 99 de la magnitud es {DN['mag_max_p99_px']} px/cuadro. La figura 3 muestra el color (dirección) y el brillo (magnitud) del campo, y los vectores sobre la imagen.")
    r.fig("figuras/fig3_dense.png", "Flujo denso de Farnebäck en cuatro instantes del ciclo. Arriba: dirección (color) y magnitud (brillo). Abajo: vectores (flechas ampliadas ×4). El movimiento se concentra en la pared y las cavidades del corazón; en el fondo casi no hay movimiento (los puntos de color oscuro son ruido de baja magnitud).", alto=150)
    r.fig("figuras/fig4_comparacion.png", "Izquierda: magnitud media del flujo denso a lo largo del ciclo. Centro: región de mayor movimiento. Derecha: acuerdo entre el desplazamiento de Lucas-Kanade y el del flujo denso integrado.", alto=70)
    r.p(f"La curva de magnitud no es constante: oscila entre {DN['curva_roi_min']} y {DN['curva_roi_max']} px/cuadro en la región de mayor movimiento, con un mínimo en el cuadro {DN['cuadro_min_movimiento']} y un máximo en el {DN['cuadro_max_movimiento']}. "
        "Esa variación es coherente con las fases del ciclo (fases de movimiento rápido y fases de reposo relativo), pero el video no trae electrocardiograma, por lo que <strong>no se asignan fases (sístole, diástole) a estos cuadros</strong>.")

    r.h2("5. Parte 4. Comparación de resultados")
    r.p("<strong>Videos generados</strong> (en <code>datos/</code>, dentro del archivo comprimido): <code>video_original.mp4</code>, <code>video_sparse_lk.mp4</code> (trayectorias de los últimos 12 cuadros) y <code>video_denso_farneback.mp4</code> (mapa de color superpuesto). "
        "Capturas del cuadro 25: <code>captura_sparse.png</code> y <code>captura_denso.png</code>.")
    r.h3("Acuerdo entre ambos métodos sobre el video real")
    r.p(f"No se conoce el movimiento verdadero del video, así que se comparó un método contra el otro: se integró el flujo denso a lo largo de {AC['cuadro']} cuadros a partir de los mismos puntos y se comparó con la trayectoria de Lucas-Kanade. "
        f"Con {AC['puntos']} puntos, el desplazamiento medio fue de {AC['movimiento_medio_px']} px, la diferencia media entre las posiciones de ambos métodos de {AC['dif_media_px']} px (mediana {AC['dif_mediana_px']} px) y la correlación entre los desplazamientos de <strong>{AC['correlacion_desplazamientos']}</strong>. "
        "Ambos métodos coinciden bien, pero que coincidan no prueba que acierten (podrían equivocarse igual), lo que motiva la validación siguiente.")
    r.h3("Validación con un movimiento conocido")
    r.p("Se aplicó al primer cuadro un campo de movimiento verdadero (una mezcla suave de contracción radial, rotación y traslación, con amplitud regulable) y se midió el <strong>error de punto final</strong> (EPE: distancia entre el vector estimado y el verdadero). "
        "Lucas-Kanade se evalúa en los puntos que rastrea; Farnebäck se evalúa en <em>esos mismos puntos</em> (comparación justa) y en toda la imagen.")
    r.fig("figuras/fig5_validacion.png", "Error de punto final frente a un flujo verdadero conocido. Izquierda: según la amplitud del movimiento. Centro: según el ruido añadido. Derecha: error de Farnebäck por píxel.", alto=70)
    filas = [[f"{b['mov_max']:.1f}", f"{b['lk_epe_media']:.3f}", f"{b['lk_puntos_ok']}/{b['lk_puntos']}", f"{b['fb_epe_mismos_puntos']:.3f}", f"{b['fb_epe_imagen']:.3f}", f"{b['fb_epe_p90']:.2f}"] for b in VA]
    r.tabla(["Movimiento máx. (px)", "LK: EPE (px)", "LK: puntos válidos", "Farnebäck: EPE en esos puntos", "Farnebäck: EPE en toda la imagen", "Farnebäck: percentil 90"], filas,
            "Error según la amplitud del movimiento (sin ruido).", num=(0, 1, 3, 4, 5))
    filas = [[str(b["ruido"]), f"{b['lk_epe_media']:.3f}", f"{b['fb_epe_mismos_puntos']:.3f}", f"{b['fb_epe_imagen']:.3f}"] for b in VR]
    r.tabla(["Ruido añadido (σ)", "LK: EPE (px)", "Farnebäck: EPE en los mismos puntos", "Farnebäck: EPE en toda la imagen"], filas, "Error según el ruido (movimiento máximo de 2.9 px).", num=(0, 1, 2, 3))
    a16, a32 = FV[0], FV[1]
    r.p(f"<strong>Resultados.</strong> (1) <strong>En los puntos con textura, ambos métodos son igual de precisos</strong> con movimientos pequeños o moderados: hasta 11.8 px de movimiento máximo los errores son de entre {VA[0]['lk_epe_media']:.2f} y {VA[4]['lk_epe_media']:.2f} px, y las diferencias entre ellos son de centésimas de píxel. "
        f"(2) <strong>El error de Farnebäck en toda la imagen es mayor</strong> ({VA[2]['fb_epe_imagen']:.2f} px con movimiento máximo de {VA[2]['mov_max']} px frente a {VA[2]['fb_epe_mismos_puntos']:.2f} px en los puntos de Lucas-Kanade), porque incluye zonas sin textura donde el movimiento no puede determinarse, sobre todo en las esquinas de la imagen, donde el movimiento es mayor y el fondo es oscuro y uniforme (figura 5, derecha). "
        f"(3) <strong>Con movimientos grandes se separan:</strong> con {VA[6]['mov_max']:.0f} px de movimiento máximo, Lucas-Kanade sigue rastreando ({VA[6]['lk_puntos_ok']} de {VA[6]['lk_puntos']} puntos, error de {VA[6]['lk_epe_media']:.2f} px), pero Farnebäck con su configuración base falla ({VA[6]['fb_epe_mismos_puntos']:.1f} px en los mismos puntos y {VA[6]['fb_epe_imagen']:.1f} px en la imagen). "
        f"Parte de ese fallo es de configuración: con una ventana mayor (winsize = 31 o 51) el error en la imagen baja de {a32['base (winsize 15)']:.1f} px a {a32['winsize 31']:.1f} y {a32['winsize 51']:.1f} px, pero sigue siendo grande. "
        f"(4) <strong>Ambos se degradan con el ruido</strong> de forma casi lineal, con errores de {VR[4]['lk_epe_media']:.2f} px (Lucas-Kanade) y {VR[4]['fb_epe_mismos_puntos']:.2f} px (Farnebäck, en los mismos puntos) con σ = 30, frente a {VR[0]['lk_epe_media']:.2f} y {VR[0]['fb_epe_mismos_puntos']:.2f} px sin ruido.")
    r.caja("<strong>Nota sobre esta validación.</strong> El movimiento sintético es suave y global, y las medidas se hicieron sobre un solo cuadro. El movimiento real del corazón es más complejo (deformaciones locales, zonas que se mueven en direcciones distintas), por lo que los errores del video real pueden ser mayores. "
           "Se sabe además que la validación en secuencias reales necesita conjuntos de referencia diseñados para ese fin (Baker et al., 2011).")

    r.h3("Preguntas de la actividad")
    r.ul([
        f"<strong>¿Qué diferencias se observaron entre ambos métodos?</strong> Lucas-Kanade produce trayectorias de unos cientos de puntos y no dice nada de las zonas sin textura; Farnebäck produce un vector por píxel, incluyendo zonas donde el movimiento no se puede determinar (el fondo casi negro), donde la estimación es poco fiable. En las zonas con textura ambos coinciden ({AC['correlacion_desplazamientos']} de correlación en el video real).",
        "<strong>¿Cuál detectó mejor el movimiento global?</strong> El <strong>denso</strong>: entrega un mapa de todo el corazón en un solo cuadro (figura 3), que permite ver <em>qué</em> regiones se mueven y hacia dónde, y calcular curvas globales como la magnitud a lo largo del ciclo. Los puntos de Lucas-Kanade solo cubren las estructuras con esquinas y dejan sin información amplias zonas del músculo cardíaco, de textura suave.",
        f"<strong>¿Cuál tuvo menor costo computacional?</strong> <strong>Lucas-Kanade</strong>: {SP['tiempo_ms_por_par']} ms por par de cuadros (con 200 puntos y la verificación adelante-atrás, es decir, dos rastreos) frente a {DN['tiempo_ms_por_par']} ms de Farnebäck, unas <strong>{cociente:.0f} veces</strong> menos. Los tiempos se midieron en un equipo, con una imagen de {V['ancho']}×{V['alto']} px; el cociente depende de la resolución y del número de puntos.",
        "<strong>¿Qué ventajas ofrece cada técnica?</strong> <em>Lucas-Kanade</em>: rápido, preciso en puntos con textura, robusto a movimientos grandes gracias a la pirámide (en esta validación, el mejor con 47 px de movimiento), y permite verificar la calidad de cada trayectoria (adelante-atrás). <em>Farnebäck</em>: cobertura completa, ideal para mapas de movimiento, deformación y segmentación por movimiento; no requiere elegir puntos.",
        "<strong>¿Qué limitaciones se encontraron?</strong> Ambos suponen <strong>constancia del brillo</strong> (en resonancia el brillo cambia con la fase del ciclo, con efectos de flujo sanguíneo) y movimientos pequeños entre cuadros; sufren el <strong>problema de apertura</strong> en bordes rectos y no funcionan en zonas sin textura. Lucas-Kanade pierde puntos (11 de 200 en este video) y acumula deriva; Farnebäck es sensible a su ventana y falla con movimientos grandes; los dos se degradan con el ruido. "
        "Además, el movimiento medido es el de la <em>imagen</em> en el plano del corte: el corazón también se mueve perpendicular a él, y ese movimiento aparece como cambio de apariencia, no como desplazamiento.",
        "<strong>¿Cómo podrían usarse en aplicaciones médicas reales?</strong> Como herramienta de análisis del movimiento cardíaco: cuantificar cuánto y en qué dirección se mueve cada segmento de la pared del ventrículo, comparar el movimiento entre un corazón sano y uno con un infarto (aquí el paciente tiene un infarto inferior crónico), estimar la deformación del miocardio y apoyar el diagnóstico; en radioterapia, seguir el movimiento respiratorio de un tumor; y en cirugía, seguir tejidos y herramientas. "
        "En la práctica clínica, el resultado se entrega como una ayuda a la persona especialista, no como diagnóstico automático.",
    ])

    r.h2("6. Parte 5. Conclusión")
    r.p("<strong>Importancia del análisis de movimiento en visión por computadora.</strong> Muchos fenómenos solo se entienden en el tiempo: una imagen fija del corazón no dice cómo se mueve. El análisis de movimiento permite seguir objetos, estimar velocidades, compensar el movimiento de la cámara y detectar cambios, y es la base de aplicaciones tan distintas como los vehículos autónomos y la cardiología.")
    r.p("<strong>Papel del flujo óptico en imagenología médica.</strong> Convierte una secuencia de imágenes en <em>medidas</em> del movimiento de los tejidos. En este trabajo permitió seguir puntos de las paredes cardíacas (el ciclo se cerró con una mediana de 0.5 px) y visualizar el movimiento del corazón en todo el plano del corte.")
    r.p("<strong>Ventajas y desafíos de analizar información temporal.</strong> Ventajas: más información que en una imagen aislada y redundancia entre cuadros para reducir el ruido. Desafíos: mayor volumen de datos y cómputo, cambios de brillo entre cuadros, ruido acumulado (la deriva de las trayectorias) y la ausencia de una verdad de referencia en datos reales, que obliga a validar con movimientos conocidos, como se hizo aquí.")
    r.p("<strong>Combinación con inteligencia artificial y aprendizaje profundo.</strong> Las redes neuronales estiman el flujo con más precisión y con movimientos grandes (Dosovitskiy et al., 2015; Teed & Deng, 2020) y se han aplicado a resonancias cardíacas para estimar el movimiento y segmentar a la vez, pero necesitan grandes conjuntos de datos. Los métodos clásicos siguen siendo útiles como referencia, como inicialización y para generar datos de entrenamiento; una arquitectura razonable combinaría ambos: flujo clásico para una estimación rápida y una red para refinarla y para interpretar el resultado (por ejemplo, clasificar el movimiento de un segmento como normal o alterado).")
    r.p("<strong>Limitaciones del trabajo:</strong> un solo paciente y un solo video; sin electrocardiograma para asignar las fases del ciclo; validación con un movimiento sintético suave; y parámetros de los métodos fijados sin una búsqueda exhaustiva.")

    r.refs([
        "Baker, S., Scharstein, D., Lewis, J. P., Roth, S., Black, M. J., &amp; Szeliski, R. (2011). A database and evaluation methodology for optical flow. <em>International Journal of Computer Vision, 92</em>(1), 1–31. https://doi.org/10.1007/s11263-010-0390-2",
        "Bouguet, J.-Y. (2001). <em>Pyramidal implementation of the affine Lucas Kanade feature tracker: Description of the algorithm</em> [Informe técnico]. Intel Corporation.",
        "Doregan. (s. f.). <em>4CH cine infarct</em> [Animación; CC BY-SA 4.0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:4CH_cine_infarct.gif",
        "Dosovitskiy, A., Fischer, P., Ilg, E., Häusser, P., Hazirbas, C., Golkov, V., van der Smagt, P., Cremers, D., &amp; Brox, T. (2015). FlowNet: Learning optical flow with convolutional networks. En <em>2015 IEEE International Conference on Computer Vision (ICCV)</em> (pp. 2758–2766). IEEE. https://doi.org/10.1109/ICCV.2015.316",
        "Farnebäck, G. (2003). Two-frame motion estimation based on polynomial expansion. En J. Bigun &amp; T. Gustavsson (Eds.), <em>Image Analysis: 13th Scandinavian Conference, SCIA 2003</em> (pp. 363–370). Springer. https://doi.org/10.1007/3-540-45103-X_50",
        "Horn, B. K. P., &amp; Schunck, B. G. (1981). Determining optical flow. <em>Artificial Intelligence, 17</em>(1–3), 185–203. https://doi.org/10.1016/0004-3702(81)90024-2",
        "Ledesma-Carbayo, M. J., Kybic, J., Desco, M., Santos, A., Sühling, M., Hunziker, P., &amp; Unser, M. (2005). Spatio-temporal nonrigid registration for ultrasound cardiac motion estimation. <em>IEEE Transactions on Medical Imaging, 24</em>(9), 1113–1126. https://doi.org/10.1109/TMI.2005.852050",
        "Lucas, B. D., &amp; Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. En <em>Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI)</em> (pp. 674–679).",
        "OpenAI. (2026). <em>Análisis de movimiento con flujo óptico</em> [Imagen generada con IA]. Material de la actividad, Visión computacional para imágenes y video. ChatGPT.",
        "Shi, J., &amp; Tomasi, C. (1994). Good features to track. En <em>1994 Proceedings of IEEE Conference on Computer Vision and Pattern Recognition</em> (pp. 593–600). IEEE. https://doi.org/10.1109/CVPR.1994.323794",
        "Teed, Z., &amp; Deng, J. (2020). RAFT: Recurrent all-pairs field transforms for optical flow. En A. Vedaldi, H. Bischof, T. Brox, &amp; J.-M. Frahm (Eds.), <em>Computer Vision – ECCV 2020</em> (pp. 402–419). Springer. https://doi.org/10.1007/978-3-030-58536-5_24",
    ])
    pdf = r.guardar("Flujo_optico_movimiento_cardiaco")
    n = zip_entrega(AQUI / "Flujo_optico_movimiento_cardiaco.zip", AQUI, ["experimentos.py", "informe.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
