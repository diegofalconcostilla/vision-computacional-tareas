"""Informe PDF: Estabilización digital utilizando procesamiento en frecuencia."""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from herramientas.reporte import Reporte, paginas_pdf, zip_entrega  # noqa: E402

R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
A, W, B, BF = R["A_estimacion"], R["A_restauracion"], R["B"], R["B_barrido_fc"]
K = W["wiener_K"]


def main():
    r = Reporte("Estabilización digital con procesamiento en frecuencia",
                "Espectro de Fourier, restauración de imágenes movidas y estabilización de video vibrado", carpeta=AQUI,
                entrega="PDF con imágenes, espectros y análisis + archivo comprimido con el código y los videos")
    r.portada()

    r.h2("1. Introducción")
    r.p("Una cámara en movimiento produce dos tipos de problemas: dentro de <em>un</em> cuadro, el desplazamiento durante la exposición <strong>desenfoca</strong> la imagen; entre <em>varios</em> cuadros, las sacudidas hacen que el video <strong>vibre</strong>. "
        "Ambos se entienden mucho mejor en el dominio de la frecuencia: el desenfoque por movimiento multiplica el espectro por una función <em>sinc</em> que deja rayas oscuras, y la vibración es una oscilación de la trayectoria de la cámara a frecuencias más altas que el movimiento intencional. "
        "Este trabajo estudia ambos casos con datos reales (fotografías CC0) a las que se les aplica un movimiento con parámetros conocidos, de modo que el resultado pueda medirse: "
        "(A) una imagen con desenfoque lineal se analiza en frecuencia y se restaura con el filtro de Wiener, y (B) un video con vibración se estabiliza estimando el movimiento por correlación de fase y filtrando la trayectoria con un pasa bajas.")
    r.caja("<strong>Sobre los datos.</strong> No se emplearon fotografías ni videos propios: la imagen movida y el video vibrado son <em>simulaciones</em> a partir de fotografías reales de dashcam con licencia libre (Rivera, 2025a, 2025b). "
           "Esto permite conocer la verdad (longitud y ángulo del desenfoque, trayectoria exacta de la cámara) y medir con precisión, pero no reproduce toda la complejidad de una sacudida real (rotaciones, obturador rodante, parallax).")

    r.h2("2. Parte 1. Observa y analiza")
    r.p("El flujo del gráfico resumen es <strong>imagen/video → transformada de Fourier → filtrado frecuencial → imagen estabilizada</strong>.")
    r.h3("¿Cómo afecta el movimiento a la calidad de una imagen?")
    r.p(f"Si la cámara se mueve durante la exposición, cada punto de la escena se «barre» a lo largo de una línea y la imagen se promedia en esa dirección. Con un desplazamiento de {A['L_real']} px a {A['angulo_real']:.0f}° y algo de ruido, la imagen de calle "
        f"cae a {W['movida']['psnr']} dB de PSNR y {W['movida']['ssim']} de SSIM respecto de la nítida (figura 2): los bordes en la dirección del movimiento desaparecen y los perpendiculares se conservan. "
        "En un video, el movimiento no deseado entre cuadros produce vibración visible aunque cada cuadro esté nítido.")
    r.h3("¿Qué diferencias existen entre ruido y movimiento?")
    r.p("El <strong>ruido</strong> es aleatorio, distinto en cada píxel y en cada toma, y su espectro es «blanco»: reparte su potencia de manera pareja en todas las frecuencias (en la práctica de filtros de Fourier se vio como una meseta de potencia en alta frecuencia). "
        "El <strong>desenfoque por movimiento</strong> es determinista y estructurado: es una convolución con una recta, por lo que multiplica el espectro por una función sinc con líneas de ceros perpendiculares a la dirección del movimiento (figura 1). "
        "Y la <strong>vibración</strong> es una oscilación temporal de la posición de la cámara: en el espectro de su trayectoria aparece como picos a unas frecuencias concretas (aquí, 3.2, 5.7 y 8.6 Hz; figura 3). "
        "Por eso el ruido se combate promediando o suavizando, mientras que el movimiento puede <em>invertirse</em> si se conoce su forma.")
    r.h3("¿Por qué es importante estabilizar imágenes en aplicaciones de visión artificial?")
    r.p("Porque casi todos los algoritmos de visión suponen que la imagen es nítida y que el movimiento entre cuadros corresponde a la escena, no a la cámara: los detectores de bordes y de puntos clave pierden respuesta con el desenfoque, "
        "el seguimiento de objetos y la odometría visual confunden vibración con movimiento real, y la medición de tamaños o distancias se degrada. En drones, vehículos y robots, la estabilización precede a la percepción.")

    r.h2("3. Parte 2. Análisis práctico")
    r.h3("A. Imagen con desenfoque por movimiento: espectro y restauración")
    r.p("<strong>Procedimiento.</strong> A una fotografía en gris (960×540 px) se le aplicó un desenfoque lineal de 31 px a 25° mediante multiplicación en el dominio de Fourier, más ruido gaussiano (σ = 2). "
        "Se calculó el espectro de la imagen nítida y el de la movida, y se estimó el movimiento con el <em>cepstrum</em> (la transformada inversa del logaritmo del espectro), donde el desenfoque lineal aparece como un pico negativo a una distancia igual a su longitud y en su misma dirección. "
        "Después se restauró con el filtro de Wiener, F̂ = H*·G / (|H|² + K), y se comparó con el filtro inverso puro (K ≈ 0).")
    r.fig("figuras/fig1_espectro_movimiento.png", "Imagen nítida y movida (foto: M. Rivera, CC0), transferencia H del desenfoque, espectros y cepstrum. Las rayas oscuras del espectro de la movida son los ceros del sinc, perpendiculares a la dirección del movimiento.", alto=150)
    r.p(f"<strong>Qué muestra el espectro.</strong> El espectro de la imagen nítida tiene un pico central y líneas suaves que provienen de bordes largos (postes, cables). En el de la imagen movida aparecen rayas oscuras paralelas: son los ceros de la función sinc del desenfoque, "
        f"con separación inversamente proporcional a la longitud del movimiento y orientación perpendicular a él. El cepstrum permitió recuperar los parámetros: longitud estimada de <strong>{A['L_estimada']} px</strong> (real, {A['L_real']}) y ángulo de <strong>{A['angulo_estimado']}°</strong> (real, {A['angulo_real']:.0f}°).")
    r.fig("figuras/fig2_restauracion.png", "Recorte de la imagen movida y de las restauraciones: filtro inverso, Wiener con la función de desenfoque real y Wiener con la estimada por cepstrum.")
    r.tabla(["Método", "PSNR (dB)", "SSIM"],
            [["Imagen movida", f"{W['movida']['psnr']}", f"{W['movida']['ssim']}"],
             ["Filtro inverso (sin regularizar)", f"{W['inverso_directo']['psnr']}", f"{W['inverso_directo']['ssim']}"]] +
            [[f"Wiener, PSF real, K = {k}", f"{v['psnr']}", f"{v['ssim']}"] for k, v in K.items()] +
            [[f"Wiener, PSF estimada (cepstrum), K = {W['mejor_K']:g}", f"{W['wiener_con_PSF_estimada']['psnr']}", f"{W['wiener_con_PSF_estimada']['ssim']}"]],
            "Calidad de la restauración frente a la imagen nítida.", num=(1, 2))
    r.p(f"<strong>Resultado.</strong> El filtro inverso puro <strong>fracasa</strong> ({W['inverso_directo']['psnr']} dB): en las frecuencias donde H es casi cero se divide entre un número minúsculo y el ruido se amplifica enormemente. "
        f"El filtro de Wiener añade el término K para no dividir entre ceros y mejora la imagen: con K = {W['mejor_K']:g} llega a {K[f'{W['mejor_K']:g}']['psnr']} dB (+{K[f'{W['mejor_K']:g}']['psnr'] - W['movida']['psnr']:.1f} dB) y {K[f'{W['mejor_K']:g}']['ssim']} de SSIM. "
        f"Con la función de desenfoque <em>estimada</em> (32.7 px, 23°) el resultado baja un poco ({W['wiener_con_PSF_estimada']['psnr']} dB), pues un error de pocos píxeles en la longitud produce una restauración menos precisa. "
        "K es un compromiso: si es muy pequeña aparece ruido y anillos; si es grande, la imagen queda borrosa (PSNR y SSIM además no coinciden en cuál es la mejor K).")

    r.h3("B. Video con vibración: correlación de fase y filtrado pasa bajas de la trayectoria")
    r.p(f"<strong>Procedimiento.</strong> A partir de una fotografía de 2400×1350 px se generaron {B['cuadros']} cuadros de 960×540 px a {B['fps']} cuadros/s (5 s), moviendo la ventana de captura con un paneo lento e intencional (600 px de recorrido horizontal) más una vibración de tres frecuencias (3.2, 5.7 y 8.6 Hz, amplitudes de hasta 7 px) y algo de ruido. "
        "(1) El desplazamiento entre cuadros consecutivos se estimó con la <strong>correlación de fase</strong>: el cociente de los espectros cruzados F₂F₁*/|F₂F₁*| tiene una transformada inversa con un pico en el desplazamiento (Kuglin & Hines, 1975). "
        f"(2) La trayectoria acumulada se filtró en el dominio de la frecuencia con un <strong>pasa bajas gaussiano de {B['fc_hz']} Hz</strong> (previa resta de la rampa del paneo, porque la FFT supone señales periódicas). (3) Cada cuadro se desplazó la diferencia entre la trayectoria suavizada y la estimada.")
    r.fig("figuras/fig3_trayectoria.png", "Arriba a la izquierda: trayectoria horizontal real, estimada y suavizada. Arriba a la derecha: espectro de la trayectoria sin la rampa, antes y después; los tres picos de la vibración desaparecen. Abajo: error respecto de la trayectoria suave deseada y elección de la frecuencia de corte.", alto=140)
    r.tabla(["Medida", "Antes", "Después"],
            [["Vibración residual, RMS (px)", f"{B['rms_vibracion_antes_px']}", f"{B['rms_vibracion_despues_px']}"],
             ["Diferencia media entre cuadros consecutivos (niveles de gris)", f"{B['dif_media_entre_cuadros_antes']}", f"{B['dif_media_entre_cuadros_despues']} (límite ideal: {B['dif_media_entre_cuadros_limite_ideal']})"]],
            f"Resultados de la estabilización con corte de {B['fc_hz']} Hz. El error de la trayectoria estimada frente a la real fue de {B['error_estimacion_trayectoria_rms_px']} px (RMS).", num=(1, 2))
    r.p(f"<strong>Resultado.</strong> La vibración residual bajó de <strong>{B['rms_vibracion_antes_px']} px a {B['rms_vibracion_despues_px']} px</strong> (RMS). En el espectro de la trayectoria, los tres picos de vibración disminuyen en más de tres órdenes de magnitud. "
        f"La diferencia media entre cuadros consecutivos baja de {B['dif_media_entre_cuadros_antes']} a {B['dif_media_entre_cuadros_despues']}, que es prácticamente el nivel de un video que solo tuviera el movimiento intencional ({B['dif_media_entre_cuadros_limite_ideal']}); "
        "que quede un poco por debajo se debe probablemente a que el remuestreo bilineal suaviza levemente las imágenes.")
    r.fig("figuras/fig4_cuadros.png", "Tres cuadros consecutivos: original (arriba) y estabilizado (abajo). Las líneas amarillas marcan una posición fija de referencia.", alto=110)
    r.h3("Elección de la frecuencia de corte")
    r.tabla(["Corte (Hz)", "Distancia al movimiento intencional verdadero (RMS, px)", "Residuo respecto de la trayectoria suavizada (RMS, px)"],
            [[k, str(v["rms_vs_pan_ideal_px"]), str(v["rms_residual_px"])] for k, v in BF.items()],
            "Efecto de la frecuencia de corte del filtro pasa bajas.", num=(1, 2))
    r.p("La segunda columna es trivial (baja siempre al subir el corte porque se compara con una versión cada vez menos suavizada); el criterio útil es la primera: **distancia al movimiento que el usuario quería hacer**. "
        f"Con un corte muy bajo (0.3 Hz: {BF['0.3']['rms_vs_pan_ideal_px']} px) el filtro «se come» parte del paneo intencional y la imagen se retrasa respecto de la mano; con un corte muy alto (7 Hz: {BF['7.0']['rms_vs_pan_ideal_px']} px) deja pasar la vibración. "
        f"El mínimo está en 0.8 Hz ({BF['0.8']['rms_vs_pan_ideal_px']} px). Este compromiso es el que ajustan los estabilizadores reales: cuánto movimiento lento se considera intencional.".replace("**", ""))

    r.h2("4. Parte 3. Relación con aplicaciones reales")
    r.h3("¿Cómo ayuda el procesamiento en frecuencia a reducir movimiento o vibraciones?")
    r.p("De dos maneras medidas en este trabajo. Para el <strong>desenfoque</strong> dentro de un cuadro, la convolución se vuelve una multiplicación en frecuencia, por lo que se puede <em>deshacer</em> dividiendo (con regularización, filtro de Wiener); "
        f"se pasó de {W['movida']['psnr']} a {K[f'{W['mejor_K']:g}']['psnr']} dB. Para la <strong>vibración</strong> entre cuadros, la trayectoria de la cámara se separa por frecuencias: el movimiento intencional es lento y la sacudida rápida, así que un pasa bajas conserva lo primero y elimina lo segundo "
        f"({B['rms_vibracion_antes_px']} px → {B['rms_vibracion_despues_px']} px). Además, la correlación de fase estima el desplazamiento entre cuadros de forma robusta a cambios de iluminación, porque usa solo la fase del espectro.")
    r.h3("¿Qué información proporciona el espectro de Fourier?")
    r.p("Qué frecuencias contiene la señal y con qué potencia: en una imagen, si predominan detalles finos (alta frecuencia) o zonas suaves; su orientación (las rayas del espectro son perpendiculares a las estructuras de la imagen); "
        "la presencia de patrones periódicos; y el nivel de ruido (meseta en alta frecuencia). En este trabajo el espectro reveló la longitud y la dirección del movimiento (por las rayas y el cepstrum) y, en la trayectoria de la cámara, las frecuencias exactas de la vibración.")
    r.h3("¿Por qué estas técnicas son importantes en cámaras modernas y visión artificial?")
    r.p("Porque las cámaras se usan en movimiento (en la mano, en un vehículo, en un dron) y la calidad de la imagen y del video depende de compensarlo. "
        "Las aproximaciones de estabilización digital suelen tener el mismo esquema que el de este trabajo: estimar la trayectoria de la cámara, suavizarla y remuestrear los cuadros (Matsushita et al., 2006; Grundmann et al., 2011; Liu et al., 2013). "
        "Y la restauración de imágenes movidas a partir de un modelo del desenfoque es un problema clásico que se resuelve con filtros como el de Wiener (Wiener, 1949) o con métodos que además estiman el desenfoque a partir de una sola foto (Fergus et al., 2006).")
    r.h3("Una aplicación real donde la estabilización es fundamental: drones")
    r.p("Un dron con cámara vibra por sus motores y por el viento, y se desplaza a velocidad considerable. Sin estabilización, el video es incómodo de ver y, más grave, inservible para tareas de percepción: "
        "la detección de obstáculos, el seguimiento de un objetivo o la cartografía por fotogrametría suponen imágenes nítidas y una relación geométrica conocida entre cuadros. "
        "La estabilización combina soportes mecánicos (gimbal) y compensación digital; en la parte digital se suaviza la trayectoria estimada de la cámara, como se hizo aquí, dejando pasar los movimientos lentos y deliberados del piloto y eliminando las sacudidas.")

    r.h2("5. Conclusión")
    r.p("El dominio de la frecuencia convierte dos problemas difíciles en operaciones sencillas: el desenfoque lineal se identifica por su patrón de rayas y se corrige dividiendo con regularización, y la vibración de video se separa del movimiento intencional por su frecuencia. "
        "Los números lo confirman (más de 3 dB de mejora al restaurar la imagen movida y una reducción de la vibración de 7.9 a 0.2 px), pero también muestran los límites: la restauración es sensible al parámetro K y a un error en la estimación del desenfoque, "
        "y la frecuencia de corte de la estabilización es un compromiso entre eliminar la sacudida y respetar el movimiento deseado. Como limitación, todos los movimientos son simulados (traslaciones puras), por lo que falta comprobar el método con video real, que añade rotaciones, obturador rodante y movimiento de la escena.")

    r.refs([
        "Fergus, R., Singh, B., Hertzmann, A., Roweis, S. T., &amp; Freeman, W. T. (2006). Removing camera shake from a single photograph. <em>ACM Transactions on Graphics, 25</em>(3), 787–794. https://doi.org/10.1145/1141911.1141956",
        "Gonzalez, R. C., &amp; Woods, R. E. (2018). <em>Digital image processing</em> (4.ª ed.). Pearson.",
        "Grundmann, M., Kwatra, V., &amp; Essa, I. (2011). Auto-directed video stabilization with robust L1 optimal camera paths. En <em>CVPR 2011</em> (pp. 225–232). IEEE. https://doi.org/10.1109/CVPR.2011.5995525",
        "Kuglin, C. D., &amp; Hines, D. C. (1975). The phase correlation image alignment method. En <em>Proceedings of the IEEE International Conference on Cybernetics and Society</em> (pp. 163–165). IEEE.",
        "Liu, S., Yuan, L., Tan, P., &amp; Sun, J. (2013). Bundled camera paths for video stabilization. <em>ACM Transactions on Graphics, 32</em>(4), Artículo 78. https://doi.org/10.1145/2461912.2461995",
        "Matsushita, Y., Ofek, E., Ge, W., Tang, X., &amp; Shum, H.-Y. (2006). Full-frame video stabilization with motion inpainting. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence, 28</em>(7), 1150–1163. https://doi.org/10.1109/TPAMI.2006.141",
        "OpenAI. (2026). <em>Estabilización de imágenes digitales</em> [Imagen generada con IA]. Material de la actividad, Visión computacional para imágenes y video. ChatGPT.",
        "Rivera, M. (2025a). <em>SB North Patterson St at College St intersection dashcam</em> [Fotografía; CC0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:SB_North_Patterson_St_at_College_St_intersection_dashcam.jpg",
        "Rivera, M. (2025b). <em>NB Bemiss Rd Traffic light ahead dashcam</em> [Fotografía; CC0]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:NB_Bemiss_Rd_Traffic_light_ahead_dashcam.jpg",
        "Wiener, N. (1949). <em>Extrapolation, interpolation, and smoothing of stationary time series</em>. MIT Press.",
    ])
    pdf = r.guardar("Estabilizacion_digital_frecuencia")
    n = zip_entrega(AQUI / "Estabilizacion_digital_frecuencia.zip", AQUI, ["experimentos.py", "informe.py", "figuras", "datos", "resultados.json", pdf.name])
    print("PDF", paginas_pdf(pdf), "páginas; zip con", n, "archivos")


if __name__ == "__main__":
    main()
