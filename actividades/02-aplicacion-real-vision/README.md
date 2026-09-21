# Simulación de la cadena de adquisición de imagen (vehículos autónomos)

Actividad "Relaciona lo aprendido: investiga sobre una aplicación real de la visión computacional" (Semana 1).

El script toma una foto de dashcam en 4K y simula cinco etapas de la adquisición de imagen:

| # | Experimento | Figura |
|---|---|---|
| 1 | Muestreo espacial (resolución) | `figuras/fig1_muestreo.png` |
| 2 | Cuantización (bits por canal) | `figuras/fig2_cuantizacion.png` |
| 3 | Mosaico Bayer y demosaico | `figuras/fig3_bayer.png` |
| 4 | Exposición y rango dinámico (fusión de Mertens) | `figuras/fig4_exposicion.png` |
| 5 | Bordes con Sobel bajo ruido | `figuras/fig5_bordes.png` |

Las métricas numéricas se guardan en `resultados.json`.

## Uso

```
pip install -r requirements.txt
python simulacion_adquisicion.py
```

Opciones: `--imagen RUTA` (por defecto `assets/dashcam_original.jpg`) y `--salida CARPETA` (por defecto, la carpeta del script).
El ruido usa una semilla fija (42), así que los resultados son reproducibles.

## Imagen de ejemplo

`assets/dashcam_original.jpg`: *NB Bemiss Rd Traffic light ahead dashcam*, de Michael Rivera, dominio público (CC0),
Wikimedia Commons: https://commons.wikimedia.org/wiki/File:NB_Bemiss_Rd_Traffic_light_ahead_dashcam.jpg

## Límites

Es una simulación sobre un JPEG de 8 bits que ya pasó por el sensor y el ISP de la dashcam. No reproduce el rango dinámico real
de la escena ni el ruido físico de un sensor; sirve para ilustrar cada fenómeno y medir su efecto relativo.
