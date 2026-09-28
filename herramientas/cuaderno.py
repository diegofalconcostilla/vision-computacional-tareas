"""Utilidades para extender, ejecutar y exportar notebooks de Jupyter/Colab sin usar servicios externos.

Flujo:  nb = cargar(orig) -> parchear(nb) -> agregar(nb, [md(..), code(..)]) -> ejecutar(nb, cwd) -> guardar -> a_pdf
"""
import html
import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .reporte import CURSO, FECHA, INTEGRANTES, esc, html_a_pdf

# Cambios de compatibilidad con Pillow/NumPy/OpenCV actuales (los notebooks originales son de Python 3.8)
PARCHES = [
    (r"Image\.ANTIALIAS", "Image.LANCZOS"),
    (r"np\.int\b", "int"),
    (r"np\.float\b", "float"),
    (r"np\.bool\b", "bool"),
]


def cargar(ruta):
    return nbformat.read(str(ruta), as_version=4)


def md(texto):
    return nbformat.v4.new_markdown_cell(texto.strip("\n"))


def code(texto):
    return nbformat.v4.new_code_cell(texto.strip("\n"))


def agregar(nb, celdas):
    nb.cells.extend(celdas)
    return nb


def parchear(nb):
    """Aplica los parches de compatibilidad y devuelve la lista de cambios hechos."""
    cambios = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        nuevo = c.source
        for pat, rep in PARCHES:
            nuevo = re.sub(pat, rep, nuevo)
        if nuevo != c.source:
            cambios.append(f"celda {i}: {c.source.strip().splitlines()[0][:40]}…")
            c.source = nuevo
    return cambios


def reemplazar(nb, viejo, nuevo, veces=1):
    """Reemplaza texto en las celdas de código; falla si no lo encuentra (para no dejar parches silenciosos)."""
    n = 0
    for c in nb.cells:
        if c.cell_type == "code" and viejo in c.source:
            c.source = c.source.replace(viejo, nuevo)
            n += 1
    if n != veces:
        raise ValueError(f"Se esperaba reemplazar {veces} celda(s) y se reemplazaron {n}: {viejo[:50]!r}")
    return n


def limpiar_salidas(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
    return nb


def ejecutar(nb, cwd, timeout=900):
    """Ejecuta el notebook con el kernel local `python3` usando `cwd` como directorio de trabajo."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    limpiar_salidas(nb)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}},
                   allow_errors=False).execute()
    return nb


def guardar(nb, ruta):
    nbformat.write(nb, str(ruta))
    return ruta


CSS_EXTRA = """
@page { size: A4; margin: 15mm 14mm 15mm 14mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10pt; margin:0; }
div.container, .container { width:auto !important; max-width:none !important; padding:0 !important; margin:0 !important; }
div.prompt, .prompt, .input_prompt, .output_prompt { display:none !important; }
div.input_area { border:1px solid #d5d9e0 !important; background:#f6f7f9 !important; border-radius:3px; }
div.input_area pre, .output_area pre { font-size:7.6pt !important; line-height:1.25; }
div.cell { break-inside: auto; }
div.output_png img, div.output_jpeg img, .output_area img { max-width:100% !important; max-height:150mm; height:auto !important; }
div.output_subarea { max-width:100% !important; }
div.text_cell_render { padding:0 !important; }
h1 { color:#0b3d91; font-size:19pt; border-bottom:2px solid #0b3d91; }
h2 { color:#0b3d91; font-size:14pt; border-bottom:1px solid #0b3d91; break-after:avoid; }
h3 { font-size:11.5pt; break-after:avoid; }
.rendered_html table, .text_cell_render table { table-layout:auto !important; width:100% !important; border-collapse:collapse !important;
  margin:6pt 0 !important; font-size:8.8pt !important; border:0 !important; }
.rendered_html th, .rendered_html td, .text_cell_render th, .text_cell_render td { text-align:left !important; vertical-align:top !important;
  border:1px solid #c9d3ea !important; padding:3pt 5pt !important; }
.rendered_html th, .text_cell_render th { background:#0b3d91 !important; color:#fff !important; font-weight:600 !important; }
.rendered_html tr:nth-child(even) td { background:#eef2fa !important; }
a.anchor-link { display:none; }
a[href]:after { content:none !important; }  /* la plantilla classic repite cada URL entre paréntesis al imprimir */
.portada { min-height:235mm; display:flex; flex-direction:column; justify-content:center; break-after:page; }
.portada .curso { color:#555; font-size:11pt; margin-bottom:22pt; }
.portada h1 { font-size:22pt; border:0; margin:0 0 6pt; }
.portada .sub { font-size:13pt; color:#333; margin-bottom:26pt; }
.portada .meta { border-left:4px solid #0b3d91; padding-left:12pt; color:#333; }
.portada .meta p { margin:0 0 3pt; }
.refs p { text-indent:-18pt; padding-left:18pt; margin:0 0 4pt; font-size:9.3pt; }
"""


def a_pdf(nb, ruta_pdf, titulo, subtitulo="", integrantes=None, entrega="Notebook ejecutado + PDF de resultados"):
    """Convierte el notebook ya ejecutado a un PDF con portada (HTML incrustado + Edge sin interfaz)."""
    ruta_pdf = Path(ruta_pdf)
    exportador = HTMLExporter(template_name="classic")
    cuerpo, _ = exportador.from_notebook_node(nb)
    integr = "<br>".join(esc(i) for i in (integrantes or INTEGRANTES))
    portada = (f'<section class="portada"><div class="curso">{esc(CURSO)}</div><h1>{titulo}</h1>'
               f'<div class="sub">{subtitulo}</div><div class="meta"><p><strong>Integrantes:</strong><br>{integr}</p>'
               f"<p><strong>Fecha:</strong> {esc(FECHA)}</p><p><strong>Entrega:</strong> {esc(entrega)}</p></div></section>")
    cuerpo = cuerpo.replace("</head>", f"<style>{CSS_EXTRA}</style></head>", 1)
    cuerpo = re.sub(r"(<body[^>]*>)", r"\1" + portada.replace("\\", "\\\\"), cuerpo, count=1)
    ruta_html = ruta_pdf.with_suffix(".html")
    ruta_html.write_text(cuerpo, encoding="utf-8")
    html_a_pdf(ruta_html, ruta_pdf, espera_js_ms=20000)
    return ruta_pdf
