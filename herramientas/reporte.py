"""Generador de informes PDF (HTML + Edge sin interfaz) y utilidades de entrega.

Uso típico:

    from herramientas.reporte import Reporte, zip_entrega
    r = Reporte("Título", "Subtítulo", carpeta=Path(__file__).parent)
    r.portada(); r.h2("1. Introducción"); r.p("Texto…"); r.fig("figuras/a.png", "Descripción")
    r.refs([...]); r.guardar("informe_mi_tarea")
"""
import html
import os
import re
import subprocess
import zipfile
from pathlib import Path

CURSO = "Visión computacional para imágenes y video · Grupo 10"
INTEGRANTES = ["Diego Falcón Costilla"]
FECHA = "septiembre de 2026"

EDGE = next((p for p in (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe") if os.path.exists(p)), None)

CSS = """
@page { size: A4; margin: 17mm 16mm 17mm 16mm; }
:root { --azul:#0b3d91; --gris:#555; --suave:#eef2fa; --borde:#c9d3ea; --ok:#1b7f3b; --alerta:#b5451b; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.3pt; line-height: 1.4; color:#1d1d1f; margin:0; }
h1 { font-size: 23pt; line-height:1.15; color:var(--azul); margin:0 0 6pt; }
h2 { font-size: 13.5pt; color:var(--azul); border-bottom: 2px solid var(--azul); padding-bottom:2pt; margin:16pt 0 6pt; break-after: avoid; }
h3 { font-size: 11.3pt; margin:10pt 0 3pt; color:#222; break-after: avoid; }
h4 { font-size: 10.3pt; margin:8pt 0 2pt; color:#333; break-after: avoid; }
p { margin: 0 0 5pt; text-align: justify; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; } li { margin-bottom:2pt; }
.portada { min-height: 240mm; display:flex; flex-direction:column; justify-content:center; break-after: page; }
.portada .curso { color:var(--gris); font-size:11pt; margin-bottom:24pt; }
.portada .sub { font-size:13.5pt; color:#333; margin-bottom:28pt; }
.portada .meta { border-left:4px solid var(--azul); padding-left:12pt; color:#333; }
.portada .meta p { margin:0 0 3pt; text-align:left; }
table { border-collapse: collapse; width:100%; margin: 5pt 0 8pt; font-size: 8.9pt; break-inside: avoid; }
th, td { border:1px solid var(--borde); padding:3.5pt 5pt; vertical-align: top; text-align:left; }
th { background: var(--azul); color:#fff; font-weight:600; }
tr:nth-child(even) td { background: var(--suave); }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
figure { margin: 7pt 0 9pt; break-inside: avoid; text-align:center; }
figure img { max-width:100%; max-height:120mm; object-fit:contain; border:1px solid var(--borde); }
figcaption { font-size: 8.7pt; color:var(--gris); margin-top:3pt; text-align:left; }
.caja { background: var(--suave); border-left:4px solid var(--azul); padding:5pt 9pt; margin: 6pt 0 8pt; break-inside: avoid; }
.caja p { margin:0 0 3pt; } .caja p:last-child { margin:0; }
.aviso { background:#fff6ee; border-left-color:var(--alerta); }
.nota { font-size:8.8pt; color:var(--gris); }
pre { background:#f5f6f8; border:1px solid #dde1e8; padding:5pt 7pt; font-size:8pt; line-height:1.3; white-space:pre-wrap; break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background:#f0f2f6; padding:0 2pt; }
.formula { text-align:center; font-family: "Cambria Math", "Times New Roman", serif; font-size:11pt; margin:4pt 0 7pt; }
.refs p { text-indent:-18pt; padding-left:18pt; text-align:left; margin-bottom:3pt; font-size:8.8pt; line-height:1.3; break-inside:avoid; }
.salto { break-before: page; }
.dos { display:flex; gap:8pt; } .dos > * { flex:1; }
"""


def esc(s):
    return html.escape(str(s), quote=False)


class Reporte:
    def __init__(self, titulo, subtitulo="", carpeta=".", curso=CURSO, integrantes=None, fecha=FECHA,
                 entrega="PDF con capturas y discusión de resultados + archivo comprimido con el código"):
        self.titulo, self.subtitulo, self.curso = titulo, subtitulo, curso
        self.integrantes = integrantes or INTEGRANTES
        self.fecha, self.entrega = fecha, entrega
        self.carpeta = Path(carpeta)
        self.partes = []
        self._fig = 0
        self._tab = 0

    # ---- bloques
    def add(self, h):
        self.partes.append(h)
        return self

    def portada(self):
        integr = "<br>".join(esc(i) for i in self.integrantes)
        return self.add(f"""<section class="portada">
<div class="curso">{esc(self.curso)}</div>
<h1>{self.titulo}</h1>
<div class="sub">{self.subtitulo}</div>
<div class="meta"><p><strong>Integrantes:</strong><br>{integr}</p>
<p><strong>Fecha:</strong> {esc(self.fecha)}</p><p><strong>Entrega:</strong> {esc(self.entrega)}</p></div></section>""")

    def h2(self, t, salto=False):
        return self.add(f'<h2{" class=salto" if salto else ""}>{t}</h2>')

    def h3(self, t):
        return self.add(f"<h3>{t}</h3>")

    def h4(self, t):
        return self.add(f"<h4>{t}</h4>")

    def p(self, t):
        return self.add(f"<p>{t}</p>")

    def ul(self, items):
        return self.add("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")

    def ol(self, items):
        return self.add("<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>")

    def caja(self, t, aviso=False):
        return self.add(f'<div class="caja{" aviso" if aviso else ""}"><p>{t}</p></div>')

    def formula(self, t):
        return self.add(f'<div class="formula">{t}</div>')

    def codigo(self, t):
        return self.add(f"<pre>{esc(t)}</pre>")

    def salto(self):
        return self.add('<div class="salto"></div>')

    def fig(self, ruta, pie, ancho=None, alto=None):
        """`ancho` en % del ancho de página; `alto` en mm (por defecto el CSS limita a 120 mm)."""
        self._fig += 1
        partes = ([f"width:{ancho}%"] if ancho else []) + ([f"max-height:{alto}mm"] if alto else [])
        estilo = f' style="{";".join(partes)}"' if partes else ""
        return self.add(f'<figure><img src="{Path(ruta).as_posix()}"{estilo}>'
                        f"<figcaption><strong>Figura {self._fig}.</strong> {pie}</figcaption></figure>")

    def tabla(self, cabeza, filas, pie=None, num=()):
        self._tab += 1
        th = "".join(f'<th class="{"num" if i in num else ""}">{c}</th>' for i, c in enumerate(cabeza))
        tr = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>'
                                       for i, c in enumerate(f)) + "</tr>" for f in filas)
        cap = f'<p class="nota"><strong>Tabla {self._tab}.</strong> {pie}</p>' if pie else ""
        return self.add(f"<table><tr>{th}</tr>{tr}</table>{cap}")

    def refs(self, lista, titulo="Referencias"):
        self.h2(titulo)
        ordenadas = sorted(lista, key=lambda s: re.sub(r"[^a-zá-ú0-9]", "", s.lower()))
        return self.add('<div class="refs">' + "".join(f"<p>{r}</p>" for r in ordenadas) + "</div>")

    # ---- salida
    def html(self):
        return (f'<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>{re.sub("<[^>]+>", "", self.titulo)}'
                f"</title><style>{CSS}</style></head><body>" + "\n".join(self.partes) + "</body></html>")

    def guardar(self, nombre):
        """Escribe <nombre>.html y <nombre>.pdf en la carpeta. Devuelve la ruta del PDF."""
        ruta_html = self.carpeta / f"{nombre}.html"
        ruta_html.write_text(self.html(), encoding="utf-8")
        return html_a_pdf(ruta_html, self.carpeta / f"{nombre}.pdf")


def html_a_pdf(ruta_html, ruta_pdf, espera_js_ms=0, apaisado=False):
    if EDGE is None:
        raise RuntimeError("No se encontró Edge ni Chrome para generar el PDF")
    ruta_pdf = Path(ruta_pdf)
    if ruta_pdf.exists():
        ruta_pdf.unlink()
    cmd = [EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={ruta_pdf}"]
    if espera_js_ms:
        cmd.append(f"--virtual-time-budget={espera_js_ms}")
    cmd.append(Path(ruta_html).resolve().as_uri())
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    if not ruta_pdf.exists():
        raise RuntimeError(f"Edge no generó {ruta_pdf}")
    return ruta_pdf


def paginas_pdf(ruta_pdf):
    import pymupdf
    with pymupdf.open(ruta_pdf) as d:
        return len(d)


def vista_previa(ruta_pdf, salida, dpi=45):
    """Guarda cada página del PDF como PNG (para revisar el diseño)."""
    import pymupdf
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(ruta_pdf) as d:
        for i, pg in enumerate(d):
            pg.get_pixmap(dpi=dpi).save(salida / f"p{i + 1}.png")
        return len(d)


def zip_entrega(ruta_zip, raiz, incluir, nombre_raiz=None, excluir=("__pycache__", ".ipynb_checkpoints", ".pyc")):
    """Comprime archivos/carpetas de `incluir` (rutas relativas a `raiz`) en `ruta_zip`."""
    raiz = Path(raiz)
    nombre_raiz = nombre_raiz or Path(ruta_zip).stem
    n = 0
    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in incluir:
            base = raiz / rel
            archivos = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
            for a in archivos:
                if any(x in str(a) for x in excluir):
                    continue
                z.write(a, f"{nombre_raiz}/{a.relative_to(raiz).as_posix()}")
                n += 1
    return n
