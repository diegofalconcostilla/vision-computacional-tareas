"""Generador de informes PDF (HTML + Edge sin interfaz) y utilidades de entrega.

Uso típico:

    from herramientas.reporte import Reporte, zip_entrega
    r = Reporte("Título", "Subtítulo", carpeta=Path(__file__).parent)
    r.portada(); r.h2("Introducción"); r.p("Texto…"); r.fig("figuras/a.png", "Descripción")
    r.refs([...]); r.guardar("informe_mi_tarea")   # PDF + .docx editable (ver herramientas/word.py)
"""
import html
import os
import re
import subprocess
import zipfile
from pathlib import Path

CURSO = "Visión computacional para imágenes y video · Grupo 10"
INTEGRANTES = ["Diego Falcón Costilla"]
# Equipo de las prácticas en Colab (actividades en equipo de Canvas)
EQUIPO = ["Diego Falcón Costilla A01139580", "Alejandro José Martínez Ubeda A01797775",
          "Ana Bonavides Aguilar A01423281", "José Andrés Orantes Guillén A01174130"]
FECHA = "septiembre de 2026"

EDGE = next((p for p in (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe") if os.path.exists(p)), None)

# Formato de los informes "Relaciona lo aprendido" (el del informe de vehículos autónomos entregado en Word):
# título de la actividad de Canvas, subtítulo con el tema, sección "Equipo" con matrículas, encabezados sin numerar
# en negro, texto justificado sin cursivas ni negritas (salvo etiquetas cortas) y declaración de uso de IA.
CSS = """
@page { size: A4; margin: 25mm 25mm 22mm 25mm; }
:root { --titulos:#000; --azul:#0b3d91; --gris:#595959; --suave:#eef2fa; --borde:#c9d3ea; }
* { box-sizing: border-box; }
body { font-family: Aptos, "Segoe UI Variable Text", "Segoe UI", Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color:#000; margin:0; }
h1 { font-family: "Aptos Display", Aptos, "Segoe UI Variable Display", "Segoe UI", sans-serif; font-weight:400; font-size: 27pt; line-height:1.2;
     color:#000; margin:0 0 8pt; }
h2 { font-family: "Aptos Display", Aptos, "Segoe UI Variable Display", "Segoe UI", sans-serif; font-weight:400; font-size: 17pt;
     color:var(--titulos); margin:20pt 0 6pt; break-after: avoid; }
h3 { font-family: "Aptos Display", Aptos, "Segoe UI Variable Display", "Segoe UI", sans-serif; font-weight:400; font-size: 13.5pt;
     color:var(--titulos); margin:12pt 0 4pt; break-after: avoid; }
h4 { font-weight:400; font-size: 12pt; letter-spacing: 0.6pt; color:var(--titulos); margin:14pt 0 2pt; break-after: avoid; }
strong { font-weight: 700; }
p { margin: 0 0 9pt; text-align: justify; }
ul, ol { margin: 0 0 9pt; padding-left: 18pt; } li { margin-bottom:3pt; text-align: justify; }
.sub { font-size:14pt; color:var(--titulos); letter-spacing: 1pt; line-height:1.4; margin-bottom:12pt; }
.equipo { text-align:left; }
table { border-collapse: collapse; width:100%; margin: 6pt 0 10pt; font-size: 8.9pt; line-height:1.35; break-inside: avoid; }
th, td { border:1px solid var(--borde); padding:3.5pt 5pt; vertical-align: top; text-align:left; }
th { background: var(--azul); color:#fff; font-weight:600; }
tr:nth-child(even) td { background: var(--suave); }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
figure { margin: 8pt 0 10pt; break-inside: avoid; text-align:center; }
figure img { max-width:100%; max-height:120mm; object-fit:contain; border:1px solid #e3e6ec; }
figcaption { font-size: 8.5pt; line-height:1.35; color:var(--gris); margin-top:3pt; text-align:left; }
.nota { font-size:8.5pt; color:var(--gris); text-align:left; }
pre { background:#f5f6f8; border:1px solid #dde1e8; padding:5pt 7pt; font-size:8pt; line-height:1.3; white-space:pre-wrap; break-inside: avoid; }
code { font-family: inherit; font-size: inherit; }
.formula { text-align:center; font-family: "Cambria Math", "Times New Roman", serif; font-size:11.5pt; margin:4pt 0 9pt; }
.refs p { text-indent:-18pt; padding-left:18pt; text-align:left; margin-bottom:4pt; line-height:1.4; break-inside:avoid; }
a { color:#467886; }
.salto { break-before: page; }
.dos { display:flex; gap:8pt; } .dos > * { flex:1; }
"""

DECLARACION_IA = ("Para la elaboración de este trabajo se utilizó Claude Opus, de Anthropic, como herramienta de apoyo en el desarrollo y la revisión "
                  "del código, así como en la redacción del documento final. Su uso fue complementario y no sustituyó el criterio académico de los "
                  "autores. El análisis, la interpretación de los resultados y la formulación de las conclusiones corresponden a los autores, quienes "
                  "asumen la responsabilidad íntegra sobre la revisión, la exactitud y la integridad académica del contenido final.")


def equipo_informe():
    """Integrantes como en el informe de ejemplo: "Nombre | Matrícula", en orden alfabético."""
    return sorted(re.sub(r"\s+(A\d{8})$", r" | \1", i) for i in EQUIPO)


def sin_enfasis(t):
    """Quita cursivas y subrayados, y las negritas que no sean una etiqueta corta al inicio ("Resultado.", "Limitación:")."""
    t = re.sub(r"</?(?:em|i|u)>", "", t)
    m = re.match(r"\s*<strong>([^<]{1,60}?)</strong>", t)
    etiqueta = m and m.group(1).rstrip()[-1:] in ".:" and len(m.group(1).split()) <= 6
    cuerpo = re.sub(r"</?strong>", "", t[m.end():] if etiqueta else t)
    return (t[:m.end()] + cuerpo) if etiqueta else cuerpo


def sin_numero(t):
    """Los encabezados van sin numerar, como en el informe de ejemplo: "2. Parte 1. Observa" -> "Parte 1. Observa"."""
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", t)


def esc(s):
    return html.escape(str(s), quote=False)


class Reporte:
    def __init__(self, titulo, subtitulo="", carpeta=".", actividad=None, integrantes=None, **_):
        """`actividad`: nombre de la actividad en Canvas (título del documento); `titulo` queda como subtítulo.
        `subtitulo` describe el trabajo (no se imprime). Otros argumentos antiguos (curso, fecha, entrega) se ignoran."""
        self.titulo, self.subtitulo = titulo, subtitulo
        self.actividad = actividad or titulo
        self.integrantes = integrantes or equipo_informe()
        self.carpeta = Path(carpeta)
        self.partes = []
        self._fig = 0
        self._tab = 0

    # ---- bloques
    def add(self, h):
        self.partes.append(h)
        return self

    def portada(self):
        sub = f'<div class="sub">{self.titulo}</div>' if self.actividad != self.titulo else ""
        integr = "<br>".join(esc(i) for i in self.integrantes)
        return self.add(f'<h1>{self.actividad}</h1>{sub}<h2>Equipo</h2><p class="equipo">{integr}</p>')

    def h2(self, t, salto=False):
        return self.add(f'<h2{" class=salto" if salto else ""}>{sin_numero(sin_enfasis(t))}</h2>')

    def h3(self, t):
        return self.add(f"<h3>{sin_numero(sin_enfasis(t))}</h3>")

    def h4(self, t):
        return self.add(f"<h4>{sin_numero(sin_enfasis(t))}</h4>")

    def p(self, t):
        return self.add(f"<p>{sin_enfasis(t)}</p>")

    def ul(self, items):
        return self.add("<ul>" + "".join(f"<li>{sin_enfasis(i)}</li>" for i in items) + "</ul>")

    def ol(self, items):
        return self.add("<ol>" + "".join(f"<li>{sin_enfasis(i)}</li>" for i in items) + "</ol>")

    def caja(self, t, aviso=False):
        """Antes era un recuadro de color; en este formato es un párrafo normal."""
        return self.p(t)

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
                        f"<figcaption><strong>Figura {self._fig}.</strong> {re.sub(r'</?(?:em|i|u|strong)>', '', pie)}</figcaption></figure>")

    def tabla(self, cabeza, filas, pie=None, num=()):
        self._tab += 1
        limpia = lambda c: re.sub(r"</?(?:em|i|u)>", "", str(c))  # noqa: E731  (las negritas de celdas destacan valores)
        th = "".join(f'<th class="{"num" if i in num else ""}">{limpia(c)}</th>' for i, c in enumerate(cabeza))
        tr = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{limpia(c)}</td>'
                                       for i, c in enumerate(f)) + "</tr>" for f in filas)
        cap = f'<p class="nota"><strong>Tabla {self._tab}.</strong> {re.sub(r"</?(?:em|i|u|strong)>", "", pie)}</p>' if pie else ""
        return self.add(f"<table><tr>{th}</tr>{tr}</table>{cap}")

    def refs(self, lista, titulo="Referencias"):
        """Referencias en APA 7: conservan la cursiva de títulos de obras y revistas; las URL quedan como enlaces."""
        self.h2(titulo)
        ordenadas = sorted(lista, key=lambda s: re.sub(r"[^a-zá-ú0-9]", "", s.lower()))
        enlace = lambda r: re.sub(r"(https?://[^\s<]+?)(\.?)$", r'<a href="\1">\1</a>\2', r)  # noqa: E731
        return self.add('<div class="refs">' + "".join(f"<p>{enlace(r)}</p>" for r in ordenadas) + "</div>")

    def declaracion_ia(self):
        return self.add(f"<h4>Declaración y uso de IA</h4><p>{DECLARACION_IA}</p>")

    # ---- salida
    def html(self):
        return (f'<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>{re.sub("<[^>]+>", "", self.titulo)}'
                f"</title><style>{CSS}</style></head><body>" + "\n".join(self.partes) + "</body></html>")

    def guardar(self, nombre, docx=True):
        """Escribe <nombre>.html, <nombre>.pdf y (si `docx`) <nombre>.docx editable. Devuelve la ruta del PDF."""
        if not any("Declaración y uso de IA" in x for x in self.partes):
            self.declaracion_ia()
        ruta_html = self.carpeta / f"{nombre}.html"
        ruta_html.write_text(self.html(), encoding="utf-8")
        pdf = html_a_pdf(ruta_html, self.carpeta / f"{nombre}.pdf")
        if docx:
            from .word import html_a_docx  # import diferido: word importa este módulo
            html_a_docx(ruta_html, self.carpeta / f"{nombre}.docx")
        return pdf


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
