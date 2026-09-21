"""Búsqueda y descarga de imágenes de Wikimedia Commons con licencia verificada.

Solo acepta licencias que permiten uso académico con atribución: dominio público, CC0, CC BY y CC BY-SA.
Cada descarga queda registrada en datos_compartidos/LICENCIAS.md.

    python herramientas/commons.py buscar "brain MRI glioma axial"
    python herramientas/commons.py bajar "File:Chest Xray PA 3-8-2010.png" chest_xray.png 1400
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "datos_compartidos"
UA = "visioncomputacional-tareas/1.0 (proyecto escolar; contacto: estudiante)"
API = "https://commons.wikimedia.org/w/api.php"
OK = re.compile(r"public domain|cc0|pd|cc[- ]by(?!-nc)|cc[- ]by-sa", re.I)


def _abrir(req, intentos=6):
    """urlopen con pausa cortés y reintentos con espera creciente ante el error 429."""
    for i in range(intentos):
        time.sleep(2.5)
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == intentos - 1:
                raise
            espera = int(e.headers.get("Retry-After", 0) or 0) or 15 * (i + 1)
            print(f"  429, espero {espera}s…", file=sys.stderr)
            time.sleep(espera)


def _api(**params):
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    return json.load(_abrir(urllib.request.Request(url, headers={"User-Agent": UA})))


def _limpio(html):
    return re.sub(r"<[^>]+>", "", html or "").replace("\n", " ").strip()


def buscar(consulta, n=12):
    d = _api(action="query", generator="search", gsrsearch=f"filetype:bitmap {consulta}", gsrnamespace=6,
             gsrlimit=n, prop="imageinfo", iiprop="size|extmetadata",
             iiextmetadatafilter="LicenseShortName|Artist|ImageDescription")
    filas = []
    for p in d.get("query", {}).get("pages", {}).values():
        ii = p["imageinfo"][0]
        lic = ii["extmetadata"].get("LicenseShortName", {}).get("value", "?")
        filas.append((p["title"], ii["width"], ii["height"], lic, bool(OK.search(lic))))
    return filas


def bajar(titulo, nombre, ancho=1600):
    d = _api(action="query", titles=titulo, prop="imageinfo", iiprop="url|size|extmetadata", iiurlwidth=ancho)
    p = list(d["query"]["pages"].values())[0]
    ii = p["imageinfo"][0]
    m = ii["extmetadata"]
    lic = m.get("LicenseShortName", {}).get("value", "?")
    if not OK.search(lic):
        raise SystemExit(f"Licencia no admitida para {titulo}: {lic}")
    url = ii.get("thumburl") if ii["width"] > ancho else ii["url"]
    DESTINO.mkdir(exist_ok=True)
    ruta = DESTINO / nombre
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    ruta.write_bytes(_abrir(req).read())
    autor = _limpio(m.get("Artist", {}).get("value", "?"))
    desc = _limpio(m.get("ImageDescription", {}).get("value", ""))[:160]
    pagina = ii["descriptionurl"]
    reg = DESTINO / "LICENCIAS.md"
    if not reg.exists():
        reg.write_text("# Licencias de las imágenes\n\n| Archivo | Autor | Licencia | Fuente | Descripción |\n|---|---|---|---|---|\n",
                       encoding="utf-8")
    with reg.open("a", encoding="utf-8") as f:
        f.write(f"| {nombre} | {autor} | {lic} | {pagina} | {desc} |\n")
    print(f"OK {nombre}  {ii['width']}x{ii['height']}  {lic}  {autor}")
    return ruta


if __name__ == "__main__":
    if sys.argv[1] == "buscar":
        for t, w, h, lic, ok in buscar(" ".join(sys.argv[2:])):
            print(("✔" if ok else "✘"), t, f"{w}x{h}", lic)
    elif sys.argv[1] == "bajar":
        bajar(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1600)
