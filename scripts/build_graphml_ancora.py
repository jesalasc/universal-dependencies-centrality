"""
Genera grafos semántico-sintácticos en formato GraphML a partir del corpus
AnCora 2.0 (XML), reutilizando la lógica de extracción de tree_balance.py.

A diferencia de tree_balance.py (que produce SVG/PNG + CSV con top-5),
este script guarda un .graphml por oración en una carpeta plana, siguiendo
la convención del dataset AnCora-ES del proyecto:

    <subcarpeta>_<archivo_sin_.tbf>_s<NNNNN>.graphml

Cada nodo lleva los atributos:
    form     : la palabra (wd del XML)
    tipo     : verbo | nucleo | predicativo | funcional
    postype  : main | auxiliary | semiauxiliary  (solo verbos)

Cada grafo lleva los atributos:
    phrase   : oración reconstruida en orden documental
    root     : verbo raíz identificado por ExtractorGrafo

USO
───
    python scripts/build_graphml_ancora.py \\
        --input "/Users/summa/Documents/Cenia/C. Riveros/data/ancora-dep-2.0/xml" \\
        --output AnCora-ES-Semantic
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx

# Importar la lógica de extracción del script hermano
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_balance import ExtractorGrafo, WORD_TAGS, es_puntuacion  # noqa: E402


# Mismas constantes que graph_centrality_store.detokenize para mantener
# coherencia con la frase guardada en los .graphml del dataset original.
OPENING_PUNCTUATION = {"¿", "¡", "(", "[", "{", "«", '"', "'"}
CLOSING_PUNCTUATION = {".", ",", ";", ":", "?", "!", ")", "]", "}", "»", '"', "'"}


def detokenize(tokens):
    """Une tokens insertando espacios excepto antes de signos de cierre."""
    phrase = ""
    previous = ""
    for token in tokens:
        if not token:
            continue
        if not phrase:
            phrase = token
        elif token in CLOSING_PUNCTUATION:
            phrase = f"{phrase}{token}"
        elif previous in OPENING_PUNCTUATION:
            phrase = f"{phrase}{token}"
        else:
            phrase = f"{phrase} {token}"
        previous = token
    return phrase


def extraer_phrase(sent):
    """Reconstruye la oración recorriendo el XML en orden documental.

    Incluye signos de puntuación para producir una frase legible aunque
    el grafo semántico los excluya (REGLA 1 de tree_balance.py).
    """
    tokens = []
    for elem in sent.iter():
        if elem.tag in WORD_TAGS and "wd" in elem.attrib:
            tokens.append(elem.attrib["wd"])
    return detokenize(tokens)


def construir_grafo(extractor, verbo_raiz, phrase):
    """Convierte la salida de ExtractorGrafo en un nx.DiGraph listo para
    serializar como GraphML."""
    grafo = nx.DiGraph()
    grafo.graph["phrase"] = phrase
    if verbo_raiz:
        grafo.graph["root"] = str(verbo_raiz)

    for wd, tipo in extractor.tipos_nodo.items():
        atributos = {"form": wd, "tipo": tipo}
        if wd in extractor.postype_nodo:
            atributos["postype"] = extractor.postype_nodo[wd]
        grafo.add_node(wd, **atributos)

    aristas_vistas = set()
    for origen, destino, _etiqueta in extractor.aristas:
        if origen in extractor.tipos_nodo and destino in extractor.tipos_nodo:
            clave = (origen, destino)
            if clave not in aristas_vistas:
                grafo.add_edge(origen, destino)
                aristas_vistas.add(clave)

    return grafo


def stem_sin_tbf(ruta):
    """Quita los sufijos .tbf.xml o .xml del nombre del archivo."""
    nombre = ruta.name
    for sufijo in (".tbf.xml", ".xml"):
        if nombre.endswith(sufijo):
            return nombre[: -len(sufijo)]
    return ruta.stem


def procesar_archivo(ruta_xml, carpeta_salida, prefijo):
    """Procesa un XML y guarda un .graphml por oración. Devuelve cuántos
    grafos se escribieron y cuántas oraciones se omitieron."""
    base = stem_sin_tbf(ruta_xml)
    try:
        tree = ET.parse(ruta_xml)
    except ET.ParseError as exc:
        print(f"  [ERROR] No se pudo parsear {ruta_xml}: {exc}")
        return 0, 0

    sentences = tree.getroot().findall(".//sentence")
    guardados = 0
    omitidos = 0

    for idx, sent in enumerate(sentences, start=1):
        extractor = ExtractorGrafo()
        verbo_raiz = extractor.extraer(sent)

        if not extractor.tipos_nodo:
            omitidos += 1
            continue

        phrase = extraer_phrase(sent)
        grafo = construir_grafo(extractor, verbo_raiz, phrase)

        nombre_archivo = f"{prefijo}_{base}_s{idx:05d}.graphml"
        ruta_salida = carpeta_salida / nombre_archivo

        try:
            nx.write_graphml(grafo, ruta_salida)
            guardados += 1
        except Exception as exc:
            print(f"    [ERROR] Oración {idx} de {base}: {exc}")
            omitidos += 1

    print(f"  {ruta_xml.name}: {guardados} grafos, {omitidos} omitidos")
    return guardados, omitidos


def listar_xml(carpeta):
    """Devuelve los XML del corpus AnCora dentro de una carpeta (no recursivo
    a subcarpetas más profundas, pero sí descendiendo dentro de la subcarpeta)."""
    archivos = sorted(carpeta.rglob("*.tbf.xml"))
    if not archivos:
        archivos = sorted(carpeta.rglob("*_tbf.xml"))
    if not archivos:
        archivos = [p for p in sorted(carpeta.rglob("*.xml"))
                    if not p.name.startswith(".")]
    return archivos


def procesar_corpus(carpeta_entrada, carpeta_salida):
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    subcarpetas = [p for p in sorted(carpeta_entrada.iterdir()) if p.is_dir()]

    total_guardados = 0
    total_omitidos = 0

    if subcarpetas:
        # Estructura esperada: <input>/<corpus>/<archivo.tbf.xml>
        for sub in subcarpetas:
            archivos = listar_xml(sub)
            if not archivos:
                continue
            print(f"\n[{sub.name}]  ({len(archivos)} archivos)")
            for ruta in archivos:
                g, o = procesar_archivo(ruta, carpeta_salida, prefijo=sub.name)
                total_guardados += g
                total_omitidos += o
    else:
        archivos = listar_xml(carpeta_entrada)
        print(f"\n({len(archivos)} archivos)")
        for ruta in archivos:
            g, o = procesar_archivo(ruta, carpeta_salida, prefijo=carpeta_entrada.name)
            total_guardados += g
            total_omitidos += o

    print(f"\n──────────────────────────────────────────────")
    print(f"Total: {total_guardados} grafos guardados, {total_omitidos} omitidos")
    print(f"Carpeta: {carpeta_salida}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera GraphML semántico-sintácticos del corpus AnCora",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Carpeta raíz con subcarpetas de XML (ej. ancora-dep-2.0/xml/)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Carpeta destino para los .graphml",
    )
    args = parser.parse_args()

    procesar_corpus(Path(args.input).expanduser(), Path(args.output).expanduser())


if __name__ == "__main__":
    main()
