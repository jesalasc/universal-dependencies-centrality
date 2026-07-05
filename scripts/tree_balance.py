"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   GRAFOS SEMÁNTICO-SINTÁCTICOS — CORPUS ANCORA 2.0 (3LB-CAST)              ║
║   Genera un grafo por oración y un CSV con las 5 palabras más importantes   ║
╚══════════════════════════════════════════════════════════════════════════════╝

DESCRIPCIÓN
───────────
Este script lee los archivos *_tbf.xml del corpus AnCora y produce:
  1. Un grafo (SVG o PNG) por cada oración.
  2. Un archivo palabras_importantes.csv con las 5 palabras clave por oración.

DEPENDENCIAS
────────────
  pip install matplotlib networkx

USO
───
  # Procesar todos los archivos de una carpeta
  python generar_grafos_ancora.py --input carpeta_xml/ --output grafos/

  # Solo un archivo
  python generar_grafos_ancora.py --input carpeta_xml/ --output grafos/ --archivo 104_c-1

  # Solo una oración (modo prueba)
  python generar_grafos_ancora.py --input carpeta_xml/ --output grafos/ --archivo 104_c-1 --oracion 5

  # Sin generar CSV
  python generar_grafos_ancora.py --input carpeta_xml/ --output grafos/ --sin-csv

  # Formato PNG en lugar de SVG
  python generar_grafos_ancora.py --input carpeta_xml/ --output grafos/ --formato png
"""

import os
import csv
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
# Solo fijar backend no-interactivo cuando se corre como script CLI;
# al importarse desde un notebook, respetar el backend inline de Jupyter.
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CONSTANTES Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# ── Colores de nodos (REGLA 2) ────────────────────────────────────────────────
#
# El color codifica el papel semántico del nodo, no su categoría léxica.
# Cada color tiene tres variantes: fondo, borde, texto.
#
COLOR_VERBO       = '#FAC775'   # Ámbar  — verbos
BORDE_VERBO       = '#BA7517'
TEXTO_VERBO       = '#412402'

COLOR_NUCLEO      = '#9FE1CB'   # Teal   — núcleos de rol semántico
BORDE_NUCLEO      = '#0F6E56'
TEXTO_NUCLEO      = '#04342C'

COLOR_PRED        = '#F5C4B3'   # Coral  — predicativos (cpred / atr)
BORDE_PRED        = '#993C1D'
TEXTO_PRED        = '#4A1B0C'

COLOR_FUNC        = '#D3D1C7'   # Gris   — palabras funcionales y sin rol
BORDE_FUNC        = '#5F5E5A'
TEXTO_FUNC        = '#2C2C2A'

# ── Tags de palabras en AnCora (nodos hoja con atributo 'wd') ─────────────────
WORD_TAGS = {'d', 'n', 'v', 's', 'c', 'r', 'a', 'z', 'w', 'p', 'f'}

# ── Verbos auxiliares de perífrasis excluidos del top 5 (REGLA CSV-1) ─────────
#
# Estos verbos aparecen con postype='main' en AnCora pero actúan como
# auxiliares de perífrasis modales o aspectuales. Se excluyen manualmente.
# Los verbos con postype='auxiliary' o 'semiauxiliary' se excluyen
# automáticamente usando el atributo del XML (más fiable).
#
MODALES_Y_PERIFRASIS = {
    'poder', 'deber', 'querer', 'soler', 'saber', 'necesitar',
    'ir', 'llevar', 'dejar', 'seguir', 'comenzar', 'empezar',
    'volver', 'acabar', 'terminar', 'cesar', 'continuar',
    'ponerse', 'echarse',
}

# ── Funciones sintácticas para el top 5 (REGLA CSV-2) ────────────────────────
#
# Se distinguen tres grupos:
#   AGENTE:     suj con rol temático agt — va primero (REGLA 3)
#   ARGUMENTO:  complementos de la estructura argumental central del verbo
#   CIRCUNST.:  complementos circunstanciales — EXCLUIDOS (REGLA CSV-3)
#
FUNCS_AGENTE = {
    'suj/agt', 'suj/tem', 'suj/pat',
}

FUNCS_ARGUMENTO = {
    'suj', 'cd', 'ci', 'creg', 'cpred', 'atr',
    'cd/pat', 'cd/tem', 'cd/ext',
    'ci/ben', 'ci/tem',
    'creg/tem', 'creg/pat',
    'cpred/atr',
}

FUNCS_CIRCUNSTANCIA = {
    'cc', 'cc/adv', 'cc/loc', 'cc/tmp',
    'cc/fin', 'cc/cau', 'cc/mnr', 'mod',
}


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — FUNCIONES AUXILIARES DE LECTURA DEL XML
# ══════════════════════════════════════════════════════════════════════════════

def es_puntuacion(elem):
    """
    Devuelve True si el elemento es signo de puntuación.
    Se detecta por el atributo 'punct' del tag <f>.
    REGLA 1: los signos de puntuación no se incluyen en el grafo.
    """
    return elem.attrib.get('punct') is not None


def primera_palabra(elem, excluir_punct=True):
    """
    Obtiene la primera palabra con contenido léxico dentro de un elemento,
    recorriendo el árbol en profundidad.
    Devuelve el valor del atributo 'wd', o None si no encuentra nada.
    """
    for e in elem.iter():
        if e.tag in WORD_TAGS and 'wd' in e.attrib:
            if excluir_punct and es_puntuacion(e):
                continue
            return e.attrib['wd']
    return None


def primer_verbo(elem):
    """
    Obtiene el primer verbo (tag <v>) dentro de un elemento.
    Devuelve (wd, postype) o (None, None).

    postype puede ser:
      'main'          → verbo léxico principal
      'auxiliary'     → auxiliar puro (haber en tiempos compuestos)
      'semiauxiliary' → semiauxiliar (ser/estar copulativos)
    """
    for e in elem.iter():
        if e.tag == 'v' and 'wd' in e.attrib:
            return e.attrib['wd'], e.attrib.get('postype', 'main')
    return None, None


def obtener_conj_o_prep(elem):
    """
    Obtiene la preposición, conjunción o pronombre relativo que
    introduce un sintagma (REGLA 5).

    Busca en los hijos directos del elemento:
      <prep> → primer <s> con wd  (preposición)
      <conj> → primer <c> con wd  (conjunción)
      <relatiu> → primer <p> con wd  (pronombre relativo)
    """
    for child in elem:
        if child.tag == 'prep':
            for e in child.iter():
                if e.tag == 's' and 'wd' in e.attrib:
                    return e.attrib['wd']
        elif child.tag == 'conj':
            for e in child.iter():
                if e.tag == 'c' and 'wd' in e.attrib:
                    return e.attrib['wd']
        elif child.tag == 'relatiu':
            for e in child.iter():
                if e.tag == 'p' and 'wd' in e.attrib:
                    return e.attrib['wd']
    return None


def obtener_verbo_raiz(sent):
    """
    Identifica el verbo raíz de la oración (REGLA 10).
    Estrategia:
      1. <grup.verb> hijo directo de <sentence> → su verbo es la raíz
      2. Primer <S> hijo de <sentence> → su verbo interno
      3. Fallback: primer verbo encontrado en cualquier lugar
    """
    # Estrategia 1
    for child in sent:
        if child.tag == 'grup.verb':
            wd, pt = primer_verbo(child)
            if wd:
                return wd, pt
    # Estrategia 2
    for child in sent:
        if child.tag == 'S':
            wd, pt = primer_verbo(child)
            if wd:
                return wd, pt
    # Estrategia 3 (fallback)
    return primer_verbo(sent)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — EXTRACTOR DE GRAFO
# ══════════════════════════════════════════════════════════════════════════════

class ExtractorGrafo:
    """
    Recorre el árbol XML de una oración y extrae:
      - aristas:       lista de (origen, destino, etiqueta)
      - tipos_nodo:    dict wd → tipo ('verbo', 'nucleo', 'predicativo', 'funcional')
      - postype_nodo:  dict wd → postype XML del verbo ('main', 'auxiliary', 'semiauxiliary')
      - correferencias: lista de (verbo, pronombre_relativo)

    Cada método 'procesar_X' maneja un tipo de sintagma del XML.
    """

    def __init__(self):
        self.aristas        = []   # (origen, destino, etiqueta)
        self.tipos_nodo     = {}   # wd → 'verbo' | 'nucleo' | 'predicativo' | 'funcional'
        self.postype_nodo   = {}   # wd → postype del XML (solo verbos)
        self.correferencias = []   # (verbo, pronombre_relativo) — REGLA 7

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _add_arista(self, origen, destino, etiq='dep'):
        """Agrega una arista evitando bucles y nulos."""
        if origen and destino and origen != destino:
            self.aristas.append((origen, destino, etiq))

    def _add_nodo(self, wd, tipo, postype=None):
        """
        Registra un nodo con su tipo semántico.
        Si el nodo ya existe, no sobreescribe el tipo (el primer registro prevalece).
        Si se proporciona postype (para verbos), lo guarda en postype_nodo.
        """
        if not wd:
            return
        if wd not in self.tipos_nodo:
            self.tipos_nodo[wd] = tipo
        if postype and wd not in self.postype_nodo:
            self.postype_nodo[wd] = postype

    # ── Procesadores por tipo de sintagma ─────────────────────────────────────

    def _proc_grup_nom(self, elem, padre, en_rol=False):
        """
        Procesa <grup.nom>: el primer sustantivo es el núcleo.
        en_rol=True cuando el sintagma padre tiene func/tem,
        lo que hace que el núcleo reciba tipo 'nucleo' (REGLA 2 — propagación).
        """
        nucleo = primera_palabra(elem)
        if not nucleo:
            return
        tipo = 'nucleo' if en_rol else 'funcional'
        self._add_nodo(nucleo, tipo)
        # Procesar modificadores internos, propagando en_rol
        for child in elem:
            if child.tag == 'sp':
                self._proc_sp(child, nucleo, en_rol_padre=en_rol)
            elif child.tag == 'sn':
                self._proc_sn(child, nucleo)
            elif child.tag == 'S':
                self._proc_S(child, nucleo)
            elif child.tag in ('grup.a', 'grup.adv'):
                w = primera_palabra(child)
                if w:
                    self._add_nodo(w, 'funcional')
                    self._add_arista(nucleo, w, 'mod')
            elif child.tag == 's.a':
                self._proc_sa_interno(child, nucleo)

    def _proc_sn(self, elem, padre, etiq='dep', en_rol=False):
        """
        Procesa <sn> (sintagma nominal).
        Estructura: padre → nucleo, nucleo ← spec, nucleo → modificadores.

        El núcleo recibe tipo 'nucleo' si:
          - el propio <sn> tiene func o tem, O
          - en_rol=True (propagado desde un <sp> padre con func/tem)

        REGLA 2 — propagación del color teal.
        """
        func = elem.attrib.get('func', '')
        tem  = elem.attrib.get('tem', '')
        # Combinar rol propio con rol heredado del padre
        tiene_rol = bool(func or tem) or en_rol

        # Buscar spec y núcleo dentro del sn
        nucleo   = None
        spec_wd  = None
        for child in elem:
            if child.tag == 'spec':
                spec_wd = primera_palabra(child)
            elif child.tag == 'grup.nom':
                nucleo = primera_palabra(child)

        if not nucleo:
            nucleo = primera_palabra(elem)
        if not nucleo:
            return nucleo

        tipo = 'nucleo' if tiene_rol else 'funcional'
        self._add_nodo(nucleo, tipo)

        # Conectar al padre y al spec
        if padre:
            self._add_arista(padre, nucleo, etiq)
        if spec_wd:
            self._add_nodo(spec_wd, 'funcional')
            self._add_arista(nucleo, spec_wd, 'spec')

        # Modificadores internos
        for child in elem:
            if child.tag == 'grup.nom':
                self._proc_grup_nom(child, nucleo, en_rol=tiene_rol)
            elif child.tag == 'sp':
                self._proc_sp(child, nucleo, en_rol_padre=tiene_rol)
            elif child.tag == 'S':
                self._proc_S(child, nucleo)
            elif child.tag == 'sn':
                self._proc_sn(child, nucleo)
            elif child.tag == 's.a':
                self._proc_sa_interno(child, nucleo)

        return nucleo

    def _proc_sp(self, elem, padre, en_rol_padre=False):
        """
        Procesa <sp> (sintagma preposicional).
        REGLA 5: la preposición es nodo intermedio entre padre y núcleo interno.
        Estructura: padre → preposición → núcleo_interno

        en_rol_padre=True cuando el padre del sp ya es un núcleo de rol,
        lo que propaga el tipo 'nucleo' al núcleo del sp (REGLA 2 — propagación).
        """
        func   = elem.attrib.get('func', '')
        tem    = elem.attrib.get('tem', '')
        etiq   = f'{func}/{tem}' if (func and tem) else (func or tem or 'sp')
        # El núcleo del sp es 'nucleo' si el sp tiene rol propio O su padre lo tiene
        en_rol = bool(func or tem) or en_rol_padre

        prep = obtener_conj_o_prep(elem)

        # Buscar núcleo interno: primer sn, S o grup.nom
        nucleo_interno = None
        for child in elem:
            if child.tag in ('sn', 'S', 'grup.nom'):
                nucleo_interno = child
                break

        if prep:
            # REGLA 5: prep como nodo intermedio
            self._add_nodo(prep, 'funcional')
            if padre:
                self._add_arista(padre, prep, etiq)
            if nucleo_interno is not None:
                if nucleo_interno.tag == 'sn':
                    self._proc_sn(nucleo_interno, prep, 'sn', en_rol=en_rol)
                elif nucleo_interno.tag == 'S':
                    self._proc_S(nucleo_interno, prep)
                elif nucleo_interno.tag == 'grup.nom':
                    nuc = primera_palabra(nucleo_interno)
                    if nuc:
                        tipo = 'nucleo' if en_rol else 'funcional'
                        self._add_nodo(nuc, tipo)
                        self._add_arista(prep, nuc, 'sn')
                        self._proc_grup_nom(nucleo_interno, nuc, en_rol=en_rol)
        else:
            # Sin preposición: conectar directo
            if nucleo_interno is not None and padre:
                if nucleo_interno.tag == 'sn':
                    self._proc_sn(nucleo_interno, padre, etiq, en_rol=en_rol)
                elif nucleo_interno.tag == 'S':
                    self._proc_S(nucleo_interno, padre)

    def _proc_sa(self, elem, padre):
        """
        Procesa <sa> (sintagma adjetival) con func explícita.
        REGLA 2: los predicativos (cpred/atr) reciben color coral.
        """
        func = elem.attrib.get('func', '')
        tem  = elem.attrib.get('tem', '')
        etiq = f'{func}/{tem}' if (func and tem) else (func or 'cpred')
        nucleo = primera_palabra(elem)
        if nucleo:
            self._add_nodo(nucleo, 'predicativo')
            if padre:
                self._add_arista(padre, nucleo, etiq)

    def _proc_sa_interno(self, elem, padre):
        """
        Procesa <s.a> internos dentro de grup.nom (adjetivos modificadores).
        Se distinguen de los <sa> predicativos: si tienen func/tem son
        predicativos, si no son funcionales.
        """
        func = elem.attrib.get('func', '')
        tem  = elem.attrib.get('tem', '')
        etiq = f'{func}/{tem}' if (func and tem) else (func or 'mod')
        nucleo = primera_palabra(elem)
        if nucleo:
            tipo = 'predicativo' if (func or tem) else 'funcional'
            self._add_nodo(nucleo, tipo)
            if padre:
                self._add_arista(padre, nucleo, etiq)

    def _proc_sadv(self, elem, padre):
        """
        Procesa <sadv> (sintagma adverbial).
        El núcleo recibe tipo 'nucleo' si el sadv tiene func/tem.
        """
        func = elem.attrib.get('func', '')
        tem  = elem.attrib.get('tem', '')
        etiq = f'{func}/{tem}' if (func and tem) else (func or 'cc')
        en_rol = bool(func or tem)

        prep   = obtener_conj_o_prep(elem)
        nucleo = primera_palabra(elem)
        if not nucleo:
            return

        tipo = 'nucleo' if en_rol else 'funcional'
        if prep and prep != nucleo:
            self._add_nodo(prep, 'funcional')
            self._add_nodo(nucleo, tipo)
            if padre:
                self._add_arista(padre, prep, etiq)
            self._add_arista(prep, nucleo, 'sn')
        else:
            self._add_nodo(nucleo, tipo)
            if padre:
                self._add_arista(padre, nucleo, etiq)

    def _proc_S(self, elem, padre, es_raiz=False):
        """
        Procesa <S> (cláusula).
        Puede ser cláusula principal, completiva, relativa o coordinada.

        REGLA 6: conjunciones subordinantes como nodos intermedios.
        REGLA 7: correferencia del pronombre relativo (arista punteada).
        REGLA 8: infinitivos encadenados.
        """
        func  = elem.attrib.get('func', '')
        tem   = elem.attrib.get('tem', '')
        etiq  = f'{func}/{tem}' if (func and tem) else (func or 'S')

        # Obtener el verbo de esta cláusula con su postype
        verbo, verbo_pt = None, 'main'
        for child in elem:
            if child.tag == 'grup.verb':
                verbo, verbo_pt = primer_verbo(child)
                break
        if not verbo:
            for child in elem:
                if child.tag == 'infinitiu':
                    verbo, verbo_pt = primer_verbo(child)
                    break

        if verbo:
            self._add_nodo(verbo, 'verbo', postype=verbo_pt)

        # REGLA 6: conjunción subordinante como nodo intermedio
        conj = obtener_conj_o_prep(elem)

        if not es_raiz and padre and verbo:
            if conj:
                self._add_nodo(conj, 'funcional')
                self._add_arista(padre, conj, etiq)
                self._add_arista(conj, verbo, 'S')
            else:
                self._add_arista(padre, verbo, etiq)

        # Procesar hijos de la cláusula
        for child in elem:
            tag     = child.tag
            ch_func = child.attrib.get('func', '')
            ch_tem  = child.attrib.get('tem', '')
            ch_etiq = f'{ch_func}/{ch_tem}' if (ch_func and ch_tem) \
                      else (ch_func or ch_tem or tag)

            if tag == 'grup.verb':
                pass  # ya procesado arriba

            elif tag == 'sn':
                self._proc_sn(child, verbo, ch_etiq)

            elif tag == 'sp':
                self._proc_sp(child, verbo)

            elif tag == 'sa':
                self._proc_sa(child, verbo)

            elif tag == 'sadv':
                self._proc_sadv(child, verbo)

            elif tag == 'S':
                self._proc_S(child, verbo)

            elif tag == 'conj':
                pass  # ya manejado por obtener_conj_o_prep

            elif tag == 'relatiu':
                # REGLA 7: pronombre relativo — doble función
                # (1) ya conectado como nodo intermedio en REGLA 6
                # (2) correferencia como sujeto del verbo
                rel_wd = primera_palabra(child)
                if rel_wd and verbo:
                    self._add_nodo(rel_wd, 'funcional')
                    self.correferencias.append((verbo, rel_wd))

            elif tag == 'infinitiu':
                # REGLA 8: verbos en cadena (infinitivos)
                inf_v, inf_pt = primer_verbo(child)
                if inf_v and verbo and inf_v != verbo:
                    self._add_nodo(inf_v, 'verbo', postype=inf_pt)
                    self._add_arista(verbo, inf_v, 'inf')
                # Procesar hijos del infinitivo
                for subchild in child:
                    self._proc_hijo_generico(subchild, inf_v or verbo)

            elif tag == 's.a':
                nuc = primera_palabra(child)
                if nuc and verbo:
                    self._add_nodo(nuc, 'nucleo')
                    self._add_arista(verbo, nuc, ch_etiq)

        return verbo

    def _proc_hijo_generico(self, child, padre):
        """Dispatcher genérico para hijos de infinitivos y otros."""
        tag     = child.tag
        ch_func = child.attrib.get('func', '')
        ch_tem  = child.attrib.get('tem', '')
        ch_etiq = f'{ch_func}/{ch_tem}' if (ch_func and ch_tem) \
                  else (ch_func or ch_tem or tag)
        if tag == 'sn':
            self._proc_sn(child, padre, ch_etiq)
        elif tag == 'sp':
            self._proc_sp(child, padre)
        elif tag == 'sa':
            self._proc_sa(child, padre)
        elif tag == 'sadv':
            self._proc_sadv(child, padre)
        elif tag == 'S':
            self._proc_S(child, padre)

    # ── Punto de entrada ──────────────────────────────────────────────────────

    def extraer(self, sent):
        """
        Extrae el grafo completo de una <sentence>.
        Aplica REGLAS 9 y 10 para identificar la raíz y manejar la coordinación.
        Devuelve el wd del verbo raíz (o None).
        """
        # REGLA 10: identificar verbo raíz
        verbo_raiz, raiz_pt = obtener_verbo_raiz(sent)
        if verbo_raiz:
            self._add_nodo(verbo_raiz, 'verbo', postype=raiz_pt)

        # REGLA 9: oraciones coordinadas sin grup.verb directo
        hijos_S = [c for c in sent if c.tag == 'S']
        tiene_grup_verb = any(c.tag == 'grup.verb' for c in sent)
        conj_coord = None
        for c in sent:
            if c.tag == 'conj':
                conj_coord = primera_palabra(c)

        if not tiene_grup_verb and len(hijos_S) >= 2 and conj_coord:
            # Primer S como raíz
            v1 = self._proc_S(hijos_S[0], None, es_raiz=True)
            if not verbo_raiz:
                verbo_raiz = v1
            # Conjunción coordinante como nodo intermedio
            if conj_coord and verbo_raiz:
                self._add_nodo(conj_coord, 'funcional')
                self._add_arista(verbo_raiz, conj_coord, 'coord')
            # S siguientes conectados a través de la conjunción
            for s_elem in hijos_S[1:]:
                v_sig, _ = primer_verbo(s_elem)
                if v_sig and conj_coord:
                    self._add_arista(conj_coord, v_sig, 'S')
                self._proc_S(s_elem, conj_coord or verbo_raiz)

        else:
            # Estructura normal: procesar cada hijo directo de sentence
            for child in sent:
                tag     = child.tag
                ch_func = child.attrib.get('func', '')
                ch_tem  = child.attrib.get('tem', '')
                ch_etiq = f'{ch_func}/{ch_tem}' if (ch_func and ch_tem) \
                          else (ch_func or ch_tem or tag)

                if tag == 'grup.verb':
                    pass  # ya identificado como raíz
                elif tag == 'sn':
                    self._proc_sn(child, verbo_raiz, ch_etiq)
                elif tag == 'sp':
                    self._proc_sp(child, verbo_raiz)
                elif tag == 'sa':
                    self._proc_sa(child, verbo_raiz)
                elif tag == 'sadv':
                    self._proc_sadv(child, verbo_raiz)
                elif tag == 'S':
                    self._proc_S(child, verbo_raiz)
                elif tag == 'neg':
                    neg_wd = primera_palabra(child)
                    if neg_wd and verbo_raiz:
                        self._add_nodo(neg_wd, 'funcional')
                        self._add_arista(verbo_raiz, neg_wd, 'neg')
                elif tag == 'morfema.verbal':
                    mv_wd = primera_palabra(child)
                    if mv_wd and verbo_raiz:
                        self._add_nodo(mv_wd, 'funcional')
                        self._add_arista(verbo_raiz, mv_wd, 'mv')
                elif tag in ('conj', 'f'):
                    pass  # conjunción coordinante y puntuación — ignorar

        return verbo_raiz


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — EXTRACTOR DE LAS 5 PALABRAS MÁS IMPORTANTES
# ══════════════════════════════════════════════════════════════════════════════

def es_auxiliar(wd, postype_nodo=None):
    """
    Determina si un verbo es auxiliar y debe excluirse del top 5.
    REGLA CSV-1.

    Estrategia 1 (preferida): usar postype del XML de AnCora.
      - 'auxiliary'     → auxiliar puro (haber)
      - 'semiauxiliary' → semiauxiliar (ser/estar copulativos)
      - 'main'          → verbo léxico principal → NO es auxiliar

    Estrategia 2 (respaldo): lista manual de modales y perífrasis aspectuales
    que AnCora etiqueta como 'main' pero semánticamente son auxiliares.

    Caso especial: estar con postype='main' seguido de gerundio es perífrasis
    progresiva. Se detecta si el lema es 'estar' y postype es 'main'.
    """
    # Estrategia 1: postype del XML
    if postype_nodo and wd in postype_nodo:
        pt = postype_nodo[wd]
        if pt in ('auxiliary', 'semiauxiliary'):
            return True

    # Estrategia 2: lista manual de modales y perífrasis
    return wd.lower() in MODALES_Y_PERIFRASIS


def extraer_top5(aristas, tipos_nodo, verbo_raiz, postype_nodo=None):
    """
    Extrae las 5 palabras semánticamente más importantes de una oración.
    Aplica REGLAS CSV-1 a CSV-6.

    ALGORITMO:
      1. Construir mapa de dependencias: para cada verbo, sus argumentos
      2. Ordenar verbos principales de mayor a menor jerarquía (raíz primero)
      3. Para cada verbo: seleccionar verbo, agente, demás argumentos
      4. Pasar al siguiente verbo subordinado y repetir hasta llegar a 5
    """
    if not verbo_raiz:
        return []

    # ── Construir mapa padre → [(hijo, etiqueta)] ──────────────────────────
    deps = defaultdict(list)
    for origen, destino, etiq in aristas:
        deps[origen].append((destino, etiq))

    # ── Identificar verbos léxicos principales ─────────────────────────────
    # REGLA CSV-1: excluir auxiliares y semiauxiliares
    verbos_principales = [
        wd for wd, tipo in tipos_nodo.items()
        if tipo == 'verbo' and not es_auxiliar(wd, postype_nodo)
    ]

    # REGLA CSV-6: si no hay ningún verbo léxico (oración copulativa pura),
    # incluir el semiauxiliar como única opción disponible
    if not verbos_principales:
        verbos_principales = [
            wd for wd, tipo in tipos_nodo.items()
            if tipo == 'verbo'
        ]

    # ── Calcular nivel jerárquico de cada verbo ────────────────────────────
    # El verbo raíz es nivel 0; los subordinados tienen nivel mayor.
    nivel_verbo = {}

    def calcular_nivel(nodo, nivel=0, visitados=None):
        if visitados is None:
            visitados = set()
        if nodo in visitados:
            return
        visitados.add(nodo)
        if tipos_nodo.get(nodo) == 'verbo' and not es_auxiliar(nodo, postype_nodo):
            nivel_verbo[nodo] = nivel
        for hijo, _ in deps.get(nodo, []):
            calcular_nivel(hijo, nivel + 1, visitados)

    calcular_nivel(verbo_raiz)

    # Verbos no alcanzados por el recorrido (estructuras atípicas)
    max_niv = max(nivel_verbo.values()) if nivel_verbo else 0
    for v in verbos_principales:
        if v not in nivel_verbo:
            nivel_verbo[v] = max_niv + 1

    # Ordenar verbos por nivel: la raíz primero, subordinados después
    verbos_ordenados = sorted(
        [v for v in verbos_principales if v in nivel_verbo],
        key=lambda v: nivel_verbo[v]
    )

    # ── Obtener argumentos de cada verbo ───────────────────────────────────
    def obtener_argumentos(verbo):
        """
        Devuelve los argumentos del verbo en orden:
          1. Agente (suj con tem=agt)
          2. Demás complementos: suj no-agt, cd, ci, creg, cpred
        REGLA CSV-3: excluye circunstancias (cc y variantes)
        REGLA CSV-4: excluye palabras funcionales (tipo='funcional')
        """
        agentes = []
        otros   = []
        for hijo, etiq in deps.get(verbo, []):
            # REGLA CSV-4: excluir funcionales
            if tipos_nodo.get(hijo) == 'funcional':
                continue
            # REGLA CSV-3: excluir circunstancias
            func_base = etiq.split('/')[0] if '/' in etiq else etiq
            if etiq in FUNCS_CIRCUNSTANCIA or func_base == 'cc':
                continue
            # Clasificar por función
            if etiq in FUNCS_AGENTE:
                agentes.append(hijo)
            elif etiq in FUNCS_ARGUMENTO or \
                 func_base in ('suj', 'cd', 'ci', 'creg', 'cpred', 'atr'):
                otros.append(hijo)

        return agentes + otros

    # ── Recolectar top 5 (REGLA CSV-2) ────────────────────────────────────
    top5 = []
    for verbo in verbos_ordenados:
        if len(top5) >= 5:
            break
        # 1. El verbo mismo
        if verbo not in top5:
            top5.append(verbo)
        # 2. Sus argumentos (agente primero, luego resto)
        for arg in obtener_argumentos(verbo):
            if len(top5) >= 5:
                break
            if arg not in top5:
                top5.append(arg)

    return top5[:5]


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — VISUALIZACIÓN DEL GRAFO
# ══════════════════════════════════════════════════════════════════════════════

def color_nodo(tipo):
    """Devuelve (fondo, borde, texto) según el tipo semántico del nodo."""
    colores = {
        'verbo':       (COLOR_VERBO,  BORDE_VERBO,  TEXTO_VERBO),
        'nucleo':      (COLOR_NUCLEO, BORDE_NUCLEO, TEXTO_NUCLEO),
        'predicativo': (COLOR_PRED,   BORDE_PRED,   TEXTO_PRED),
        'funcional':   (COLOR_FUNC,   BORDE_FUNC,   TEXTO_FUNC),
    }
    return colores.get(tipo, colores['funcional'])


def calcular_layout(G, verbo_raiz, tipos_nodo, aristas):
    """
    Layout jerárquico semántico propio (sin dependencias externas).

    REGLAS DE POSICIONAMIENTO:
      L1. El verbo raíz se coloca en la parte superior central (nivel 0).
      L2. El agente (tem=agt) se eleva medio nivel sobre los otros argumentos
          del mismo verbo (REGLA 3 del diseño).
      L3. Los nodos se organizan por niveles BFS desde la raíz.
      L4. Los subárboles se posicionan en el espacio horizontal asignado
          proporcional a su complejidad (ancho de subárbol).
    """
    if not verbo_raiz or verbo_raiz not in G.nodes:
        nodos = list(G.nodes)
        return {n: (i * 2.0, 0) for i, n in enumerate(nodos)}

    # Mapa de hijos desde las aristas del grafo
    hijos = defaultdict(list)
    for u, v in G.edges():
        hijos[u].append(v)

    # Identificar agentes para elevarlos (REGLA L2)
    nodos_agente = set()
    for origen, destino, etiq in aristas:
        if 'agt' in etiq or etiq in ('suj', 'suj/agt', 'suj/tem'):
            nodos_agente.add(destino)

    # Calcular ancho de subárbol para cada nodo (bottom-up)
    ANCHO_MIN = 2.2

    def ancho_subarbol(nodo, visitados=None):
        if visitados is None:
            visitados = set()
        if nodo in visitados:
            return ANCHO_MIN
        visitados.add(nodo)
        hijos_nodo = [h for h in hijos.get(nodo, []) if h not in visitados]
        if not hijos_nodo:
            return ANCHO_MIN
        return sum(ancho_subarbol(h, visitados) for h in hijos_nodo)

    # Asignar posiciones recursivamente (top-down)
    pos    = {}
    SEP_Y  = 1.8  # separación vertical entre niveles

    def asignar_pos(nodo, x_centro, y, visitados=None):
        if visitados is None:
            visitados = set()
        if nodo in visitados:
            return
        visitados.add(nodo)
        pos[nodo] = (x_centro, y)

        hijos_nodo = [h for h in hijos.get(nodo, []) if h not in visitados]
        if not hijos_nodo:
            return

        anchos = [ancho_subarbol(h, set(visitados)) for h in hijos_nodo]
        total  = sum(anchos)
        x_init = x_centro - total / 2

        for hijo, ancho in zip(hijos_nodo, anchos):
            x_hijo = x_init + ancho / 2
            # REGLA L2: elevar al agente medio nivel
            y_hijo = y - SEP_Y
            if hijo in nodos_agente and nodo == verbo_raiz:
                y_hijo = y - SEP_Y * 0.5
            asignar_pos(hijo, x_hijo, y_hijo, visitados)
            x_init += ancho

    asignar_pos(verbo_raiz, 0, 0)

    # Nodos no alcanzados (ciclos o nodos aislados)
    y_extra = min(p[1] for p in pos.values()) - SEP_Y if pos else -SEP_Y
    for nodo in G.nodes:
        if nodo not in pos:
            pos[nodo] = (len(pos) * ANCHO_MIN, y_extra)

    return pos


def generar_grafo(aristas, tipos_nodo, correferencias, verbo_raiz,
                  titulo, ruta_salida):
    """
    Genera y guarda el grafo como imagen (SVG o PNG).

    REGLAS VISUALES:
      - Sin etiquetas en las aristas (REGLA 4): la información semántica
        está en los colores de los nodos y la posición vertical.
      - Nodos rectangulares redondeados (FancyBboxPatch) para que las
        palabras largas quepan cómodamente.
      - Correferencias con línea punteada (REGLA 7).
      - Fondo blanco para legibilidad.
    """
    import networkx as nx

    G = nx.DiGraph()

    # Agregar nodos
    for nodo, tipo in tipos_nodo.items():
        G.add_node(nodo, tipo=tipo)

    # Agregar aristas sin duplicados
    aristas_vistas = set()
    for origen, destino, etiq in aristas:
        if origen in tipos_nodo and destino in tipos_nodo:
            clave = (origen, destino)
            if clave not in aristas_vistas:
                G.add_edge(origen, destino, tipo=etiq)
                aristas_vistas.add(clave)

    if len(G.nodes) == 0:
        print(f'  [AVISO] Oración sin nodos: {titulo}')
        return

    # Layout jerárquico semántico
    pos = calcular_layout(G, verbo_raiz, tipos_nodo, aristas)

    # Dimensiones del canvas proporcionales al grafo
    xs    = [p[0] for p in pos.values()]
    ys    = [p[1] for p in pos.values()]
    fig_w = max(14, (max(xs) - min(xs)) * 2.2 + 4)
    fig_h = max(10, (max(ys) - min(ys)) * 2.0 + 4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    ax.set_title(titulo, fontsize=11, fontweight='bold', pad=14, color='#333333')

    NODE_H  = 0.35   # alto del nodo en unidades del grafo
    CHAR_W  = 0.13   # ancho aproximado por carácter

    def nodo_ancho(wd):
        return max(0.8, len(wd) * CHAR_W)

    # Dibujar aristas normales ANTES de los nodos (para que queden debajo)
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2) ** 0.5
        if dist == 0:
            continue
        # Retraer la punta al borde del nodo destino
        retro = NODE_H / 2
        x2r   = x2 - (dx / dist) * retro
        y2r   = y2 - (dy / dist) * retro
        ax.annotate('', xy=(x2r, y2r), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color='#999999',
                                   lw=0.8, mutation_scale=14))

    # REGLA 7: correferencias con línea punteada
    for verbo, relativo in correferencias:
        if verbo not in pos or relativo not in pos:
            continue
        ax.annotate('', xy=pos[relativo], xytext=pos[verbo],
                    arrowprops=dict(arrowstyle='-|>', color='#aaaaaa',
                                   lw=0.6, linestyle='dashed',
                                   mutation_scale=11,
                                   connectionstyle='arc3,rad=0.3'))

    # Dibujar nodos como rectángulos redondeados
    for nodo, (x, y) in pos.items():
        tipo             = tipos_nodo.get(nodo, 'funcional')
        fill, borde, txt = color_nodo(tipo)
        w   = nodo_ancho(nodo)
        lw  = 1.5 if tipo == 'verbo' else (1.1 if tipo == 'nucleo' else 0.7)
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - NODE_H / 2), w, NODE_H,
            boxstyle='round,pad=0.06',
            facecolor=fill, edgecolor=borde, linewidth=lw, zorder=3
        )
        ax.add_patch(box)
        fs = 8 if len(nodo) > 14 else (9 if len(nodo) > 9 else 10)
        fw = 'bold' if tipo in ('verbo', 'nucleo') else 'normal'
        ax.text(x, y, nodo, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=txt, zorder=4)

    # Leyenda
    leyenda = [
        mpatches.Patch(facecolor=COLOR_VERBO,  edgecolor=BORDE_VERBO,  label='verbo'),
        mpatches.Patch(facecolor=COLOR_NUCLEO, edgecolor=BORDE_NUCLEO, label='núcleo de rol'),
        mpatches.Patch(facecolor=COLOR_PRED,   edgecolor=BORDE_PRED,   label='predicativo'),
        mpatches.Patch(facecolor=COLOR_FUNC,   edgecolor=BORDE_FUNC,   label='pal. funcional'),
        mpatches.Patch(facecolor='white', edgecolor='#aaaaaa',
                       linestyle='dashed', label='correferencia'),
    ]
    ax.legend(handles=leyenda, loc='lower right', fontsize=8,
              framealpha=0.92, edgecolor='#cccccc')

    plt.tight_layout()
    ext = Path(ruta_salida).suffix.lower()
    fmt = 'svg' if ext == '.svg' else 'png'
    plt.savefig(ruta_salida, format=fmt, bbox_inches='tight',
                dpi=150 if fmt == 'png' else None, facecolor='white')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — PROCESAMIENTO DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════

def procesar_archivo(ruta_xml, carpeta_salida, formato='svg', filas_csv=None):
    """
    Procesa un archivo XML del corpus:
      - Lee todas las <sentence>
      - Extrae el grafo de cada oración y lo guarda
      - Si filas_csv es una lista, agrega una fila por oración con el top 5

    Parámetros:
      ruta_xml       : ruta al archivo *_tbf.xml
      carpeta_salida : carpeta donde guardar los grafos
      formato        : 'svg' o 'png'
      filas_csv      : lista acumuladora (o None si no se quiere CSV)
    """
    nombre_base = Path(ruta_xml).stem

    try:
        tree = ET.parse(ruta_xml)
    except ET.ParseError as e:
        print(f'  [ERROR] No se pudo parsear {ruta_xml}: {e}')
        return

    root      = tree.getroot()
    sentences = root.findall('.//sentence')
    print(f'  {nombre_base}: {len(sentences)} oraciones')

    for idx, sent in enumerate(sentences, start=1):

        # Extraer grafo
        extractor  = ExtractorGrafo()
        verbo_raiz = extractor.extraer(sent)

        if not extractor.tipos_nodo:
            print(f'    [AVISO] Oración {idx} vacía, omitida')
            continue

        # Guardar grafo
        titulo         = f'{nombre_base} — oración {idx}'
        nombre_archivo = f'{nombre_base}_oracion_{idx:02d}.{formato}'
        ruta_grafo     = os.path.join(carpeta_salida, nombre_archivo)

        try:
            generar_grafo(
                extractor.aristas,
                extractor.tipos_nodo,
                extractor.correferencias,
                verbo_raiz,
                titulo,
                ruta_grafo
            )
            print(f'    Guardado: {nombre_archivo}')
        except Exception as e:
            print(f'    [ERROR] Oración {idx}: {e}')

        # Agregar fila al CSV
        if filas_csv is not None:
            top5 = extraer_top5(
                extractor.aristas,
                extractor.tipos_nodo,
                verbo_raiz,
                extractor.postype_nodo
            )
            top5 += [''] * (5 - len(top5))  # rellenar si hay menos de 5
            filas_csv.append({
                'archivo':   nombre_base,
                'oracion':   idx,
                'palabra_1': top5[0],
                'palabra_2': top5[1],
                'palabra_3': top5[2],
                'palabra_4': top5[3],
                'palabra_5': top5[4],
            })


def procesar_corpus(carpeta_entrada, carpeta_salida, formato='svg',
                    solo_archivo=None, solo_oracion=None, generar_csv=True):
    """
    Punto de entrada principal: procesa todos los *_tbf.xml de una carpeta.

    Parámetros:
      carpeta_entrada : carpeta con los archivos XML
      carpeta_salida  : carpeta de salida para grafos y CSV
      formato         : 'svg' o 'png'
      solo_archivo    : nombre parcial para filtrar un archivo concreto
      solo_oracion    : número de oración para pruebas (procesa solo esa)
      generar_csv     : True para generar palabras_importantes.csv
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    # Buscar archivos XML recursivamente
    archivos = sorted(Path(carpeta_entrada).rglob('*_tbf.xml'))
    if not archivos:
        archivos = sorted(Path(carpeta_entrada).rglob('*.xml'))

    if solo_archivo:
        archivos = [f for f in archivos if solo_archivo in f.name]

    if not archivos:
        print(f'No se encontraron archivos XML en: {carpeta_entrada}')
        return

    print(f'Archivos encontrados: {len(archivos)}')

    # Acumulador de filas para el CSV
    filas_csv = [] if generar_csv else None

    for ruta in archivos:
        print(f'\nProcesando: {ruta.name}')

        if solo_oracion is not None:
            # Modo prueba: procesar solo una oración específica
            try:
                tree      = ET.parse(ruta)
                root      = tree.getroot()
                sentences = root.findall('.//sentence')
                if solo_oracion > len(sentences):
                    print(f'  Oración {solo_oracion} no existe '
                          f'(total: {len(sentences)})')
                    continue
                sent       = sentences[solo_oracion - 1]
                extractor  = ExtractorGrafo()
                verbo_raiz = extractor.extraer(sent)
                nombre_base    = ruta.stem
                titulo         = f'{nombre_base} — oración {solo_oracion}'
                nombre_archivo = (f'{nombre_base}_oracion_'
                                  f'{solo_oracion:02d}.{formato}')
                ruta_grafo = os.path.join(carpeta_salida, nombre_archivo)
                generar_grafo(
                    extractor.aristas, extractor.tipos_nodo,
                    extractor.correferencias, verbo_raiz,
                    titulo, ruta_grafo
                )
                print(f'  Guardado: {nombre_archivo}')
                if filas_csv is not None:
                    top5 = extraer_top5(
                        extractor.aristas, extractor.tipos_nodo,
                        verbo_raiz, extractor.postype_nodo)
                    top5 += [''] * (5 - len(top5))
                    filas_csv.append({
                        'archivo':   nombre_base,
                        'oracion':   solo_oracion,
                        'palabra_1': top5[0], 'palabra_2': top5[1],
                        'palabra_3': top5[2], 'palabra_4': top5[3],
                        'palabra_5': top5[4],
                    })
            except Exception as e:
                print(f'  [ERROR]: {e}')
        else:
            procesar_archivo(str(ruta), carpeta_salida, formato, filas_csv)

    # Guardar CSV consolidado
    if filas_csv:
        ruta_csv = os.path.join(carpeta_salida, 'palabras_importantes.csv')
        campos   = ['archivo', 'oracion',
                    'palabra_1', 'palabra_2', 'palabra_3',
                    'palabra_4', 'palabra_5']
        with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(filas_csv)
        print(f'\nCSV guardado: {ruta_csv}  ({len(filas_csv)} filas)')

    print(f'\nProceso completado. Grafos en: {carpeta_salida}')


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — PUNTO DE ENTRADA (línea de comandos)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Genera grafos semántico-sintácticos del corpus AnCora',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input',   '-i', required=True,
                        help='Carpeta con archivos *_tbf.xml')
    parser.add_argument('--output',  '-o', default='grafos_ancora',
                        help='Carpeta de salida (defecto: grafos_ancora/)')
    parser.add_argument('--formato', '-f', choices=['svg', 'png'], default='svg',
                        help='Formato de salida: svg (defecto) o png')
    parser.add_argument('--archivo', '-a', default=None,
                        help='Procesar solo este archivo (nombre parcial)')
    parser.add_argument('--oracion', '-n', type=int, default=None,
                        help='Procesar solo esta oración (número)')
    parser.add_argument('--sin-csv', action='store_true', default=False,
                        help='No generar el CSV de palabras importantes')

    args = parser.parse_args()

    procesar_corpus(
        carpeta_entrada=args.input,
        carpeta_salida=args.output,
        formato=args.formato,
        solo_archivo=args.archivo,
        solo_oracion=args.oracion,
        generar_csv=not args.sin_csv
    )


if __name__ == '__main__':
    main()
