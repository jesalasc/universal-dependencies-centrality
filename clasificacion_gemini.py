import stanza
from stanza.utils.conll import CoNLL
import pandas as pd
import logging
# Silenciar los logs informativos de Stanza
logging.getLogger('stanza').setLevel(logging.ERROR)
stanza.download('es')
# --- Configuración ---

# Inicializa el pipeline de Stanza para español con los procesadores necesarios
# 'tokenize': segmenta el texto en oraciones y palabras
# 'mwt': procesa multi-word tokens (ej. 'al', 'del')
# 'pos': Part-of-Speech tagging (categoría gramatical)
# 'lemma': lematización
# 'depparse': análisis de dependencias sintácticas (el corazón de la clasificación)
# nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma,depparse')

# --- Funciones de Clasificación ---

def is_verb(word):
    """Verifica si una palabra es un verbo o un auxiliar."""
    return word.upos in ['VERB', 'AUX']

def get_depth(word, id2word):
        # Root has head = 0 → depth 0
        if word.head == 0:
            return 0
        # Otherwise, 1 + depth of its head
        return 1 + get_depth(id2word[word.head], id2word)

def has_subordination(sentence_words):
    """
    Detecta si una oración contiene relaciones de dependencia que indican subordinación.
    Basado en las relaciones de dependencia de UD relevantes para subordinación.
    """
    subordination_deps = [
        'advcl',    # Cláusulas adverbiales (ej. cuando, porque, si)
        'acl',      # Cláusulas adjetivas (ej. el que)
        'acl:relcl',# Cláusulas relativas (subtipo de acl)
        'ccomp',    # Complementos oracionales (ej. dijo que)
        'csubj',    # Sujetos oracionales (ej. que venga es bueno)
        'csubj:pass', # Sujetos oracionales pasivos
        'xcomp'     # Complementos abiertos (infinitivos, gerundios con sujeto compartido)
    ]
    
    for word in sentence_words:
        if is_verb(word) and word.deprel in subordination_deps:
            # Aseguramos que el verbo no es simplemente un auxiliar del head,
            # lo cual indicaría una perífrasis y no una cláusula subordinada independiente.
            # También comprobamos que el 'head' no es el propio verbo (evitar auto-dependencias si el parser las produce).
            head_word = next((w for w in sentence_words if w.id == word.head), None)
            if head_word and word.deprel != 'aux': # Solo auxiliares directos
                # Para xcomp, ccomp, etc., el verbo subordinado es el que tiene el deprel.
                # Para acl/advcl, el verbo es el head de la cláusula.
                # Esta lógica simplificada asume que el 'deprel' ya marca la subordinación del verbo.
                return True
    return False

def has_coordination(sentence_words, id2word=None):
    """
    Detecta si una oración contiene relaciones de dependencia que indican coordinación
    entre dos o más elementos verbales o cláusulas verbales.
    """
    for word in sentence_words:
        if word.deprel == 'conj' and is_verb(word):
            # Si un verbo tiene una relación 'conj', su 'head' (el elemento coordinado)
            # también debería ser un verbo para indicar coordinación de cláusulas.
            head_word = next((w for w in sentence_words if w.id == word.head), None)
            if head_word and is_verb(head_word):
                return True # Dos verbos coordinados
    return False

# def has_coordination(sentence_words, id2word=None):
#     """
#     Detecta si una oración contiene relaciones de dependencia que indican coordinación
#     entre dos o más elementos verbales o cláusulas verbales.
#     """
#     verbs = [word for word in sentence_words if is_verb(word)]
#     verb_levels = [get_depth(word, id2word) for word in verbs]
#     if len(set(verb_levels)) < len(verb_levels):
#         return True
#     # Verifica si hay conjunciones que conectan verbos
#     return False

def classify_sentence(stanza_sentence_obj):
    """
    Clasifica una oración (objeto Stanza.Sentence) como Simple, Compuesta o Compleja
    basándose en el análisis de dependencias de Stanza.
    """
    sentence_words = stanza_sentence_obj.words

    # Contar verbos principales, excluyendo los auxiliares de perífrasis.
    # Un auxiliar (deprel='aux') no cuenta como un predicado independiente.
    main_verbs_count = 0
    # IDs de los verbos que son el 'head' de una relación 'aux' (es decir, el verbo principal de una perífrasis)
    # y que no deben ser contados como un predicado independiente si solo son 'aux'.
    
    # Obtener IDs de auxiliares que 'apuntan' a un verbo principal
    # y los IDs de los verbos que son 'aux' themselves.
    verb_ids = {w.id for w in sentence_words if is_verb(w)}
    aux_heads_ids = {w.head for w in sentence_words if w.deprel == 'aux' and w.head in verb_ids}
    id2word = {w.id: w for w in sentence_words}

    for word in sentence_words:
        if is_verb(word) and word.id not in aux_heads_ids:
            # Si un verbo no es un auxiliar y no es el 'head' de un auxiliar,
            # entonces lo consideramos un verbo principal de una cláusula.
            main_verbs_count += 1
            
    # Lógica de clasificación
    has_sub = has_subordination(sentence_words)
    has_coord = has_coordination(sentence_words, id2word)

    if main_verbs_count == 1:
        # Con un solo verbo principal, si hay subordinación (ej. Quiero comer), es compleja.
        # Si no hay subordinación, es simple.
        return "Compleja" if has_sub else "Simple"
    
    if main_verbs_count == 0:
        return "Unimembre"  # Oraciones sin verbos principales (ej. ¡Hola!, ¡Qué bonito!)
    # Si hay más de un verbo principal
    if has_sub and has_coord:
        return "Compuesta-Compleja"
    elif has_sub:
        return "Compleja"
    elif has_coord:
        return "Compuesta"
    else:
        # Esto podría indicar múltiples verbos no clasificados por las reglas anteriores
        # (ej. en oraciones muy elípticas o con estructuras no estándar en UD).
        return "Indeterminada (múltiples verbos sin subordinación/coordinación clara)"

# --- Ejecución principal ---
# Define la ruta a tu archivo CoNLL-U del corpus UD 2.8 en español.
# ASEGÚRATE DE QUE ESTE ARCHIVO EXISTE EN LA RUTA ESPECIFICADA.
# Por ejemplo: 'es_gsd-ud-train.conllu' si está en la misma carpeta que tu script.
# O una ruta completa: '/ruta/a/tu/corpus/es_gsd-ud-train.conllu'
corpus_file_path = '../data/deep/UD_Spanish-GSD/es_gsd-ud-train.conllu' # <--- CAMBIA ESTO A LA RUTA DE TU ARCHIVO
output_name = "clasificacion_resultados_filtered.csv"
try:
    # Abrir y leer el archivo CoNLL-U
    # CoNLL.conll2doc() convierte el texto CoNLL-U a un objeto Document de Stanza
    print(f"Abriendo y procesando el archivo: {corpus_file_path}...")
    # Leer directamente el archivo CoNLL-U usando CoNLL.conll2doc
    doc = CoNLL.conll2doc(corpus_file_path)

    print("\n--- Resultados de la clasificación ---")
    rows = []
    for i, sentence in enumerate(doc.sentences):
        classification = classify_sentence(sentence)
        sentence_text = sentence.text  # Obtiene el texto original de la oración
        rows.append({'ID': i+1, 'Clasificación': classification, 'Oración': sentence_text})

    # Crear DataFrame y exportar resultados con ID
    df = pd.DataFrame(rows)
    df.to_csv(output_name, index=False, encoding='utf-8-sig')
    print(f"\nProcesamiento terminado. Resultados guardados en '{output_name}'.")

except FileNotFoundError:
    print(f"Error: El archivo '{corpus_file_path}' no se encontró.")
    print("Asegúrate de que la ruta al archivo es correcta y que el archivo existe.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
