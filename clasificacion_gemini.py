import stanza
from stanza.utils.conll import CoNLL

# --- Configuración ---
# Asegúrate de haber descargado el modelo de español al menos una vez:
# stanza.download('es')

# Inicializa el pipeline de Stanza para español con los procesadores necesarios
# 'tokenize': segmenta el texto en oraciones y palabras
# 'mwt': procesa multi-word tokens (ej. 'al', 'del')
# 'pos': Part-of-Speech tagging (categoría gramatical)
# 'lemma': lematización
# 'depparse': análisis de dependencias sintácticas (el corazón de la clasificación)
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma,depparse')

# --- Funciones de Clasificación ---

def is_verb(word):
    """Verifica si una palabra es un verbo o un auxiliar."""
    return word.upos in ['VERB', 'AUX']

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
        # Solo consideramos palabras que son verbos o auxiliares y tienen una dependencia de subordinación
        if is_verb(word) and word.deprel in subordination_deps:
            # Aseguramos que el verbo no es simplemente un auxiliar del head,
            # lo cual indicaría una perífrasis y no una cláusula subordinada independiente.
            # También comprobamos que el 'head' no es el propio verbo (evitar auto-dependencias si el parser las produce).
            head_word = next((w for w in sentence_words if w.id == word.head.id), None)
            if head_word and word.deprel != 'aux': # Solo auxiliares directos
                # Para xcomp, ccomp, etc., el verbo subordinado es el que tiene el deprel.
                # Para acl/advcl, el verbo es el head de la cláusula.
                # Esta lógica simplificada asume que el 'deprel' ya marca la subordinación del verbo.
                return True
    return False

def has_coordination(sentence_words):
    """
    Detecta si una oración contiene relaciones de dependencia que indican coordinación
    entre dos o más elementos verbales o cláusulas verbales.
    """
    for word in sentence_words:
        if word.deprel == 'conj' and is_verb(word):
            # Si un verbo tiene una relación 'conj', su 'head' (el elemento coordinado)
            # también debería ser un verbo para indicar coordinación de cláusulas.
            head_word = next((w for w in sentence_words if w.id == word.head.id), None)
            if head_word and is_verb(head_word):
                return True # Dos verbos coordinados
    return False

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
    aux_heads_ids = {w.head.id for w in sentence_words if w.deprel == 'aux' and w.head.id in verb_ids}
    
    for word in sentence_words:
        if is_verb(word) and word.id not in aux_heads_ids:
            # Si un verbo no es un auxiliar y no es el 'head' de un auxiliar,
            # entonces lo consideramos un verbo principal de una cláusula.
            main_verbs_count += 1
            
    # Lógica de clasificación
    has_sub = has_subordination(sentence_words)
    has_coord = has_coordination(sentence_words)

    if main_verbs_count <= 1:
        # Con un solo verbo principal, si hay subordinación (ej. Quiero comer), es compleja.
        # Si no hay subordinación, es simple.
        return "Compleja" if has_sub else "Simple"
    
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
if __name__ == "__main__":
    # Define la ruta a tu archivo CoNLL-U del corpus UD 2.8 en español.
    # ASEGÚRATE DE QUE ESTE ARCHIVO EXISTE EN LA RUTA ESPECIFICADA.
    # Por ejemplo: 'es_gsd-ud-train.conllu' si está en la misma carpeta que tu script.
    # O una ruta completa: '/ruta/a/tu/corpus/es_gsd-ud-train.conllu'
    corpus_file_path = 'es_gsd-ud-train.conllu' # <--- CAMBIA ESTO A LA RUTA DE TU ARCHIVO

    try:
        # Abrir y leer el archivo CoNLL-U
        # CoNLL.conll2doc() convierte el texto CoNLL-U a un objeto Document de Stanza
        print(f"Abriendo y procesando el archivo: {corpus_file_path}...")
        with open(corpus_file_path, 'r', encoding='utf-8') as f:
            # Lee todo el contenido del archivo
            conllu_text = f.read()
            # Convierte el texto CoNLL-U en un objeto Document de Stanza
            doc = CoNLL.conll2doc(conllu_text)

        # Iterar sobre cada oración en el documento y clasificarla
        print("\n--- Resultados de la clasificación ---")
        classified_results = {}
        for i, sentence in enumerate(doc.sentences):
            classification = classify_sentence(sentence)
            sentence_text = sentence.text # Obtiene el texto original de la oración
            
            if classification not in classified_results:
                classified_results[classification] = []
            classified_results[classification].append(sentence_text)

            # Opcional: Imprimir cada oración y su clasificación
            # print(f"Oración {i+1}: '{sentence_text}' -> {classification}")
        
        print("\n--- Resumen de Clasificación ---")
        for category, sentences in classified_results.items():
            print(f"\nCategoría: {category} ({len(sentences)} oraciones)")
            # Imprimir las primeras 5 oraciones de cada categoría para revisión
            for j, sent_text in enumerate(sentences[:5]):
                print(f"  - '{sent_text}'")
            if len(sentences) > 5:
                print(f"  ... y {len(sentences) - 5} más.")
                
        print("\nProcesamiento terminado.")

    except FileNotFoundError:
        print(f"Error: El archivo '{corpus_file_path}' no se encontró.")
        print("Asegúrate de que la ruta al archivo es correcta y que el archivo existe.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
