import os
from typing import Set
import pandas as pd
from collections import Counter
import nltk
from glob import glob
import logging

logger = logging.getLogger()


def generar_catalogo(path, cols_to_consider, cols_a_dividr,
                     min_freq=3,
                     nombres_ignorables=tuple(),
                     sep=",") -> Set[str]:
    """
    Genera un catálogo de nombres a partir de un archivo csv.
    :param path: Path al archivo csv con los nombres
    :param cols_to_consider: Nombres de las columnas a considerar
    :param cols_a_dividr: Nombres de las columnas que se dividirán en palabras
    :param min_freq: Frecuencia mínima para considerar un nombre
    :param nombres_ignorables: Nombres que no se considerarán
    :param sep: Separador de columnas
    :return: Conjunto de nombres
    """
    if os.path.isdir(path):
        csv_files = glob(os.path.join(path, "*.csv"))
        df = pd.concat((pd.read_csv(f, encoding='utf-8', sep=sep,
                                    low_memory=False) for f in csv_files))
    if os.path.isfile(path):
        df = pd.read_csv(path, encoding='utf-8', sep=sep, low_memory=False)
    # df.columns = map(str.lower, df.columns)
    # df = df.iloc[:, 3:6]
    nombres_apellidos = set()
    logger.debug(f"Colanmes: {df.columns}")
    for colname in set(cols_to_consider).union(set(cols_a_dividr)):
        lista_columna = [x for x in df[colname].to_list()
                         if isinstance(x, str)]
        if colname in cols_a_dividr:
            lista_divididos = [x.split() for x in lista_columna]
            lista_columna = [x.strip() for y in lista_divididos for x in y]

        # lista_columna contiene una lista de nombres posibles para el catálogo
        freq = Counter(lista_columna)
        conjunto_columna = set([x for x in lista_columna
                                if freq[x] >= min_freq])
        conjunto_columna = conjunto_columna.difference(set(nombres_ignorables))
        nombres_apellidos = nombres_apellidos.union(conjunto_columna)
        logger.debug(f"\t{colname} : "
                     f"{len(lista_columna)} -> "
                     f"{len(conjunto_columna)}")

    return nombres_apellidos


def ngrams_gen(sentence: str, n: int):  # faltaria quitar signos de puntuacion
    '''
    Input:
    -----------
    sentence: A string.
    n: The size of the q-grams to consider.
    Output:
    -----------
    A object of type Counter that contains the n-grams an their occurrence on the bi and three grams.
    '''

    sent_lc = sentence.lower()
    tokens = nltk.word_tokenize(sent_lc)  # generate the word tokens
    two_grams = nltk.ngrams(tokens, 2)  # obtain the 2-token-grams
    two_grams = list(two_grams)
    three_grams = nltk.ngrams(tokens, 3)  # obtain the 3-token-grams
    three_grams = list(three_grams)
    # concatenate the 2-grams and 3-grams lists
    grams = list(two_grams) + list(three_grams) + [(x,) for x in tokens]
    conc_tuples = [''.join(t) for t in grams]  # for each 2,3-gram join them
    # obtain the q-grams and their ocurrence
    count_grams = [Counter(list(nltk.ngrams(tup, n))) for tup in conc_tuples]
    return count_grams, two_grams


