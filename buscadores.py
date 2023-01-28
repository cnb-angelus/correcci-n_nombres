
from abc import ABC, abstractmethod
import nltk
from collections import Counter
import pandas as pd

class Buscador(ABC):

    @abstractmethod
    def __init__(self,file, col_names, dist):
        pass

    @abstractmethod
    def similares(self, text):
        pass


class Buscador_Levenshtein(Buscador):
    def __init__(self, file, num_candidates, threshold, dist):
        df=pd.read_csv(file)
        full_name= df['nombre']+df['apellido_paterno']+df['apellido_materno']


def ngrams_gen(text, n): #faltaria quitar signos de puntuacion
    '''
    Input:
    -----------
    sentence: A string.
    n: The size of the q-grams to consider.
    Output:
    -----------
    A object of type Counter that contains the n-grams an their occurrence on the bi and three grams.
    '''
    
    sent_lc = text.lower()
    tokens = nltk.word_tokenize(sent_lc) # generate the word tokens
    twograms = nltk.ngrams(tokens,2) # obtain the 2-grams
    twograms = list(twograms)
    threegrams = nltk.ngrams(tokens,3) # obtain the 3-grams
    threegrams = list(threegrams)
    grams = list(twograms) + list(threegrams) # concatenate the 2-grams and 3-grams lists
    conc_tuples=[''.join(t) for t in grams] # for each 2,3-gram join them
    count_grams=[Counter(list(nltk.ngrams(tup,n))) for tup in conc_tuples] # obtain the q-grams and their ocurrence
    return count_grams
