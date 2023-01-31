from abc import ABC, abstractclassmethod
from typing import List,Tuple
import textdistance
import os
import pandas as pd
from collections import Counter
import nltk
from glob import glob

class Buscador(ABC):
    @abstractclassmethod
    def __init__(self, path, ncol_name,thr_dist):
        pass
    
    @abstractclassmethod
    def similares(self, texto: str) -> List[Tuple[str, float]]:
        pass


class BuscadorDistancias(Buscador):
    
    def __init__(path,thr_dist,fun_dist):
        self.thr_distancia=thr_dist
        self.funcion=fun_dist
        #Crear catálogo de nombres y apellidos
        #csv_files = [f for f in os.listdir(os.path.join(path,"raw_data/csv")) if f.endswith('.csv')]
        csv_files  = glob(os.path.join(path,"*.csv"))
        df = pd.concat((pd.read_csv(f, encoding='utf-8', sep=',', low_memory=False) for f in csv_files))
        df.columns = map(str.lower, df.columns)
        df = df.iloc[:, 3:6]
        df.columns = ["primer_apellido", "segundo_apellido", "nombre"]
        for i in df.columns:
            df = df[df[i] != "MENOR"]
            df = df.dropna()
        df["nombre"] = df["nombre"].str.split(" ").str[0]
        #Crear una lista de apellidos y nombres"
        apellidos = df.primer_apellido.tolist() + df.segundo_apellido.tolist()
        nombres = df.nombre.tolist()
        #Eliminar registros que se repitan 3 veces o menos"
        freq_apellidos = Counter(apellidos)
        freq_nombres = Counter(nombres)
        apellidos = [x for x in apellidos if freq_apellidos[x] >= 2]
        nombres = [x for x in nombres if freq_nombres[x] >= 2]
        #Dejar valores únicos
        apellidos = list(set(apellidos))
        nombres = list(set(nombres))
        #Hacer una lista general de nombres y apellidos
        nombres_apellidos = apellidos + nombres
        self.catalogo = nombres_apellidos

    def similares(self, texto):
        distancias=[self.funcion(texto, na) for na in self.catalogo]
        distancias.sort(
            key=lambda x: x[1])
        return[(d["nombre"],d["distancia"]) for d in distancias if d["distancia"]<self.thr_distancia]


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
