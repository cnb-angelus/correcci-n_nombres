import datetime
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from collections import Counter
import nltk
from utils import generar_catalogo, ngrams_gen

from wide_hash import WideHash
import textdistance
import numpy as np

import logging

logger = logging.getLogger()


class Buscador(ABC):
    @abstractmethod
    def __init__(self, path, ncol_name, thr_dist, **kwargs):
        pass

    @abstractmethod
    def similares(self, texto: str) -> List[Tuple[str, float]]:
        pass


class BuscadorDistancias(Buscador):
    def __init__(self, path, thr_dist, fun_dist,
                 n=3,
                 min_freq=2,
                 ncol_name=('PRIMER APELLIDO',
                            'SEGUNDO APELLIDO',
                            'NOMBRE'),
                 nombres_ignorables=('MENOR',)):
        super().__init__(path, ncol_name, thr_dist)
        self.thr_distancia = thr_dist
        self.funcion = fun_dist
        self.n = 3
        self.catalogo = generar_catalogo(path,
                                         cols_to_consider=ncol_name,
                                         min_freq=min_freq,
                                         nombres_ignorables=nombres_ignorables,
                                         cols_a_dividr=['NOMBRE'])

    def similares(self, texto):
        distancias = [(na,self.funcion(texto, na)) for na in self.catalogo]
        distancias.sort(
            key=lambda x: x[1])
        return [(d[0], d[1])
                for d in distancias
                if d[1] < self.thr_distancia]


class BuscadorLSH(Buscador):
    def __init__(self, path, ncol_name, thr_dist,
                 fundist=None,
                 q=3,
                 min_nombre_freq=3,
                 nombres_ignorables=('MENOR',),
                 split_cols=('NOMBRE',),
                 max_ngram_df=0.7,
                 min_ngram_df=0.001,
                 **kwargs):
        super().__init__(path, ncol_name, thr_dist, **kwargs)
        noigs = nombres_ignorables
        self.aprox_por_bucket = 15
        self.num_hashtables = 7

        if fundist is None:
            fundist = textdistance.levenshtein.distance
        self.fundist = fundist
        self.q = q
        self.catalogo = generar_catalogo(path,
                                         cols_to_consider=ncol_name,
                                         min_freq=min_nombre_freq,
                                         nombres_ignorables=noigs,
                                         cols_a_dividr=split_cols)
        self.catalogo = [c.upper() for c in self.catalogo]

        logger.info(f"Se agregaron {len(self.catalogo)} "
                    f"nombres al catalogo de nombres")
        # Este diccionario relaciona 'f','e','r' -> 123
        self.ngram2idx = self.seleccionar_ngramas_caracter(max_df=max_ngram_df,
                                                           min_df=min_ngram_df)
        self.dim = len(self.ngram2idx.keys())
        self.idx2ngram = list(self.ngram2idx.keys())
        # Esta lista tiene como element 123 la tupla 'f','e','r'
        self.idx2ngram.sort(key=lambda k: self.ngram2idx[k])
        self.lsh: Optional[WideHash] = None

    def similares(self, texto: str,
                  num_sim=10,
                  distancia="euclidean",
                  hash_results: Optional[int] = 400,
                  max_width=1) -> List[Tuple[str, float]]:
        if self.lsh is None:
            logger.warning("Indice sin inicializar. Se indizaran ahora los "
                           "nombres del catálogo")
            self.indizar_catalogo()
        texto = texto.upper()
        v = self._vectorizar(texto)
        nn = self.lsh.query(v,
                            num_results=hash_results,
                            distance_func=distancia,
                            max_width=max_width)

        logger.debug(f"El widehash con max_width {max_width}"
                     f" regreso un total de {len(nn)} vecinos cercanoss")
        if self.fundist is not None:
            result = [(nombre, self.fundist(nombre, texto))
                      for ((vect, nombre), distan) in nn]
        else:
            result = [(nombre, distan)
                      for ((vect, nombre), distan) in nn]
        result.sort(key=lambda x: x[1])

        return result[:num_sim] if num_sim else result

    def _vectorizar(self, nombre: str):
        ngrams = set(nltk.ngrams(nombre.upper(), n=self.q))
        ngrams = list(ngrams.intersection(self.ngram2idx.keys()))
        ngram_counts = Counter(ngrams)
        v = np.zeros(self.dim)
        indices = [self.ngram2idx[k] for k in ngrams]
        values = [ngram_counts[i] for i in ngrams]
        v[indices] = values
        v = v / np.linalg.norm(v)

        return v

    def indizar_catalogo(self):
        numbuquets = min(5, int(math.log2(len(
            self.catalogo) / self.aprox_por_bucket)))

        self.lsh = WideHash(hash_size=numbuquets,
                            input_dim=self.dim,
                            num_hashtables=self.num_hashtables)

        st = datetime.datetime.now()
        for nombre in self.catalogo:
            v = self._vectorizar(nombre=nombre)
            self.lsh.index(v, extra_data=nombre)
        et = datetime.datetime.now()
        logger.info(f"Indizar {len(self.catalogo)} nombres "
                    f"tomo {(et - st).total_seconds()} segundos")

    def seleccionar_ngramas_caracter(self,
                                     max_df: float = 0.7,
                                     min_df=0.005):
        all_ngram_dfs = dict()
        for nap in self.catalogo:
            if not isinstance(nap, str):
                logger.error(f"{str(nap)} <-está en el catalogo!")
                break
            ngrams = list(nltk.ngrams(nap.upper(), n=self.q))
            ngram_counts = Counter(ngrams)
            for ng, count in ngram_counts.items():
                all_ngram_dfs[ng] = 1 + all_ngram_dfs.get(ng, 0)

        logger.debug(f"Ngramas antes de filtrar "
                     f"{len(all_ngram_dfs)}")

        to_remove = set()
        totdocs = len(self.catalogo)
        for ngram, dcount in all_ngram_dfs.items():
            if (dcount > (totdocs * max_df) or dcount < (totdocs * min_df)):
                to_remove.add(ngram)

        logger.debug(f"Se van a remover {len(to_remove)} ngramas porque son "
                     f"o muy raros o muy frecuentes. Solo se conservan entre "
                     f"{totdocs * min_df} y {totdocs * max_df}")

        validngrams = set(all_ngram_dfs.keys()).difference(to_remove)
        ngramindex = {ngram: i for i, ngram in enumerate(validngrams)}
        return ngramindex
