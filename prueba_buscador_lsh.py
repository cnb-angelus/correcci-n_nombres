import datetime
import logging

import textdistance

from buscadores import BuscadorLSH

logging.basicConfig(level=logging.WARNING)

path = "./raw_data/csv"
colnames = ('PRIMER APELLIDO', 'SEGUNDO APELLIDO', 'NOMBRE')
splitcols = ('NOMBRE',)
buscador = BuscadorLSH(path=path,
                       thr_dist=0,
                       fundist=textdistance.lcsstr.distance,
                       ncol_name=colnames,
                       cols_dividir=splitcols,
                       min_nombre_freq=3,
                       max_ngram_df=0.5,
                       min_ngram_df=0.001,
                       num_hashtables=20
                       )

text2query = "FERNANDES"
for fun in [textdistance.lcsstr.distance, textdistance.jaccard.distance,
            textdistance.hamming.distance, textdistance.levenshtein.distance,
            textdistance.sorensen.distance, textdistance.cosine.distance]:
    buscador.fundist = fun
    print(f"Funcion: {fun}")
    print(buscador.similares(text2query))
    print("")
