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
                       fundist=textdistance.levenshtein.distance,
                       ncol_name=colnames,
                       cols_dividir=splitcols,
                       min_nombre_freq=3,
                       max_ngram_df=0.08,
                       min_ngram_df=0.001,
                       )

buscador.indizar_catalogo()

numqueries = 10
top = 5
for mw in [0,1]:
    for hr in [10,100,200,1000]:
        st = datetime.datetime.now()
        for i in range(numqueries):
            results = buscador.similares("FERN4N O",
                                         num_sim=45,
                                         max_width=mw,
                                         hash_results=hr)
        et = datetime.datetime.now()
        print(f"\nmw:{mw}\thr{hr}\tt:{(et-st).total_seconds()} \t in {numqueries} "
              f"queries")
        for n,s in results[:top]:
            print("\t",n)

for n, s in results:
    print(f"{n}  \t:  {s}")
