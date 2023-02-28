

buscador = Buscador(....)
rangos : Dict[nombre, rango]  = dict()

def costo_palabra(palabra) -> float:
    """
        :return que tan lejos está la palabra del nombre más cercano
    """
    if palabra in buscador.catalogo:
        return 1
    
    d = buscador.similares(palabra)
    d.sort(key : lambda x:x[1])
    distancia_al_mas_cercano = d[0][1]
    nombre_mas_cercano =  d[0][0]
    # distancia va entre 0 y 1
    
    costo = log(1 + 1000*distancia_al_mas_cercano)  +  alpha*log(rangos[nombre_mas_cercano])
    
    #  Escoger alpha que garantice que 
    # Hipolit0  ->  Hipolito en vez de Hiram
    # Aunque Hipolito sea mucho menos frecuente que Hiram


    return costo

def mejor_nombre(palabra) -> str:
    d = buscador.similares(palabra)
    d.sort(key : lambda x:x[1])
    best_distance = d[0][1]
    empatados = [x[0] in d if x[1] == best_distance]
    empatados.sort(key = lambda x: rango[x])
    best = empatados[0]
    return  best


def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + costo_palabra(s[i-k-1:i]), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))

#nombre_a_limpiar = "Fern4ndof 3rnand ezLop z"
def predictor(nombre_a_limpiar):
    sinespacios = "".join(nombre_a_limpiar.split())  # Fern4ndof3rnandezLopz
    mejor_conespacios = infer_spaces(sinespacios) #  Fern4ndo f3rnandez Lopz
    nombrelimpio = " ".join([mejor_nombre(pedazo)
                            for pedazo in mejor_conespacios.split()])

import random
def predictor_dummy(nombre_a_limpiar):
    if len(nombre_a_limpiar)<2:
        return nombre_a_limpiar
    if random.random() < 0.8:
        return nombre_a_limpiar
    else:
        LL = [l for l in nombre_a_limpiar]
        random.shuffle(LL)
        return "".join(LL)