from abc import ABC, abstractclassmethod
from typing import List,Tuple
import textdistance
import os
import pandas as pd
from collections import Counter

class BuscadorGenerico(ABC):
    @abstractclassmethod
    def __init__(self, archivo, nombre_columna,thr_distancia):
        pass
    
    @abstractclassmethod
    def buscar(self, texto: str) -> List[Tuple[str, float]]:
        pass


class BuscadorDistancias(BuscadorGenerico):
    
    def __init__(thr_distancia,funciondistancia):
        os.chdir("D:/github/correcci-n_nombres")
        #Crear catálogo de nombres y apellidos
        csv_files = [f for f in os.listdir("raw_data/csv") if f.endswith('.csv')]
        df = pd.concat((pd.read_csv("raw_data/csv/"+f, encoding='utf-8', sep=',', low_memory=False) for f in csv_files))       
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
        self.thr_distancia=thr_distancia
        self.funcion=funciondistancia
      


    def similares (self, texto):
        distancias=[self.funcion(texto, i) for i in self.catalogo]
        distancias.sort(
            key=lambda x: x[1])
        return[(d["nombre"],d["distancia"]) for d in distancias if d["distancia"]<self.thr_distancia]