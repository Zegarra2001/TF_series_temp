import os
import numpy as np
import wfdb
from utils.etiquetas import extraer_etiqueta_snomed, snomed_map
from collections import Counter

# Mismo que en tu entrenamiento
def construir_dataset(ruta_base, canal='II'):
    X = []
    y = []
    for subdir, _, files in os.walk(ruta_base):
        for file in files:
            if file.endswith('.hea'):
                nombre = os.path.splitext(file)[0]
                path = os.path.join(subdir, nombre)
                etiqueta = extraer_etiqueta_snomed(path)
                if etiqueta in snomed_map.values():
                    record = wfdb.rdrecord(path)
                    if canal not in record.sig_name:
                        continue
                    fs = record.fs
                    idx = record.sig_name.index(canal)
                    signal = record.p_signal[:, idx]
                    if len(signal) >= fs * 10:
                        segmento = signal[:fs*10]
                        segmento = (segmento - np.mean(segmento)) / np.std(segmento)
                        X.append(segmento)
                        y.append(etiqueta)
    return np.array(X), np.array(y)

# Corre para ver la distribución
ruta = r'C:\Users\Sergio\Documents\Academic-miscelaneous\INFOPUCP\Dimplomatura de Inteligencia Artificial\Módulo 3 (Feb-Abr)\Redes Neuronales\Trabajo final\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords'

X_data, y_data = construir_dataset(ruta)

# Mostrar la distribución de las etiquetas
conteo = Counter(y_data)
print("Distribución de etiquetas:")
for etiqueta, cantidad in conteo.items():
    print(f"{etiqueta}: {cantidad} ejemplos")