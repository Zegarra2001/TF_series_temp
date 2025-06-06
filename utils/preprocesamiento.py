import os
import numpy as np
import wfdb
from utils.etiquetas import extraer_etiqueta_snomed, snomed_map

def construir_dataset(ruta_base, canal='II'):
    """
    Recorre todos los registros del dataset y construye X (señales) y Y (etiquetas).
    
    - ruta_base: ruta al dataset (local/no incluido en el github)
    - canal: derivada a usar
    - duracion: duración máxima del segmento en segundos (por defecto 10)
    """
    
    X = []
    y = []
    clases = ['Sinus Bradycardia', 'Sinus Rhythm', 'Atrial Fibrillation', 'Sinus Tachycardia']

    for subdir, _, files in os.walk(ruta_base):
        for file in files:
            if file.endswith('.hea'):
                nombre = os.path.splitext(file)[0]
                path = os.path.join(subdir, nombre)
                etiquetas = extraer_etiqueta_snomed(path)

                if etiquetas:
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

                        vector = [1 if clase in etiquetas else 0 for clase in clases]
                        y.append(vector)

    return np.array(X), np.array(y)
