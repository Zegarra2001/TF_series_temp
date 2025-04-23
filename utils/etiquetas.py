import os
import wfdb

# Diccionario SNOMED
snomed_map = {
    '426177001': 'Sinus Bradycardia',
    '426783006': 'Sinus Rhythm',
    '164889003': 'Atrial Fibrillation',
    '427084000': 'Sinus Tachycardia'
}

# Función para extraer etiquetas
def extraer_etiqueta_snomed(hea_path):
    record = wfdb.rdheader(hea_path)
    for comment in record.comments:
        if 'SNOMED CT' in comment:
            for code in snomed_map:
                if code in comment:
                    return snomed_map[code]
    return None