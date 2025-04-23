import os
import wfdb
import numpy as np
from tqdm import tqdm

snomed_map = {
    '426783006': 'Sinus Rhythm',
    '426177001': 'Sinus Bradycardia',
    '427084000': 'Sinus Tachycardia',
    '164889003': 'Atrial Fibrillation'
}

def extraer_etiqueta_snomed(hea_path):
    record = wfdb.rdheader(hea_path)
    for comment in record.comments:
        if 'SNOMED CT' in comment:
            for code in snomed_map:
                if code in comment:
                    return snomed_map[code]
    return None