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
    hea_file = hea_path if hea_path.endswith('.hea') else hea_path + '.hea'
    try:
        with open(hea_file, 'r') as f:
            for line in f:
                for code in snomed_map:
                    if code in line:
                        return snomed_map[code]
    except Exception as e:
        print(f"Error leyendo {hea_file}: {e}")
    return None