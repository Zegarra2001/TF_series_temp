import wfdb

# Diccionario SNOMED
snomed_map = {
    '426177001': 'Sinus Bradycardia',
    '426783006': 'Sinus Rhythm',
    '164889003': 'Atrial Fibrillation',
    '427084000': 'Sinus Tachycardia'
}

# Función para extraer etiquetas
def extraer_etiqueta_snomed(path):
    record = wfdb.rdheader(path)
    etiquetas = []
    for comment in record.comments:
        if comment.startswith('Dx:'):
            codigos = comment.replace('Dx:', '').strip().split(',')
            for code in codigos:
                if code in snomed_map:
                    etiquetas.append(snomed_map[code])
    return etiquetas
