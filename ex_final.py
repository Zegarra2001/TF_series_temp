import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt

# Función para graficar canal de registro
def graficar_registro_canal(record, nombre, canal = 'I'):

    # Selección de canales
    canales = {i:j for (j,i) in enumerate(record.sig_name)}

    # Parámetros de configuración
    fs = record.fs
    signal = record.p_signal[:, canales[canal]]

    # Duración del segmento a mostrar (en segundos)
    duracion = 10
    n_muestras = int(fs * duracion)
    signal = signal[:n_muestras]
    t = np.linspace(0, duracion, n_muestras)

    # Crear la figura con fondo estilo papel ECG
    fig, ax = plt.subplots(figsize=(24, 4))  # Tamaño similar a papel ECG

    # Dibujar cuadrícula fina (cada 1 mm = 0.04 s, 0.1 mV)
    for x in np.arange(0, duracion, 0.04):
        ax.axvline(x=x, color='lightgrey', linewidth=0.5)
    for y in np.arange(-2, 2, 0.1):
        ax.axhline(y=y, color='lightgrey', linewidth=0.5)

    # Dibujar cuadrícula gruesa (cada 5 mm = 0.20 s, 0.5 mV)
    for x in np.arange(0, duracion, 0.20):
        ax.axvline(x=x, color='grey', linewidth=1)
    for y in np.arange(-2, 2, 0.5):
        ax.axhline(y=y, color='grey', linewidth=1)

    # Trazar señal ECG
    ax.plot(t, signal, color='black', linewidth=1)

    # Etiquetas y formato
    ax.set_xlim([0, duracion])
    ax.set_ylim([-2, 2])  # ajustar según amplitud de señal
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud (mV)')
    ax.set_title(f'ECG - Registro {nombre}')

    # Quitar fondo blanco
    ax.set_facecolor('#fffafa')

    # Mostrar en Streamlit
    st.pyplot(fig)

st.title('Visualización y Análisis de Electrocardiograma')

# Selección de registro
dir = st.selectbox(
    'Seleccione una carpeta:',
    (wfdb.get_record_list('ecg-arrhythmia/1.0.0/')),
)

nombre = st.selectbox(
    'Seleccione el registro a visualizar',
    (wfdb.get_record_list('ecg-arrhythmia/1.0.0/' + dir)),
)

# Selección de canal
record = wfdb.rdrecord(nombre, pn_dir='ecg-arrhythmia/1.0.0/' + dir)
canal = st.selectbox(
    'Canal',
    (record.sig_name),
)

graficar_registro_canal(record, nombre, canal)
graficar_registro_canal(record, nombre, canal)
