import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt

# Obtener la señal de un canal
def extraer_senal(record, canal, duracion=10):
    fs = record.fs
    idx_canal = record.sig_name.index(canal)
    signal = record.p_signal[:, idx_canal]
    n_muestras = int(fs * duracion)
    return signal[:n_muestras], np.linspace(0, duracion, n_muestras)

# Dibujar cuadrícula de papel ECG
def dibujar_cuadricula(ax, duracion, vmin=-2, vmax=2):
    # Cuadrícula fina
    for x in np.arange(0, duracion, 0.04):
        ax.axvline(x=x, color='lightgrey', linewidth=0.5)
    for y in np.arange(vmin, vmax, 0.1):
        ax.axhline(y=y, color='lightgrey', linewidth=0.5)
    # Cuadrícula gruesa
    for x in np.arange(0, duracion, 0.20):
        ax.axvline(x=x, color='grey', linewidth=1)
    for y in np.arange(vmin, vmax, 0.5):
        ax.axhline(y=y, color='grey', linewidth=1)

# Función para graficar un canal
def graficar_registro_canal(record, nombre, canal='I', duracion=10):
    signal, t = extraer_senal(record, canal, duracion)
    fig, ax = plt.subplots(figsize=(12, 4))
    dibujar_cuadricula(ax, duracion)
    ax.plot(t, signal, color='black', linewidth=1)
    ax.set_xlim([0, duracion])
    ax.set_ylim([-2, 2])
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud (mV)')
    ax.set_title(f'ECG - Registro {nombre} - Canal {canal}')
    ax.set_facecolor('#fffafa')
    st.pyplot(fig)

# Función para graficar un registro completo o solo un canal
def graficar_registro(record, nombre, canal='Todos', duracion=10):
    if canal != 'Todos':
        graficar_registro_canal(record, nombre, canal, duracion)
    else:
        for canal_individual in record.sig_name:
            graficar_registro_canal(record, nombre, canal_individual, duracion)


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
    (['Todos'] + record.sig_name),
)

graficar_registro_canal(record, nombre, canal)
graficar_registro_canal(record, nombre, canal)
