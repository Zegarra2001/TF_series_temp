import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import neurokit2 as nk

# Obtener la señal de un canal
def extraer_senal(record, canal):
    fs = record.fs
    idx_canal = record.sig_name.index(canal)
    signal = record.p_signal[:, idx_canal]
    n_muestras = int(fs * 10) # 10 segundos
    return signal[:n_muestras], np.linspace(0, 10, n_muestras)

# Dibujar cuadrícula de papel ECG
def dibujar_cuadricula(ax, vmin=-2, vmax=2):
    # Cuadrícula fina
    for x in np.arange(0, 10, 0.04): # 10 segundos
        ax.axvline(x=x, color='lightgrey', linewidth=0.5)
    for y in np.arange(vmin, vmax, 0.1):
        ax.axhline(y=y, color='lightgrey', linewidth=0.5)
    # Cuadrícula gruesa
    for x in np.arange(0, 10, 0.20): # 10 segundos
        ax.axvline(x=x, color='grey', linewidth=1)
    for y in np.arange(vmin, vmax, 0.5):
        ax.axhline(y=y, color='grey', linewidth=1)

# Función para graficar un canal
def graficar_registro_canal(record, nombre, canal='I'):
    signal, t = extraer_senal(record, canal)
    fig, ax = plt.subplots(figsize=(24, 4))
    dibujar_cuadricula(ax)
    ax.plot(t, signal, color='black', linewidth=1)
    ax.set_xlim([0, 10])
    ax.set_ylim([-2, 2])
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud (mV)')
    ax.set_title(f'ECG - Registro {nombre} - Canal {canal}')
    ax.set_facecolor('#fffafa')
    return fig, ax

# Función para graficar un registro completo o solo un canal
def graficar_registro(record, nombre, canal='Todos'):
    if canal != 'Todos':
        fig, ax = graficar_registro_canal(record, nombre, canal)
        st.pyplot(fig)
    else:
        for canal_individual in record.sig_name:
            fig, ax = graficar_registro_canal(record, nombre, canal_individual)
            st.pyplot(fig) 

#Función para graficar la señal junto con sus picos
def graficar_picos(picos, record, nombre, canal):
    idx_canal = record.sig_name.index(canal)
    sig_seleccionada = record.p_signal[:, idx_canal]
    t = np.linspace(0, 10, len(sig_seleccionada))
    picos_scatt = np.array([(t[i], sig_seleccionada[i]) 
                            for i in picos['ECG_R_Peaks']])
    fig, ax = graficar_registro_canal(record, nombre, canal)
    ax.scatter(picos_scatt[:, 0], picos_scatt[:, 1])
    st.pyplot(fig)

# Función para obtener frecuencia cardiaca
def obtener_frecuenciacardiaca(picos):
    posiciones_pulsos = picos['ECG_R_Peaks']
    intervalo_RR = (posiciones_pulsos[4] - posiciones_pulsos[3]) * (1/500)
    fc = int(1/intervalo_RR)

    return fc

st.title('Visualización y Análisis de Electrocardiograma')

# Selección de registro
dir = st.selectbox(
    'Seleccione una carpeta:',
    (wfdb.get_record_list('ecg-arrhythmia/1.0.0/')), ## Descargar db: agregar botón
)

nombre = st.selectbox(
    'Seleccione el registro a visualizar',
    (wfdb.get_record_list('ecg-arrhythmia/1.0.0/' + dir)),
)

# Selección de canal
record = wfdb.rdrecord(nombre, pn_dir = 'ecg-arrhythmia/1.0.0/' + dir)
canal = st.selectbox(
    'Canal',
    (['Todos'] + record.sig_name),
)

graficar_registro(record, nombre, canal)

# Obtener rítmo cardiaco
select_confirmada = False
if canal != 'Todos':
    select_confirmada = st.button('Calcular FC usando esta derivada', type = 'primary')

if select_confirmada:
    idx_canal = record.sig_name.index(canal)
    record_limpio = nk.ecg_clean(record.p_signal[:, idx_canal], sampling_rate = 500)
    _, picos = nk.ecg_peaks(record_limpio, sampling_rate = 500) # Para obtener picos

    graficar_picos(picos, record, nombre, canal)
    frec_cardiaca = obtener_frecuenciacardiaca(picos)

    st.text(f'Frecuencia cardiaca conseguida del canal {canal}: ')
    st.write(frec_cardiaca)
