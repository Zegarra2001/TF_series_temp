import wfdb
import torch
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import neurokit2 as nk
from models.cnn_model import ECGClassifier
from sklearn.preprocessing import LabelEncoder

# Obtener la señal de un derivada
def extraer_senal(record, derivada):
    fs = record.fs #500 Hz
    idx_derivada = record.sig_name.index(derivada)
    signal = record.p_signal[:, idx_derivada]
    n_muestras = int(fs * 10)  # 10 segundos
    return signal[:n_muestras], np.linspace(0, 10, n_muestras)

# Graficar ECG con cuadrícula tipo papel
def graficar_plotly(signal, t, derivada, nombre, picos=None):
    fig = go.Figure()

    # Señal ECG
    fig.add_trace(go.Scattergl(
        x=t,
        y=signal,
        mode='lines',
        name=f'Derivada {derivada}',
        line=dict(color='black')  # Línea negra
    ))

    # Picos R (opcional)
    if picos is not None:
        picos_scatt = [(t[i], signal[i]) for i in picos["ECG_R_Peaks"]]
        if picos_scatt:
            x_picos, y_picos = zip(*picos_scatt)
            fig.add_trace(go.Scattergl(
                x=x_picos,
                y=y_picos,
                mode='markers',
                name='Picos R',
                marker=dict(color='red', size=8)
            ))

    # Configuración de layout
    fig.update_layout(
        title=dict(
            text=f"ECG - Registro {nombre} - derivada {derivada}",
            font=dict(color='white', size=20)  # Título en blanco
        ),
        plot_bgcolor="white",         # Fondo de área de trazado blanco
        paper_bgcolor="rgba(0,0,0,0)", # Fondo exterior transparente
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            title=dict(text="Tiempo (s)", font=dict(color="white", size=16)),
            range=[0, 10],
            showgrid=True,
            gridcolor="lightpink",  # Cuadrícula fina
            gridwidth=0.5,
            tickmode="linear",
            tick0=0,
            dtick=0.2,
            tickfont=dict(color="white", size=12),  # Ticks blancos
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Amplitud (mV)", font=dict(color="white", size=16)),
            range=[-2, 2],
            showgrid=True,
            gridcolor="lightpink",  # Cuadrícula fina
            gridwidth=0.5,
            tickmode="linear",
            tick0=0,
            dtick=0.5,
            tickfont=dict(color="white", size=12),  # Ticks blancos
            zeroline=False
        )
    )

    # Cuadrícula pequeña (cada 0.04s y 0.1mV)
    for x in np.arange(0, 10.01, 0.04):
        fig.add_shape(type="line", x0=x, x1=x, y0=-2, y1=2,
                      line=dict(color="lightpink", width=0.5), layer="below")
    for y in np.arange(-2, 2.01, 0.1):
        fig.add_shape(type="line", x0=0, x1=10, y0=y, y1=y,
                      line=dict(color="lightpink", width=0.5), layer="below")

    # Cuadrícula gruesa (cada 0.2s y 0.5mV)
    for x in np.arange(0, 10.01, 0.2):
        fig.add_shape(type="line", x0=x, x1=x, y0=-2, y1=2,
                      line=dict(color="white", width=1), layer="below")
    for y in np.arange(-2, 2.5, 0.5):
        fig.add_shape(type="line", x0=0, x1=10, y0=y, y1=y,
                      line=dict(color="white", width=1), layer="below")

    return fig

# Función para graficar registro
def graficar_registro(record, nombre, derivada='Todos'):
    if derivada != 'Todos':
        signal, t = extraer_senal(record, derivada)
        fig = graficar_plotly(signal, t, derivada, nombre)
        st.plotly_chart(fig, use_container_width=True)
    else:
        for derivada_individual in record.sig_name:
            signal, t = extraer_senal(record, derivada_individual)
            fig = graficar_plotly(signal, t, derivada_individual, nombre)
            st.plotly_chart(fig, use_container_width=True)

# Función para obtener frecuencia cardiaca
def obtener_frecuenciacardiaca(picos, fs=500):
    r_peaks_idx = picos['ECG_R_Peaks']
    rr_intervals = np.diff(r_peaks_idx) / fs
    if len(rr_intervals) > 0:
        return int(60 / np.mean(rr_intervals))
    return 0

# Función para predecir etiqueta
def predecir_clase_ecg(signal):
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = modelo(signal)
        pred_idx = output.argmax(dim=1).item()
        
        probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
        df_probs = pd.DataFrame({"Ritmo": classes, "Probabilidad": probs})
        st.bar_chart(df_probs.set_index("Ritmo"))

        return classes[pred_idx]

# Clasificación original desde el .hea
def obtener_clasificacion_real(record):
    clases_validas = {
        '426177001': 'Sinus Bradycardia',
        '426783006': 'Sinus Rhythm',
        '164889003': 'Atrial Fibrillation',
        '427084000': 'Sinus Tachycardia'
    }
    codigos = []
    for comment in record.comments:
        if comment.startswith('#Dx:'):
            codigos = comment.replace('#Dx:', '').strip().split(',')
            break
    etiquetas = [clases_validas[c] for c in codigos if c in clases_validas]
    return etiquetas if etiquetas else ['-']

# Ruta al modelo entrenado y al codificador
modelo_path = 'models/modelo_CNN_MLP.pt'
label_encoder_path = 'models/label_encoder_classes.npy'

# Cargar modelo entrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = ECGClassifier().to(device)
modelo.load_state_dict(torch.load(modelo_path, map_location=device))
modelo.eval()

# Cargar el codificador de etiquetas
classes = np.load(label_encoder_path)

# App en sí
st.title('Visualización y Análisis de Electrocardiograma')

with st.expander("ℹ️ ¿Cómo funciona esta aplicación?"):
    st.markdown("""
    Esta aplicación permite explorar registros de electrocardiogramas (ECG) del dataset *ecg-arrhythmia*.  
    Podrás visualizar las señales en formato similar al papel milimetrado utilizado en cardiología.  
    También puedes seleccionar derivadas específicas y calcular la frecuencia cardíaca automáticamente.  
    El análisis se realiza sobre los fragmentos de 10 segundos.
    """)

col1, col2 = st.columns(2)
with col1:
    dir = st.selectbox(
        'Carpeta:',
        (wfdb.get_record_list('ecg-arrhythmia/1.0.0/'))
    )
with col2:
    nombre = st.selectbox(
        'Registro:',
        (wfdb.get_record_list('ecg-arrhythmia/1.0.0/' + dir))
    )

record = wfdb.rdrecord(nombre, pn_dir='ecg-arrhythmia/1.0.0/' + dir)

col3, col4 = st.columns([1, 3])
with col3:
    seleccion_derivada_manual = st.checkbox('Elegir derivada')

derivada = 'II' # Derivada mostrada por defecto
with col4:
    if seleccion_derivada_manual:
        derivada = st.selectbox('Derivada', ['Todos'] + record.sig_name, index=record.sig_name.index(derivada)+1, key="derivada_manual")

graficar_registro(record, nombre, derivada)

# Sección de cálculo y opciones
_, __, ___, col_fc_der = st.columns([0.25, 0.25, 0.3, 0.2])

with col_fc_der:
    calcular = st.button("Calcular FC", type="primary")

# Acción al presionar el botón
if calcular:
    derivada_elegido = 'II'
    if seleccion_derivada_manual:
        if derivada == 'Todos':
            st.error('Por favor, elija una derivada válida')
            st.stop()
        else:
            derivada_elegido = derivada

    # Extraer exactamente los mismos 10s de señal
    signal_raw, t = extraer_senal(record, derivada_elegido)
    record_limpio = nk.ecg_clean(signal_raw, sampling_rate=500)
    _, picos = nk.ecg_peaks(record_limpio, sampling_rate=500)

    # Visualizar con picos detectados
    fig = graficar_plotly(signal_raw, t, derivada_elegido, nombre, picos=picos)
    st.plotly_chart(fig, use_container_width=True)

    # Calcular frecuencia cardíaca
    frec_cardiaca = obtener_frecuenciacardiaca(picos)

    st.markdown(f'**Frecuencia cardíaca del derivada {derivada_elegido}:** `{frec_cardiaca} lpm`')
    if frec_cardiaca < 60 or frec_cardiaca > 100:
        st.error('⚠️ Frecuencia cardíaca fuera del rango normal (60–100 lpm)')

# Clasificación con modelo
_, __, ___, col_clase_der = st.columns([0.25, 0.25, 0.3, 0.2])

with col_clase_der:
    clasificar = st.button("📊 Clasificar ritmo cardíaco")
if clasificar:
    derivada_pred = derivada if derivada != 'Todos' else 'II'
    signal, _ = extraer_senal(record, derivada_pred)
    clase_predicha = predecir_clase_ecg(signal)
    st.success(f"✅ Ritmo clasificado: **{clase_predicha}**")

    etiquetas_real = obtener_clasificacion_real(record)
    st.info(f"📌 Clasificación original: **{', '.join(etiquetas_real)}**")