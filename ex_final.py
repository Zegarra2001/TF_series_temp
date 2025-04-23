import wfdb
import torch
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import neurokit2 as nk
from models.cnn_model import ECGClassifier
from sklearn.preprocessing import LabelEncoder

# Obtener la señal de un canal
def extraer_senal(record, canal):
    fs = record.fs
    idx_canal = record.sig_name.index(canal)
    signal = record.p_signal[:, idx_canal]
    n_muestras = int(fs * 10)  # 10 segundos
    return signal[:n_muestras], np.linspace(0, 10, n_muestras)

# Graficar ECG con cuadrícula tipo papel
def graficar_plotly(signal, t, canal, nombre, picos=None):
    fig = go.Figure()

    # Señal
    fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name=f'Derivada {canal}', line=dict(color='black')))

    # Añadir picos si existen
    if picos is not None:
        picos_scatt = [(t[i], signal[i]) for i in picos["ECG_R_Peaks"]]
        if picos_scatt:
            x_picos, y_picos = zip(*picos_scatt)
            fig.add_trace(go.Scatter(x=x_picos, y=y_picos, mode='markers', name='Picos R', marker=dict(color='red', size=8)))

    # Cuadrícula tipo ECG
    for x in np.arange(0, 10, 0.04):
        fig.add_shape(type="line", x0=x, x1=x, y0=-2, y1=2,
                      line=dict(color="LightPink", width=0.5), layer="below")
    for y in np.arange(-2, 2.1, 0.1):
        fig.add_shape(type="line", x0=0, x1=10, y0=y, y1=y,
                      line=dict(color="LightPink", width=0.5), layer="below")
    for x in np.arange(0, 10, 0.2):
        fig.add_shape(type="line", x0=x, x1=x, y0=-2, y1=2,
                      line=dict(color="LightPink", width=1.5), layer="below")
    for y in np.arange(-2, 2.5, 0.5):
        fig.add_shape(type="line", x0=0, x1=10, y0=y, y1=y,
                      line=dict(color="LightPink", width=1.5), layer="below")

    fig.update_layout(
        title=f"ECG - Registro {nombre} - Canal {canal}",
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud (mV)",
        xaxis=dict(range=[0, 10], dtick=0.2),
        yaxis=dict(range=[-2, 2]),
        plot_bgcolor="white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    return fig

# Función para graficar registro
def graficar_registro(record, nombre, canal='Todos'):
    if canal != 'Todos':
        signal, t = extraer_senal(record, canal)
        fig = graficar_plotly(signal, t, canal, nombre)
        st.plotly_chart(fig, use_container_width=True)
    else:
        for canal_individual in record.sig_name:
            signal, t = extraer_senal(record, canal_individual)
            fig = graficar_plotly(signal, t, canal_individual, nombre)
            st.plotly_chart(fig, use_container_width=True)

# Función para obtener frecuencia cardiaca
def obtener_frecuenciacardiaca(picos):
    fc_array = nk.ecg_rate(picos, sampling_rate=500)
    return int(np.mean(fc_array))

# Función para predecir etiqueta
def predecir_clase_ecg(signal):
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = modelo(signal)
        pred_idx = output.argmax(dim=1).item()
        return classes[pred_idx]


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
    seleccion_canal_manual = st.checkbox('Elegir canal')

canal = 'V4'
with col4:
    if seleccion_canal_manual:
        canal = st.selectbox('Derivada', ['Todos'] + record.sig_name, index=record.sig_name.index(canal)+1, key="canal_manual")

graficar_registro(record, nombre, canal)

# Sección de cálculo y opciones
_, __, col_fc_izq, col_fc_der = st.columns([0.25, 0.25, 0.3, 0.2])

with col_fc_der:
    calcular = st.button("Calcular FC", type="primary")

# Acción al presionar el botón
if calcular:
    canal_elegido = 'V4'
    if seleccion_canal_manual:
        if canal == 'Todos':
            st.error('Por favor, elija una derivada válida')
            st.stop()
        else:
            canal_elegido = canal

    idx_canal = record.sig_name.index(canal_elegido)
    record_limpio = nk.ecg_clean(record.p_signal[:, idx_canal], sampling_rate=500)
    _, picos = nk.ecg_peaks(record_limpio, sampling_rate=500)

    signal, t = extraer_senal(record, canal_elegido)
    fig = graficar_plotly(signal, t, canal_elegido, nombre, picos=picos)
    st.plotly_chart(fig, use_container_width=True)

    frec_cardiaca = obtener_frecuenciacardiaca(picos)
    st.markdown(f'**Frecuencia cardíaca del canal {canal_elegido}:** `{frec_cardiaca} lpm`')
    if frec_cardiaca < 60 or frec_cardiaca > 100:
        st.error('⚠️ Frecuencia cardíaca fuera del rango normal (60–100 lpm)')

# Clasificación con modelo
_, __, ___, col_clase_der = st.columns([0.25, 0.25, 0.3, 0.2])

with col_clase_der:
    clasificar = st.button("📊 Clasificar ritmo cardíaco")
if clasificar:
    canal_pred = canal if canal != 'Todos' else 'II'
    signal, _ = extraer_senal(record, canal_pred)
    clase_predicha = predecir_clase_ecg(signal)
    st.success(f"✅ Ritmo clasificado: **{clase_predicha}**")

    probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
    df_probs = pd.DataFrame({"Ritmo": classes, "Probabilidad": probs})
    st.bar_chart(df_probs.set_index("Ritmo"))
