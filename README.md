# TF_series_temp

# ECG Arrhythmia Classifier and Analyzer

Este repositorio contiene una aplicación interactiva desarrollada con Streamlit para el análisis y clasificación automática de señales de electrocardiograma (ECG). Está basado en el dataset `ecg-arrhythmia` de PhysioNet y un modelo de red neuronal CNN+MLP entrenado sobre fragmentos de 10 segundos.

## Funcionalidades principales

- Visualización de señales ECG con cuadrícula tipo papel milimetrado.
- Selección de derivadas específicas por parte del usuario.
- Cálculo automático de la frecuencia cardíaca (FC) a partir de intervalos RR.
- Clasificación del ritmo cardíaco en una de las siguientes clases:
  - Sinus Bradycardia
  - Sinus Rhythm
  - Atrial Fibrillation
  - Sinus Tachycardia
- Comparación entre la clasificación predicha por el modelo y la etiqueta real del archivo `.hea`.

## Estructura del repositorio

TF_series_temp/
│
├── models/
│   ├── cnn_model.py                # Arquitectura de la red neuronal CNN+MLP
│   ├── entrenamiento.py           # Script de entrenamiento del modelo
│   ├── modelo_CNN_MLP.pt          # Modelo entrenado (PyTorch)
│   └── label_encoder_classes.npy  # Codificador de clases (para decodificar la predicción)
│
├── utils/
│   ├── etiquetas.py               # Función para extraer etiquetas desde archivos .hea
│   ├── preprocesamiento.py        # Función para construir dataset desde señales
│
├── ecg-arrhythmia/                # Carpeta descargada desde PhysioNet (no se incluye por tamaño)
│
└── ex_final.py                    # Script principal de la app Streamlit

## Dataset

Este proyecto utiliza el dataset público de PhysioNet:

Nombre: A large-scale 12-lead electrocardiogram database for arrhythmia study  
Link: https://physionet.org/content/ecg-arrhythmia/1.0.0/

> Debido al tamaño del dataset, no está incluido directamente en este repositorio. Se espera que esté disponible en una ruta local para el entrenamiento. Para al funcionamiento de la app, se hace la consulta a tiempo real

## Modelo

El modelo utilizado es una combinación de Convolutional Neural Network (CNN) y Multi-Layer Perceptron (MLP) entrenado sobre la derivada II del ECG. Clasifica en 4 categorías basadas en códigos SNOMED.
