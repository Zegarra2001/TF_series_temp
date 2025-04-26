import torch
import numpy as np
import torch.nn as nn
from utils.preprocesamiento import construir_dataset
from models.cnn_model import ECGClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Preparar datos
ruta = r'C:\Users\Sergio\Documents\Academic-miscelaneous\INFOPUCP\Dimplomatura de Inteligencia Artificial\Módulo 3 (Feb-Abr)\Redes Neuronales\Trabajo final\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords'
X_data, y_data = construir_dataset(ruta)

clases = ['Atrial Fibrillation', 'Sinus Bradycardia', 'Sinus Rhythm', 'Sinus Tachycardia']

y_encoded = []
for etiquetas in y_data:
    multi_label = [1 if clase in etiquetas else 0 for clase in clases]
    y_encoded.append(multi_label)
y_encoded = np.array(y_encoded)

X_tensor = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_encoded, dtype=torch.float32)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

np.save("models/label_encoder_classes.npy", clases)

# Entrenar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

epocas = 10
for epoch in range(epocas):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    total = correct = 0
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        pred = torch.sigmoid(output) > 0.5  # Umbral de decisión
        correct += (pred == y_batch.bool()).all(dim=1).sum().item()
        total += y_batch.size(0)
    print(f"Epoch {epoch+1}/{epocas}: Accuracy {correct/total:.2f}")

torch.save(model.state_dict(), "models/modelo_CNN_MLP.pt")
