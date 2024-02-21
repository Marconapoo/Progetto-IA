import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE


# Carica il dataset da un file CSV
datasetPath = "dataset.csv"
df = pd.read_csv(datasetPath)

# Seleziona le colonne da utilizzare come caratteristiche per l'addestramento
features_columns = ["RIAGENDR", "PAQ605", "BMXBMI", "LBXGLU", "DIQ010" , "LBXGLT", "LBXIN"]
features = df[features_columns]
# Definisce la variabile target da prevedere
target = df["age_group"]

# Mappa i valori target da stringhe a numeri interi per la classificazione
mapTarget = {"Adult": 0, "Senior": 1}
targetMapped = target.map(mapTarget)

# Divide il dataset in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(features, targetMapped, random_state=42)

# Applica SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Inizializza lo StandardScaler per normalizzare le caratteristiche
scaler = StandardScaler()
# Applica la normalizzazione al set di addestramento e di test
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train_smote
)

class_weights_dict = dict(enumerate(class_weights))


# Definisce il modello di rete neurale sequenziale
model = Sequential([
    Dense(64, activation='relu', input_shape=(7,)),  # Primo strato con funzione di attivazione ReLU
    Dense(64, activation='relu'),  # Secondo strato nascosto con funzione di attivazione ReLU
    Dense(1, activation='sigmoid')  # Strato di output con funzione di attivazione sigmoide per classificazione binaria
])

# Compila il modello specificando ottimizzatore, funzione di perdita e metriche
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Funzione di perdita per la classificazione binaria
              metrics=['accuracy', Precision(), Recall()])

# Addestra il modello sui dati normalizzati
history = model.fit(X_train_scaled, y_train_smote, epochs=8, validation_split=0.2, class_weight = class_weights_dict)

# Prevede le probabilità di appartenenza alla classe per il set di test
y_pred_probs = model.predict(X_test_scaled)

# Converte le probabilità in classificazioni binarie usando una soglia di 0.5
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# Calcola la matrice di confusione per valutare le prestazioni del modello
cm = confusion_matrix(y_test, y_pred)

# Crea e visualizza la matrice di confusione
cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=["Adult", "Senior"])
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Matrice di Confusione')
plt.show()

# Valuta il modello sul set di test per ottenere perdita e accuratezza
loss, accuracy, precision, recall = model.evaluate(X_test_scaled, y_test)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Test set accuracy: {accuracy * 100:.2f}%')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Estrae i dati di accuratezza e perdita dall'addestramento per visualizzarli
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Visualizza l'accuratezza di addestramento e validazione
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Visualizza la perdita di addestramento e validazione
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()