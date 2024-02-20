from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from imblearn.over_sampling import SMOTE

# Carica il dataset da un file CSV
df = pd.read_csv("dataset.csv")

# Seleziona le colonne da utilizzare come caratteristiche e come target
features_columns = ["RIAGENDR", "PAQ605", "BMXBMI", "LBXGLU", "DIQ010", "LBXGLT", "LBXIN"]
features = df[features_columns]
target = df["age_group"]

# Mappa i valori target da stringhe a numeri interi per la classificazione
mapTarget = {"Adult": 0, "Senior": 1}
targetMapped = target.map(mapTarget)

# Divide il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(features, targetMapped, test_size=0.3, random_state=0, stratify=targetMapped)

smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Addestra il classificatore Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')
rf.fit(X_train_smote, y_train_smote)

# Effettua previsioni sul set di test
predictions = rf.predict(X_test)

# Calcola metriche di valutazione di base
accuracy = metrics.accuracy_score(y_test, predictions)
precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
f1 = metrics.f1_score(y_test, predictions)

# Stampa le metriche di valutazione
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Calcola la matrice di confusione
cm = confusion_matrix(y_test, predictions)

# Crea e visualizza la matrice di confusione
cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=["Adult", "Senior"])
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
