from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pydotplus
from IPython.display import Image
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Importa il dataset da un file CSV
datasetPath = "dataset.csv"
df = pd.read_csv(datasetPath)

# Seleziona le colonne da utilizzare come caratteristiche e come target
features_columns = ["RIAGENDR", "PAQ605", "BMXBMI", "LBXGLU", "DIQ010", "LBXGLT", "LBXIN"]
features = df[features_columns]
target = df["age_group"]

# Mappa i valori target da stringhe a numeri interi per la classificazione
mapTarget = {"Adult": 0, "Senior": 1}
targetMapped = target.map(mapTarget)

# Divide il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(features, targetMapped, random_state=0)

smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Addestra il classificatore ad albero di decisione
dtree = DecisionTreeClassifier(random_state=0, class_weight="balanced")
dtree.fit(X_train_smote, y_train_smote)

# Effettua previsioni sul set di test
predictions = dtree.predict(X_test)

# Nomi delle caratteristiche per la visualizzazione dell'albero di decisione
featureNames = ["Genere", "Attivita' sportiva", "Indice Massa Corporea", "Glucosio", "Diabete", "Test di tolleranza al glucosio orale", "Insulina"]

# Visualizza l'albero di decisione
dot_data = export_graphviz(dtree, out_file=None, feature_names=featureNames, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
# Salva l'immagine dell'albero di decisione
graph.write_png('tree.png')

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
plt.title('Matrice di Confusione')
plt.show()
