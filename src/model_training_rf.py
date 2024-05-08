import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Trænings- og testdatasæt
train_data = pd.read_csv("./data/train_data.csv")
test_data = pd.read_csv("./data/test_data.csv")

# Adskilning af fun
X_train = train_data.drop('Outcome', axis=1)
y_train = train_data['Outcome']
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']

# Træning af Random Forest-model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction på testdatasættet med features
y_pred = clf.predict(X_test)

# Evaluering af modelpræstation
print("##### Confusion Matrix #####")
print(confusion_matrix(y_test, y_pred))
print("\n##### Classification Report #####")
print(classification_report(y_test, y_pred))

# Visualisering af Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
