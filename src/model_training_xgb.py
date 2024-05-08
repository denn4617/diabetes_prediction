import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
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

# Konverter funktioner og labels til DMatrix-format
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# Opret XGBClassifier og definer parametre
clf = xgb.XGBClassifier(random_state=42)

# Opsæt hyperparameter-gitteret for Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Udfør Grid Search med 5-fold cross-validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,
                           scoring='accuracy', verbose=1, n_jobs=-1)

# Træn Grid Search på træningsdataene
grid_search.fit(X_train, y_train)

# Udskriv de bedste parametre og den tilhørende nøjagtighed
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Brug den bedste model til at lave forudsigelser på testdataene
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# clf.fit(X_train, y_train)

# # Prediction på testdatasættet med features
# y_pred = clf.predict(X_test)

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
