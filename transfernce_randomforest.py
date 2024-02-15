import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Chemins des fichiers
temp_file_path = "C:/Users/Rhola/Downloads/Temp_noeuds1.txt"
coord_file_path = "C:/Users/Rhola/Downloads/Coord_noeuds1.txt"

# Chargement des données
temp_data = pd.read_csv(temp_file_path, sep='\t', header=None)
coord_data = pd.read_csv(coord_file_path, sep='\t', header=None)

# Nettoyage et préparation des données
temp_data_cleaned = temp_data.drop(index=0)
coord_data_cleaned = coord_data.drop(index=0)
temp_data_cleaned.columns = ['Node', 'Temperature']
coord_data_cleaned.columns = ['Node', 'Unknown', 'X', 'Y', 'Z']
temp_data_cleaned['Temperature'] = temp_data_cleaned['Temperature'].str.replace(',', '.').astype(float)
coord_data_cleaned['X'] = coord_data_cleaned['X'].str.replace(',', '.').astype(float)
coord_data_cleaned['Y'] = coord_data_cleaned['Y'].str.replace(',', '.').astype(float)
coord_data_cleaned['Z'] = coord_data_cleaned['Z'].str.replace(',', '.').astype(float)

# Normalisation des coordonnées
scaler = StandardScaler()
coord_normalized = scaler.fit_transform(coord_data_cleaned[['X', 'Y', 'Z']])
coord_data_normalized = pd.DataFrame(coord_normalized, columns=['X_norm', 'Y_norm', 'Z_norm'])
coord_data_normalized['Node'] = coord_data_cleaned['Node']

# Fusion des données
merged_data = pd.merge(coord_data_normalized, temp_data_cleaned, on='Node')
X = merged_data[['X_norm', 'Y_norm', 'Z_norm']]
y = merged_data['Temperature']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test et calcul de l'erreur MSE
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE sur l'ensemble de test:", mse)

# Analyse des erreurs
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Optimisation des hyperparamètres avec RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
clf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', error_score=0, verbose=3, n_jobs=-1)
clf.fit(X_train, y_train)

# Meilleurs paramètres
print("Meilleurs paramètres:", clf.best_params_)

# Validation croisée plus approfondie
scores = cross_val_score(clf.best_estimator_, X, y, cv=5, scoring='neg_mean_squared_error')
print("MSE moyen de la validation croisée plus approfondie:", -scores.mean(), "avec un écart-type de", scores.std())

# Analyse de l'importance des caractéristiques
feature_importances = pd.Series(clf.best_estimator_.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()