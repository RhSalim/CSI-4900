import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Chemins des fichiers
temp_file_path = "C:/Users/Rhola/Downloads/Temp_noeuds1.txt"
coord_file_path = "C:/Users/Rhola/Downloads/Coord_noeuds1.txt"

# Chargement des données
temp_data = pd.read_csv(temp_file_path, sep='\t', header=None)
coord_data = pd.read_csv(coord_file_path, sep='\t', header=None)

# Suppression des premières lignes qui ne sont pas des en-têtes de colonnes
temp_data_cleaned = temp_data.drop(index=0)
coord_data_cleaned = coord_data.drop(index=0)

# Renommage des colonnes
temp_data_cleaned.columns = ['Node', 'Temperature']
coord_data_cleaned.columns = ['Node', 'Unknown', 'X', 'Y', 'Z']

# Remplacement des virgules par des points dans les données de température
temp_data_cleaned['Temperature'] = temp_data_cleaned['Temperature'].str.replace(',', '.').astype(float)


# Extraction des colonnes numériques pour la normalisation
coord_numeric_data = coord_data_cleaned[['X', 'Y', 'Z']]

# Standardisation des données
scaler = StandardScaler()
coord_normalized = scaler.fit_transform(coord_numeric_data)

# Création d'un DataFrame avec les valeurs normalisées
coord_data_normalized = pd.DataFrame(coord_normalized, columns=['X_norm', 'Y_norm', 'Z_norm'])
coord_data_normalized['Node'] = coord_data_cleaned['Node']


# Statistiques descriptives pour les données de température
temp_descriptive_stats = temp_data_cleaned['Temperature'].describe()

# Statistiques descriptives pour les données de coordonnées normalisées
coord_descriptive_stats = coord_data_normalized[['X_norm', 'Y_norm', 'Z_norm']].describe()

# Visualisation de la distribution des températures
plt.figure(figsize=(10, 5))
sns.histplot(temp_data_cleaned['Temperature'], kde=True)
plt.title('Distribution of Temperatures')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
