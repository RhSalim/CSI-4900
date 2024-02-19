
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


temp_data = pd.read_csv('/System/Volumes/Data/Users/maroua/Downloads/CSI-4900-main/Cleaned_Temperature_Data.csv')
coord_data = pd.read_csv('/System/Volumes/Data/Users/maroua/Downloads/CSI-4900-main/Normalized_Coordinate_Data.csv')


combined_data = pd.merge(temp_data, coord_data, on='Node')


features = combined_data[['Temperature', 'X_norm', 'Y_norm', 'Z_norm']]


_, X_test = train_test_split(features, test_size=0.2, random_state=42)

# Save X_test as CSV
X_test.to_csv('X_test.csv', index=False)
