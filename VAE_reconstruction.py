import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from VAEs import sampling
# Load the trained VAE model
vae = load_model("C:/Users/Rhola/Desktop/miscalaneous/CSI 4900/vae_model", custom_objects={'sampling': sampling},compile=False)

# Step 1: Load the Test Data
test_data_path = "C:/Users/Rhola/Downloads/X_test.csv" 
test_df = pd.read_csv(test_data_path)

scaler = StandardScaler().fit(test_df)
X_test_scaled = scaler.transform(test_df)

# Step 2: Model Prediction
X_test_pred = vae.predict(X_test_scaled)

# Step 3: Compute Reconstruction Error
reconstruction_error = mean_squared_error(X_test_scaled, X_test_pred, multioutput='raw_values')

# Step 4: Visual and Quantitative Evaluation
plt.figure(figsize=(10, 5))
sns.histplot(reconstruction_error, bins=50)
plt.xlabel('Reconstruction error')
plt.ylabel('Number of samples')
plt.title('Distribution of Reconstruction Errors')
plt.show()

print("Original Values:", X_test_scaled[0])
print("Reconstructed Values:", X_test_pred[0])
