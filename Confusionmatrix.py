import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def categorize_temperature(temp):
    if temp <= 25:
        return 'Low'
    elif temp <= 30:
        return 'Medium'
    else:
        return 'High'

# Function to load, categorize, and compute confusion matrix
def compare_temperatures(cleaned_file_path, predictions_file_path, title):
    cleaned_df = pd.read_csv(cleaned_file_path)
    predictions_df = pd.read_csv(predictions_file_path)

    # Ensure the dataframes are the same length for comparison
    min_length = min(len(cleaned_df), len(predictions_df))
    cleaned_df = cleaned_df.iloc[:min_length]
    predictions_df = predictions_df.iloc[:min_length]

    # Categorize temperatures
    cleaned_df['Actual_Category'] = cleaned_df['TEMPERATURE'].apply(categorize_temperature)
    predictions_df['Predicted_Category'] = predictions_df['Predicted'].apply(categorize_temperature)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(cleaned_df['Actual_Category'], predictions_df['Predicted_Category'], labels=["Low", "Medium", "High"])

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted Categories')
    plt.ylabel('Actual Categories')
    plt.show()

# Paths to the cleaned and prediction files
cleaned_file_path = "C:/Users/Rhola/Desktop/CSI 4900/cleaned_heat_transfer_analysis.csv"
rf_predictions_file_path = "C:/Users/Rhola/Desktop/CSI 4900/rf_predictions.csv"
vae_predictions_file_path = "C:/Users/Rhola/Desktop/CSI 4900/vae_predictions.csv"
xgb_predictions_file_path = "C:/Users/Rhola/Desktop/CSI 4900/xgb_predictions.csv"

# Perform comparisons
compare_temperatures(cleaned_file_path, rf_predictions_file_path, 'RF Predictions')
compare_temperatures(cleaned_file_path, vae_predictions_file_path, 'VAE Predictions')
compare_temperatures(cleaned_file_path, xgb_predictions_file_path, 'XGB Predictions')
