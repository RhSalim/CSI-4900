import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers, Model

def load_and_clean_data(temp_file_path, coord_file_path):
    # Load data
    temp_data = pd.read_csv(temp_file_path, sep='\t', header=None)
    coord_data = pd.read_csv(coord_file_path, sep='\t', header=None)

    # Clean data by dropping first row and renaming columns
    temp_data_cleaned = temp_data.drop(index=0).rename(columns={0: 'Node', 1: 'Temperature'})
    coord_data_cleaned = coord_data.drop(index=0).rename(columns={0: 'Node', 1: 'Unknown', 2: 'X', 3: 'Y', 4: 'Z'})

    # Convert temperature data to float
    temp_data_cleaned['Temperature'] = temp_data_cleaned['Temperature'].str.replace(',', '.').astype(float)

    return temp_data_cleaned, coord_data_cleaned

def standardize_coordinates(coord_data_cleaned):
    # Extract numeric columns for normalization
    coord_numeric_data = coord_data_cleaned[['X', 'Y', 'Z']]

    # Standardize data
    scaler = StandardScaler()
    coord_normalized = scaler.fit_transform(coord_numeric_data)

    # Create a DataFrame with normalized values
    coord_data_normalized = pd.DataFrame(coord_normalized, columns=['X_norm', 'Y_norm', 'Z_norm'])
    coord_data_normalized['Node'] = coord_data_cleaned['Node'].values

    return coord_data_normalized

def save_data_as_excel_and_csv(data, excel_file_name, csv_file_name):
    # Save the data to Excel and CSV formats
    data.to_excel(excel_file_name, index=False)
    data.to_csv(csv_file_name, index=False)

def plot_temperature_distribution(temp_data_cleaned):
    # Visualize the distribution of temperatures
    plt.figure(figsize=(10, 5))
    sns.histplot(temp_data_cleaned['Temperature'], kde=True)
    plt.title('Distribution of Temperatures')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def main(temp_file_path, coord_file_path):
    # Load and clean data
    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)

    # Standardize coordinates
    coord_data_normalized = standardize_coordinates(coord_data_cleaned)

    # Save cleaned and normalized data
    save_data_as_excel_and_csv(temp_data_cleaned, 'Cleaned_Temperature_Data.xlsx', 'Cleaned_Temperature_Data.csv')
    save_data_as_excel_and_csv(coord_data_normalized, 'Normalized_Coordinate_Data.xlsx', 'Normalized_Coordinate_Data.csv')

    # Plot temperature distribution
    plot_temperature_distribution(temp_data_cleaned)

# Load the datasets
temp_file_path = "C:/Users/Rhola/Desktop/CSI 4900/Temp_noeuds1.txt"
coord_file_path = "C:/Users/Rhola/Desktop/CSI 4900/Coord_noeuds1.txt"

# Run the main function with the file paths
main(temp_file_path, coord_file_path)

def merge_and_prepare_data(coord_data_normalized, temp_data_cleaned):
    # Merge coordinate and temperature data on 'Node'
    merged_data = pd.merge(coord_data_normalized, temp_data_cleaned, on='Node')
    X = merged_data[['X_norm', 'Y_norm', 'Z_norm']]
    y = merged_data['Temperature']
    return X, y

def train_test_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and calculate MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE on the test set: {mse}")

    return model, X_train, y_train, X_test, y_test, y_pred

def plot_residuals(y_test, y_pred):
    # Plot residuals
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()

def hyperparameter_optimization(X_train, y_train):
    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0]
    }

    # Initialize RandomizedSearchCV
    clf = RandomizedSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', error_score=0, verbose=3, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f"Best parameters: {clf.best_params_}")

    return clf.best_estimator_

def main(temp_file_path, coord_file_path):
    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)
    coord_data_normalized = standardize_coordinates(coord_data_cleaned)
    save_data_as_excel_and_csv(temp_data_cleaned, 'Cleaned_Temperature_Data.xlsx', 'Cleaned_Temperature_Data.csv')
    save_data_as_excel_and_csv(coord_data_normalized, 'Normalized_Coordinate_Data.xlsx', 'Normalized_Coordinate_Data.csv')

    X, y = merge_and_prepare_data(coord_data_normalized, temp_data_cleaned)
    model, X_train, y_train, X_test, y_test, y_pred = train_test_model(X, y)

    best_model = hyperparameter_optimization(X_train, y_train)

main(temp_file_path, coord_file_path)

def train_test_random_forest_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions and calculate MSE
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE on the test set with RandomForest: {mse}")

    return rf_model, X_train, y_train, X_test, y_test, y_pred

def hyperparameter_optimization_rf(X_train, y_train):
    # Parameter distribution for RandomizedSearchCV for RandomForest
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize RandomizedSearchCV for RandomForest
    rf = RandomForestRegressor(random_state=42)
    clf_rf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', error_score=0, verbose=3, n_jobs=-1)
    clf_rf.fit(X_train, y_train)

    print(f"Best parameters for RandomForest: {clf_rf.best_params_}")

    return clf_rf.best_estimator_

def feature_importance_analysis(model, X):
    # Analyze feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance with RandomForest')
    plt.show()

# Integrating the RandomForest functions into the main workflow

def main(temp_file_path, coord_file_path):
    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)
    coord_data_normalized = standardize_coordinates(coord_data_cleaned)
    save_data_as_excel_and_csv(temp_data_cleaned, 'Cleaned_Temperature_Data.xlsx', 'Cleaned_Temperature_Data.csv')
    save_data_as_excel_and_csv(coord_data_normalized, 'Normalized_Coordinate_Data.xlsx', 'Normalized_Coordinate_Data.csv')

    X, y = merge_and_prepare_data(coord_data_normalized, temp_data_cleaned)

    # XGBoost Model
    xgb_model, X_train, y_train, X_test, y_test, xgb_pred = train_test_model(X, y)
    plot_residuals(y_test, xgb_pred)
    best_xgb_model = hyperparameter_optimization(X_train, y_train)

    # RandomForest Model
    rf_model, rf_X_train, rf_y_train, rf_X_test, rf_y_test, rf_pred = train_test_random_forest_model(X, y)
    plot_residuals(rf_y_test, rf_pred)
    best_rf_model = hyperparameter_optimization_rf(rf_X_train, rf_y_train)
    feature_importance_analysis(best_rf_model, X)

main(temp_file_path, coord_file_path)

def prepare_data_for_vae(combined_data):
    features = combined_data[['Temperature', 'X_norm', 'Y_norm', 'Z_norm']]
    X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)
    return X_train, X_test

def build_vae(latent_dim=2, input_shape=4):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(16, activation='relu')(inputs)
    x = layers.Dense(8, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(8, activation='relu')(latent_inputs)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(input_shape, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    vae_outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, vae_outputs, name='vae_mlp')

    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, vae_outputs) * input_shape
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1)) * -0.5
    vae.add_loss(reconstruction_loss + kl_loss)
    vae.compile(optimizer='adam')

    return vae

def train_vae(vae, X_train, X_test, epochs=10, batch_size=32):
    vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))

def evaluate_vae(vae, X_test_scaled):
    X_test_pred = vae.predict(X_test_scaled)
    reconstruction_error = mean_squared_error(X_test_scaled, X_test_pred, multioutput='raw_values')

    plt.figure(figsize=(10, 5))
    sns.histplot(reconstruction_error, bins=50)
    plt.xlabel('Reconstruction error')
    plt.ylabel('Number of samples')
    plt.title('Distribution of Reconstruction Errors')
    plt.show()

    print("Original Values:", X_test_scaled)
    print("Reconstructed Values:", X_test_pred)

def main_vae_workflow(temp_file_path, coord_file_path):
    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)
    coord_data_normalized = standardize_coordinates(coord_data_cleaned)

    combined_data = pd.merge(temp_data_cleaned, coord_data_normalized, on='Node')
    X_train, X_test = prepare_data_for_vae(combined_data)

    vae = build_vae()
    train_vae(vae, X_train, X_test)
    evaluate_vae(vae, X_test)

main_vae_workflow(temp_file_path, coord_file_path)