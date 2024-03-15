import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers, Model

def load_and_clean_data(temp_file_path, coord_file_path):

    temp_data = pd.read_csv(temp_file_path, sep='\t', header=None)
    coord_data = pd.read_csv(coord_file_path, sep='\t', header=None)


    temp_data_cleaned = temp_data.drop(index=0).rename(columns={0: 'Node', 1: 'Temperature'})
    coord_data_cleaned = coord_data.drop(index=0).rename(columns={0: 'Node', 1: 'Unknown', 2: 'X', 3: 'Y', 4: 'Z'})


    temp_data_cleaned['Temperature'] = temp_data_cleaned['Temperature'].str.replace(',', '.').astype(float)

    return temp_data_cleaned, coord_data_cleaned

def standardize_coordinates(coord_data_cleaned):

    coord_numeric_data = coord_data_cleaned[['X', 'Y', 'Z']]


    scaler = StandardScaler()
    coord_normalized = scaler.fit_transform(coord_numeric_data)


    coord_data_normalized = pd.DataFrame(coord_normalized, columns=['X_norm', 'Y_norm', 'Z_norm'])
    coord_data_normalized['Node'] = coord_data_cleaned['Node'].values

    return coord_data_normalized

def save_data_as_excel_and_csv(data, excel_file_name, csv_file_name):

    data.to_excel(excel_file_name, index=False)
    data.to_csv(csv_file_name, index=False)

def plot_temperature_distribution(temp_data_cleaned):
    plt.figure(figsize=(10, 5))
    sns.histplot(temp_data_cleaned['Temperature'], kde=True)
    plt.title('Distribution of Temperatures')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def save_predictions_to_csv(y_true, y_pred, file_name):
   
    df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})

    df.to_csv(file_name, index=False)
def main(temp_file_path, coord_file_path):

    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)

    coord_data_normalized = standardize_coordinates(coord_data_cleaned)

    save_data_as_excel_and_csv(temp_data_cleaned, 'Cleaned_Temperature_Data.xlsx', 'Cleaned_Temperature_Data.csv')
    save_data_as_excel_and_csv(coord_data_normalized, 'Normalized_Coordinate_Data.xlsx', 'Normalized_Coordinate_Data.csv')

    plot_temperature_distribution(temp_data_cleaned)


temp_file_path = "C:/Users/Rhola/Desktop/CSI 4900/Temp_noeuds1.txt"
coord_file_path = "C:/Users/Rhola/Desktop/CSI 4900/Coord_noeuds1.txt"


main(temp_file_path, coord_file_path)

def merge_and_prepare_data(coord_data_normalized, temp_data_cleaned):

    merged_data = pd.merge(coord_data_normalized, temp_data_cleaned, on='Node')
    X = merged_data[['X_norm', 'Y_norm', 'Z_norm']]
    y = merged_data['Temperature']
    return X, y

def train_test_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE on the test set: {mse}")
    x=mse

    return model, X_train, y_train, X_test, y_test, y_pred  

def plot_residuals(y_test, y_pred):

    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()

def hyperparameter_optimization(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0]
    }

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
    xgb_model, X_train, y_train, X_test, y_test, xgb_pred = train_test_model(X, y)
    save_predictions_to_csv(y_test, xgb_pred, 'xgb_predictions.csv')

    best_model = hyperparameter_optimization(X_train, y_train)

main(temp_file_path, coord_file_path)


def train_test_random_forest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE on the test set with RandomForest: {mse}")
    

    return rf_model, X_train, y_train, X_test, y_test, y_pred
def hyperparameter_optimization_rf(X_train, y_train):

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)
    clf_rf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', error_score=0, verbose=3, n_jobs=-1)
    clf_rf.fit(X_train, y_train)

    print(f"Best parameters for RandomForest: {clf_rf.best_params_}")

    return clf_rf.best_estimator_

def feature_importance_analysis(model, X):
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance with RandomForest')
    plt.show()

def main(temp_file_path, coord_file_path):
    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)
    coord_data_normalized = standardize_coordinates(coord_data_cleaned)
    save_data_as_excel_and_csv(temp_data_cleaned, 'Cleaned_Temperature_Data.xlsx', 'Cleaned_Temperature_Data.csv')
    save_data_as_excel_and_csv(coord_data_normalized, 'Normalized_Coordinate_Data.xlsx', 'Normalized_Coordinate_Data.csv')

    X, y = merge_and_prepare_data(coord_data_normalized, temp_data_cleaned)

    xgb_model, X_train, y_train, X_test, y_test, xgb_pred = train_test_model(X, y)
    plot_residuals(y_test, xgb_pred)
    best_xgb_model = hyperparameter_optimization(X_train, y_train)

    rf_model, rf_X_train, rf_y_train, rf_X_test, rf_y_test, rf_pred = train_test_random_forest_model(X, y)
    plot_residuals(rf_y_test, rf_pred)
    best_rf_model = hyperparameter_optimization_rf(rf_X_train, rf_y_train)
    feature_importance_analysis(best_rf_model, X)
    rf_model, rf_X_train, rf_y_train, rf_X_test, rf_y_test, rf_pred = train_test_random_forest_model(X, y)

    save_predictions_to_csv(rf_y_test, rf_pred, 'rf_predictions.csv')

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

def evaluate_vae(vae, X_test):
    X_test_scaled = X_test if isinstance(X_test, np.ndarray) else X_test.to_numpy()
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
    
    return X_test_pred  

def save_predictions_to_csv_vae(X_test_scaled, X_test_pred, file_name):

    if not isinstance(X_test_scaled, np.ndarray):
        X_test_scaled = X_test_scaled.to_numpy()
        
    if X_test_scaled.shape != X_test_pred.shape:
        raise ValueError("Shapes of original and predicted data do not match.")

    df = pd.DataFrame({
        'Original_Feature_1': X_test_scaled[:, 0], 
        'Original_Feature_2': X_test_scaled[:, 1], 
        'Original_Feature_3': X_test_scaled[:, 2], 
        'Original_Feature_4': X_test_scaled[:, 3],
        'Predicted_Feature_1': X_test_pred[:, 0], 
        'Predicted_Feature_2': X_test_pred[:, 1], 
        'Predicted_Feature_3': X_test_pred[:, 2], 
        'Predicted_Feature_4': X_test_pred[:, 3]
    })
    df.to_csv(file_name, index=False)

def main_vae_workflow(temp_file_path, coord_file_path):
    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)
    coord_data_normalized = standardize_coordinates(coord_data_cleaned)
    combined_data = pd.merge(temp_data_cleaned, coord_data_normalized, on='Node')
    X_train, X_test = prepare_data_for_vae(combined_data)
    vae = build_vae(latent_dim=2, input_shape=4)
    train_vae(vae, X_train, X_test, epochs=10, batch_size=32)
    X_test_pred = evaluate_vae(vae, X_test)  
    save_predictions_to_csv_vae(X_test, X_test_pred, 'vae_predictions.csv')


main_vae_workflow(temp_file_path, coord_file_path)


models = ['XGBoost', 'RandomForest', 'VAE (Initial Run)', 'VAE (Second Run)']
mse_values = [10.278996486981319, 0.11351744047035806, 731.6599, 933.7374]


plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Performance Comparison')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')

# Display the plot
plt.show()



