import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the datasets
temp_data = pd.read_csv("/System/Volumes/Data/Users/maroua/Downloads/CSI-4900-main/Cleaned_Temperature_Data.csv")
coord_data = pd.read_csv("/System/Volumes/Data/Users/maroua/Downloads/CSI-4900-main/Normalized_Coordinate_Data.csv")

coord_data['Node'] = pd.to_numeric(coord_data['Node'], errors='coerce').dropna().astype(int)
temp_data['Node'] = pd.to_numeric(temp_data['Node'], errors='coerce').dropna().astype(int)

combined_data = pd.merge(temp_data, coord_data, on='Node', how='inner')

features = combined_data[['Temperature', 'X_norm', 'Y_norm', 'Z_norm']]

X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)


latent_dim = 2
inputs = layers.Input(shape=(4,))
x = layers.Dense(16, activation='relu')(inputs)
x = layers.Dense(8, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
def sampling(args): z_mean, z_log_var = args; epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim)); return z_mean + tf.exp(0.5 * z_log_var) * epsilon
z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(8, activation='relu')(latent_inputs)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(4, activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name='decoder')
vae_outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, vae_outputs, name='vae_mlp')
reconstruction_loss = mse(inputs, vae_outputs) * 4
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1)) * -0.5
vae.add_loss(reconstruction_loss + kl_loss)
vae.compile(optimizer='adam')

vae.fit(X_train, epochs=10, batch_size=32, validation_data=(X_test, None))