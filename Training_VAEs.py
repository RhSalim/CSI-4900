import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import mse
from sklearn.model_selection import train_test_split

# Load the datasets
temp_data = pd.read_csv("C:/Users/Rhola/Cleaned_Temperature_Data.csv")
coord_data = pd.read_csv("C:/Users/Rhola/Normalized_Coordinate_Data.csv")

# Drop the first row of the coordinate data if it contains NaN in the 'Node' column
coord_data = coord_data.dropna(subset=['Node'])

# Convert 'Node' columns to integer type in both datasets for proper merging
temp_data['Node'] = temp_data['Node'].astype(int)
coord_data['Node'] = coord_data['Node'].astype(int)

# Merge the datasets on the 'Node' column
combined_data = pd.merge(temp_data, coord_data, on='Node', how='inner')

# Splitting the data into features (Temperature, X_norm, Y_norm, Z_norm)
features = combined_data[['Temperature', 'X_norm', 'Y_norm', 'Z_norm']]

# Splitting the data into training and testing sets
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

# Define the dimension of the latent space
latent_dim = 2

# Encoder
inputs = layers.Input(shape=(4,))
x = layers.Dense(16, activation='relu')(inputs)
x = layers.Dense(8, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(8, activation='relu')(latent_inputs)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(4, activation='sigmoid')(x)

# Decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# VAE loss
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= 4  # scale up by the number of features
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Training the VAE model
vae.fit(X_train, epochs=10, batch_size=32, validation_data=(X_test, None))
