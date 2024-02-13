import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import random 

# Load the datasets
normalized_coordinate_data = pd.read_csv("C:/Users/Rhola/Normalized_Coordinate_Data.csv")
cleaned_temperature_data = pd.read_csv("C:/Users/Rhola/Cleaned_Temperature_Data.csv")

# Merge the datasets on the 'Node' column and drop rows with missing values
combined_data = pd.merge(cleaned_temperature_data, normalized_coordinate_data, on='Node', how='inner').dropna()

# Splitting the data into features (Temperature, X_norm, Y_norm, Z_norm)
features = combined_data[['Temperature', 'X_norm', 'Y_norm', 'Z_norm']]

# Splitting the data into training and testing sets
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

# Define the dimensions of the latent space
latent_dim = 2

# Encoder architecture
inputs = Input(shape=(4,))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder architecture
decoder_input = Input(shape=(latent_dim,))
x = Dense(64, activation='relu')(decoder_input)
x = Dense(128, activation='relu')(x)
outputs = Dense(4, activation='linear')(x)

# Models
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_input, outputs, name='decoder')
vae_output = decoder(encoder(inputs)[2])
vae = Model(inputs, vae_output, name='vae_mlp')

# Loss function
reconstruction_loss = MeanSquaredError()(inputs, vae_output)
reconstruction_loss *= 4
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile the VAE
vae.compile(optimizer=Adam())

# Display the model summary
vae.summary()

# Note: Due to the environment, the actual training command is commented out, but would look like this:
vae.fit(X_train, epochs=50, batch_size=32, validation_split=0.2)

vae.save("C:/Users/Rhola/vae_model")
