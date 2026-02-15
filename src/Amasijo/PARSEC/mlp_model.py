import sys
import os
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

print(f"Running Tensoflow {tf.__version__}")


# Defining model archiquetures
# ----------------------------
def create_custom_model(
	input_shape: int,
	num_layers: int,
	size_layers: int,
	activation_layers: str,
	output_units: int,
	output_activation: str = "linear",
	learning_rate: float = 1e-3
) -> Sequential:
	"""
	Create a Keras Sequential model with the specified number of layers.

	Parameters:
	- input_shape (int): The number of features in the input data.
	- num_layers (int): The number of hidden layers in the model.
	- units (List[int]): A list containing the number of units for each hidden layer.
	- activations (List[str]): A list containing the activation function for each hidden layer.
	- output_units (int): The number of units in the output layer.
	- output_activation (str): The activation function for the output layer (default is 'linear').

	Returns:
	- model (Sequential): The compiled Keras Sequential model.
	"""

	model = Sequential()

	# Input layer
	model.add(Dense(size_layers, activation=activation_layers, input_shape=(input_shape,)))

	# Hidden layers
	for i in range(1, num_layers):
		model.add(
			Dense(
				size_layers,
				activation=activation_layers,
				kernel_initializer=keras.initializers.HeNormal(),
				bias_initializer=keras.initializers.HeNormal(),
			)
		)

	# Output layer
	model.add(
		Dense(
			output_units,
			activation=output_activation,
			kernel_initializer=keras.initializers.HeNormal(),
			bias_initializer=keras.initializers.HeNormal()
		)
	)

	# Compile the model
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="mean_squared_error",
		metrics=["mean_squared_error"],
	)

	return model