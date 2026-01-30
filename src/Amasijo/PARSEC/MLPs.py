"""Pre-Trained Neural Network as BT-Settl model interpolator. PyTensor implementation.

This module wraps a pre-trained feed-forward neural network used to interpolate
stellar model outputs (mass and absolute photometry) as a function of age and a
single stellar parameter (theta). The network parameters and scalers are read
from a dill/pickle file and the forward pass is implemented using PyTensor
(pytensor) tensors so the outputs can be used inside PyMC models.
"""
import os
from pickle import load
import numpy as np
import pytensor.tensor as pt


# def relu(x, alpha=0):
#     """
#     Compute the element-wise rectified linear activation function.

#     Parameters
#     ----------
#     x : symbolic tensor
#         Tensor to compute the activation function for.
#     alpha : `scalar or tensor, optional`
#         Slope for negative input, usually between 0 and 1. The default value
#         of 0 will lead to the standard rectifier, 1 will lead to
#         a linear activation function, and any value in between will give a
#         leaky rectifier. A shared variable (broadcastable against `x`) will
#         result in a parameterized rectifier with learnable slope(s).

#     Returns
#     -------
#     symbolic tensor
#         Element-wise rectifier applied to `x`.

#     Notes
#     -----
#     This is numerically equivalent to ``pt.switch(x > 0, x, alpha * x)``
#     (or ``pt.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
#     formulation or an optimized Op, so we encourage to use this function.

#     """
#     if alpha == 0:
#         return 0.5 * (x + abs(x))
#     else:
#         # We can't use 0.5 and 1 for one and half.  as if alpha is a
#         # numpy dtype, they will be considered as float64, so would
#         # cause upcast to float64.
#         alpha = pt.as_tensor_variable(alpha)
#         f1 = 0.5 * (1 + alpha)
#         f2 = 0.5 * (1 - alpha)
#         return f1 * x + f2 * abs(x)

def relu(x): #With alpha set to zero to improve speed
    """
    Compute the element-wise rectified linear activation function.

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : `scalar or tensor, optional`
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.

    Notes
    -----
    This is numerically equivalent to ``pt.switch(x > 0, x, alpha * x)``
    (or ``pt.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.

    """
    # if alpha == 0:
    return 0.5 * (x + abs(x))
    # else:
	    # We can't use 0.5 and 1 for one and half.  as if alpha is a
	    # numpy dtype, they will be considered as float64, so would
	    # cause upcast to float64.
	    # alpha = pt.as_tensor_variable(alpha)
	    # f1 = 0.5 * (1 + alpha)
	    # f2 = 0.5 * (1 - alpha)
	    # return f1 * x + f2 * abs(x)

class MLP_phot:
	"""Wrapper around a pretrained multilayer perceptron.

	Responsibilities:
	- load NN weights and preprocessing scalers from disk
	- expose a callable that accepts (age, theta, n_stars) and returns
	  PyTensor expressions for mass and photometry suitable for inclusion
	  inside probabilistic models.

	Notes on shapes and conventions used in this code:
	- age: scalar (float) or PyTensor scalar
	- theta: 1-D array-like of length n_stars (PyTensor or numpy)
	- n_stars: integer number of samples in theta
	- The implementation stacks age and theta into a 2xN array and applies
	  the same preprocessing (BoxCox + MinMax) used at training time.
	- The network returns an array "targets" where the first output column
	  corresponds to stellar mass and the remaining columns correspond to
	  absolute magnitudes (photometry) in the trained band order.
	"""

	def __init__(self, 
		file_mlp : str = 'mlp_iso_BT-Settl.pkl'
		):
		"""Load optimal weights and scalers from trained MLP.

		The serialized file is expected to contain a dict-like object with keys:
		  - "weights": list/array of weight and bias matrices in the expected order
		  - "scalers": list of sklearn-like scalers [BoxCox, MinMax]
		  - "phot_min": per-band lower limit used elsewhere for filtering
		  - "age_domain": (min_age, max_age)
		  - "par_domain" (theta domain)
		"""
		# Load weights and scalers from provided file path
		try:
			with open(file_mlp, 'rb') as file:
				mlp = load(file)
				weights = mlp["weights"]
				scalers = mlp["scalers"]
				num_layers = mlp["num_layers"]
				self.targets = mlp["targets"]
				self.phot_min = mlp["phot_min"]
				self.age_domain  = mlp["age_domain"]
				self.mass_domain = mlp["mass_domain"]
				self.teff_domain = mlp["teff_domain"]

			assert num_layers == 12, "Error: mismatch in number of layers!"

		except FileNotFoundError as error:
			# Provide a clear error for the user if file missing
			raise FileNotFoundError("The file containing optimal weights and scalers cannot be found. Please, provide a valid path") from error
		else:
			# Unpack weights/biases according to the trained network layout.
			# The code expects exactly this many items and ordering.
			self.W = weights[::2]
			self.b = weights[1::2]

			# Scalers used for preprocessing the inputs
			self.BoxCox = scalers[0]
			self.MinMax = scalers[1]

			# Box-Cox lambdas are stored per input dimension and reshaped to column vectors.
			self.lamb = np.array(self.BoxCox.lambdas_).reshape(-1, 1)
			
			# MinMax scaling metadata (data_min_, data_max_) used to scale inputs.
			self.min_val = self.MinMax.data_min_.reshape(-1, 1)
			self.max_val = self.MinMax.data_max_.reshape(-1, 1)
			self.dff_val = self.max_val - self.min_val

		# Print a small summary for user awareness when instantiating the MLP.
		print("Age domain: [{0:2.1f},{1:2.1f}]".format(*self.age_domain))
		print("Mass domain: [{0:2.2f},{1:2.3f}]".format(*self.mass_domain))
		print("Teff domain: [{0:2.2f},{1:2.3f}]".format(*self.teff_domain))

	def __call__(self, age, mass, teff, n_stars):
		"""Compute NN predictions for given age and theta grid.

		Returns:
		  - mass: PyTensor vector of length n_stars
		  - photometry: PyTensor matrix shape (n_stars, n_bands)

		Detailed steps:
		1. Stack age and theta into a 2xN tensor so each column corresponds to one star.
	 2. Apply Box-Cox transform (or log for lambda==0) elementwise.
	 3. Apply MinMax scaling using stored min/max values.
	 4. Run forward pass through 10 hidden layers using relu activations.
	 5. Final linear mapping produces targets: [mass, phot_band_1, phot_band_2, ...].
		"""
		# Stack age and tiled age with theta to form input columns:
		# amt has shape (2, n_stars) using pytensor operations.
		amt = pt.stack([pt.tile(age, n_stars),mass, teff])

		# Apply Box-Cox transform: (x^lambda - 1)/lambda, and log if lambda==0.
		# The comparison and switch is done elementwise using pytensor operations.
		x = pt.switch(pt.neq(self.lamb, 0), (amt ** self.lamb - 1) / self.lamb, pt.log(amt))

		# Normalize using MinMax stored values and transpose to shape (n_stars, n_features)
		# so that a forward pass with weight matrices (features x hidden) works naturally.
		A0 = ((x - self.min_val) / self.dff_val).T

		# Forward pass through the network: linear -> relu -> linear -> relu -> ...
		# Each Z* is the pre-activation and A* the post-activation (relu).
		A1  = relu(pt.dot(A0, self.W[0])  + self.b[0])
		A2  = relu(pt.dot(A1, self.W[1])  + self.b[1])
		A3  = relu(pt.dot(A2, self.W[2])  + self.b[2])
		A4  = relu(pt.dot(A3, self.W[3])  + self.b[3])
		A5  = relu(pt.dot(A4, self.W[4])  + self.b[4])
		A6  = relu(pt.dot(A5, self.W[5])  + self.b[5])
		A7  = relu(pt.dot(A6, self.W[6])  + self.b[6])
		A8  = relu(pt.dot(A7, self.W[7])  + self.b[7])
		A9  = relu(pt.dot(A8, self.W[8])  + self.b[8])
		A10 = relu(pt.dot(A9, self.W[9])  + self.b[9])
		A11 = relu(pt.dot(A10, self.W[10])  + self.b[10])
		A12 = relu(pt.dot(A11, self.W[11])  + self.b[11])
		# Final linear output (no activation)
		targets = pt.dot(A12, self.W[12]) + self.b[12]

		return targets

class MLP_logg:
	"""Wrapper around a pretrained multilayer perceptron.

	Responsibilities:
	- load NN weights and preprocessing scalers from disk
	- expose a callable that accepts (age, theta, n_stars) and returns
	  PyTensor expressions for mass and photometry suitable for inclusion
	  inside probabilistic models.

	Notes on shapes and conventions used in this code:
	- age: scalar (float) or PyTensor scalar
	- theta: 1-D array-like of length n_stars (PyTensor or numpy)
	- n_stars: integer number of samples in theta
	- The implementation stacks age and theta into a 2xN array and applies
	  the same preprocessing (BoxCox + MinMax) used at training time.
	- The network returns an array "targets" where the first output column
	  corresponds to stellar mass and the remaining columns correspond to
	  absolute magnitudes (photometry) in the trained band order.
	"""

	def __init__(self, 
		file_mlp : str = 'mlp_iso_BT-Settl.pkl'
		):
		"""Load optimal weights and scalers from trained MLP.

		The serialized file is expected to contain a dict-like object with keys:
		  - "weights": list/array of weight and bias matrices in the expected order
		  - "scalers": list of sklearn-like scalers [BoxCox, MinMax]
		  - "phot_min": per-band lower limit used elsewhere for filtering
		  - "age_domain": (min_age, max_age)
		  - "par_domain" (theta domain)
		"""
		# Load weights and scalers from provided file path
		try:
			with open(file_mlp, 'rb') as file:
				mlp = load(file)
				weights = mlp["weights"]
				scalers = mlp["scalers"]
				num_layers = mlp["num_layers"]
				self.targets = mlp["targets"]
				self.age_domain  = mlp["age_domain"]
				self.mass_domain = mlp["mass_domain"]
				self.teff_domain = mlp["teff_domain"]

			assert num_layers == 7, "Error: mismatch in number of layers!"

		except FileNotFoundError as error:
			# Provide a clear error for the user if file missing
			raise FileNotFoundError("The file containing optimal weights and scalers cannot be found. Please, provide a valid path") from error
		else:
			# Unpack weights/biases according to the trained network layout.
			# The code expects exactly this many items and ordering.
			self.W = weights[::2]
			self.b = weights[1::2]

			# Scalers used for preprocessing the inputs
			self.BoxCox = scalers[0]
			self.MinMax = scalers[1]

			# Box-Cox lambdas are stored per input dimension and reshaped to column vectors.
			self.lamb = np.array(self.BoxCox.lambdas_).reshape(-1, 1)
			
			# MinMax scaling metadata (data_min_, data_max_) used to scale inputs.
			self.min_val = self.MinMax.data_min_.reshape(-1, 1)
			self.max_val = self.MinMax.data_max_.reshape(-1, 1)
			self.dff_val = self.max_val - self.min_val

		# Print a small summary for user awareness when instantiating the MLP.
		print("Age domain: [{0:2.1f},{1:2.1f}]".format(*self.age_domain))
		print("Mass domain: [{0:2.2f},{1:2.3f}]".format(*self.mass_domain))
		print("Teff domain: [{0:2.2f},{1:2.3f}]".format(*self.teff_domain))

	def __call__(self, age, mass, teff, n_stars):
		"""Compute NN predictions for given age and theta grid.

		Returns:
		  - mass: PyTensor vector of length n_stars
		  - photometry: PyTensor matrix shape (n_stars, n_bands)

		Detailed steps:
		1. Stack age and theta into a 2xN tensor so each column corresponds to one star.
	 2. Apply Box-Cox transform (or log for lambda==0) elementwise.
	 3. Apply MinMax scaling using stored min/max values.
	 4. Run forward pass through 10 hidden layers using relu activations.
	 5. Final linear mapping produces targets: [mass, phot_band_1, phot_band_2, ...].
		"""
		# Stack age and tiled age with theta to form input columns:
		# amt has shape (2, n_stars) using pytensor operations.
		amt = pt.stack([pt.tile(age, n_stars),mass, teff])

		# Apply Box-Cox transform: (x^lambda - 1)/lambda, and log if lambda==0.
		# The comparison and switch is done elementwise using pytensor operations.
		x = pt.switch(pt.neq(self.lamb, 0), (amt ** self.lamb - 1) / self.lamb, pt.log(amt))

		# Normalize using MinMax stored values and transpose to shape (n_stars, n_features)
		# so that a forward pass with weight matrices (features x hidden) works naturally.
		A0 = ((x - self.min_val) / self.dff_val).T

		# Forward pass through the network: linear -> relu -> linear -> relu -> ...

		A1  = relu(pt.dot(A0, self.W[0])  + self.b[0])
		A2  = relu(pt.dot(A1, self.W[1])  + self.b[1])
		A3  = relu(pt.dot(A2, self.W[2])  + self.b[2])
		A4  = relu(pt.dot(A3, self.W[3])  + self.b[3])
		A5  = relu(pt.dot(A4, self.W[4])  + self.b[4])
		A6  = relu(pt.dot(A5, self.W[5])  + self.b[5])
		A7  = relu(pt.dot(A6, self.W[6])  + self.b[6])
		# A8  = relu(pt.dot(A7, self.W[7])  + self.b[7])
		# A9  = relu(pt.dot(A8, self.W[8])  + self.b[8])
		# A10 = relu(pt.dot(A9, self.W[9])  + self.b[9])
		# A11 = relu(pt.dot(A10,self.W[10]) + self.b[10])
		# A12 = relu(pt.dot(A11,self.W[11]) + self.b[11])
		# A13 = relu(pt.dot(A12,self.W[12]) + self.b[12])
		# A14 = relu(pt.dot(A13,self.W[13]) + self.b[13])
		# A15 = relu(pt.dot(A14,self.W[14]) + self.b[14])
		# A16 = relu(pt.dot(A15,self.W[15]) + self.b[15])
		# Final linear output (no activation) gives targets for mass + photometry bands
		targets = pt.dot(A7, self.W[7]) + self.b[7]

		return targets


# The block below is an example usage / quick visual test when running the file
# directly. It is not required for the library functionality and will only run
# in interactive/script mode.
if __name__ == "__main__":

	import pandas as pn
	import matplotlib.pyplot as plt
	import seaborn as sns

	dir_base = "/home/jolivares/Models/PARSEC/Gaia_EDR3_10-400Myr/"

	file_iso      = dir_base + "Gaia_EDR3_10-400Myr.dat"
	file_mlp_phot = dir_base + "MLPs/Phot_l12_s256/mlp.pkl"
	file_mlp_logg = dir_base + "MLPs/Logg_l7_s256/mlp.pkl"

	mlp_phot = MLP_phot(file_mlp=file_mlp_phot)
	mlp_logg = MLP_logg(file_mlp=file_mlp_logg)

	# Example: load an isochrone from a parametrized CSV and overlay predicted photometry
	age = 50.

	df_iso = pd.read_csv(file_iso,
					skiprows=13,
					delimiter=r"\s+",
					header="infer",
					comment="#")
	df_iso = df_iso.loc[df_iso["label"]<= max_label]
	df_iso["age"] = np.pow(10.,df_iso["logAge"])/1.0e6
	df_iso["Teff"] = np.pow(10.,df_iso["logTe"])
	df_iso = df_iso.loc[:,sum([features,targets],[])]
	print(df_iso.describe())
	df_iso = df_iso.groupby("age").get_group(age).copy()

	mass  = df_iso["Mini"].to_numpy()
	teff  = df_iso["Teff"].to_numpy()
	n_stars = len(theta) 

	absolute_photometry = mlp_phot(age,mass,teff,n_stars)
	df_prd = pn.DataFrame(
		data=absolute_photometry.eval(),
		columns=mlp_phot.targets)
	df_prd["Mini"] = mass
	df_prd["Teff"] = teff
	logg = mlp_logg(age,mass,teff,n_stars)
	df_prd["logg"] = logg.eval()

	# Simple diagnostic plot to visually compare trained MLP predictions with input isochrone
	for target in mlp_phot.targets:
		plt.figure(0)
		ax = sns.scatterplot(data=df_iso,
							x="Mini",y=target,
							legend=True,
							zorder=0)
		ax = sns.lineplot(data=df_prd,
							x="Mini",y=target,
							legend=True,
							sort=False,
							zorder=1,
							ax=ax)

		ax.set_xlabel("Mini")
		ax.set_ylabel(target)
		plt.show()

	for target in mlp_logg.targets:
		plt.figure(0)
		ax = sns.scatterplot(data=df_iso,
							x="Mini",y=target,
							legend=True,
							zorder=0)
		ax = sns.lineplot(data=df_prd,
							x="Mini",y=target,
							legend=True,
							sort=False,
							zorder=1,
							ax=ax)

		ax.set_xlabel("Mini")
		ax.set_ylabel(target)
		plt.show()