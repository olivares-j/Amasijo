import sys
import os
import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from mlp_model import create_custom_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train = False

# ------------------------------ Model properties --------------------------------
max_label = 1
features = ["age","Mini"]
targets  = ["Teff"]
list_of_num_layers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # Number of hidden layers
list_of_size_layers = [128,256,512]  # Units in each hidden layer
training_epochs = [9999,9999,9999]
learning_rates  = [1e-3,1e-4,1e-5]
activation_layers = "relu" # Activation functions for each hidden layer
output_activation = "linear"  # Activation function for the output layer
batch_fraction = 1.0
#--------------------------------------------------------------------------

#--------------- Directories and files ---------------------------------------------
dir_base  = "/home/jolivares/Models/PARSEC/Gaia_EDR3_15-400Myr/"
# Remove the # from the row contain the header in the input file
file_iso  = dir_base + "output.dat" 
dir_mlps  = dir_base + "MLPs/"

file_mad  = dir_mlps +"Teff_MAD.png"
base_fld  = "Teff_l{0}_s{1}/"
base_dat  = "{0}data.csv"
base_mlp  = "{0}mlp.pkl"
base_lss  = "{0}loss.png"
base_fit  = "{0}fit.png"
base_err  = "{0}error.png"
#------------------------------------------------------------------------------------

#------------- Load data ----------------------------
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
#----------------------------------------------------

#--------------------------------------------------- Select certain ages -----------------------------------------------------------------
# df_iso = df_iso.query("age_Myr == [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200]")
# df_iso = df_iso.query("age_Myr == [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]")
# df_iso = df_iso.query("age_Myr == [20,40,60,80,100,120,140,160,180,200]")
# df_iso = df_iso.query("age_Myr == [20,200]")
# print(df_iso.describe())
#------------------------------------------------------------------------------------------------------------------------------------------------------------

mads = []
for size_layers in list_of_size_layers:
	for num_layers in list_of_num_layers:
		dir_case   = dir_mlps  + base_fld.format(num_layers,size_layers)
		os.makedirs(dir_case,exist_ok=True)

		file_mlp = base_mlp.format(dir_case)

		#------------------ Train ---------------------------
		if train:
			#-------------- Save data -------------- 
			df_iso.to_csv(base_dat.format(dir_case))
			#---------------------------------------

			#---------- Numpy arrays ------------------
			raw_features = df_iso[features].to_numpy()
			raw_targets  = df_iso[targets].to_numpy()
			#-------------------------------------------

			#---------------- Transformations ------------------------
			BoxCox = PowerTransformer(
						method="box-cox", 
						standardize=False).fit(raw_features)
			inputs = BoxCox.transform(raw_features)
			MinMax = MinMaxScaler().fit(inputs)
			scalers = [BoxCox, MinMax]
			#---------------------------------------------------------

			#------------ Domains ---------------------------------------------
			phot_min = df_iso[targets].min()
			age_domain = [df_iso["age"].min(),df_iso["age"].max()]
			mass_domain = [df_iso["Mini"].min(),df_iso["Mini"].max()]
			#-------------------------------------------------------------------------

			batch_size   = int(batch_fraction*df_iso.shape[0])
			transformed_features = scalers[1].transform(scalers[0].transform(raw_features))

			losses = []
			for learning_rate,epochs in zip(learning_rates,training_epochs):

				#--------------- Instantiate model -----------------------
				model = create_custom_model(
				input_shape=len(features), 
				num_layers=num_layers,
				size_layers=size_layers, 
				activation_layers=activation_layers, 
				output_units=len(targets), 
				output_activation=output_activation,
				learning_rate=learning_rate)
				#------------------------------------------------------------

				#------------ Load previous fit --------
				if os.path.isfile(file_mlp):
					with open(file_mlp, "rb") as file:
						mlp = dill.load(file)
						model.set_weights(mlp["weights"])
				#-----------------------------------------

				#------------ Fit -----------
				fit = model.fit(
					transformed_features,
					raw_targets,
					epochs=epochs,
					batch_size=batch_size,
					verbose=1)
				#----------------------------

				#--------- Save --------------------
				mlp = {
					"num_layers":num_layers,
					"size_layers":size_layers,
					"scalers":scalers,
					"weights":model.get_weights(),
					"phot_min":phot_min,
					"targets":targets,
					"age_domain":age_domain,
					"mass_domain":mass_domain
					}

				with open(file_mlp, "wb") as file:
					dill.dump(mlp, file)
				#------------------------------------

				#------------- Data for loss plot ----------
				losses.append(fit.history["loss"])
				#---------------------------------------------

			loss = np.concatenate(losses,axis=0)
			df = pd.DataFrame(data={
						"loss":loss,
						"iter":np.arange(len(loss)),
						})

			#------------ Plot Loss --------------------------
			fig, ax = plt.subplots(1, 1, figsize=(16, 8))
			ax = sns.lineplot(data=df,
								x="iter",
								y="loss",
								zorder=0)
			ax.set_xlabel("Iteration")
			ax.set_ylabel("Loss")
			ax.set_yscale('log')
			# plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
			fig.savefig(base_lss.format(dir_case))
			plt.close()

		#---------- Instantiate model ---------------------
		model = create_custom_model(
				input_shape=len(features), 
				num_layers=num_layers,
				size_layers=size_layers, 
				activation_layers=activation_layers, 
				output_units=len(targets), 
				output_activation=output_activation)
		#------------------------------------------------

		#------------ Load existing fit --------
		with open(file_mlp, "rb") as file:
			mlp = dill.load(file)
			scalers = mlp["scalers"]
			model.set_weights(mlp["weights"])
		#----------------------------------------

		#-------------- Plot predictions ---------------------
		df_tst = df_iso.copy()
		df_tst.reset_index(drop=True,inplace=True)

		df_features = df_tst[features]
		input_tst = scalers[1].transform(scalers[0].transform(df_features.to_numpy()))
		ouput_tst = model(input_tst,training=False)

		df_tar = pd.DataFrame(columns=targets,data=ouput_tst._numpy())
		df_prd = pd.concat([df_features,df_tar],axis=1,ignore_index=False)

		df = pd.merge(left=df_tst,right=df_tar,left_index=True,right_index=True,suffixes=["_tst","_prd"])
		df.set_index(features,drop=True,inplace=True)

		for trgt in targets:
			df[trgt] = df.apply(lambda x: (x[trgt+"_prd"]-x[trgt+"_tst"])/x[trgt+"_tst"],axis=1)

		#--------- Save for general plot---------------------
		tmp = pd.DataFrame(data={
			"MAD":df.loc[:,targets].abs().max().to_numpy(),
			"num_layers":num_layers,
			"size_layers":size_layers})
		tmp.set_index("num_layers",inplace=True)
		mads.append(tmp)
		#----------------------------------------------------

		df = df.loc[:,targets].stack().reset_index()
		df.columns = ["age","Mini","target","value"]
		print(df.describe())


		fig, ax = plt.subplots(1, 1, figsize=(16, 8))
		ax = sns.scatterplot(data=df,
							x="Mini",
							y="value",
							hue="age",
							style="target",
							zorder=0)
		ax.set_xlabel("Mass [Msun]")
		ax.set_ylabel("(Predicted-Test)/Test")
		plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
		fig.savefig(base_err.format(dir_case))
		plt.close()

		ax = sns.scatterplot(data=df_tst,
				x="Mini",
				y="Teff",
				hue="age",
				zorder=0)
		ax = sns.lineplot(data=df_prd,
				x="Mini",
				y="Teff",
				hue="age",
				legend=False,
				sort=False,
				zorder=1,
				ax=ax)
		ax.set_xlabel("Mini")
		ax.set_ylabel("Teff")
		plt.savefig(base_fit.format(dir_case))
		plt.close()

df_mad = pd.concat(mads,ignore_index=False)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax = sns.lineplot(data=df_mad,
					x="num_layers",
					y="MAD",
					hue="size_layers",
					zorder=0)
ax.set_xlabel("Number of layers")
ax.set_ylabel("MAD of the relative error")
ax.set_yscale("log")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
fig.savefig(file_mad)
plt.close()
