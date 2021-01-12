import sys
import os
import numpy  as np
import pandas as pd
import scipy.stats as st
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Functions import AngularSeparation,CovarianceParallax
from distributions import eff,king,toCartesian

from pygaia.astrometry.vectorastrometry import cartesian_to_spherical
from pygaia.errors.astrometric import parallax_uncertainty,total_position_uncertainty

from isochrones import get_ichrone
from isochrones.priors import ChabrierPrior


###############################################################################
class Amasijo(object):
	"""This class intends to construct synthetic clusters with simple 
		astrometric distributions and photometry from stellar models"""
	def __init__(self,astrometric_args,
					  photometric_args, 
					  seed=1234):

		#------ Set Seed -----------------
		np.random.seed(seed=seed)
		self.random_state = np.random.RandomState(seed=seed)

		#---------- Arguments --------------------------
		self.astrometric_args = astrometric_args
		self.photometric_args = photometric_args

		#--------------- Tracks -----------------------------------
		self.tracks = get_ichrone('mist', tracks=True,bands=photometric_args["bands"])

		#------- Labels -----------------------------------------------------
		self.labels_true_as = ["X","Y","Z"]
		self.labels_obs_as  = ["ra","dec","parallax"]
		self.labels_unc_as  = ["ra_error","dec_error","parallax_error"]

	#====================== Generate Astrometric Data ==================================================

	def _generate_phase_space(self,n_stars,astrometric_args):
		position_args = astrometric_args["position"]
		velocity_args = astrometric_args["velocity"]

		#======================= Verification ==============================================================
		msg_0 = "Error in position arguments: loc and scale must have same dimension."
		msg_1 = "Error in position arguments: family {0} is not implemented".format(position_args["family"]) 

		assert len(position_args["loc"]) == len(position_args["scl"]), msg_0
		assert position_args["family"] in ["Gaussian","EFF","King","GMM"], msg_1

		#=============================== Positions ========================================================

		#------- Sample the radial distance  ----------------------------------------------
		if (position_args["family"] == "Gaussian") or (position_args["family"] == "GMM"):
			r = st.norm.rvs(size=n_stars)
		elif position_args["family"] == "EFF":
			r = eff.rvs(gamma=position_args["gamma"],size=n_stars)
		elif position_args["family"] == "King":
			r = king.rvs(rt=position_args["tidal_radius"],size=n_stars)
		#----------------------------------------------------------------------------------


		#------ Samples from the angles -------
		samples = toCartesian(r,3,random_state=self.random_state).reshape(n_stars,3)

		chol = np.linalg.cholesky(position_args["scl"])
		samples = np.dot(samples,chol)

		if position_args["family"] == "GMM":
			sys.exit("Not yet implemented")
			assert np.sum(position_args["weights"]) == 1.0,"weights must be a simplex"
			n_cmp = len(position_args["weights"])
			n_stars_cmp = np.floor(position_args["weights"]*n_stars).astype('int')
			n_res = n_stars - np.sum(n_stars_cmp)
			residual = np.ones(n_cmp)
			residual[n_res:] = 0
			n_stars_cmp += residual
			assert np.sum(n_stars_cmp) == n_stars, "Check division of sources in GMM!"

			init = 0
			X = np.empty((n_stars,3))
			for n,l,s in zip(n_stars_cmp,position_args["loc"],position_args["scl"]):
				X[init:(init+n)] = np.array(l) + np.matmul(r[init:(init+n)],np.array(s))
				init += n

		else:
			X = position_args["loc"] + samples
		#--------------------------------------------------------

		return X

	def _generate_true_values(self,X,photometric_args,parallax_spatial_correlations="Vasiliev+2019"):
		N,D = X.shape

		assert D == 3, "Error, this code works only with ra,dec and parallax"
		
		#------- Sky coordinates -----------
		r,ra,dec = cartesian_to_spherical(X[:,0],X[:,1],X[:,2])

		#----- Transform ----------------------
		plx  = 1000.0/r             # In mas
		ra   = np.rad2deg(ra)       # In degrees
		dec  = np.rad2deg(dec)      # In degrees
		true = np.column_stack((ra,dec,plx))

		#------------ Total covariance of correlation -------------
		cov_corr = np.zeros((D*N,D*N))

		#------ Parallax spatial correlations -------------------
		if parallax_spatial_correlations is not None:

			#------ Angular separations ----------
			theta = AngularSeparation(true[:,:2])

			#-------- Covariance ------------------------------
			cov_corr_plx = CovarianceParallax(theta,
							case=parallax_spatial_correlations)

			#----- Fill parallax part ---------------
			idx = np.arange(2,D*N,step=D)
			cov_corr[np.ix_(idx,idx)] = cov_corr_plx

		#----------------- Photometric data ---------------------------------------

		#------- Sample from Chbarier prior-------
		masses  = ChabrierPrior().sample(10*N)

		#------- Only stars less massive than limit ------
		masses  = masses[np.where(masses < photometric_args["mass_limit"])[0]]
		masses  = np.random.choice(masses,N)
		df = self.tracks.generate(masses, photometric_args["log_age"], photometric_args["metallicity"], 
									 distance=r, 
									 AV=photometric_args["Av"])
		df.dropna(inplace=True)

		return true,df,cov_corr
	#======================================================================================

	def _assign_uncertainty(self,true_as,true_ph,cov_corr,release='dr3'):
		N,D = true_as.shape

		#-------Computes ra,dec, and parallax uncertainty ------------------------
		parallax_error = 1e-3*parallax_uncertainty(true_ph["G_mag"], release=release)
		position_error = 1e-3*total_position_uncertainty(true_ph["G_mag"], release=release)

		#-------------- Stack  ------------------------------------------------
		unc_as = np.column_stack((position_error,position_error,parallax_error))

		#------ Covariance of observational uncertainties --------
		cov_obs = np.diag(unc_as.flatten()**2)

		#------ Total covariance is the convolution of the previous ones
		cov =  cov_obs + cov_corr 

		#------- Correlated observations -----------------------------------------
		obs_as = st.multivariate_normal.rvs(mean=true_as.flatten(),cov=cov,size=1).reshape((N,D))
		#------------------------------------------------------------------------

		#--------- Data Frames -----------------------------------
		df_obs_as = pd.DataFrame(data=obs_as,columns=self.labels_obs_as)
		df_unc_as = pd.DataFrame(data=unc_as,columns=self.labels_unc_as)

		df_true = pd.DataFrame(data=true_as,columns=self.labels_true_as).join(true_ph)

		#-------- Join data frames -----------------------------
		df = df_true.join(df_obs_as).join(df_unc_as)
		
		return df

	#================= Generate cluster ==========================
	def generate_cluster(self,file,n_stars=100,
						parallax_spatial_correlations="Vasiliev+2019",
						index_label="source_id",
						release='dr3'):

		print("Generating synthetic data ...")

		#---------- Phase space coordinates --------------------------------------------------
		X = self._generate_phase_space(n_stars=n_stars,
							astrometric_args=self.astrometric_args)

		#--------------------- True values -----------------------------------------------------
		true,df_ph,cov_corr = self._generate_true_values(X,
									photometric_args=self.photometric_args,
									parallax_spatial_correlations=parallax_spatial_correlations)

		#------------- Adds uncertainty ---------------------------------------
		self.df = self._assign_uncertainty(true,df_ph,cov_corr,release=release)

		#----------- Save data frame ----------------------------
		self.df.to_csv(path_or_buf=file,index_label=index_label)


	#=========================Plot =====================================================
	def plot_cluster(self,file_plot,figsize=(10,10),color="blue",markersize=1):
		print("Plotting ...")
		pdf = PdfPages(filename=file_plot)
		n_bins = 100

		#----------------- X Y Z -----------------------------
		coords = { 2:["X","Z"], 3:["Z","Y"], 4:["X","Y"]}

		fig = plt.figure(figsize=figsize)
		count = 2

		for i in range(2):
			for j in range(2):
				if (i + j) != 0 :

					ax = fig.add_subplot(2, 2, count)
					#============ Data =========================
					x = coords[count][0]
					y = coords[count][1]

					ax.scatter(self.df[x],self.df[y],
								s=markersize,color=color)
					ax.set_xlabel(x + " [pc]")
					ax.set_ylabel(y + " [pc]")

					#----- Avoids crowded ticks --------------
					ax.locator_params(nbins=4)
					#-----------------------------------------

					count += 1


		ax = fig.add_subplot(2, 2, 1, projection='3d')
		ax.scatter(self.df["X"], self.df["Y"], self.df["Z"], 
								s=markersize, color=color)
		ax.set_xlabel("X [pc]")
		ax.set_ylabel("Y [pc]")
		ax.set_zlabel("Z [pc]")
		ax.view_init(25,-135)
		#----- Avoids crowded ticks --------------
		ax.locator_params(nbins=4)
		#-----------------------------------------

		fig.tight_layout()
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure()
		plt.scatter(self.df["ra"],self.df["dec"],
						s=markersize, color=color)
		plt.ylabel("Dec. [deg]")
		plt.xlabel("R.A. [deg]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure()
		plt.hist(self.df["parallax"],density=False,bins=n_bins,
					histtype="step",color=color)
		plt.ylabel("Density")
		plt.xlabel("parallax [mas]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure()
		plt.hist(self.df["parallax_error"],density=False,bins=n_bins,
					log=True,histtype="step",color=color)
		plt.ylabel("Density")
		plt.xlabel("parallax_error [mas]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure()
		plt.scatter(self.df["V_mag"]-self.df["I_mag"],self.df["G_mag"],
						s=markersize, color=color)
		plt.ylabel("G [mag]")
		plt.xlabel("V - I [mag]")
		plt.gca().invert_yaxis()
		pdf.savefig(bbox_inches='tight')
		plt.close()

		pdf.close()
	#------------------------------------------------------------------------------

if __name__ == "__main__":

	dir_main      =  "/home/javier/Repositories/Amasijo/Data/"
	file_plot     = dir_main + "Plot.pdf"
	file_data     = dir_main + "synthetic.csv"
	random_seeds  = [1]    # Random state for the synthetic data
	n_stars       = 100

	astrometric_args = {
		"position":{
			"family":"Gaussian",
			"loc":np.array([50.,50.,50.]),
			"scl":np.eye(3)*10.0,
			"loc2":np.array([150.0,0.0,0.0]),
			"scl2":np.eye(3)*20.0,
			"fraction":0.5,
			"gamma": 5.0,
			"tidal_radius": 5.0
			},
		"velocity":{
			"family":"Gaussian",
			"loc":np.array([50.,50.,50.]),
			"scl":np.eye(3)*10.0
			}
	}
	photometric_args = {
		"log_age": 8.2,     # Solar metallicity
		"metallicity":0.02, # Typical value of Bossini+2019
		"Av": 0.0,          # No extinction
		"mass_limit":4.0,   # Avoids NaNs in photometry
		"bands":"VIG"
	}

	for seed in random_seeds:
		ama = Amasijo(astrometric_args=astrometric_args,
					  photometric_args=photometric_args,
					  seed=seed,)

		ama.generate_cluster(file_data,n_stars=n_stars)

		ama.plot_cluster(file_plot=file_plot)











