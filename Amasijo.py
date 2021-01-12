import sys
import os
import numpy  as np
import pandas as pd
import scipy.stats as st
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Functions import AngularSeparation,CovarianceParallax
from distributions import mveff,mvking

from pygaia.astrometry.vectorastrometry import cartesian_to_spherical
from pygaia.errors.astrometric import parallax_uncertainty,total_position_uncertainty
from pygaia.errors.photometric import g_magnitude_uncertainty,bp_magnitude_uncertainty,rp_magnitude_uncertainty

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
		self.labels_phase_space = ["X","Y","Z"]
		self.labels_true_as = ["ra_true","dec_true","parallax_true"]
		self.labels_true_ph = [band+"_mag" for band in photometric_args["bands"]]
		self.labels_obs_as  = ["ra","dec","parallax"]
		self.labels_unc_as  = ["ra_error","dec_error","parallax_error"]
		self.labels_obs_ph  = ["g","bp","rp"]
		self.labels_unc_ph  = ["g_error","bp_error","rp_error"]

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
		if position_args["family"] == "Gaussian":
			XYZ = st.multivariate_normal.rvs(mean=position_args["loc"],cov=position_args["scl"],
												size=n_stars)

		elif position_args["family"] == "EFF":
			XYZ = mveff.rvs(loc=position_args["loc"],scale=position_args["scl"],
							gamma=position_args["gamma"],size=n_stars)

		elif position_args["family"] == "King":
			XYZ = mvking.rvs(loc=position_args["loc"],scale=position_args["scl"],
							rt=position_args["tidal_radius"],size=n_stars)

		elif position_args["family"] == "GMM":
			assert np.sum(position_args["weights"]) == 1.0,"weights must be a simplex"
			n_cmp = len(position_args["weights"])
			n_stars_cmp = np.floor(position_args["weights"]*n_stars).astype('int')
			n_res = n_stars - np.sum(n_stars_cmp)
			residual = np.ones(n_cmp)
			residual[n_res:] = 0
			n_stars_cmp += residual
			assert np.sum(n_stars_cmp) == n_stars, "Check division of sources in GMM!"

			l = []
			for n,loc,scl in zip(n_stars_cmp,position_args["loc"],position_args["scl"]):
				l.append(st.multivariate_normal.rvs(mean=loc,cov=scl,size=n))

			XYZ = np.concatenate(l,axis=0)

		else:
			sys.exit("Error: incorrect family argument")
		#===============================================================================================

		return XYZ

	def _generate_true_values(self,XYZ,photometric_args,parallax_spatial_correlations="Vasiliev+2019"):
		N,D = XYZ.shape

		assert D == 3, "Error, this code works only with ra,dec and parallax"
		
		#------- Sky coordinates -----------
		r,ra,dec = cartesian_to_spherical(XYZ[:,0],XYZ[:,1],XYZ[:,2])

		#----- Transform ----------------------
		plx  = 1000.0/r             # In mas
		ra   = np.rad2deg(ra)       # In degrees
		dec  = np.rad2deg(dec)      # In degrees
		true = np.column_stack((ra,dec,plx))

		#-------- Data Frame --------------------------------------
		df_as = pd.DataFrame(data=true,columns=self.labels_true_as)

		#------------ Total covariance of correlation -------------
		cov_as = np.zeros((D*N,D*N))

		#------ Parallax spatial correlations -------------------
		if parallax_spatial_correlations is not None:

			#------ Angular separations ----------
			theta = AngularSeparation(true[:,:2])

			#-------- Covariance ------------------------------
			cov_corr_plx = CovarianceParallax(theta,
							case=parallax_spatial_correlations)

			#----- Fill parallax part ---------------
			idx = np.arange(2,D*N,step=D)
			cov_as[np.ix_(idx,idx)] = cov_corr_plx
		#--------------------------------------------------------

		#=============== Photometric data ===================================
		#------- Sample from Chabrier prior-------
		masses  = ChabrierPrior().sample(3*N)

		#------- Only stars less massive than mass_limit ---------------------
		masses  = masses[np.where(masses < photometric_args["mass_limit"])[0]]

		#---------- Indices -----------
		idx = np.arange(start=0,stop=N)

		#------- Obtains photometry --------------------------------
		df_ph = self.tracks.generate(masses[idx], 
									photometric_args["log_age"], 
									photometric_args["metallicity"], 
									distance=r, 
									AV=photometric_args["Av"])
		#-----------------------------------------------------------

		#------- Ensure that sources are within Gaia limit ---------
		while any(df_ph["G_mag"] > 21.0):
			#-------- Find indices and index -----
			idx = np.where(df_ph["G_mag"] > 21.0)[0]
			index = df_ph.iloc[idx].index.values
			#-------------------------------------
			
			#----------- Choose new masses -------------
			idy = np.random.choice(np.arange(start=N,
									stop=len(masses)),
									size=len(idx))
			#-------------------------------------------

			#---------- Generates photometry -----------------------
			tmp = self.tracks.generate(masses[idy], 
									photometric_args["log_age"], 
									photometric_args["metallicity"], 
									distance=r[idx], 
									AV=photometric_args["Av"])
			tmp.set_index(index,inplace=True)
			#-------------------------------------------------------

			print("Replacing {0} sources beyond Gaia limit".format(len(idx)))
			#------------------------------------------------
			df_ph.update(tmp)
		#-------------------------------------------------------------

		#- Drop missing values ----
		df_ph.dropna(subset=["G_mag"],inplace=True)

		#---- Join true values -------
		df_true = df_as.join(df_ph)

		return df_true,cov_as
	#======================================================================================

	def _generate_observed_values(self,true,cov_as,release='dr3'):
		N = len(true)

		#---- Astrometric and photometric values ------
		true_as = true.loc[:,self.labels_true_as]
		true_ph = true.loc[:,["G_mag","BP_mag","RP_mag"]]
		#-----------------------------------------------

		#------- Uncertainties -------------------------------------
		plx_unc = 1e-3*parallax_uncertainty(true_ph["G_mag"],
								release=release)

		pos_unc = 1e-3*total_position_uncertainty(true_ph["G_mag"], 
								release=release)

		g_unc   = g_magnitude_uncertainty(true_ph["G_mag"])

		bp_unc  = bp_magnitude_uncertainty(true_ph["G_mag"],
								true["V_mag"] - true["I_mag"])

		rp_unc  = rp_magnitude_uncertainty(true_ph["G_mag"], 
								true["V_mag"] - true["I_mag"])
		#------------------------------------------------------------

		#------ Stack values ------------------------------
		unc_as = np.column_stack((pos_unc,pos_unc,plx_unc))
		unc_ph = np.column_stack((g_unc,bp_unc,rp_unc))
		#---------------------------------------------------

		#======= Astrometry ======================
		#------ Covariance ---------------
		u_as = unc_as.copy()
		u_as[:,0] /= 3.6e6
		u_as[:,1] /= 3.6e6
		cov = np.diag(u_as.flatten()**2)
		# Plus correlations
		cov +=  cov_as 
		#----------------------------------

		#------- Observed values ------------
		obs_as = st.multivariate_normal.rvs(
				mean=true_as.to_numpy().flatten(),
				cov=cov,
				size=1).reshape((N,3))
		#-------------------------------------

		#--------- Data Frames ------------------
		df_obs_as = pd.DataFrame(data=obs_as,
					columns=self.labels_obs_as)
		df_unc_as = pd.DataFrame(data=unc_as,
					columns=self.labels_unc_as)

		df_as = df_obs_as.join(df_unc_as)
		#-----------------------------------------
		#===========================================

		#======= Photometry ======================
		#------ Covariance ---------------
		cov = np.diag(unc_ph.flatten()**2)
		#----------------------------------

		#------- Observed values ------------
		obs_ph = st.multivariate_normal.rvs(
				mean=true_ph.to_numpy().flatten(),
				cov=cov,
				size=1).reshape((N,3))
		#-------------------------------------

		#--------- Data Frames ------------------
		df_obs_ph = pd.DataFrame(data=obs_ph,
					columns=self.labels_obs_ph)
		df_unc_ph = pd.DataFrame(data=unc_ph,
					columns=self.labels_unc_ph)

		df_ph = df_obs_ph.join(df_unc_ph)
		#-----------------------------------------
		#===========================================
		
		return df_as.join(df_ph)

	#================= Generate cluster ==========================
	def generate_cluster(self,file,n_stars=100,
						parallax_spatial_correlations="Vasiliev+2019",
						index_label="source_id",
						release='dr3'):

		print("Generating synthetic data ...")

		#---------- Phase space coordinates --------------------------------------------------
		XYZ = self._generate_phase_space(n_stars=n_stars,
							astrometric_args=self.astrometric_args)
		df_xyz = pd.DataFrame(data=XYZ,columns=self.labels_phase_space)

		#--------------------- True values -----------------------------------------------------
		df_true,cov_as = self._generate_true_values(XYZ,
									photometric_args=self.photometric_args,
									parallax_spatial_correlations=parallax_spatial_correlations)

		#------------- Observed values ---------------------------------------
		df_obs = self._generate_observed_values(df_true,cov_as,release=release)

		#-------- Join data frames -----------------------------
		self.df = df_xyz.join(df_true).join(df_obs)

		#----------- Save data frame ----------------------------
		self.df.to_csv(path_or_buf=file,index_label=index_label)


	#=========================Plot =====================================================
	def plot_cluster(self,file_plot,figsize=(10,10),n_nins=50,
			cases={
					"true":    {"color":"red", "ms":5,  "label":"True values"},
					"observed":{"color":"blue","ms":10, "label":"Observed values"}
				}
				):
		print("Plotting ...")
		pdf = PdfPages(filename=file_plot)

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
								s=cases["true"]["ms"],
								color=cases["true"]["color"],
								label=cases["true"]["label"])
					ax.set_xlabel(x + " [pc]")
					ax.set_ylabel(y + " [pc]")

					#----- Avoids crowded ticks --------------
					ax.locator_params(nbins=4)
					#-----------------------------------------

					count += 1


		ax = fig.add_subplot(2, 2, 1, projection='3d')
		ax.scatter(self.df["X"], self.df["Y"], self.df["Z"], 
								s=cases["true"]["ms"], 
								color=cases["true"]["color"],
								label=cases["true"]["label"])
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

		plt.figure(figsize=figsize)
		plt.scatter(self.df["ra"],self.df["dec"],
						s=cases["observed"]["ms"], 
						color=cases["observed"]["color"],
						label=cases["observed"]["label"])
		plt.scatter(self.df["ra_true"],self.df["dec_true"],
						s=cases["true"]["ms"], 
						color=cases["true"]["color"],
						label=cases["true"]["label"])
		plt.ylabel("ra [deg]")
		plt.xlabel("dec [deg]")
		plt.legend(title="Cases",loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.hist(self.df["parallax"],density=False,
					bins=n_bins,histtype="step",
					color=cases["observed"]["color"],
					label=cases["observed"]["label"])
		plt.hist(self.df["parallax_true"],density=False,
					bins=n_bins,histtype="step",
					color=cases["true"]["color"],
					label=cases["true"]["label"])
		plt.ylabel("Density")
		plt.xlabel("parallax [mas]")
		plt.legend(title="Cases",loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.hist(self.df["ra_error"],density=True,bins=n_bins,
					log=True,histtype="step",color=cases["observed"]["color"])
		plt.ylabel("Density")
		plt.xlabel("ra_error [mas]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.hist(self.df["parallax_error"],density=True,bins=n_bins,
					log=True,histtype="step",color=cases["observed"]["color"])
		plt.ylabel("Density")
		plt.xlabel("parallax_error [mas]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.hist(self.df["g_error"],density=True,bins=n_bins,
					log=True,histtype="step",color=cases["observed"]["color"])
		plt.ylabel("Density")
		plt.xlabel("g_error [mag]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.hist(self.df["bp_error"],density=True,bins=n_bins,
					log=True,histtype="step",color=cases["observed"]["color"])
		plt.ylabel("Density")
		plt.xlabel("bp_error [mag]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.hist(self.df["rp_error"],density=True,bins=n_bins,
					log=True,histtype="step",color=cases["observed"]["color"])
		plt.ylabel("Density")
		plt.xlabel("rp_error [mag]")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.scatter(self.df["bp"]-self.df["rp"],
					self.df["g"],
					s=cases["observed"]["ms"],
					color=cases["observed"]["color"],
					label=cases["observed"]["label"])
		plt.scatter(self.df["BP_mag"]-self.df["RP_mag"],
					self.df["G_mag"],
					s=cases["true"]["ms"],
					color=cases["true"]["color"],
					label=cases["true"]["label"])
		plt.ylabel("G [mag]")
		plt.xlabel("BP - RP [mag]")
		plt.gca().invert_yaxis()
		plt.legend(title="Cases",loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		plt.figure(figsize=figsize)
		plt.scatter(self.df["g"]-self.df["rp"],
					self.df["g"],
					s=cases["observed"]["ms"],
					color=cases["observed"]["color"],
					label=cases["observed"]["label"])
		plt.scatter(self.df["G_mag"]-self.df["RP_mag"],
					self.df["G_mag"],
					s=cases["true"]["ms"],
					color=cases["true"]["color"],
					label=cases["true"]["label"])
		plt.ylabel("G [mag]")
		plt.xlabel("G - RP [mag]")
		plt.gca().invert_yaxis()
		plt.legend(title="Cases",loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()

		pdf.close()
	#------------------------------------------------------------------------------

if __name__ == "__main__":

	dir_main      =  "/home/javier/Repositories/Amasijo/Data/"
	file_plot     = dir_main + "Plot.pdf"
	file_data     = dir_main + "synthetic.csv"
	random_seeds  = [1]    # Random state for the synthetic data
	n_stars       = 1000

	astrometric_args = {
		"position":{
			"family":"Gaussian",
				"loc":np.array([500.,500.,500.]),
				"scl":np.eye(3)*10.0
			},
		"velocity":{
			"family":"Gaussian",
				"loc":np.array([0.,0.,0.]),
				"scl":np.eye(3)*2.0
			}
	}
	photometric_args = {
		"log_age": 8.2,     # Solar metallicity
		"metallicity":0.02, # Typical value of Bossini+2019
		"Av": 0.0,          # No extinction
		"mass_limit":4.0,   # Avoids NaNs in photometry
		"bands":["V","I","G","BP","RP"]
	}

	for seed in random_seeds:
		ama = Amasijo(astrometric_args=astrometric_args,
					  photometric_args=photometric_args,
					  seed=seed,)

		ama.generate_cluster(file_data,n_stars=n_stars)

		ama.plot_cluster(file_plot=file_plot)











