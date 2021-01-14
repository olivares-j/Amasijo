import sys
import os
import numpy  as np
import pandas as pd
import scipy.stats as st
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Functions import AngularSeparation,covariance_parallax,covariance_proper_motion
from distributions import mveff,mvking

from pygaia.astrometry.vectorastrometry import phase_space_to_astrometry
from pygaia.errors.astrometric import parallax_uncertainty,total_position_uncertainty,proper_motion_uncertainty
from pygaia.errors.photometric import g_magnitude_uncertainty,bp_magnitude_uncertainty,rp_magnitude_uncertainty
from pygaia.errors.spectroscopic import vrad_error_sky_avg 

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
		self.labels_phase_space = ["X","Y","Z","U","V","W"]
		self.labels_true_as = ["ra_true","dec_true","parallax_true",
								"pmra_true","pmdec_true","radial_velocity_true"]
		self.labels_true_ph = [band+"_mag" for band in photometric_args["bands"]]
		self.labels_obs_as  = ["ra","dec","parallax","pmra","pmdec","radial_velocity"]
		self.labels_unc_as  = ["ra_error","dec_error","parallax_error",
								"pmra_error","pmdec_error","radial_velocity_error"]
		self.labels_obs_ph  = ["g","bp","rp"]
		self.labels_unc_ph  = ["g_error","bp_error","rp_error"]
		self.labels_rvl     = ["radial_velocity","radial_velocity_error"]

	#====================== Generate Astrometric Data ==================================================

	def _generate_phase_space(self,n_stars):
		'''	The phase space coordinates are
    		assumed to represent barycentric (i.e. centred on the Sun) positions and velocities.
    	'''
		position_args = self.astrometric_args["position"]
		velocity_args = self.astrometric_args["velocity"]

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
			residual = np.ones(n_cmp).astype('int')
			residual[n_res:] = 0
			n_stars_cmp += residual
			assert np.sum(n_stars_cmp) == n_stars, "Check division of sources in GMM!"

			l = []
			for n,loc,scl in zip(n_stars_cmp,position_args["loc"],position_args["scl"]):
				l.append(st.multivariate_normal.rvs(mean=loc,cov=scl,size=n))

			XYZ = np.concatenate(l,axis=0)

		else:
			sys.exit("Error: incorrect position family argument")
		#===============================================================================================

		#=============================== Velocities ========================================================
		if velocity_args["family"] == "Gaussian":
			UVW = st.multivariate_normal.rvs(mean=velocity_args["loc"],cov=velocity_args["scl"],
												size=n_stars)

		elif velocity_args["family"] == "GMM":
			assert np.sum(velocity_args["weights"]) == 1.0,"weights must be a simplex"
			n_cmp = len(velocity_args["weights"])
			n_stars_cmp = np.floor(velocity_args["weights"]*n_stars).astype('int')
			n_res = n_stars - np.sum(n_stars_cmp)
			residual = np.ones(n_cmp).astype('int')
			residual[n_res:] = 0
			n_stars_cmp += residual
			assert np.sum(n_stars_cmp) == n_stars, "Check division of sources in GMM!"

			l = []
			for n,loc,scl in zip(n_stars_cmp,velocity_args["loc"],velocity_args["scl"]):
				l.append(st.multivariate_normal.rvs(mean=loc,cov=scl,size=n))

			UVW = np.concatenate(l,axis=0)

		else:
			sys.exit("Error: incorrect velocity family argument")
		#===============================================================================================

		XYZUVW = np.hstack((XYZ,UVW))

		return XYZUVW

	def _generate_true_values(self,n_stars):

		#================= Astrometric data =================================
		#---------- Phase space coordinates --------------
		X = self._generate_phase_space(n_stars=n_stars)
		#------------------------------------------------
		
		#------- Astrometry & Radial velocity --------------------
		ra,dec,plx,mua,mud,rvel = phase_space_to_astrometry(
						X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5])
		#---------------------------------------------------------

		#----- Transform ----------------------
		ra   = np.rad2deg(ra)       # In degrees
		dec  = np.rad2deg(dec)      # In degrees
		r    = 1000.0/plx           # Distance
		true = np.column_stack((ra,dec,plx,mua,mud,rvel))

		#-------- Data Frames ---------------------------
		df_as = pd.DataFrame(data=true,
						columns=self.labels_true_as)

		df_ps = pd.DataFrame(data=X,
						columns=self.labels_phase_space)
		#------------------------------------------------
		#====================================================================

		#=============== Photometric data ===================================
		#------- Sample from Chabrier prior-------
		masses  = ChabrierPrior().sample(3*n_stars)

		#------- Only stars less massive than mass_limit ---------------------
		masses  = masses[np.where(masses < photometric_args["mass_limit"])[0]]

		#---------- Indices -----------
		idx = np.arange(start=0,stop=n_stars)

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
			idy = np.random.choice(np.arange(start=n_stars,
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

			#print("Replacing {0} sources beyond Gaia limit".format(len(idx)))
			#------------------------------------------------
			df_ph.update(tmp)
		#-------------------------------------------------------------

		#- Drop missing values --------------------
		df_ph.dropna(subset=["G_mag"],inplace=True)

		#---- Join true values -------
		df_true = df_as.join(df_ph)

		return df_true,df_ps
	#======================================================================================

	def _generate_observed_values(self,true,
				release='dr3',
				angular_correlations="Vasiliev+2019",
				radial_velocity_gmag_limits=[4.0,13.0]):
		N = len(true)

		rng = np.random.default_rng()

		#---- Astrometric and photometric values ------
		true_as = true.loc[:,self.labels_true_as]
		true_ph = true.loc[:,["G_mag","BP_mag","RP_mag"]]
		#-----------------------------------------------

		#=================== Angular correlations =========================
		angular_corr = np.zeros((6*N,6*N))

		#------ Angular separations ----------
		theta = AngularSeparation(true[self.labels_true_as[:2]].to_numpy())

		if angular_correlations is not None:
			#-- Parallax angular correlations --------
			corr_plx = covariance_parallax(theta,
							case=angular_correlations)

			#-- Proper motions angular correlations --
			corr_ppm = covariance_proper_motion(theta,
							case=angular_correlations)

			#----- Fill parallax part ---------------
			idx = np.arange(2,6*N,step=6)
			angular_corr[np.ix_(idx,idx)] = corr_plx
			#----------------------------------------

			#----- Fill pmra part ---------------
			idx = np.arange(3,6*N,step=6)
			angular_corr[np.ix_(idx,idx)] = corr_ppm

			#----- Fill pmdec part ---------------
			idx = np.arange(4,6*N,step=6)
			angular_corr[np.ix_(idx,idx)] = corr_ppm
		#=====================================================================

		#================== Uncertainties ================================
		pos_unc = total_position_uncertainty(true_ph["G_mag"],
							release=release)

		plx_unc = parallax_uncertainty(true_ph["G_mag"],
							release=release)

		mua_unc,mud_unc = proper_motion_uncertainty(true_ph["G_mag"], 
							release=release)

		rvl_unc = vrad_error_sky_avg(true["V_mag"],
							'B0V',extension=0.0) #FIXME

		g_unc   = g_magnitude_uncertainty(true_ph["G_mag"])

		bp_unc  = bp_magnitude_uncertainty(true_ph["G_mag"],
							true["V_mag"] - true["I_mag"])

		rp_unc  = rp_magnitude_uncertainty(true_ph["G_mag"], 
							true["V_mag"] - true["I_mag"])
		#------------------------------------------------------------

		#--- Correct units ----
		pos_unc *= 1e-3
		plx_unc *= 1e-3
		mua_unc *= 1e-3
		mud_unc *= 1e-3
		#----------------------

		#------ Stack values ------------------------------
		unc_as = np.column_stack((pos_unc,pos_unc,plx_unc,
								  mua_unc,mud_unc,rvl_unc))
		unc_ph = np.column_stack((g_unc,bp_unc,rp_unc))
		#---------------------------------------------------
		#==================================================================

		#======= Astrometry =================================
		#------ Covariance ---------------
		u_as = unc_as.copy()
		# Uncertainty must be in same units as value
		u_as[:,0] /= 3.6e6 # mas to degrees
		u_as[:,1] /= 3.6e6 # mas to degrees
		cov_as = np.diag(u_as.flatten()**2)
		# Add angular correlations
		cov_as +=  angular_corr 
		#----------------------------------

		#------- Observed values ---------------------
		obs_as = rng.multivariate_normal(
					mean=true_as.to_numpy().flatten(),
					cov=cov_as,
					tol=1e-8,
					method='cholesky',
					size=1).reshape((N,6))
		#---------------------------------------------

		#--------- Data Frames ------------------
		df_obs_as = pd.DataFrame(data=obs_as,
					columns=self.labels_obs_as)
		df_unc_as = pd.DataFrame(data=unc_as,
					columns=self.labels_unc_as)

		df_as = df_obs_as.join(df_unc_as)
		#-----------------------------------------
		#===========================================================

		#======= Photometry ======================
		#------ Covariance ---------------
		cov_ph = np.diag(unc_ph.flatten()**2)
		#----------------------------------

		#------- Observed values ------------
		obs_ph = rng.multivariate_normal(
					mean=true_ph.to_numpy().flatten(),
					cov=cov_ph,
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

		#------- Join ------------
		df_obs = df_as.join(df_ph)

		#--------- Radial velocity limits ---------------------
		mask  = df_obs['g'] < radial_velocity_gmag_limits[0] 
		mask |= df_obs['g'] > radial_velocity_gmag_limits[1]
		df_obs.loc[mask,self.labels_rvl] = np.nan
		#------------------------------------------------------
		
		return df_obs

	#================= Generate cluster ==========================
	def generate_cluster(self,file,n_stars=100,
						angular_correlations="Vasiliev+2019",
						index_label="source_id",
						release='dr3'):

		#--------- True values -----------------------------------------------------
		print("Generating true values ...")
		df_true,df_ps = self._generate_true_values(n_stars)

		#------------- Observed values ---------------------------------------
		print("Generating observed values ...")
		df_obs = self._generate_observed_values(df_true,
						release=release,
						angular_correlations=angular_correlations)

		#-------- Join data frames -----------------------------
		self.df = df_obs.join(df_true).join(df_ps)

		#----------- Save data frame ----------------------------
		self.df.to_csv(path_or_buf=file,index_label=index_label)


	#=========================Plot =====================================================
	def plot_cluster(self,file_plot,figsize=(10,10),n_bins=50,
			cases={
					"true":    {"color":'#377eb8', "ms":5,  "label":"True values"},
					"observed":{"color":'#ff7f00',"ms":10, "label":"Observed values"}
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
					x = coords[count][0]
					y = coords[count][1]

					ax.scatter(self.df[x],self.df[y],
								s=cases["true"]["ms"],
								color=cases["true"]["color"],
								label=cases["true"]["label"])
					ax.set_xlabel(x + " [pc]")
					ax.set_ylabel(y + " [pc]")
					ax.locator_params(nbins=4)
					count += 1


		ax = fig.add_subplot(2, 2, 1, projection='3d')
		ax.scatter(self.df["X"], self.df["Y"], self.df["Z"], 
								s=cases["true"]["ms"], 
								color=cases["true"]["color"],
								label=cases["true"]["label"])
		ax.set_xlabel("X [pc]")
		ax.set_ylabel("Y [pc]")
		ax.set_zlabel("Z [pc]")
		# View point
		ax.view_init(25,-135)
		# Avoids crowded ticks 
		ax.locator_params(nbins=4)

		fig.tight_layout()
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------

		#----------------- U V W -----------------------------
		coords = { 2:["U","W"], 3:["W","V"], 4:["U","V"]}

		fig = plt.figure(figsize=figsize)
		count = 2

		for i in range(2):
			for j in range(2):
				if (i + j) != 0 :
					ax = fig.add_subplot(2, 2, count)
					x = coords[count][0]
					y = coords[count][1]

					ax.scatter(self.df[x],self.df[y],
								s=cases["true"]["ms"],
								color=cases["true"]["color"],
								label=cases["true"]["label"])
					ax.set_xlabel(x + " [km/s]")
					ax.set_ylabel(y + " [km/s]")
					ax.locator_params(nbins=4)
					count += 1


		ax = fig.add_subplot(2, 2, 1, projection='3d')
		ax.scatter(self.df["U"], self.df["V"], self.df["W"], 
								s=cases["true"]["ms"], 
								color=cases["true"]["color"],
								label=cases["true"]["label"])
		ax.set_xlabel("U [km/s]")
		ax.set_ylabel("V [km/s]")
		ax.set_zlabel("W [km/s]")
		# View point
		ax.view_init(25,-135)
		# Avoids crowded ticks 
		ax.locator_params(nbins=4)

		fig.tight_layout()
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------

		#---------- Sky coordinates -----------------------
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
		plt.legend(loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#--------------------------------------------------

		#---------- Proper motions -----------------------
		plt.figure(figsize=figsize)
		plt.scatter(self.df["pmra"],self.df["pmdec"],
						s=cases["observed"]["ms"], 
						color=cases["observed"]["color"],
						label=cases["observed"]["label"])
		plt.scatter(self.df["pmra_true"],self.df["pmdec_true"],
						s=cases["true"]["ms"], 
						color=cases["true"]["color"],
						label=cases["true"]["label"])
		plt.ylabel("pmra [mas/yr]")
		plt.xlabel("pmdec [mas/yr]")
		plt.legend(loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#--------------------------------------------------

		#---------- Proper motions -----------------------
		plt.figure(figsize=figsize)
		plt.scatter(self.df["parallax"],self.df["pmdec"],
						s=cases["observed"]["ms"], 
						color=cases["observed"]["color"],
						label=cases["observed"]["label"])
		plt.scatter(self.df["parallax_true"],self.df["pmdec_true"],
						s=cases["true"]["ms"], 
						color=cases["true"]["color"],
						label=cases["true"]["label"])
		plt.xlabel("parallax [mas]")
		plt.ylabel("pmdec [mas/yr]")
		plt.legend(loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#--------------------------------------------------

		#------------ Parallax ----------------------------
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
		plt.legend(loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#--------------------------------------------------

		#------------ Radial velocity ----------------------------
		plt.figure(figsize=figsize)
		plt.hist(self.df["radial_velocity"],density=False,
					bins=n_bins,histtype="step",
					color=cases["observed"]["color"],
					label=cases["observed"]["label"])
		plt.hist(self.df["radial_velocity_true"],density=False,
					bins=n_bins,histtype="step",
					color=cases["true"]["color"],
					label=cases["true"]["label"])
		plt.ylabel("Density")
		plt.xlabel("radial_velocity [km/s]")
		plt.legend(loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#--------------------------------------------------

		#----------- CMDs -----------------------------------
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
		plt.legend(loc="best")
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
		plt.legend(loc="best")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#-------------------------------------------------

		#---------- Uncertainties ----------------------
		features = ["ra_error","pmra_error","parallax_error",
					"radial_velocity_error",
					"g_error","bp_error","rp_error"]
		labels = [f+u for f,u in zip(features,[" [mas]"," [mas/yr]",
					" [mas]"," [km/s]"," [mag]"," [mag]"," [mag]"])]

		fig, axs = plt.subplots(nrows=7, ncols=1,figsize=figsize,
					gridspec_kw={"hspace":0.4})
		for ax,feature,label in zip(axs,features,labels):
			ax.hist(self.df[feature],density=True,
					bins=n_bins,log=True,histtype="step",
					color=cases["observed"]["color"])
			ax.set_ylabel("Density")
			ax.set_xlabel(label,labelpad=0)
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#--------------------------------------------------------

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
				"loc":np.array([100.,100.,100.]),
				"scl":np.eye(3)*10.0
			# "family":"EFF",
			# 	"loc":np.array([500.,500.,500.]),
			# 	"scl":np.eye(3)*10.0,
			# 	"gamma":3.5
			# "family":"King",
			# 	"loc":np.array([500.,500.,500.]),
			# 	"scl":np.eye(3)*10.0,
			# 	"tidal_radius":5.
			# "family":"GMM",
			# 	"weights":np.array([0.4,0.6]),
			# 	"loc":[np.array([500.,500.,500.]),
			# 		   np.array([550.,550.,550.])],
			# 	"scl":[np.eye(3)*10.0,np.eye(3)*20.0]
			},
		"velocity":{
			"family":"Gaussian",
				"loc":np.array([10.,10.,10.]),
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











