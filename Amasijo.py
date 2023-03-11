import sys
import os
import numpy  as np
import pandas as pd
import scipy.stats as st
from time import time
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import splrep,splev

from Functions import AngularSeparation,covariance_parallax,covariance_proper_motion
from distributions import mveff,mvking

from pygaia.astrometry.vectorastrometry import phase_space_to_astrometry
from pygaia.errors.astrometric import parallax_uncertainty,position_uncertainty,proper_motion_uncertainty
from pygaia.errors.photometric import magnitude_uncertainty
from pygaia.errors.spectroscopic import radial_velocity_uncertainty

from isochrones import get_ichrone
from isochrones.priors import ChabrierPrior


###############################################################################
class Amasijo(object):
	"""This class intends to construct synthetic clusters with simple 
		astrometric distributions and photometry from stellar models"""
	def __init__(self,photometric_args,
					astrometric_args=None,
					mcluster_args=None,
					kalkayotl_file=None,
					release="dr3",
					seed=1234):

		#------ Set Seed -----------------------------------
		np.random.seed(seed=seed)
		self.random_state = np.random.RandomState(seed=seed)
		self.seed = seed
		self.release = release
		#---------------------------------------------------

		#--------------- Tracks ----------------------------
		self.tracks = get_ichrone('mist', tracks=True,
						bands=photometric_args["bands"])
		#---------------------------------------------------

		#---------- Arguments ---------------------------------------------------------------
		self.photometric_args = photometric_args

		if astrometric_args is not None:
			self.astrometric_args = astrometric_args
			case = "astrometric"
		elif mcluster_args is not None:
			self.mcluster_args = mcluster_args
			case = "McLuster"
		elif kalkayotl_file is not None:
			assert os.path.exists(kalkayotl_file), "Input Kalkayotl file does not exists!"
			self.astrometric_args = self._read_kalkayotl(kalkayotl_file)
			case = "Kalkayotl"
		else:
			sys.exit("You must provide astrometric_args, mcluster_args or a Kalkayotl file!")

		print("Astrometry will be generated from the provided {0} arguments!".format(case))
		#-----------------------------------------------------------------------------------
		

		#------- Labels ----------------------------------------------------------------------------------
		self.labels_phase_space = ["X","Y","Z","U","V","W"]
		self.labels_true_as = ["ra_true","dec_true","parallax_true",
								"pmra_true","pmdec_true","radial_velocity_true"]
		self.labels_true_ph = [band+"_mag" for band in photometric_args["bands"]]
		self.labels_obs_as  = ["ra","dec","parallax","pmra","pmdec","{0}_radial_velocity".format(release)]
		self.labels_unc_as  = ["ra_error","dec_error","parallax_error",
								"pmra_error","pmdec_error","{0}_radial_velocity_error".format(release)]
		self.labels_cor_as  = ["ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
							   "dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
							   "parallax_pmra_corr","parallax_pmdec_corr",
							   "pmra_pmdec_corr"]
		self.labels_obs_ph  = ["g","bp","rp"]
		self.labels_unc_ph  = ["g_error","bp_error","rp_error"]
		self.labels_rvl     = ["{0}_radial_velocity".format(release),"{0}_radial_velocity_error".format(release)]
		#--------------------------------------------------------------------------------------------

	def _read_kalkayotl(self,file):
		#-------- Read file ----------------------------------
		param = pd.read_csv(file,usecols=["Parameter","mode"])
		#-----------------------------------------------------

		if any(param["Parameter"].str.contains("weights")):
			#===================== GMM and CGMM =======================================

			#--------- Weights -------------------------------------------------------
			wghs = param.loc[param["Parameter"].str.contains("weights"),"mode"].values
			#------------------------------------------------------------------------

			#------------- Location ----------------------------------
			loc = param[param["Parameter"].str.contains("loc")]


			if loc.shape[0]/len(wghs) == 6:
				#------------- GMM ------------------------------------
				family = "GMM"
				locs = []
				for i in range(len(wghs)):
					selection = loc["Parameter"].str.contains(
								"[{0}]".format(i),regex=False)
					locs.append(loc.loc[selection,"mode"].values)
				#---------------------------------------------------------
			else:
				#-------------- CGMM -------------------------------------------------
				family = "CGMM"
				loc = param.loc[param["Parameter"].str.contains("loc"),"mode"].values
				locs = [loc for w in wghs]
				#---------------------------------------------------------------------

			#------------- Covariances -----------------------
			scl = param.fillna(value=1.0)

			stds = []
			cors = []
			covs = []
			for i in range(len(wghs)):
				#---------- Select component parameters --------
				mask_std = scl["Parameter"].str.contains(
							"{0}_stds".format(i),regex=False)
				mask_cor = scl["Parameter"].str.contains(
							"{0}_corr".format(i),regex=False)
				#-----------------------------------------------

				#------Extract parameters -------------------
				std = scl.loc[mask_std,"mode"].values
				cor = scl.loc[mask_cor,"mode"].values
				#--------------------------------------------

				stds.append(std)

				#---- Construct covariance --------------
				std = np.diag(std)
				cor = np.reshape(cor,(6,6))
				cov = np.dot(std,cor.dot(std))
				#-----------------------------------------

				#--- Append -------
				cors.append(cor)
				covs.append(cov)
				#------------------
			#-------------------------------------------------

			astrometric_args = {
			"position+velocity":{
					"family":family,
					"weights":wghs,
					"location":locs,
					"covariance":covs}
				}
			#========================================================================
		else:
			#=================== Gaussian ===========================================
			#---- Extract parameters ------------------------------------------------
			loc  = param.loc[param["Parameter"].str.contains("loc"),"mode"].values
			param.fillna(value=1.0,inplace=True)
			stds = param.loc[param["Parameter"].str.contains('stds'),"mode"].values
			corr = param.loc[param["Parameter"].str.contains('corr'),"mode"].values
			#------------------------------------------------------------------------

			#---- Construct covariance --------------
			stds = np.diag(stds)
			corr = np.reshape(corr,(6,6))
			cov  = np.dot(stds,corr.dot(stds))
			#-----------------------------------------

			astrometric_args = {
			"position+velocity":{
					"family":"Gaussian",
					"location":loc,
					"covariance":cov}
				}
			#==========================================================================

		return astrometric_args

	#====================== Generate Astrometric Data ==================================================
	def _generate_phase_space(self,n_stars):
		'''	The phase space coordinates are
			assumed to represent barycentric (i.e. centred on the Sun) positions and velocities.
		'''
		#=============== Joined =================================================================
		if "position+velocity" in self.astrometric_args:
			join_args = self.astrometric_args["position+velocity"]
			if join_args["family"] == "Gaussian":
				XYZUVW = st.multivariate_normal.rvs(
							mean=join_args["location"],
							cov=join_args["covariance"],
							size=n_stars)

			elif self.astrometric_args["position+velocity"]["family"] in ["GMM", "CGMM"]:
				assert np.sum(join_args["weights"]) == 1.0,"weights must be a simplex"
				n_cmp = len(join_args["weights"])
				n_stars_cmp = np.floor(join_args["weights"]*n_stars).astype('int')
				n_res = n_stars - np.sum(n_stars_cmp)
				residual = np.ones(n_cmp).astype('int')
				residual[n_res:] = 0
				n_stars_cmp += residual
				assert np.sum(n_stars_cmp) == n_stars, "Check division of sources in GMM!"

				l = []
				for n,loc,cov in zip(n_stars_cmp,join_args["location"],join_args["covariance"]):
					l.append(st.multivariate_normal.rvs(mean=loc,cov=cov,size=n))

				XYZUVW = np.concatenate(l,axis=0)

			else:
				sys.exit("Specified family currently not supported for 'position+velocity' type")


		#========================================================================================
		else:
			position_args = self.astrometric_args["position"]
			velocity_args = self.astrometric_args["velocity"]

			#======================= Verification ==============================================================
			msg_0 = "Error in position arguments: loc and scale must have same dimension."
			msg_1 = "Error in position arguments: family {0} is not implemented".format(position_args["family"]) 

			assert len(position_args["location"]) == len(position_args["covariance"]), msg_0
			assert position_args["family"] in ["Gaussian","EFF","King","GMM"], msg_1

			#=============================== Positions ========================================================
			if position_args["family"] == "Gaussian":
				XYZ = st.multivariate_normal.rvs(mean=position_args["location"],cov=position_args["covariance"],
													size=n_stars)

			elif position_args["family"] == "EFF":
				XYZ = mveff.rvs(loc=position_args["location"],scale=position_args["covariance"],
								gamma=position_args["gamma"],size=n_stars)

			elif position_args["family"] == "King":
				XYZ = mvking.rvs(loc=position_args["location"],scale=position_args["covariance"],
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
				for n,loc,scl in zip(n_stars_cmp,position_args["location"],position_args["covariance"]):
					l.append(st.multivariate_normal.rvs(mean=loc,cov=scl,size=n))

				XYZ = np.concatenate(l,axis=0)

			else:
				sys.exit("Error: incorrect position family argument")
			#===============================================================================================

			#=============================== Velocities ========================================================
			if velocity_args["family"] == "Gaussian":
				UVW = st.multivariate_normal.rvs(mean=velocity_args["location"],cov=velocity_args["covariance"],
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
				for n,loc,scl in zip(n_stars_cmp,velocity_args["location"],velocity_args["covariance"]):
					l.append(st.multivariate_normal.rvs(mean=loc,cov=scl,size=n))

				UVW = np.concatenate(l,axis=0)

			else:
				sys.exit("Error: incorrect velocity family argument")
			#===============================================================================================

			XYZUVW = np.hstack((XYZ,UVW))

		return XYZUVW

	def _generate_true_astrometry(self,X):
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
		df_as = pd.DataFrame(data=true,columns=self.labels_true_as)

		return df_as,r

	def _generate_true_photometry(self,masses,distances,mist_lower_mass=0.1):
		#-------- Split --------------------------------
		idx_mist  = np.where(masses >= mist_lower_mass)[0]
		idx_faint = np.where(masses <  mist_lower_mass)[0]
		n_faint   = len(idx_faint)
		#-----------------------------------------------

		#------- MIST photometry -----------------------------------------
		df_ph_mist = self.tracks.generate(masses[idx_mist], 
									self.photometric_args["log_age"], 
									self.photometric_args["metallicity"], 
									distance=distances[idx_mist], 
									AV=self.photometric_args["Av"])
		#-----------------------------------------------------------------

		#------- Faint photometry ---------------------------------------
		if n_faint > 0:
			print("Warning: objects with mass lower than MIST mass " +\
							"limit will have wrong photometry!!!!!")

			#--------- Minimal mass -----------------------
			idx_limit = np.argmin(df_ph_mist["mass"])
			df_limit  = df_ph_mist.iloc[[idx_limit]].copy()
			#---------------------------------------------

			#----- Maximum mist G band ---------
			mist_lower_mag = df_limit["G_mag"]
			#-----------------------------------

			#--------------Repeat n_faint times -----------------------
			df_ph_faint = pd.concat([df_limit for n in range(n_faint)],
									ignore_index=True)
			#---------------------------------------------------------

			#--------- Replace G values --------------------
			df_ph_faint.loc[:,"G_mag"] = np.random.uniform(
								low=mist_lower_mag,
								high=21.0,
								size=n_faint)
			#-----------------------------------------------

			#--------- Append -------------------------------------------
			df_ph = pd.concat([df_ph_mist,df_ph_faint],ignore_index=True)
			#------------------------------------------------------------

		else:
			df_ph = df_ph_mist

		#------- Assert valid magnitude and masses ------------------------------------------------
		bad =  (df_ph["G_mag"]  >21.0) | (df_ph["G_mag"]  <4.0) | np.isnan(df_ph["G_mag"])  | \
			   (df_ph["BP_mag"] >21.0) | (df_ph["BP_mag"] <4.0) | np.isnan(df_ph["BP_mag"]) | \
			   (df_ph["RP_mag"] >21.0) | (df_ph["RP_mag"] <4.0) | np.isnan(df_ph["RP_mag"]) 
		idx_bad = np.where(bad)[0]
		#---------------------------------------------------------------------------------------

		if len(idx_bad) > 0:
			msg_error = "ERROR: Modify the mass interval!\n" + \
			"Stars are being generated outside the PyGaia limits [4,21]\n" +\
			"or above MIST photometric limits!\n"
			print(msg_error)
			print("Bad photometric sources:")
			print(df_ph.iloc[idx_bad])
			sys.exit()
		#--------------------------------------------------------------------

		return df_ph
	#======================================================================================

	def _generate_observed_values(self,true,
				angular_correlations="Lindegren+2020"
				):

		N = len(true)

		rng = np.random.default_rng()

		#---- Astrometric and photometric values ------
		true_as = true.loc[:,self.labels_true_as]
		true_ph = true.loc[:,["G_mag","BP_mag","RP_mag"]]
		true_sp = true.loc[:,["G_mag","BP_mag","RP_mag","Teff","logg"]]
		index = true.index
		del true
		#------------------------------------------------

		#-------- G RVS -----------------------------------------------------------------------
		# https://dms.cosmos.esa.int/COSMOS/doc_fetch.php?id=2760608
		# file:///home/jolivares/Downloads/GAIA-C5-TN-UB-CJ-041.pdf
		# G − GRVS = 0.0386 + 0.9457*(BP-RP) - 0.1149*(BP-RP)**2 + 0.0022 (BP-RP)**3 + E(0.06)
		# GRVS = G - 0.0386 - 0.9457*(BP-RP) + 0.1149*(BP-RP)**2 - 0.0022 (BP-RP)**3
		true_sp["GRVS_mag"] = true_sp.apply(lambda x: 
			x["G_mag"] - 0.0386 - 0.9457*(x["BP_mag"]-x["RP_mag"]) + 
			0.1149*((x["BP_mag"]-x["RP_mag"])**2) - 0.0022+((x["BP_mag"]-x["RP_mag"])**3),
			axis=1)
		#--------------------------------------------------------------------------------------

		#================== Uncertainties ================================
		ra_unc,dec_unc = position_uncertainty(true_ph["G_mag"],
							release=self.release)

		plx_unc = parallax_uncertainty(true_ph["G_mag"],
							release=self.release)

		mua_unc,mud_unc = proper_motion_uncertainty(true_ph["G_mag"], 
							release=self.release)

		rvl_unc = radial_velocity_uncertainty(grvs=true_sp["GRVS_mag"],
							teff=true_sp["Teff"],
							logg=true_sp["logg"],
							release=self.release)

		g_unc   = magnitude_uncertainty(band="g",maglist=true_ph["G_mag"],
							release=self.release)

		bp_unc  = magnitude_uncertainty(band="bp",maglist=true_ph["BP_mag"],
							release=self.release)

		rp_unc  = magnitude_uncertainty(band="rp",maglist=true_ph["RP_mag"], 
							release=self.release)
		del true_sp
		#------------------------------------------------------------

		#--- Correct units ----
		# Micro to mili arcsec
		ra_unc  *= 1e-3
		dec_unc *= 1e-3
		plx_unc *= 1e-3
		mua_unc *= 1e-3
		mud_unc *= 1e-3
		# mmag to mag
		g_unc   *= 1e-3
		bp_unc  *= 1e-3
		rp_unc  *= 1e-3
		#----------------------

		#------ Stack values ------------------------------
		unc_as = np.column_stack((ra_unc,dec_unc,plx_unc,
								  mua_unc,mud_unc,rvl_unc))
		unc_ph = np.column_stack((g_unc,bp_unc,rp_unc))
		#---------------------------------------------------
		#==================================================================

		#================= Astrometry =================================
		#-------------- Missing rvel -----------------
		idx_nan_rvl = np.where(np.isnan(rvl_unc))[0]
		unc_as[idx_nan_rvl,5] = 99
		#--------------------------------------------

		#------ Fix units --------------------------
		# Uncertainty must be in same units as value
		unc_as[:,0] /= 3.6e6 # mas to degrees
		unc_as[:,1] /= 3.6e6 # mas to degrees
		#-------------------------------------------

		if angular_correlations is not None:
			print("Using {} angular correlation function".format(
				angular_correlations))
			#-------------- Allocate matrix --------------
			cov_as = np.zeros((6*N,6*N),dtype=np.float32)

			#------ Angular separations ------------------
			theta = AngularSeparation(
				true_as[self.labels_true_as[:2]].to_numpy())
			#---------------------------------------------

			#-- Parallax angular correlations --------
			corr_plx = covariance_parallax(theta,
							case=angular_correlations)

			#-- Proper motions angular correlations --
			corr_ppm = covariance_proper_motion(theta,
							case=angular_correlations)

			#----- Fill parallax part ---------------
			idx = np.arange(2,6*N,step=6)
			cov_as[np.ix_(idx,idx)] = corr_plx
			#----------------------------------------

			#----- Fill pmra part ---------------
			idx = np.arange(3,6*N,step=6)
			cov_as[np.ix_(idx,idx)] = corr_ppm
			#----------------------------------------

			#----- Fill pmdec part ---------------
			idx = np.arange(4,6*N,step=6)
			cov_as[np.ix_(idx,idx)] = corr_ppm
			#---------------------------------------

			del theta,corr_plx,corr_ppm

			# Add variances to angular correlations
			cov_as += np.diag(unc_as.flatten()**2)
		
			#------- Observed values ---------------------
			obs_as = rng.multivariate_normal(
						mean=true_as.to_numpy().flatten(),
						cov=cov_as,
						tol=1e-8,
						method='cholesky',
						size=1).reshape((N,6))
			#---------------------------------------------

		else:
			#----- Loop over stars -------------------------
			obs_as = np.empty((N,6))
			for i,mu in true_as.iterrows():
				obs_as[i] = rng.multivariate_normal(
						mean=mu,cov=np.diag(unc_as[i]**2),
						tol=1e-8,method="cholesky",size=1)
			#-----------------------------------------------
		#=====================================================================

		#-- Replace nan rvel by nan -------
		obs_as[idx_nan_rvl,5] = np.nan
		unc_as[idx_nan_rvl,5] = np.nan
		#--------------------------------

		#------ Fix units --------------------------
		unc_as[:,0] *= 3.6e6 #degrees back to mas
		unc_as[:,1] *= 3.6e6 #degrees back to mas
		#-------------------------------------------

		#--------- Data Frames ------------------
		df_obs_as = pd.DataFrame(data=obs_as,
					columns=self.labels_obs_as)
		df_unc_as = pd.DataFrame(data=unc_as,
					columns=self.labels_unc_as)

		df_as = df_obs_as.join(df_unc_as)
		#-----------------------------------------

		#-------- Correlations ------------------
		df_cor_as = pd.DataFrame(data=np.zeros(
					(N,len(self.labels_cor_as))),
					columns=self.labels_cor_as)
		df_as = df_as.join(df_cor_as)
		#----------------------------------------
		del df_obs_as,df_unc_as,df_cor_as,obs_as,unc_as
		#===========================================================

		#======= Photometry ===================================
		#----- Loop over stars -------------------------
		obs_ph = np.empty((N,3))
		for i,mu in true_ph.iterrows():
			obs_ph[i] = rng.multivariate_normal(
					mean=mu,cov=np.diag(unc_ph[i]**2),
					tol=1e-8,method="cholesky",size=1)
		#-----------------------------------------------

		#--------- Data Frames ------------------
		df_obs_ph = pd.DataFrame(data=obs_ph,
					columns=self.labels_obs_ph)
		df_unc_ph = pd.DataFrame(data=unc_ph,
					columns=self.labels_unc_ph)

		df_ph = df_obs_ph.join(df_unc_ph)
		#-----------------------------------------
		del df_obs_ph,df_unc_ph,obs_ph,unc_ph
		#===========================================

		#------- Join ------------
		df_obs = df_as.join(df_ph)
		del df_as,df_ph

		#------- Set index -------
		df_obs.set_index(index,inplace=True)

		return df_obs

	def read_mcluster(self,file):
		''' Reads mcluster code input files '''
		mc = pd.read_csv(file,sep="\t",header=None,skiprows=1,
				names=sum([["Mass"],self.labels_phase_space],[]))
		masses = mc["Mass"].to_numpy()
		X = mc[self.labels_phase_space].to_numpy()

		return masses,X

	#================= Generate cluster ==========================
	def generate_cluster(self,file,n_stars=100,
						angular_correlations="Lindegren+2020",
						index_label="source_id"):

		if hasattr(self,"mcluster_args"):
			#--------------- Generate McLuster file ------------
			path = self.mcluster_args["path"]
			base_cmd = "{0}mcluster -f0 -s {1} -N {2} "
			base_cmd = base_cmd.format(path,self.seed,n_stars)

			if self.mcluster_args["family"] == "EFF":
				rc = self.mcluster_args["core_radius"]
				rt = self.mcluster_args["truncation_radius"]
				g  = self.mcluster_args["gamma"]
				name = "EFF_n{0}_r{1}_g{2}_c{3}"
				args = "-P 3 -u 1 -r {0} -g {1} -c {2} -o {3}"
				name = name.format(n_stars,rc,g,rt)
				args = args.format(rc,g,rt,name)
				cmd  = base_cmd + args

			elif self.mcluster_args["family"] == "King":
				rc = self.mcluster_args["core_radius"]
				rt = self.mcluster_args["tidal_radius"]
				w  = self.mcluster_args["W0"]
				name = "King_n{0}_r{1}_w{3}"
				args = "-P 1 -u 1 -r {0} -W {1} -c {2} -o {3}"
				name = name.format(n_stars,rc,w)
				args = args.format(rc,w,name)
				cmd  = base_cmd + args
			
			mcluster_file = "{0}/{1}.txt".format(os.getcwd(),name)
			print("Running McLuster command:")
			print(cmd)
			call(cmd, shell=True)
			#---------------------------------------------------

			#--------------- Read mcluster file ----------------
			print("Reading McLuster phase-space values ...")
			_,X = self.read_mcluster(file=mcluster_file)
			#--------------------------------------------------

			#-------- Shift positions and velocities ------------------
			X += np.array(mcluster_args["position+velocity"])
			#----------------------------------------------------------
			
		elif hasattr(self,"astrometric_args"):

			#---------- Phase space coordinates --------------
			print("Generating phase-space values ...")
			X = self._generate_phase_space(n_stars=n_stars)
			#------------------------------------------------

		#------------ Masses ------------------------------
		if self.photometric_args["mass_prior"] == "Chabrier":
			masses  = ChabrierPrior(
					bounds=self.photometric_args["mass_limits"]
						).sample(n_stars)
		elif self.photometric_args["mass_prior"] == "Uniform":
			masses = np.random.uniform(
				low=self.photometric_args["mass_limits"][0],
				high=self.photometric_args["mass_limits"][1],
				size=n_stars)
		else:
			sys.exit("Photometric prior not implemented")
		#--------------------------------------------------

		#---------- Phase-space ------------------------------------
		df_ps = pd.DataFrame(data=X,columns=self.labels_phase_space)

		#--------- True astrometry -----------------------
		print("Generating true astrometry ...")
		df_as,distances = self._generate_true_astrometry(X)
		#--------------------------------------------------

		#--------- True photometry -------------------------------
		print("Generating true photometry ...")
		df_ph = self._generate_true_photometry(masses,distances)
		#---------------------------------------------------------

		#---- Join true values ----------------
		df_true = df_as.join(df_ph,how="inner")
		#--------------------------------------

		#------------- Observed values ----------------------------
		print("Generating observed values ...")
		df_obs = self._generate_observed_values(df_true,
						angular_correlations=angular_correlations)
		#----------------------------------------------------------

		#-------- Join data frames -----------------------------
		self.df = df_obs.join(df_true).join(df_ps)

		#-------- Reset index -----------------------
		self.df.reset_index(drop=True,inplace=True)

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
		plt.hist(self.df[self.labels_rvl[0]],density=False,
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

		#------------- Mass -------------
		plt.figure(figsize=figsize)
		plt.hist(self.df["mass"],bins=50)
		plt.xlabel("Mass [$M_{\\odot}$]")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------

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

		#---------- Uncertainties --------------------------------
		features = sum([self.labels_unc_as,self.labels_unc_ph],[])
		features.remove("dec_error")
		labels_features = [" [mas]"," [mas]",
							" [mas/yr]"," [mas/yr]",
							" [km/s]"," [mag]",
							" [mag]"," [mag]"]
		labels = [f+u for f,u in zip(features,labels_features)]

		fig, axs = plt.subplots(nrows=4, ncols=2,figsize=figsize,
					gridspec_kw={"hspace":0.3})
		for ax,feature,label in zip(axs.flatten(),features,labels):
			ax.hist(self.df[feature],density=True,
					bins=n_bins,log=True,histtype="step",
					color=cases["observed"]["color"])
			ax.set_ylabel("Density",labelpad=0)
			ax.set_xlabel(label,labelpad=0)
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#----------------------------------------------------------

		pdf.close()
	#------------------------------------------------------------------------------

if __name__ == "__main__":

	seed      = 9
	n_stars   = int(1e3)
	dir_main  = "/home/jolivares/OCs/TWH/Mecayotl/runs/iter_0/Kalkayotl/Gaussian_central/"
	base_name = "TWH_n{0}".format(n_stars)
	file_plot = dir_main + base_name + ".pdf"
	file_data = dir_main + base_name + ".csv"
	
	astrometric_args = {
		"position":{"family":"Gaussian",
					"location":np.array([0.0,100.0,0.0]),
					"covariance":np.diag([2.,2.,2.])},
		"velocity":{"family":"Gaussian",
					"location":np.array([0.0,0.0,0.0]),
					"covariance":np.diag([1.,1.,1.])}}

	photometric_args = {
	"log_age": 7.0,    
	"metallicity":0.012,
	"Av": 0.0,         
	"mass_limits":[0.01,2.5], 
	"bands":["G","BP","RP"],
	"mass_prior":"Uniform"
	}

	mcluster_args = {
		"path":"/home/jolivares/Repos/mcluster/",
		"family":"EFF","gamma":5.0,"core_radius":1.0,
		"position+velocity":[0.0,100.0,0.0,0.0,0.0,0.0],
		"truncation_radius":20.0}

	kalkayotl_file = dir_main + "Cluster_statistics.csv"

	ama = Amasijo(
				# astrometric_args=astrometric_args,
				# mcluster_args=mcluster_args,
				kalkayotl_file=kalkayotl_file,
				photometric_args=photometric_args,
				seed=seed)

	ama.generate_cluster(file_data,n_stars=n_stars)

	ama.plot_cluster(file_plot=file_plot)











