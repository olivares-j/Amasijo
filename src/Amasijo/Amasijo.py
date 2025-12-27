import sys
import os
import numpy  as np
import pandas as pd
import string
import scipy.stats as st
from time import time
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import splrep,splev
from scipy.spatial import distance

from .functions import AngularSeparation,covariance_parallax,covariance_proper_motion

from pygaia.errors.astrometric import parallax_uncertainty,position_uncertainty,proper_motion_uncertainty
from pygaia.errors.photometric import magnitude_uncertainty
from pygaia.errors.spectroscopic import radial_velocity_uncertainty

from pygaia.astrometry.vectorastrometry import astrometry_to_phase_space,phase_space_to_astrometry
from pygaia.astrometry.coordinates import CoordinateTransformation
from pygaia.astrometry.coordinates import Transformations 

###############################################################################
class Amasijo(object):
	"""This class intends to construct synthetic clusters with simple 
		phase-space distributions and photometry from stellar models"""
	def __init__(self,isochrones_args,
					phasespace_args=None,
					mcluster_args=None,
					kalkayotl_args={
						"file":None,
						"statistic":"mean"},
					photometry={
						"labels":{
							"phot_g_mean_mag":"phot_g_mean_mag",
							"phot_bp_mean_mag":"phot_bp_mean_mag",
							"phot_rp_mean_mag":"phot_rp_mean_mag"},
						"family":"Gaia"},
					radial_velocity={
						"labels":{"radial_velocity":"radial_velocity"},
						"family":"Gaia"},
					additional_columns={"ruwe":1.0},
					reference_system="Galactic",
					release="dr3",
					seed=1234):

		assert radial_velocity["family"] in ["Gaia","Uniform"],"Error: only Gaia or Uniform in radial_velocity family!"

		#------ Set Seed -----------------------------------
		np.random.seed(seed=seed)
		self.random_state = np.random.RandomState(seed=seed)
		self.seed = seed
		self.radial_velocity = radial_velocity
		self.release = release
		self.reference_system = reference_system
		#-----------------------------------------------------

		#------------------- Validate input arguments -------------------------------------------------
		assert reference_system in ["Galactic","ICRS"],\
		"ERROR:reference_system must be Galactic or ICRS"
		assert set(["G","BP","RP"]).issubset(set(isochrones_args["bands"])),\
		"The three Gaia bands (G,BP,RP, in capital letters) must be present in isochrones_args['bands']!"

		if isochrones_args["model"] == "MIST":
			assert np.all(isochrones_args["mass_limits"][0]>= 0.1),\
			"Error: The lower mass allowed by the MIST model is 0.1. Adjust mass_limits!"
		elif isochrones_args["model"] == "PARSEC":
			assert np.all(isochrones_args["mass_limits"][0]>= 0.1),\
			"Error: The lower mass allowed by the PARSEC model is 0.01. Adjust mass_limits!"
		else:
			sys.exit("ERROR: Amasijo currently supports only MIST or PARSEC models!")
		#---------------------------------------------------------------------------------------------


		#---------------- Mapper -----------------------------------
		self.mapper = photometry["labels"]
		for key,value in radial_velocity["labels"].items():
			self.mapper[key] = value

		for key,value in self.mapper.copy().items():
			self.mapper[key+"_error"] = "{0}_error".format(value)
		#-----------------------------------------------------------

		#---------- Arguments ---------------------------------------------------------------
		self.isochrones_args = isochrones_args

		if phasespace_args is not None:
			self.phasespace_args = phasespace_args
			case = "phasespace"
		elif mcluster_args is not None:
			self.mcluster_args = mcluster_args
			case = "McLuster"
		elif kalkayotl_args is not None:
			self.phasespace_args = self._read_kalkayotl(kalkayotl_args)
			case = "Kalkayotl"
		else:
			sys.exit("You must provide phasespace_args, mcluster_args or a Kalkayotl_args!")

		print("Astrometry will be generated from the provided {0} arguments!".format(case))
		#-----------------------------------------------------------------------------------
		

		#------- Labels ----------------------------------------------------------------------------------
		self.labels_phase_space = ["X","Y","Z","U","V","W"]
		self.labels_true_as = ["ra_true","dec_true","parallax_true",
								"pmra_true","pmdec_true","radial_velocity_true"]
		self.labels_true_bands = [band+"_mag" for band in isochrones_args["bands"]]
		self.labels_cor_as  = ["ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
							   "dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
							   "parallax_pmra_corr","parallax_pmdec_corr",
							   "pmra_pmdec_corr"]
		self.labels_obs_ph  = ["phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag"]
		self.labels_unc_ph  = ["phot_g_mean_mag_error","phot_bp_mean_mag_error","phot_rp_mean_mag_error"]
		self.labels_obs_rv  = ["radial_velocity"]
		self.labels_unc_rv  = ["radial_velocity_error"]
		self.labels_obs_as  = ["ra","dec","parallax","pmra","pmdec","radial_velocity"]
		self.labels_unc_as  = ["ra_error","dec_error","parallax_error",
								"pmra_error","pmdec_error","radial_velocity_error"]
		self.labels_obs_bands = [band for band in isochrones_args["bands"]]
		self.labels_unc_bands = ["e_"+band for band in isochrones_args["bands"]]
		self.additional_columns = additional_columns
		#-----------------------------------------------------------------------------------------------------

	def _read_kalkayotl(self,args):
		assert os.path.exists(args["file"]), "Input Kalkayotl file does not exists!\n{0}".format(args["file"])
		statistic = args["statistic"]
		do_replace = True if "replace" in args else False

		#-------- Read file ---------------------------------------------
		param = pd.read_csv(args["file"],usecols=["Parameter",statistic])
		param.set_index("Parameter",inplace=True)
		if do_replace:
			for key,value in args["replace"].items():
				if key in param.index:
					old = param.loc[key,statistic]
				else:
					old = np.nan
				print("Parameter {0}: {1:2.1f} -> {2:2.3f}".format(key,old,value))
				param.loc[key,statistic] = value
		#-----------------------------------------------------------------

		has_UVW = any(param.index.str.contains("U")) &\
				  any(param.index.str.contains("V")) &\
				  any(param.index.str.contains("W"))
		assert has_UVW, "ERROR: The Kalkayotl file does not contains velocities UVW"

		is_mixture = any(param.index.str.contains("weights"))

		is_linear = any(param.index.str.contains("kappa"))

		if is_mixture:
			print("A mixture model was identified")
			#===================== GMM and CGMM =======================================

			#--------- Weights -----------------------------------------------------------
			wghs = param.loc[param.index.str.contains("weights"),statistic].values
			#-----------------------------------------------------------------------------

			n_components = len(wghs)
			names_components = list(string.ascii_uppercase)[:n_components]

			#------------- Location ----------------------------------
			loc = param[param.index.str.contains("loc")]


			if loc.shape[0]/n_components == 6:
				#------------- GMM ------------------------------------
				family = "GMM"
				locs = []
				for c in names_components:
					selection = loc.index.str.contains(
								"{0}".format(c),regex=False)
					locs.append(loc.loc[selection,statistic].values)
				#---------------------------------------------------------
			else:
				#-------------- CGMM -------------------------------------------------
				family = "CGMM"
				loc = param.loc[param.index.str.contains("loc"),statistic].values
				locs = [loc for w in wghs]
				#---------------------------------------------------------------------

			print("Family type: {0}".format(family))

			#------------- Covariances -----------------------
			scl = param.fillna(value=1.0)

			stds = []
			cors = []
			covs = []
			for c in names_components:
				#---------- Select component parameters --------
				mask_std = scl.index.str.contains(
							"std[{0}".format(c),regex=False)
				mask_cor = scl.index.str.contains(
							"corr[{0}".format(c),regex=False)
				#-----------------------------------------------

				#------Extract parameters -------------------
				std = scl.loc[mask_std,statistic].values
				cor = scl.loc[mask_cor,statistic].values
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

			phasespace_args = {
			"position+velocity":{
					"family":family,
					"weights":wghs,
					"location":locs,
					"covariance":covs}
				}
			#========================================================================

		#=================== Student T =================================================
		elif any(param.index.str.contains("nu")):
			print("A StudentT model was identified")
			
			#---- Extract parameters ------------------------------------------------
			loc  = param.loc[param.index.str.contains("loc"),statistic].values
			std  = param.loc[param.index.str.contains('std'),statistic].values
			nu   = param.loc[param.index.str.contains('nu'),statistic].values
			#-------------------------------------------------------------------------

			param.fillna(value=1.0,inplace=True)

			if is_linear:
				kappa = np.array([
					param.loc[param.index=="6D::kappa[X]",statistic].values,
					param.loc[param.index=="6D::kappa[Y]",statistic].values,
					param.loc[param.index=="6D::kappa[Z]",statistic].values
					]).flatten()

				if param.index.str.contains("omega"):
					omega = param.loc[param.index.str.contains("omega"),statistic].values
				else:
					omega = np.zeros((2,3))

				#---------------------- Extract -------------------------------------------
				corr_pos = param.loc[param.index.str.contains('corr_pos'),statistic].values
				corr_vel = param.loc[param.index.str.contains('corr_vel'),statistic].values
				#------------------------------------------------------------------------

				#---- Construct covariance ---------
				std_pos = np.diag(std[:3])
				std_vel = np.diag(std[3:])
				corr_pos = np.reshape(corr_pos,(3,3))
				corr_vel = np.reshape(corr_vel,(3,3))
				cov_pos  = np.dot(std_pos,corr_pos.dot(std_pos))
				cov_vel  = np.dot(std_vel,corr_vel.dot(std_vel))
				#----------------------------------

				phasespace_args = {
				"position":{
						"family":"StudentT",
						"nu":nu,
						"location":loc[:3],
						"covariance":cov_pos},
				"velocity":{
						"family":"StudentT",
						"nu":nu,
						"location":loc[3:],
						"covariance":cov_vel,
						"kappa":kappa,
						"omega":omega}
						}
			else:
				#---------------------- Extract -------------------------------------------
				std = param.loc[param.index.str.contains('std'),statistic].values
				corr = param.loc[param.index.str.contains('corr'),statistic].values
				#------------------------------------------------------------------------

				#---- Construct covariance ---------
				std  = np.diag(std)
				corr = np.reshape(corr,(6,6))
				cov  = np.dot(std,corr.dot(std))
				#----------------------------------

				phasespace_args = {
				"position+velocity":{
						"family":"StudentT",
						"nu":nu,
						"location":loc,
						"covariance":cov}
					}
		#================================================================================

		else:
		#=================== Gaussian ====================================================
			print("A Gaussian model was identified")
			#---- Extract parameters ------------------------------------------------
			loc  = param.loc[param.index.str.contains("loc"),statistic].values
			std  = param.loc[param.index.str.contains('std'),statistic].values
			#------------------------------------------------------------------------

			param.fillna(value=1.0,inplace=True)

			if is_linear:
				kappa = np.array([
					param.loc[param.index=="6D::kappa[X]",statistic].values,
					param.loc[param.index=="6D::kappa[Y]",statistic].values,
					param.loc[param.index=="6D::kappa[Z]",statistic].values
					]).flatten()

				if any(param.index.str.contains("omega")):
					omega = param.loc[param.index.str.contains("omega"),statistic].values
					omega = omega.reshape((2,3))
				else:
					omega = np.zeros((2,3))

				#---------------------- Extract correlations -------------------------------------
				corr_pos = param.loc[param.index.str.contains('corr_pos'),statistic].values
				corr_vel = param.loc[param.index.str.contains('corr_vel'),statistic].values
				#---------------------------------------------------------------------------------

				#---- Construct covariance ---------
				std_pos = np.diag(std[:3])
				std_vel = np.diag(std[3:])
				corr_pos = np.reshape(corr_pos,(3,3))
				corr_vel = np.reshape(corr_vel,(3,3))
				cov_pos  = np.dot(std_pos,corr_pos.dot(std_pos))
				cov_vel  = np.dot(std_vel,corr_vel.dot(std_vel))
				#----------------------------------

				phasespace_args = {
				"position":{
						"family":"Gaussian",
						"location":loc[:3],
						"covariance":cov_pos},
				"velocity":{
						"family":"Gaussian",
						"location":loc[3:],
						"covariance":cov_vel,
						"kappa":kappa,
						"omega":omega}
						}
			else:
				#---------------------- Extract -------------------------------------------
				corr = param.loc[param.index.str.contains('corr'),statistic].values
				#------------------------------------------------------------------------

				#---- Construct covariance ---------
				std  = np.diag(std)
				corr = np.reshape(corr,(6,6))
				cov  = np.dot(std,corr.dot(std))
				#----------------------------------

				phasespace_args = {
				"position+velocity":{
						"family":"Gaussian",
						"location":loc,
						"covariance":cov}
					}
			#==========================================================================
		return phasespace_args

	#====================== Generate Astrometric Data ==================================================
	def _generate_phase_space(self,n_stars,max_mahalanobis_distance=np.inf,max_n=2):
		'''	The phase space coordinates are
			assumed to represent barycentric (i.e. centred on the Sun) positions and velocities.
		'''
		if max_mahalanobis_distance != np.inf :
			msg_maha = "Error: max_mahalanobis_distance argument only valid for Gaussian iid positions and velocities"
			if "position" in self.phasespace_args:
				assert (self.phasespace_args["position"]["family"] == "Gaussian") and \
						(self.phasespace_args["velocity"]["family"] == "Gaussian"), msg_maha
			elif "position+velocity" in self.phasespace_args:
				assert self.phasespace_args["position+velocity"]["family"] == "Gaussian",msg_maha
			else:
				sys.exit(msg_maha)
			

		#=============== Joined =================================================================
		if "position+velocity" in self.phasespace_args:
			join_args = self.phasespace_args["position+velocity"]
			if join_args["family"] == "Gaussian":
				XYZUVW = st.multivariate_normal.rvs(
							mean=join_args["location"],
							cov=join_args["covariance"],
							size=n_stars,
							random_state=self.seed)

			elif self.phasespace_args["position+velocity"]["family"] in ["GMM", "CGMM"]:
				np.testing.assert_almost_equal(np.sum(join_args["weights"]),1.0,
						err_msg="ERROR: sum of weights must be 1.0. It is ",verbose=True)
				n_cmp = len(join_args["weights"])
				n_stars_cmp = np.floor(join_args["weights"]*n_stars).astype('int')
				n_res = n_stars - np.sum(n_stars_cmp)
				residual = np.ones(n_cmp).astype('int')
				residual[n_res:] = 0
				n_stars_cmp += residual
				assert np.sum(n_stars_cmp) == n_stars, "Check division of sources in GMM!"

				l = []
				for n,loc,cov in zip(n_stars_cmp,join_args["location"],join_args["covariance"]):
					l.append(st.multivariate_normal.rvs(mean=loc,cov=cov,size=n,random_state=self.seed))

				XYZUVW = np.concatenate(l,axis=0)

			else:
				sys.exit("Specified family currently not supported for 'position+velocity' type")


		#========================================================================================
		else:
			position_args = self.phasespace_args["position"]
			velocity_args = self.phasespace_args["velocity"]

			#======================= Verification ==============================================================
			msg_0 = "Error in position arguments: loc and scale must have same dimension."
			msg_1 = "Error in position arguments: family {0} is not implemented".format(position_args["family"]) 

			assert len(position_args["location"]) == len(position_args["covariance"]), msg_0
			assert position_args["family"] in ["Gaussian","StudentT","GMM"], msg_1
			#===================================================================================================

			#=============================== Positions ========================================================
			if position_args["family"] == "Gaussian":
				max_n_stars = int(max_n*n_stars)
				xyz = st.multivariate_normal.rvs(
											mean=position_args["location"],
											cov=position_args["covariance"],
											size=max_n_stars,
											random_state=self.seed)
				if max_mahalanobis_distance == np.inf:
					idx = np.arange(n_stars)
				else:
					mhl_pos = np.zeros(max_n_stars)
					loc_pos = position_args["location"]
					inv_pos = np.linalg.inv(position_args["covariance"])
					for i in range(max_n_stars):
						mhl_pos[i] = distance.mahalanobis(xyz[i],loc_pos, inv_pos)

					idx = np.where(mhl_pos <= max_mahalanobis_distance)[0]

					if len(idx) >= n_stars:
						idx = np.random.choice(idx,size=n_stars,replace=False)
					else:
						sys.exit("Error: The number of available stars is smaller than the requested\n"+
							"Try increasing the 'max_n' factor")

					print("Maximum Mahalanobis distance in position: {0:2.1f}".format(mhl_pos[idx].max()))

				XYZ = xyz[idx]

			elif position_args["family"] == "StudentT":
				XYZ = st.multivariate_t.rvs(
											loc=position_args["location"],
											shape=position_args["covariance"],
											df=position_args["nu"],
											size=n_stars)

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

			#=============================== Velocities ========================================
			if velocity_args["family"] == "Gaussian":
				uvw = st.multivariate_normal.rvs(
											mean=velocity_args["location"],
											cov=velocity_args["covariance"],
											size=max_n_stars,
											random_state=self.seed)
				if max_mahalanobis_distance == np.inf:
					idx = np.arange(n_stars)
				else:
					mhl_vel = np.zeros(max_n_stars)
					loc_vel = velocity_args["location"]
					inv_vel = np.linalg.inv(velocity_args["covariance"])
					for i in range(max_n_stars):
						mhl_vel[i] = distance.mahalanobis(uvw[i],loc_vel, inv_vel)

					idx = np.where(mhl_vel <= max_mahalanobis_distance)[0]

					if len(idx) >= n_stars:
						idx = np.random.choice(idx,size=n_stars,replace=False)
					else:
						sys.exit("Error: The number of available stars is smaller than the requested\n"+
							"Try increasing the 'max_n' factor")

					print("Maximum Mahalanobis distance in velocity: {0:2.1f}".format(mhl_vel[idx].max()))
				
				UVW = uvw[idx]

			elif velocity_args["family"] == "StudentT":
				UVW = st.multivariate_t.rvs(
									loc=velocity_args["location"],
									shape=velocity_args["covariance"],
									df=velocity_args["nu"],
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


			if "kappa" in velocity_args:
				lnv = np.zeros((3,3))

				lnv[np.diag_indices(3)]    = velocity_args["kappa"]
				lnv[np.triu_indices(3,1)]  = velocity_args["omega"][0]
				lnv[np.tril_indices(3,-1)] = velocity_args["omega"][1]

				offset_pos = XYZ - position_args["location"]

				offset_vel = np.dot(lnv,offset_pos.T).T

				print("Mean velocity offset: {0:2.3f}".format(np.mean(offset_vel)))

				UVW += offset_vel
			#===============================================================================================

			XYZUVW = np.hstack((XYZ,UVW))

		return XYZUVW

	def _generate_true_astrometry(self,X):

		#--------------  Galactic to ICRS --------------------------------
		if self.reference_system == "Galactic":
			GAL2ICRS = CoordinateTransformation(Transformations.GAL2ICRS)

			x,y,z = GAL2ICRS.transform_cartesian_coordinates(
							X[:,0],X[:,1],X[:,2])
			u,v,w = GAL2ICRS.transform_cartesian_coordinates(
							X[:,3],X[:,4],X[:,5])
		else:
			x,y,z,u,v,w = X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5]
		
		#-----------------------------------------------------------------
		
		#------- Astrometry & Radial velocity ---------------------------
		ra,dec,plx,mua,mud,rvel = phase_space_to_astrometry(x,y,z,u,v,w)
		#----------------------------------------------------------------

		#----- Transform ----------------------
		ra   = np.rad2deg(ra)       # In degrees
		dec  = np.rad2deg(dec)      # In degrees
		r    = 1000.0/plx           # Distance
		true = np.column_stack((ra,dec,plx,mua,mud,rvel))

		#-------- Data Frames ---------------------------
		df_as = pd.DataFrame(data=true,columns=self.labels_true_as)

		return df_as,r

	def _generate_true_photometry(self,distances):
		n_stars = len(distances)

		if self.isochrones_args["model"] == "MIST":

			from isochrones import get_ichrone

			#--------------- Tracks ----------------------------
			tracks = get_ichrone('mist', tracks=True,
							bands=isochrones_args["bands"])
			#---------------------------------------------------

			#------------ Masses ------------------------------
			masses = np.random.uniform(
					low=self.isochrones_args["mass_limits"][0],
					high=self.isochrones_args["mass_limits"][1],
					size=n_stars)
			#--------------------------------------------------

			#------- Photometry -----------------------------------------
			df_ph = tracks.generate(
					mass=masses, 
					age=np.log10(self.isochrones_args["age"]*1.e6), 
					feh=self.isochrones_args["MIST_args"]["metallicity"], 
					distance=distances, 
					AV=self.isochrones_args["MIST_args"]["Av"],
					return_df=True)
			#-----------------------------------------------------------------
		elif self.isochrones_args["model"] == "PARSEC":
			from PARSEC import MLP
			mlp = MLP(
				file_mlp=self.isochrones_args["PARSEC_args"]["file_mlp"])

			assert (self.isochrones_args["age"] >= float(mlp.age_domain[0])) &\
				(self.isochrones_args["age"] <= float(mlp.age_domain[1])),"ERROR:"+\
				"Input age outside the PARSEC mlp domain"

			#---------- Selected bands --------------------------------------------------------------
			parsec_bands = np.array([band.replace("G_","").replace("mag","") for band in mlp.bands])
			requested_bands = np.array(self.isochrones_args["bands"])

			cnd = sum(np.isin(requested_bands,parsec_bands)) == requested_bands.shape[0]
			msg = "Error: requested bands not present in PARSEC bands:\n"+\
			"{0}".format(parsec_bands)
			assert cnd,msg
			idx_bands = np.where(np.isin(requested_bands,parsec_bands))[0]
			#-------------------------------------------------------------------------------------------

			#------------ Theta ------------------------------
			theta = np.random.uniform(
				low=mlp.theta_domain[0],
				high=mlp.theta_domain[1],
				size=n_stars)
			#--------------------------------------------------

			#------ Mass and absolute photometry ----------
			mass,absolute_photometry = mlp(
				age=self.isochrones_args["age"],
				theta=theta,
				n_stars=n_stars)
			#----------------------------------------------

			#---------- Apparent photometry -----------
			apparent_photometry = absolute_photometry.eval() \
					+ 5.0*np.log10(distances)[:,np.newaxis] - 5.0
			#-----------------------------------------

			#--------- Selected bands -----------------------------
			photometry = apparent_photometry[:,idx_bands]
			#------------------------------------------------------

			#------------- Data frame ----------------------
			df_ph = pd.DataFrame(
				data=photometry,
				columns=[band+"_mag" for band in requested_bands])
			df_ph["mass"] = mass.eval()
			df_ph["Teff"] = 5500.
			df_ph["logg"] = 2.7
			#----------------------------------------------

		else:
			sys.exit("ERROR: model currently not supported")



		assert np.all(np.isfinite(df_ph["G_mag"])),"ERROR: There are missing values in the photometry!"

		#------- Assert valid magnitudes------------------------------------------------
		bad =  (df_ph["G_mag"] > 21.0) | (df_ph["G_mag"] < 4.0)
		#---------------------------------------------------------------------------------------

		if sum(bad) > 0:
			print("WARNING: The following sources were generated outside the PyGaia limits [4,21]:\n")
			print(df_ph.loc[bad,["G_mag","mass"]])
		#--------------------------------------------------------------------

		return df_ph
	#======================================================================================

	def _generate_observed_values(self,true,
				angular_correlations="Lindegren+2020",
				soil_mag_uncertainty=1.0,
				g_mag_shift_for_uncertainty=None,
				impute_radial_velocity=False,
				fraction_radial_velocities_observed=None,
				):
		frac_rvs_obs = fraction_radial_velocities_observed

		g_mag_shift_for_uncertainty = g_mag_shift_for_uncertainty if \
		g_mag_shift_for_uncertainty is not None else {"astrometry":0.0,"spectroscopy":0.0}

		assert frac_rvs_obs is None or isinstance(frac_rvs_obs,float),\
		"Error: fraction_radial_velocities_observed must be float or None"

		N = len(true)

		rng = np.random.default_rng()

		#---- Astrometric and photometric values ------
		true_as = true.loc[:,self.labels_true_as].copy()
		true_ph = true.loc[:,["G_mag","BP_mag","RP_mag"]].copy()
		true_sp = true.loc[:,["G_mag","BP_mag","RP_mag","Teff","logg"]].copy()
		true_bands = true.loc[:,self.labels_true_bands].copy()
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
		#------- Apply photometric shift to compute astrometric uncertainties------
		shifted_G_mag_as = true_ph["G_mag"]    + g_mag_shift_for_uncertainty["astrometry"]
		shifted_G_mag_sp = true_sp["GRVS_mag"] + g_mag_shift_for_uncertainty["spectroscopy"]
		#---------------------------------------------------------------------------
		

		ra_unc,dec_unc = position_uncertainty(shifted_G_mag_as,
							release=self.release) # In micro-arcseconds

		plx_unc = parallax_uncertainty(shifted_G_mag_as,
							release=self.release) # In micro-arcseconds

		mua_unc,mud_unc = proper_motion_uncertainty(shifted_G_mag_as, 
							release=self.release) # In micro-arcseconds per year

		if self.radial_velocity["family"] == "Gaia":
			rvl_unc = radial_velocity_uncertainty(grvs=shifted_G_mag_sp,
								teff=true_sp["Teff"],
								logg=true_sp["logg"],
								release=self.release) # km per second
		elif self.radial_velocity["family"] == "Uniform":
			print("rv_uncertainty ~ Uniform(limits) [km.s-1]")
			rvl_unc = np.random.uniform(
				low=self.radial_velocity["limits"][0],
				high=self.radial_velocity["limits"][1],
				size=len(shifted_G_mag_sp))
		else:
			sys.exit("Error: The radial velocity family '{0}' is not implemented".format(
				self.radial_velocity["family"]))
		
		g_unc  = np.full(N,fill_value=soil_mag_uncertainty)
		bp_unc = np.full(N,fill_value=soil_mag_uncertainty)
		rp_unc = np.full(N,fill_value=soil_mag_uncertainty)

		idx_g = np.where(
			(true_ph["G_mag"] > 4.0) & 
			(true_ph["G_mag"] < 21.0))[0]

		idx_bp = np.where(
			(true_ph["BP_mag"] > 4.0) & 
			(true_ph["BP_mag"] < 21.0))[0]

		idx_rp = np.where(
			(true_ph["RP_mag"] > 4.0) & 
			(true_ph["RP_mag"] < 21.0))[0]

		g_unc[idx_g]   = magnitude_uncertainty(band="g",
								maglist=true_ph["G_mag"][idx_g],
								release=self.release) # In mmag

		bp_unc[idx_bp]  = magnitude_uncertainty(band="bp",
							maglist=true_ph["BP_mag"][idx_bp],
							release=self.release) # In mmag

		rp_unc[idx_rp]  = magnitude_uncertainty(band="rp",
							maglist=true_ph["RP_mag"][idx_rp], 
							release=self.release) # In mmag
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
		#-------------- Missing rvel ----------------------------------------------------
		idx_nan_rvl = np.where(np.isnan(rvl_unc))[0]
		idx_obs_rvl = np.where(np.isfinite(rvl_unc))[0]

		if frac_rvs_obs is not None:
			target_n_obs_rvl = int(np.ceil(N*frac_rvs_obs))
			actual_n_obs_rvl = len(idx_obs_rvl)
			if actual_n_obs_rvl < target_n_obs_rvl:
				print("WARNING: The number of non-missing radial velocities "+
				"is smaller than the target one")
				print("Actual: {0}. Target: {1}".format(
					actual_n_obs_rvl,target_n_obs_rvl))
				print("Try increasing the minimum mass")
			else:
				idx_nan_rvl_extra = np.random.choice(idx_obs_rvl,
									size=actual_n_obs_rvl - target_n_obs_rvl,
									replace=False)

				idx_nan_rvl = np.union1d(idx_nan_rvl,idx_nan_rvl_extra)
			print("The fraction of missing radial velocities is {0:2.2f}: target {1:2.2f}".format(
			float(len(idx_nan_rvl)/N),1.0-frac_rvs_obs))

		unc_as[idx_nan_rvl,5] = 99
		#---------------------------------------------------------------------------------------

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

		if impute_radial_velocity:
			#------ Radial velocity mean and std -------
			mu_rv = np.nanmedian(obs_as[idx_obs_rvl,5])
			sd_rv = 2.*np.nanstd(obs_as[idx_obs_rvl,5])
			#------------------------------------------

			#-- Replace nan rvel by median -------
			obs_as[idx_nan_rvl,5] = mu_rv
			unc_as[idx_nan_rvl,5] = sd_rv
			#--------------------------------
		else:
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
		#----- Gaia -------------------------
		obs_ph = np.empty((N,3))
		for i,mu in true_ph.iterrows():
			obs_ph[i] = rng.multivariate_normal(
					mean=mu,cov=np.diag(unc_ph[i]**2),
					tol=1e-8,method="cholesky",size=1)
		#-----------------------------------------------

		#--------- Additional (including Gaia) ------------------
		obs_bands = np.empty_like(true_bands)
		unc_bands = np.repeat(rp_unc.reshape((-1,1)),
						obs_bands.shape[1],
						axis=1)
		print("WARNING: All isochrone bands will have the same uncertainty as the Gaia_RP band")
		for i,mu in true_bands.iterrows():
			obs_bands[i] = rng.multivariate_normal(
					mean=mu,cov=np.diag(unc_bands[i]**2),
					tol=1e-8,method="cholesky",size=1)
		#-----------------------------------------------

		#--------- Data Frames ------------------
		df_obs_ph = pd.DataFrame(data=obs_ph,
					columns=self.labels_obs_ph)
		df_unc_ph = pd.DataFrame(data=unc_ph,
					columns=self.labels_unc_ph)

		df_obs_bands = pd.DataFrame(data=obs_bands,
					columns=self.labels_obs_bands)
		df_unc_bands = pd.DataFrame(data=unc_bands,
					columns=self.labels_unc_bands)
		df_ph = pd.concat(
			[df_obs_ph,df_unc_ph,df_obs_bands,df_unc_bands],
			ignore_index=False,axis=1)
		#-----------------------------------------

		del df_obs_ph,df_unc_ph,obs_ph,unc_ph
		del df_obs_bands,df_unc_bands,obs_bands,unc_bands
		#===========================================

		#------- Join ------------
		df_obs = df_as.join(df_ph)
		del df_as,df_ph
		#-------------------------

		#--------------- Additional columns -------------
		for key,value in self.additional_columns.items():
			df_obs[key] = value
		#------------------------------------------------

		#------- Set index ------------------
		df_obs.set_index(index,inplace=True)
		#--------------------------------------

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
						index_label="source_id",
						soil_mag_uncertainty=4.0,
						impute_radial_velocity=False,
						g_mag_shift_for_uncertainty=None,
						fraction_radial_velocities_observed=None,
						max_mahalanobis_distance=np.inf,max_n=2):

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
			
		elif hasattr(self,"phasespace_args"):

			#---------- Phase space coordinates --------------
			print("Generating phase-space values ...")
			X = self._generate_phase_space(n_stars=n_stars,
				max_mahalanobis_distance=max_mahalanobis_distance,
				max_n=max_n)
			#------------------------------------------------

		#---------- Phase-space ------------------------------------
		df_ps = pd.DataFrame(data=X,columns=self.labels_phase_space)

		#--------- True astrometry -----------------------
		print("Generating true astrometry ...")
		df_as,distances = self._generate_true_astrometry(X)
		#--------------------------------------------------

		#--------- True photometry -------------------------------
		print("Generating true photometry ...")
		df_ph = self._generate_true_photometry(distances)
		#---------------------------------------------------------

		#---- Join true values ----------------
		df_true = df_as.join(df_ph,how="inner")
		#--------------------------------------

		#------------- Observed values ----------------------------
		print("Generating observed values ...")
		df_obs = self._generate_observed_values(df_true,
		angular_correlations=angular_correlations,
		soil_mag_uncertainty=soil_mag_uncertainty,
		impute_radial_velocity=impute_radial_velocity,
		g_mag_shift_for_uncertainty=g_mag_shift_for_uncertainty,
		fraction_radial_velocities_observed=fraction_radial_velocities_observed)
		#----------------------------------------------------------

		#-------- Join data frames -----------------------------
		df = df_obs.join(df_true).join(df_ps)

		#-------- Reset index -----------------------
		df.reset_index(drop=True,inplace=True)

		self.df = df.copy()

		#----------- Rename columns ---------------
		df.rename(columns=self.mapper,inplace=True)

		#----------- Save data frame ----------------------------
		df.to_csv(path_or_buf=file,index_label=index_label)


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

		#------------- Mass -------------
		plt.figure(figsize=figsize)
		plt.hist(self.df["mass"],bins=50)
		plt.xlabel("Mass [$M_{\\odot}$]")
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------

		#----------- CMDs -----------------------------------
		plt.figure(figsize=figsize)
		plt.scatter(self.df["phot_bp_mean_mag"]-self.df["phot_rp_mean_mag"],
					self.df["phot_g_mean_mag"],
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
		plt.scatter(self.df["phot_g_mean_mag"]-self.df["phot_rp_mean_mag"],
					self.df["phot_g_mean_mag"],
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

	seed      = 0
	n_stars   = 100
	distance  = 200.0
	model     = "PARSEC"
	dir_main  = "/home/jolivares/Repos/Amasijo/Validation/Test/"
	base_name = "{0}_n{1}_d{2}_s{3}".format(model,n_stars,int(distance),seed)
	file_plot = dir_main + base_name + ".pdf"
	file_data = dir_main + base_name + ".csv"
	
	phasespace_args = {
		"position":{"family":"Gaussian",
					"location":np.array([distance,0.0,0.0]),
					"covariance":np.diag([9.,9.,9.])},
		"velocity":{"family":"Gaussian",
					"location":np.array([10.0,10.0,10.0]),
					"covariance":np.diag([1.,1.,1.]),
					"kappa":np.ones(3),
					"omega":np.array([[-1,-1,-1],[1,1,1]])
					}}

	isochrones_args = {
	"model":model,
	"age": 120.0,# [Myr]
	"MIST_args":{"metallicity":0.012,"Av": 0.0},
	"PARSEC_args":{
		"file_mlp":"/home/jolivares/Repos/Huehueti/mlps/PARSEC_10x100/mlp.pkl"},    
	"mass_limits":[0.1,2.5],
	"bands":["G","BP","RP"]
	}

	mcluster_args = {
		"path":"/home/jolivares/Repos/mcluster/",
		"family":"EFF","gamma":5.0,"core_radius":1.0,
		"position+velocity":[0.0,100.0,0.0,0.0,0.0,0.0],
		"truncation_radius":20.0}

	# kalkayotl_file = dir_main + "Cluster_statistics.csv"

	ama = Amasijo(
				phasespace_args=phasespace_args,
				# mcluster_args=mcluster_args,
				# kalkayotl_args={"file":kalkayotl_file},
				isochrones_args=isochrones_args,
				seed=seed)
	ama.generate_cluster(file_data,n_stars=n_stars,angular_correlations=None)
	ama.plot_cluster(file_plot=file_plot)











