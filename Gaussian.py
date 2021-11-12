import numpy as np
from Amasijo import Amasijo

dir_main =  "/home/jolivares/Repos/Kalkayotl/article/v2.0/Validation/Gaussian/"
name     = "Gaussian"
seeds    = [0,1,2,3,4,5,6,7,8,9]
loc      = 1000.0
scl      = 5.0
n_stars  = 100

astrometric_args = {
	"position":{
			"family":"Gaussian",
			"loc":np.array([0.0,loc,0.0]),
			"scl":np.diag([scl**2,scl**2,scl**2])},
	"velocity":{
			"family":"Gaussian",
			"loc":np.array([0.0,10.0,0.0]),
			"scl":np.diag([1.,1.,1.])}
}
photometric_args = {
		"log_age": 8.2,     # Solar metallicity
		"metallicity":0.02, # Typical value of Bossini+2019
		"Av": 0.0,          # No extinction
		"mass_limit":4.0,   # Avoids NaNs in photometry
		"bands":["V","I","G","BP","RP"]
}

for seed in seeds:
	file_name = dir_main + "{0}_s{1}_n{2}_loc{3}_scl{4}".format(
							name,seed,n_stars,int(loc),int(scl))

	ama = Amasijo(astrometric_args=astrometric_args,
				  photometric_args=photometric_args,
				  seed=seed)

	ama.generate_cluster(file_name+".csv",n_stars=n_stars,m_factor=4)

	ama.plot_cluster(file_plot=file_name+".pdf")